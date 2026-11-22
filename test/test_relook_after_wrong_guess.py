"""对方指出她认图里的角色认错了、或要求再试一次：原图再仔细看一遍，Qwen 直接看图认角色，再按特征搜一次。"""
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

from taoyao import identify as identify_module
from taoyao.app import TaoYaoBot
from taoyao.identify import CharacterIdentifier
from taoyao.vision import MediaDescriber

HISTORY = [{"role": "user", "content": "[群友] 现在能不能认出来\n[图片内容] 动漫少女"},
           {"role": "assistant", "content": "是东方Project里的爱丽丝吧？"}]


def make_group(tmp, ages=(60,)):
    """在群 1 的图片目录里放若干张图（ages 是各自多少秒前发的，新到旧排列写进 index.csv）。"""
    base = Path(tmp) / "group" / "1"
    base.mkdir(parents=True)
    lines = ["time,user_id,group_id,message_id,path"]
    paths = []
    for i, age in enumerate(sorted(ages, reverse=True)):      # index.csv 里旧的在前
        path = base / f"2026112{i}-000000_7000{i}_m{i}.bin"
        Image.new("RGB", (400, 300), (200, 200, 50)).save(path, "PNG")
        os.utime(path, (time.time() - age, time.time() - age))
        lines.append(f"2026112{i}-000000,7000{i},1,m{i},{path.name}")
        paths.append(path)
    (base / "index.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def relook_bot(tmp, judge='{"kind":"wrong"}', features="橙金渐变长发，黑色星形头冠，紫色心形宝石", named="像荧（原神）",
               ages=(60,), extra_chat=()):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {"relook": {"timeout_seconds": 5}}
    bot.profiles_enabled = True
    bot.profiles = SimpleNamespace(texts_by_message_ids=AsyncMock(return_value={
        "m0": "[群友] 图\n[图片内容] 昨天那张：金发少女", "m1": "[群友] 图\n[图片内容] 今天这张：猫"}))
    paths = make_group(tmp, ages)
    prepared = Path(tmp) / "prepared.jpg"
    prepared.write_bytes(b"x")
    bot.vision = SimpleNamespace(work_dir=Path(tmp), prepare_relook=AsyncMock(return_value=(prepared, [])),
                                 describe_for_relook=AsyncMock(return_value=features), cleanup=Mock())
    bot.character_identifier = SimpleNamespace(enabled=True, identify=AsyncMock(return_value=named))
    bot.utility_llm = SimpleNamespace(chat=AsyncMock(side_effect=[judge, *extra_chat, "黑色星形头冠 紫色心形宝石 角色"]))
    bot.llm = bot.utility_llm
    bot.web_search = SimpleNamespace(search=AsyncMock(return_value=["r"]),
                                     format_results=lambda q, r: f"查询：{q}\n1. 某角色")
    return bot, paths


class WrongGuessTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_wrong_guess_triggers_a_second_look_with_qwen_and_a_new_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, paths = relook_bot(tmp)
            note = await bot._relook_note("可惜不是", HISTORY, 7, True, 1)
            self.assertIn("【再次观察】", note)
            self.assertIn("黑色星形头冠", note)              # DeepSeek 逐项描述的特征
            self.assertIn("直接看图的判断（可能有误）：像荧（原神）", note)   # Qwen 直接看图
            self.assertIn("1. 某角色", note)                  # 搜索结果
            self.assertIn("不要重复已经被否定的名字", note)
            bot.character_identifier.identify.assert_awaited_once()
            self.assertIn("爱丽丝", bot.character_identifier.identify.await_args.args[1])     # 已被否定的猜测
            bot.vision.cleanup.assert_called_once()

    async def test_the_same_image_is_only_looked_at_again_once_after_it_worked(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp)
            await bot._relook_note("可惜不是", HISTORY, 7, True, 1)
            self.assertEqual(await bot._relook_note("还是不对", HISTORY, 7, True, 1), "")
            bot.character_identifier.identify.assert_awaited_once()

    async def test_without_a_qwen_answer_the_note_still_has_the_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp, named="")
            note = await bot._relook_note("可惜不是", HISTORY, 7, True, 1)
            self.assertIn("黑色星形头冠", note)
            self.assertNotIn("直接看图的判断（可能有误）", note)

    async def test_nothing_happens_for_ordinary_talk_the_first_question_or_a_no_from_the_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp)
            self.assertEqual(await bot._relook_note("你认得出来吗", HISTORY, 7, True, 1), "")      # 第一次问：不触发
            self.assertEqual(await bot._relook_note("哈哈好看", HISTORY, 7, True, 1), "")
            self.assertEqual(await bot._relook_note("可惜不是", [{"role": "user", "content": "你好"}], 7, True, 1), "")
            bot.utility_llm.chat.assert_not_awaited()
            bot.character_identifier.identify.assert_not_awaited()
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp, judge='{"kind":"none"}')
            self.assertEqual(await bot._relook_note("不是在说这个", HISTORY, 7, True, 1), "")
            bot.character_identifier.identify.assert_not_awaited()

    async def test_failures_are_retried_at_most_twice_and_never_break_the_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp, features="", named="", extra_chat=())
            bot.utility_llm.chat = AsyncMock(return_value='{"kind":"wrong"}')
            for _ in range(4):
                self.assertEqual(await bot._relook_note("可惜不是", HISTORY, 7, True, 1), "")
            self.assertEqual(bot.vision.describe_for_relook.await_count, 2)


class RetryRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_retry_request_works_without_any_correction_word_and_picks_the_older_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, paths = relook_bot(tmp, judge='{"kind":"retry"}', ages=(100000, 60), extra_chat=('{"pick":2}',))
            note = await bot._relook_note("再试一试，能不能认出来昨天那张图里的角色", HISTORY, 7, True, 1)
            self.assertIn("对方要求你再认一次图里的角色", note)
            self.assertEqual(bot.vision.prepare_relook.await_args.args[0], paths[0])        # 选了昨天那张
            self.assertNotIn("爱丽丝", str(bot.character_identifier.identify.await_args))   # 再试一次时不带“被否定的猜测”

    async def test_a_retry_without_a_time_word_uses_the_newest_recent_image_without_asking_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, paths = relook_bot(tmp, judge='{"kind":"retry"}', ages=(100000, 60))
            await bot._relook_note("你再试试看呢？", [{"role": "user", "content": "x"}], 7, True, 1)
            self.assertEqual(bot.vision.prepare_relook.await_args.args[0], paths[1])
            self.assertEqual(bot.utility_llm.chat.await_count, 2)       # 意图判断 + 搜索词，没有多出“选图”这次

    async def test_no_saved_image_means_nothing_to_look_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp, judge='{"kind":"retry"}', ages=(10 ** 7,))
            self.assertEqual(await bot._relook_note("再试一试", HISTORY, 7, True, 1), "")
            bot.vision.prepare_relook.assert_not_awaited()


class CharacterIdentifierTests(unittest.IsolatedAsyncioTestCase):
    def config(self, tmp, **relook):
        key = Path(tmp) / "k.txt"
        key.write_text("secret", encoding="utf-8")
        return {"qwen": {"api_base": "http://x/v1", "model": "qwen-test", "key_file": str(key)}, "relook": relook}

    async def test_it_sends_the_image_and_the_rejected_guess_and_returns_the_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            Image.new("RGB", (50, 50)).save(image)
            identifier = CharacterIdentifier(self.config(tmp))
            captured = {}

            async def fake(client, url, headers, payload):
                captured.update(url=url, payload=payload)
                return {"choices": [{"message": {"content": " 像荧 "}}]}
            with patch.object(identify_module, "collect_stream", fake):
                self.assertEqual(await identifier.identify(image, "爱丽丝"), "像荧")
            content = captured["payload"]["messages"][0]["content"]
            self.assertIn("爱丽丝", content[0]["text"])
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/"))
            self.assertEqual(captured["payload"]["model"], "qwen-test")

    async def test_it_prefers_the_stronger_model_with_thinking_off_and_falls_back_to_the_main_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.jpg"
            Image.new("RGB", (50, 50)).save(image)
            identifier = CharacterIdentifier(self.config(tmp, identify_model="big-model"))
            self.assertEqual(identifier.models, ["big-model", "qwen-test"])
            tried = []

            async def fake(client, url, headers, payload):
                tried.append((payload["model"], payload["enable_thinking"]))
                if payload["model"] == "big-model":
                    raise RuntimeError("404")
                return {"choices": [{"message": {"content": "主模型的答案"}}]}
            with patch.object(identify_module, "collect_stream", fake):
                self.assertEqual(await identifier.identify(image), "主模型的答案")
            self.assertEqual(tried, [("big-model", False), ("qwen-test", False)])

    async def test_it_is_off_without_a_key_or_when_disabled_and_failures_return_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(CharacterIdentifier({"qwen": {"key_file": str(Path(tmp) / "none.txt"), "model": "m"}}).enabled)
            self.assertFalse(CharacterIdentifier(self.config(tmp, identify=False)).enabled)
            identifier = CharacterIdentifier(self.config(tmp))
            image = Path(tmp) / "a.jpg"
            Image.new("RGB", (50, 50)).save(image)
            with patch.object(identify_module, "collect_stream", AsyncMock(side_effect=RuntimeError("down"))):
                self.assertEqual(await identifier.identify(image), "")


class VisionRelookTests(unittest.IsolatedAsyncioTestCase):
    def describer(self, tmp):
        d = MediaDescriber({"enabled": True, "provider": "api", "api_base": "http://x", "model": "m",
                            "work_dir": tmp})
        d._api_ready = lambda: True
        return d

    def test_crop_uses_the_original_resolution_with_a_margin_and_rejects_tiny_areas(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.png"
            Image.new("RGB", (3000, 2000), (10, 10, 10)).save(src)
            out = Path(tmp) / "c.png"
            self.assertTrue(MediaDescriber._crop_original(src, [200, 100, 700, 900], out))
            width, height = Image.open(out).size
            self.assertGreater(width, 1500)                 # 取自原图 3000 像素宽，而不是缩略图
            self.assertFalse(MediaDescriber._crop_original(src, [0, 0, 120, 120], Path(tmp) / "d.png"))

    async def test_relook_crops_the_found_region_then_describes_it_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.describer(tmp)
            src = Path(tmp) / "a.png"
            Image.new("RGB", (3000, 2000), (10, 10, 10)).save(src)
            seen = []

            async def fake(paths, prompt):
                if "千分比" in prompt:
                    return '{"box":[200,100,700,900]}'
                seen.append(Image.open(paths[0]).size)
                return "特征描述"
            d._api_describe = fake
            self.assertEqual(await d.relook(src), "特征描述")
            self.assertEqual(len(seen), 1)
            self.assertLess(seen[0][0], 3000)               # 是裁出来的区域，不是整图
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["a.png"])   # 临时文件已清理

    async def test_relook_falls_back_to_the_whole_image_when_no_region_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.describer(tmp)
            src = Path(tmp) / "a.png"
            Image.new("RGB", (800, 600), (10, 10, 10)).save(src)

            async def fake(paths, prompt):
                return '{"box":null}' if "千分比" in prompt else "整图特征"
            d._api_describe = fake
            self.assertEqual(await d.relook(src), "整图特征")


if __name__ == "__main__":
    unittest.main()
