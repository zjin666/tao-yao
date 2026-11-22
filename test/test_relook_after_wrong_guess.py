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


def make_group(tmp, ages=(60,), uids=None):
    """在群 1 的图片目录里放若干张图（ages 是各自多少秒前发的，新到旧排列写进 index.csv）。"""
    base = Path(tmp) / "group" / "1"
    base.mkdir(parents=True)
    lines = ["time,user_id,group_id,message_id,path"]
    paths = []
    for i, age in enumerate(sorted(ages, reverse=True)):      # index.csv 里旧的在前
        path = base / f"2026112{i}-000000_7000{i}_m{i}.bin"
        Image.new("RGB", (400, 300), (200, 200, 50)).save(path, "PNG")
        os.utime(path, (time.time() - age, time.time() - age))
        sender = uids[i] if uids else f"7000{i}"
        lines.append(f"2026112{i}-000000,{sender},1,m{i},{path.name}")
        paths.append(path)
    (base / "index.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def relook_bot(tmp, judge='{"kind":"wrong","pick":1}', features="橙金渐变长发，黑色星形头冠，紫色心形宝石", named="像荧（原神）",
               ages=(60,), extra_chat=(), uids=None):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {"relook": {"timeout_seconds": 5}}
    bot.profiles_enabled = True
    bot.profiles = SimpleNamespace(texts_by_message_ids=AsyncMock(return_value={
        "m0": "[群友] 图\n[图片内容] 昨天那张：金发少女", "m1": "[群友] 图\n[图片内容] 今天这张：猫"}))
    paths = make_group(tmp, ages, uids)
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
            self.assertIn("1. 像荧（原神）", note)             # Qwen 直接看图，问了两次
            self.assertIn("2. 像荧（原神）", note)
            self.assertIn("1. 某角色", note)                  # 搜索结果
            self.assertIn("不要重复已经被否定的名字", note)
            self.assertIn("两次直接看图的判断指向同一个角色", note)
            self.assertEqual(bot.character_identifier.identify.await_count, 2)
            self.assertIn("爱丽丝", bot.character_identifier.identify.await_args.args[1])     # 已被否定的猜测
            bot.vision.cleanup.assert_called_once()

    async def test_the_same_image_is_only_looked_at_again_once_after_it_worked(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp, extra_chat=("x",))
            bot.utility_llm.chat = AsyncMock(side_effect=['{"kind":"wrong","pick":1}', "搜索词", '{"kind":"wrong","pick":1}'])
            await bot._relook_note("可惜不是", HISTORY, 7, True, 1)
            self.assertEqual(await bot._relook_note("还是不对", HISTORY, 7, True, 1), "")
            self.assertEqual(bot.character_identifier.identify.await_count, 2)      # 只有第一次做了（问两次）

    async def test_it_searches_by_features_alone_and_by_candidate_at_the_same_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp)
            bot.utility_llm.chat = AsyncMock(side_effect=[
                '{"kind":"wrong","pick":1}', "黑色星形头冠 紫色宝石 金发 角色" + chr(10) + "荧 原神 黑色星形头冠"])
            note = await bot._relook_note("可惜不是", HISTORY, 7, True, 1)
            asked = sorted(call.args[0] for call in bot.web_search.search.await_args_list)
            self.assertEqual(asked, ["荧 原神 黑色星形头冠", "黑色星形头冠 紫色宝石 金发 角色"])     # 一条不带名字，一条带候选名字
            self.assertIn("只按外观特征搜到的：", note)
            self.assertIn("按候选名字核对搜到的：", note)

    async def test_without_a_qwen_answer_the_note_still_has_the_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp, named="")
            note = await bot._relook_note("可惜不是", HISTORY, 7, True, 1)
            self.assertIn("黑色星形头冠", note)
            self.assertNotIn("直接看图的判断（可能有误", note)

    async def test_the_model_decides_by_meaning_and_nothing_happens_when_it_says_no(self):
        for judge in ('{"kind":"none","pick":0}', '{"kind":"wrong","pick":0}', "不是json"):
            with tempfile.TemporaryDirectory() as tmp:
                bot, _ = relook_bot(tmp, judge=judge)
                self.assertEqual(await bot._relook_note("你认得出来吗", HISTORY, 7, True, 1), "", judge)
                bot.character_identifier.identify.assert_not_awaited()

    async def test_candidates_carry_the_image_or_sticker_description_so_the_model_can_tell_them_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, paths = relook_bot(tmp, ages=(100, 50))
            bot.profiles.texts_by_message_ids = AsyncMock(return_value={
                "m0": "[群友] 图\n[图片内容] 猜图题：粉发少女", "m1": "[群友] 图\n[表情包内容] 形象：蓝发女仆"})
            candidates = await bot._relook_candidates(7, True, 1, 3600, limit=5)
            self.assertEqual([c["desc"] for c in candidates], ["表情包：形象：蓝发女仆", "猜图题：粉发少女"])   # 新到旧
            self.assertEqual(len(await bot._relook_candidates(7, True, 1, 3600, limit=1)), 1)

    async def test_without_any_saved_image_no_model_is_even_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp, ages=(10 ** 7,))
            self.assertEqual(await bot._relook_note("再试一试", HISTORY, 7, True, 1), "")
            bot.utility_llm.chat.assert_not_awaited()
            bot.vision.prepare_relook.assert_not_awaited()

    async def test_failures_are_retried_at_most_twice_and_never_break_the_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, _ = relook_bot(tmp, features="", named="")
            bot.utility_llm.chat = AsyncMock(return_value='{"kind":"wrong","pick":1}')
            for _ in range(4):
                self.assertEqual(await bot._relook_note("可惜不是", HISTORY, 7, True, 1), "")
            self.assertEqual(bot.vision.describe_for_relook.await_count, 2)


class RetryRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_any_wording_of_asking_again_works_and_the_model_can_pick_an_older_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 候选按新到旧排：1 是 60 秒前的，2 是 100000 秒前的
            bot, paths = relook_bot(tmp, judge='{"kind":"retry","pick":2}', ages=(100000, 60))
            note = await bot._relook_note("我相信你可以的", HISTORY, 7, True, 1)       # 没有任何“再、试、认”之类的字
            self.assertIn("对方要求你再认一次图里的角色", note)
            self.assertEqual(bot.vision.prepare_relook.await_args.args[0], paths[0])        # 选了较旧的那张
            self.assertNotIn("爱丽丝", str(bot.character_identifier.identify.await_args))   # 再试一次时不带“被否定的猜测”

    async def test_the_models_pick_is_honored_even_when_the_speaker_sent_an_image_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 发言的人 7 自己有一张较旧的图；模型认为他说的是别人刚发的那张，就按模型的来，不靠“谁发的”这种规则
            bot, paths = relook_bot(tmp, judge='{"kind":"retry","pick":1}', ages=(120, 30), uids=["7", "9"])
            await bot._relook_note("继续猜这张图", HISTORY, 7, True, 1)
            self.assertEqual(bot.vision.prepare_relook.await_args.args[0], paths[1])
            asked = bot.utility_llm.chat.await_args_list[0].args[0][1]["content"]
            self.assertIn("现在说话的人自己发的图", asked)          # 只是提供信息，不是硬规则
            self.assertIn("不要只因为某张图最新就选它", asked)
            self.assertEqual(bot.utility_llm.chat.await_count, 2)   # 一次判断（意图加选图）+ 一次搜索词

    async def test_a_vague_retry_is_resolved_from_the_conversation_not_only_the_sentence(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot, paths = relook_bot(tmp, judge='{"kind":"retry","pick":1}', ages=(100000, 90000))
            talk = [{"role": "user", "content": "[哥哥] 再试一试，能不能认出来昨天那张图里的角色"},
                    {"role": "assistant", "content": "金发星形发饰加黑黄上衣白裙，我没对上号"}]
            await bot._relook_note("我相信你可以的，再试一试", talk, 7, True, 1)
            asked = bot.utility_llm.chat.await_args_list[0].args[0][1]["content"]
            self.assertIn("金发星形发饰加黑黄上衣白裙", asked)                         # 带上了最近对话
            self.assertIn("昨天那张图里的角色", asked)
            self.assertEqual(bot.vision.prepare_relook.await_args.args[0], paths[1])   # 编号 1 是列表里最新的那张


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

    async def test_relook_uses_the_whole_image_downscaled_without_cropping_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.describer(tmp)
            src = Path(tmp) / "a.png"
            Image.new("RGB", (3840, 2088), (10, 10, 10)).save(src)
            seen = []
            prompts = []

            async def fake(paths, prompt):
                prompts.append(prompt)
                seen.append(Image.open(paths[0]).size)
                return "特征描述"
            d._api_describe = fake
            self.assertEqual(await d.relook(src), "特征描述")
            self.assertEqual(len(prompts), 1)                      # 不再多一次“框出角色位置”的调用
            width, height = seen[0]
            self.assertEqual(max(width, height), 1280)             # 降低像素：长边 1280
            self.assertAlmostEqual(width / height, 3840 / 2088, places=1)   # 整张图，比例不变，没有裁剪
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["a.png"])   # 临时文件已清理

    async def test_the_longest_side_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.describer(tmp)
            src = Path(tmp) / "a.png"
            Image.new("RGB", (3000, 2000), (10, 10, 10)).save(src)
            prepared, temporary = await d.prepare_relook(src, 800)
            try:
                self.assertEqual(max(Image.open(prepared).size), 800)
            finally:
                d.cleanup(temporary)


if __name__ == "__main__":
    unittest.main()
