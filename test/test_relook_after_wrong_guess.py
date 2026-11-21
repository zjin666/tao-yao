"""对方指出她认图里的角色认错了：把原图再仔细看一遍，再按新特征搜一次；每张图只做一次。"""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from taoyao.app import TaoYaoBot
from taoyao.vision import MediaDescriber

HISTORY = [{"role": "user", "content": "[群友] 现在能不能认出来\n[图片内容] 动漫少女"},
           {"role": "assistant", "content": "是东方Project里的爱丽丝吧？"}]


def relook_bot(tmp, judge='{"wrong":true}', features="橙金渐变长发，黑色星形头冠，紫色心形宝石", image=True):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {"relook": {"timeout_seconds": 5}}
    bot.profiles_enabled = True
    work = Path(tmp)
    (work / "group" / "1").mkdir(parents=True)
    if image:
        Image.new("RGB", (400, 300), (200, 200, 50)).save(work / "group" / "1" / "20261122-000553_7_9.bin", "PNG")
    bot.vision = SimpleNamespace(work_dir=work, relook=AsyncMock(return_value=features))
    bot.utility_llm = SimpleNamespace(chat=AsyncMock(side_effect=[judge, "黑色星形头冠 紫色心形宝石 橙金渐变长发 角色"]))
    bot.llm = bot.utility_llm
    bot.web_search = SimpleNamespace(search=AsyncMock(return_value=["r"]),
                                     format_results=lambda q, r: f"查询：{q}\n1. 某角色")
    return bot


class RelookGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_wrong_guess_triggers_a_careful_second_look_and_a_new_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = relook_bot(tmp)
            note = await bot._relook_note("可惜不是", HISTORY, 7, True, 1)
            self.assertIn("【再次观察】", note)
            self.assertIn("黑色星形头冠", note)
            self.assertIn("1. 某角色", note)
            self.assertIn("不要再重复刚才猜错的名字", note)
            bot.vision.relook.assert_awaited_once()
            bot.web_search.search.assert_awaited_once()

    async def test_the_same_image_is_only_looked_at_again_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = relook_bot(tmp)
            await bot._relook_note("可惜不是", HISTORY, 7, True, 1)
            self.assertEqual(await bot._relook_note("还是不对", HISTORY, 7, True, 1), "")
            bot.vision.relook.assert_awaited_once()

    async def test_nothing_happens_without_a_correction_a_recent_image_or_when_the_judge_says_no(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = relook_bot(tmp)
            self.assertEqual(await bot._relook_note("哈哈好看", HISTORY, 7, True, 1), "")             # 不是指出认错
            self.assertEqual(await bot._relook_note("可惜不是", [{"role": "user", "content": "你好"}], 7, True, 1), "")  # 上下文没有图
            bot.vision.relook.assert_not_awaited()
        with tempfile.TemporaryDirectory() as tmp:
            bot = relook_bot(tmp, judge='{"wrong":false}')
            self.assertEqual(await bot._relook_note("不是在说这个", HISTORY, 7, True, 1), "")        # 否定别的话题
            bot.vision.relook.assert_not_awaited()
        with tempfile.TemporaryDirectory() as tmp:
            bot = relook_bot(tmp, image=False)
            self.assertEqual(await bot._relook_note("可惜不是", HISTORY, 7, True, 1), "")            # 找不到原图
            bot.vision.relook.assert_not_awaited()

    async def test_failures_do_not_break_the_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = relook_bot(tmp, features="")
            self.assertEqual(await bot._relook_note("可惜不是", HISTORY, 7, True, 1), "")
        with tempfile.TemporaryDirectory() as tmp:
            bot = relook_bot(tmp)
            bot.web_search.search = AsyncMock(side_effect=RuntimeError("down"))
            note = await bot._relook_note("可惜不是", HISTORY, 7, True, 1)
            self.assertIn("没有搜到可用结果", note)               # 搜索失败时仍给出看到的特征


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
