import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image, ImageDraw

from taoyao.app import TaoYaoBot
from taoyao.sticker_memory import StickerMemory


def make_image(path: Path, shift: int = 0, size: int = 128) -> Path:
    img = Image.new("RGB", (size, size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse((20 + shift, 20, 100 + shift, 100), fill=(30, 30, 30))
    draw.rectangle((70, 80 + shift, 120, 120), fill=(200, 80, 80))
    img.save(path)
    return path


class StickerMemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.memory = StickerMemory({"db": str(self.root / "s.db")})

    def tearDown(self):
        self.memory.close()
        self.folder.cleanup()

    def test_same_image_is_recognised_and_counted(self):
        first = make_image(self.root / "a.png")
        fp = self.memory.fingerprint(first)
        self.assertIsNone(self.memory.lookup("private:1", fp))
        sticker_id = self.memory.remember("private:1", 1, fp, "灰色圆脸角色")
        # 重新保存（体积/编码不同）后依然认得出
        again = self.root / "b.jpg"
        Image.open(first).save(again, quality=70)
        hit = self.memory.lookup("private:1", self.memory.fingerprint(again))
        self.assertEqual(hit["id"], sticker_id)
        self.assertEqual(hit["seen_count"], 2)

    def test_different_image_and_other_scope_do_not_match(self):
        first = make_image(self.root / "a.png")
        other = Image.new("RGB", (128, 128), (255, 255, 255))
        ImageDraw.Draw(other).line((0, 0, 127, 127), fill=(0, 0, 0), width=9)
        other.save(self.root / "o.png")
        fp = self.memory.fingerprint(first)
        self.memory.remember("group:9", 1, fp, "灰色圆脸角色")
        self.assertIsNone(self.memory.lookup("group:9", self.memory.fingerprint(self.root / "o.png")))
        self.assertIsNone(self.memory.lookup("private:1", fp))   # 群里的记录不带进私聊

    def test_meaning_is_stored_with_evidence_and_shown_in_note(self):
        fp = self.memory.fingerprint(make_image(self.root / "a.png"))
        sticker_id = self.memory.remember("private:1", 1, fp, "灰褐色头发的角色")
        self.assertTrue(self.memory.set_meaning(sticker_id, 1, "表达开心的心情", "不是，只是表达我开心的心情"))
        hit = self.memory.lookup("private:1", fp)
        note = StickerMemory.note(hit, [], 1)
        self.assertIn("已经认识", note)
        self.assertNotIn("见过", note)
        self.assertIn("不要向对方提起", note)
        self.assertIn("表达开心的心情", note)

    def test_taught_meaning_is_offered_for_a_new_picture_of_same_sender(self):
        fp = self.memory.fingerprint(make_image(self.root / "a.png"))
        sticker_id = self.memory.remember("private:1", 1, fp, "灰褐色头发的角色")
        self.memory.set_meaning(sticker_id, 1, "表达开心的心情", "只是表达我开心")
        self.assertEqual(self.memory.taught_by("private:2", 2), [])
        taught = self.memory.taught_by("private:1", 1, exclude_id=999)
        note = StickerMemory.note(None, taught, 1)
        self.assertIn("表达开心的心情", note)
        self.assertIn("同一个角色", note)

    def test_pending_requires_explanation_hint_and_expires(self):
        self.memory.mark_pending("private:1", 1, 5)
        self.assertEqual(self.memory.take_pending("private:1", 1, "在做腿举"), 0)
        self.assertEqual(self.memory.take_pending("private:1", 1, "只是表达我开心的心情"), 5)
        self.memory.pending_seconds = -1
        self.assertEqual(self.memory.take_pending("private:1", 1, "这个表情是开心"), 0)

    def test_scope_is_bounded(self):
        self.memory.max_per_scope = 10
        for i in range(15):
            self.memory.remember("g", 1, f"{i:016x}", f"d{i}")
        count = self.memory._conn.execute("SELECT COUNT(*) FROM stickers WHERE scope='g'").fetchone()[0]
        self.assertEqual(count, 10)


class StickerLearningTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, reply):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.sticker_memory = StickerMemory({"db": str(Path(folder.name) / "s.db")})
        self.addCleanup(bot.sticker_memory.close)
        bot._sticker_tasks = set()
        bot.utility_llm = bot.llm = SimpleNamespace(chat=AsyncMock(return_value=reply))
        return bot

    async def test_explanation_after_sticker_is_learned_in_background(self):
        bot = self.make_bot('{"meaning":"表达开心的心情"}')
        memory = bot.sticker_memory
        sticker_id = memory.remember("private:7", 7, "0" * 16, "灰褐色头发的角色")
        memory.mark_pending("private:7", 7, sticker_id)
        data = {"message_type": "private", "user_id": 7}
        bot._watch_sticker_explanation(data, "不是，只是表达我开心的心情")
        await asyncio.gather(*list(bot._sticker_tasks))
        self.assertEqual(memory.lookup("private:7", "0" * 16)["meaning"], "表达开心的心情")
        self.assertEqual(memory.take_pending("private:7", 7, "只是"), 0)   # 学到后不再等

    async def test_unrelated_chat_is_not_learned(self):
        bot = self.make_bot('{"meaning":""}')
        memory = bot.sticker_memory
        sticker_id = memory.remember("private:7", 7, "0" * 16, "灰褐色头发的角色")
        memory.mark_pending("private:7", 7, sticker_id)
        bot._watch_sticker_explanation({"message_type": "private", "user_id": 7}, "在做腿举")
        bot.utility_llm.chat.assert_not_awaited()          # 前置过滤直接挡掉，不调模型
        bot._watch_sticker_explanation({"message_type": "private", "user_id": 7}, "这是我今天第一组")
        await asyncio.gather(*list(bot._sticker_tasks))
        self.assertEqual(memory.lookup("private:7", "0" * 16)["meaning"], "")

    async def test_other_senders_message_does_not_teach(self):
        bot = self.make_bot('{"meaning":"乱写"}')
        memory = bot.sticker_memory
        sticker_id = memory.remember("group:1", 7, "0" * 16, "角色")
        memory.mark_pending("group:1", 7, sticker_id)
        bot._watch_sticker_explanation({"message_type": "group", "group_id": 1, "user_id": 8}, "这个表情的意思是乱写")
        self.assertFalse(bot._sticker_tasks)


class StickerContextTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, reply='{"guess":"撒娇求饶"}'):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.sticker_memory = StickerMemory({"db": str(root / "s.db")})
        self.addCleanup(bot.sticker_memory.close)
        bot._sticker_tasks = set()
        bot.utility_llm = bot.llm = SimpleNamespace(chat=AsyncMock(return_value=reply))
        bot.vision = SimpleNamespace(
            enabled=True,
            fetch_segment=AsyncMock(return_value=(make_image(root / "p.png"), False)),
            describe=AsyncMock(return_value="形象：灰色圆脸猫；可能想表达：委屈"))
        bot.user_memes = SimpleNamespace(maybe_store=lambda *a, **k: False)
        bot.memes = SimpleNamespace(_keyword_match=lambda text: "")
        bot._remember_private_image = lambda *a, **k: None
        bot._describe_kept_image = lambda *a, **k: None
        return bot

    async def test_previous_line_goes_to_vision_and_next_line_refines_guess(self):
        bot = self.make_bot()
        data = {"message_type": "private", "user_id": 7, "message": [{"type": "mface", "data": {}}]}
        bot._watch_sticker_explanation(data, "你怎么还不回我")          # 上一句
        await bot._media_text(data)
        self.assertEqual(bot.vision.describe.await_args.args[2], "你怎么还不回我")
        bot._watch_sticker_explanation({"message_type": "private", "user_id": 7}, "好啦好啦我错了")   # 下一句
        await asyncio.gather(*list(bot._sticker_tasks))
        user_prompt = bot.utility_llm.chat.await_args.args[0][1]["content"]
        self.assertIn("你怎么还不回我", user_prompt)
        self.assertIn("好啦好啦我错了", user_prompt)
        self.assertIn("可能想表达：撒娇求饶", bot.sticker_memory.lookup("private:7", bot.sticker_memory.fingerprint(
            make_image(Path(tempfile.gettempdir()) / "same.png")))["description"])

    async def test_only_the_adjacent_lines_are_used_and_only_once(self):
        bot = self.make_bot()
        data = {"message_type": "private", "user_id": 7, "message": [{"type": "mface", "data": {}}]}
        await bot._media_text(data)
        bot._watch_sticker_explanation({"message_type": "private", "user_id": 7}, "第一句")
        bot._watch_sticker_explanation({"message_type": "private", "user_id": 7}, "第二句")
        await asyncio.gather(*list(bot._sticker_tasks))
        self.assertEqual(bot.utility_llm.chat.await_count, 1)     # 只看紧邻的一句，其余上下文不看

    async def test_missing_before_line_is_left_out_not_sent_empty(self):
        import json
        bot = self.make_bot()
        data = {"message_type": "private", "user_id": 7, "message": [{"type": "mface", "data": {}}]}
        await bot._media_text(data)                     # 之前没有任何聊天
        bot._watch_sticker_explanation({"message_type": "private", "user_id": 7}, "好啦我错了")
        await asyncio.gather(*list(bot._sticker_tasks))
        sent = json.loads(bot.utility_llm.chat.await_args.args[0][1]["content"])
        self.assertNotIn("before", sent)
        self.assertEqual(sent["after"], "好啦我错了")

    async def test_no_next_line_means_no_refinement(self):
        bot = self.make_bot()
        data = {"message_type": "private", "user_id": 7, "message": [{"type": "mface", "data": {}}]}
        bot._watch_sticker_explanation(data, "你怎么还不回我")
        await bot._media_text(data)
        await asyncio.gather(*list(bot._sticker_tasks))
        bot.utility_llm.chat.assert_not_awaited()
        self.assertIn("可能想表达：委屈", bot.sticker_memory.description_of(1))   # 保持原猜想

    async def test_large_image_sent_as_picture_is_classified_by_the_model(self):
        bot = self.make_bot()
        big = make_image(Path(tempfile.gettempdir()) / "big_sticker.png", size=1078)   # 以图片发来、尺寸很大
        bot.vision.fetch_segment = AsyncMock(return_value=(big, False))
        bot.vision.describe = AsyncMock(return_value="表情包|形象：灰褐卷发角色；可能想表达：开心")
        data = {"message_type": "private", "user_id": 7, "message": [{"type": "image", "data": {}}]}
        first = await bot._media_text(data)
        self.assertEqual(bot.vision.describe.await_args.args[1], "image_or_sticker")
        self.assertIn("形象：灰褐卷发角色；可能想表达：开心", first)
        self.assertNotIn("表情包|", first)                  # 分类标记不进入文本
        self.assertEqual(bot.sticker_memory.take_pending("private:7", 7, "代表我很开心"), 1)
        await bot._media_text(data)
        self.assertEqual(bot.vision.describe.await_count, 1)   # 第二次直接认出，不再识图

    async def test_echoed_prompt_opening_is_not_stored(self):
        bot = self.make_bot()
        bot.vision.describe = AsyncMock(
            return_value="这是聊天里发的表情包。形象：羊角萌系角色；可能想表达：卖萌撒娇")
        data = {"message_type": "private", "user_id": 7, "message": [{"type": "mface", "data": {}}]}
        text = await bot._media_text(data)
        self.assertNotIn("这是聊天里发的表情包", text)
        self.assertTrue(bot.sticker_memory.description_of(1).startswith("形象：羊角萌系角色"))
        # 以图片发来、模型打了标记后同样抄回开头的情况
        bot.vision.describe = AsyncMock(
            return_value="表情包|这是聊天里发的表情包。形象：动漫少女；可能想表达：慌张")
        other = Path(tempfile.gettempdir()) / "other_sticker.png"
        img = Image.new("RGB", (128, 128), (255, 255, 255))
        ImageDraw.Draw(img).line((0, 0, 127, 127), fill=(0, 0, 0), width=9)
        img.save(other)
        bot.vision.fetch_segment = AsyncMock(return_value=(other, False))
        data["message"] = [{"type": "image", "data": {}}]
        text = await bot._media_text(data)
        self.assertNotIn("这是聊天里发的表情包", text)
        self.assertIn("形象：动漫少女", text)

    async def test_ordinary_photo_is_described_normally_and_not_remembered(self):
        bot = self.make_bot()
        bot.vision.describe = AsyncMock(return_value="一张傍晚的街道照片，路边停着几辆自行车。")
        data = {"message_type": "private", "user_id": 7, "message": [{"type": "image", "data": {}}]}
        text = await bot._media_text(data)
        self.assertIn("傍晚的街道照片", text)
        self.assertEqual(bot.sticker_memory.take_pending("private:7", 7, "这是我拍的"), 0)
        count = bot.sticker_memory._conn.execute("SELECT COUNT(*) FROM stickers").fetchone()[0]
        self.assertEqual(count, 0)

    async def test_stated_meaning_is_not_overwritten_by_guess_refinement(self):
        bot = self.make_bot()
        memory = bot.sticker_memory
        sticker_id = memory.remember("private:7", 7, "0" * 16, "形象：猫；可能想表达：委屈")
        memory.set_meaning(sticker_id, 7, "表达开心的心情", "只是表达我开心")
        self.assertFalse(memory.refine_guess(sticker_id, "撒娇"))

    def test_stale_context_is_ignored(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        memory = StickerMemory({"db": str(Path(folder.name) / "s.db"), "context_seconds": 0})
        self.addCleanup(memory.close)
        memory.note_text("g", "很久以前的话")
        self.assertEqual(memory.before_text("g"), "")


class StickerPromptTests(unittest.TestCase):
    def test_sticker_prompt_asks_for_meaning_not_appearance(self):
        from taoyao.vision import IMAGE_PROMPT, STICKER_PROMPT
        self.assertIn("可能想表达", STICKER_PROMPT)
        self.assertIn("不要描述画面细节", STICKER_PROMPT)
        self.assertNotIn("可能想表达", IMAGE_PROMPT)
        from taoyao.vision import CLASSIFY_PROMPT, STICKER_TAG, MediaDescriber
        self.assertIn(STICKER_TAG, CLASSIFY_PROMPT)
        self.assertIn("照片、截图", CLASSIFY_PROMPT)
        self.assertIn(IMAGE_PROMPT, CLASSIFY_PROMPT)               # 不是表情包时按原要求描述
        self.assertIn("上一句聊天是“在吗”", MediaDescriber._sticker_prompt("在吗", classify=True))


class MediaTextIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_sighting_skips_vision_and_carries_taught_meaning(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        picture = make_image(root / "pic.png")
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.sticker_memory = StickerMemory({"db": str(root / "s.db")})
        self.addCleanup(bot.sticker_memory.close)
        bot._sticker_tasks = set()
        bot.vision = SimpleNamespace(
            enabled=True,
            fetch_segment=AsyncMock(return_value=(picture, False)),
            describe=AsyncMock(return_value="灰褐色头发的角色张着嘴"))
        bot.user_memes = SimpleNamespace(maybe_store=lambda *a, **k: False)
        bot.memes = SimpleNamespace(_keyword_match=lambda text: "")
        bot._remember_private_image = lambda *a, **k: None
        bot._describe_kept_image = lambda *a, **k: None
        data = {"message_type": "private", "user_id": 7,
                "message": [{"type": "mface", "data": {"summary": "[开心]"}}]}

        first = await bot._media_text(data)
        self.assertIn("灰褐色头发的角色张着嘴", first)
        self.assertNotIn("见过", first)
        self.assertEqual(bot.vision.describe.await_args.args[1], "sticker")   # 首次按“猜意思”识图
        bot.sticker_memory.set_meaning(1, 7, "表达开心的心情", "只是表达我开心")

        second = await bot._media_text(data)
        self.assertEqual(bot.vision.describe.await_count, 1)     # 第二次没有再识图
        self.assertIn("已经认识", second)
        self.assertNotIn("2次", second)
        self.assertIn("表达开心的心情", second)
        self.assertTrue(second.startswith("[表情包内容]"))

    async def test_guess_is_kept_until_sender_states_meaning_then_replaced(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        picture = make_image(root / "pic.png")
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.sticker_memory = StickerMemory({"db": str(root / "s.db")})
        self.addCleanup(bot.sticker_memory.close)
        bot._sticker_tasks = set()
        bot.vision = SimpleNamespace(
            enabled=True,
            fetch_segment=AsyncMock(return_value=(picture, False)),
            describe=AsyncMock(return_value="形象：灰色圆脸猫；可能想表达：委屈"))
        bot.user_memes = SimpleNamespace(maybe_store=lambda *a, **k: False)
        bot.memes = SimpleNamespace(_keyword_match=lambda text: "")
        bot._remember_private_image = lambda *a, **k: None
        bot._describe_kept_image = lambda *a, **k: None
        data = {"message_type": "private", "user_id": 7,
                "message": [{"type": "mface", "data": {}}]}
        await bot._media_text(data)
        guessed = await bot._media_text(data)
        self.assertIn("可能想表达：委屈", guessed)            # 没人说明：沿用模型的猜想
        bot.sticker_memory.set_meaning(1, 7, "表达开心的心情", "只是表达我开心")
        stated = await bot._media_text(data)
        self.assertNotIn("委屈", stated)                      # 说明后猜想被取代
        self.assertIn("形象：灰色圆脸猫", stated)
        self.assertIn("表达开心的心情", stated)
        self.assertEqual(bot.vision.describe.await_count, 1)


if __name__ == "__main__":
    unittest.main()
