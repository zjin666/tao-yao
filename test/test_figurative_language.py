"""非字面表达（比喻、象征、反话、玩笑等）：提示要求按解释理解；解释过的含义记下来，群聊私聊按可见范围想起。"""
import asyncio
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.figure_memory import FigureMemory
from taoyao.persona import build_system_prompt
from taoyao.profiles import ProfileStore


class PromptTests(unittest.TestCase):
    def test_prompt_covers_every_non_literal_kind_and_explanations(self):
        for kwargs in ({"role": "owner"}, {"role": "user", "is_group": True}):
            prompt = build_system_prompt("人设", [], **kwargs)
            self.assertIn("听懂话里不是字面意思的说法", prompt)
            self.assertIn("暗示、委婉、代称、夸张", prompt)
            self.assertIn("按他解释的意思理解", prompt)

    def test_group_summary_keeps_the_real_meaning_of_non_literal_phrases(self):
        source = inspect.getsource(ProfileStore.maybe_summarize_group)
        self.assertIn("不是字面意思的说法要写明真实含义", source)


class FigureStoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.memory = FigureMemory({"db": str(Path(self.folder.name) / "f.db"), "max_per_speaker": 10})

    def tearDown(self):
        self.memory.close()
        self.folder.cleanup()

    def test_group_meaning_is_recalled_in_that_group_and_in_the_speakers_private_chat(self):
        self.memory.add(7, "group:1", "生活的苦水", "上班的苦")
        self.assertEqual(self.memory.recall(8, 1, "你们在喝生活的苦水吗")[0]["meaning"], "上班的苦")   # 本群别人
        self.assertEqual(self.memory.recall(7, 0, "今天又喝了点生活的苦水")[0]["meaning"], "上班的苦")  # 本人私聊
        self.assertEqual(self.memory.recall(8, 0, "生活的苦水"), [])        # 别人的私聊不带出
        self.assertEqual(self.memory.recall(8, 2, "生活的苦水"), [])        # 别的群不带出

    def test_private_meaning_stays_in_private(self):
        self.memory.add(7, "private:7", "那个人", "我前任")
        self.assertEqual(self.memory.recall(7, 0, "那个人又来了")[0]["meaning"], "我前任")
        self.assertEqual(self.memory.recall(7, 1, "那个人又来了"), [])

    def test_similar_phrase_is_recalled_and_unrelated_text_is_not(self):
        self.memory.add(7, "group:1", "生活的苦水", "上班的苦")
        self.assertTrue(self.memory.recall(7, 1, "再灌点生活的苦"))
        self.assertEqual(self.memory.recall(7, 1, "今天天气不错"), [])

    def test_a_new_explanation_replaces_the_old_meaning_and_storage_is_bounded(self):
        self.memory.add(7, "group:1", "火", "怨气")
        self.memory.add(7, "group:1", "火", "加班的动力")
        self.assertEqual(self.memory.recall(7, 1, "这火")[0]["meaning"], "加班的动力")
        for i in range(30):
            self.memory.add(7, "group:1", f"说法{i:02d}", "含义")
        total = self.memory._conn.execute("SELECT COUNT(*) FROM figures").fetchone()[0]
        self.assertLessEqual(total, 10)

    def test_forgetting_a_user_removes_their_entries(self):
        self.memory.add(7, "group:1", "生活的苦水", "上班的苦")
        self.memory.forget_user(7)
        self.assertEqual(self.memory.recall(7, 1, "生活的苦水"), [])


def figure_bot(tmp, reply):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.figure_memory = FigureMemory({"db": str(Path(tmp) / "f.db")})
    bot.profiles_enabled = True
    rows = [
        {"role": "user", "user_id": 7, "nickname": "哈哩", "content": "[群友] 正在喝生活的苦水（不是）"},
        {"role": "assistant", "user_id": 7, "nickname": "", "content": "我这儿只有桃花酿"},
    ]
    bot.profiles = SimpleNamespace(group_context=AsyncMock(return_value=rows),
                                   private_context=AsyncMock(return_value=rows))
    bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value=reply))
    bot.llm = bot.utility_llm
    bot._figure_tasks = set()
    return bot


class LearnTests(unittest.IsolatedAsyncioTestCase):
    async def test_explained_phrase_is_stored_with_the_authors_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = figure_bot(tmp, '{"items":[{"phrase":"生活的苦水","meaning":"上班的苦","author":7}]}')
            await bot._learn_figure(7, 1, "是象征意义啊，两个笨蛋")
            self.assertEqual(bot.figure_memory.recall(9, 1, "生活的苦水")[0]["meaning"], "上班的苦")
            bot.figure_memory.close()

    async def test_invented_phrases_and_other_peoples_explanations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = figure_bot(tmp, '{"items":[{"phrase":"原文里没有的词","meaning":"x","author":7},'
                                  '{"phrase":"生活的苦水","meaning":"上班的苦","author":99}]}')
            await bot._learn_figure(7, 1, "是象征意义啊，两个笨蛋")
            self.assertEqual(bot.figure_memory.recall(7, 1, "生活的苦水 原文里没有的词"), [])
            bot.figure_memory.close()

    async def test_unparseable_or_failed_judgement_stores_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = figure_bot(tmp, "不是json")
            await bot._learn_figure(7, 1, "是象征意义啊")
            bot.utility_llm.chat = AsyncMock(side_effect=RuntimeError("down"))
            await bot._learn_figure(7, 1, "是象征意义啊")
            self.assertEqual(bot.figure_memory.recall(7, 1, "生活的苦水"), [])
            bot.figure_memory.close()

    async def test_only_messages_that_look_like_explanations_start_a_judgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = figure_bot(tmp, '{"items":[]}')
            for text in ("今天好累", "/帮助", "象" * 130):
                bot._watch_figure_explanation({"message_type": "group", "user_id": 7, "group_id": 1}, text)
            self.assertEqual(bot._figure_tasks, set())
            bot._watch_figure_explanation({"message_type": "group", "user_id": 7, "group_id": 1}, "是象征意义啊")
            self.assertEqual(len(bot._figure_tasks), 1)
            await asyncio.gather(*bot._figure_tasks)
            bot.figure_memory.close()

    async def test_recalled_meaning_is_injected_into_the_prompt_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = figure_bot(tmp, "")
            bot.figure_memory.add(7, "group:1", "生活的苦水", "上班的苦")
            note = bot._figure_context("又在喝生活的苦水了", [], 7, True, 1)
            self.assertIn("【表达含义】", note)
            self.assertIn("上班的苦", note)
            self.assertEqual(bot._figure_context("今天天气不错", [], 7, True, 1), "")
            bot.figure_memory.close()


if __name__ == "__main__":
    unittest.main()
