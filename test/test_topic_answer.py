"""桃夭自己开的群话题：群友之后的回答要能触发接话，不再只靠概率。"""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import ReplyIntent, TaoYaoBot


def topic_bot(answer='{"refers":true}'):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {"group_ambient": {}}
    bot.profiles_enabled = False
    bot._group_last_bot_reply, bot._group_open_topic = {}, {}
    bot._group_reply_budget = SimpleNamespace(paused=lambda *a, **k: False)
    bot.utility_llm = bot.llm = SimpleNamespace(chat=AsyncMock(return_value=answer))
    return bot


class OpenTopicTests(unittest.IsolatedAsyncioTestCase):
    async def test_sending_a_topic_opens_it_and_registers_her_last_words(self):
        bot = topic_bot()
        bot._group_topic_state = {}
        bot._group_topic_text = AsyncMock(return_value="你们最近在追什么剧？")
        bot._send_segments = AsyncMock(return_value=True)
        bot.profiles = SimpleNamespace(add_group_topic=AsyncMock())
        await bot._send_group_topic(100)
        self.assertEqual(bot._group_open_topic[100][1], "你们最近在追什么剧？")
        self.assertEqual(bot._group_last_bot_reply[100][2], "你们最近在追什么剧？")

    async def test_an_answer_to_her_topic_counts_and_the_model_is_told_it_is_her_topic(self):
        bot = topic_bot()
        now = time.monotonic()
        bot._group_open_topic[100] = (now - 600, "你们最近在追什么剧？")      # 超过默认的 180 秒也还算
        self.assertTrue(await bot._refers_to_bot_words(100, 7, "我在追庆余年"))
        prompt = bot.utility_llm.chat.await_args.args[0][1]["content"]
        self.assertIn("主动发起的话题", prompt)

    async def test_an_old_topic_is_closed_and_a_model_no_means_no(self):
        bot = topic_bot()
        bot._group_open_topic[100] = (time.monotonic() - 5000, "你们最近在追什么剧？")
        self.assertFalse(await bot._refers_to_bot_words(100, 7, "我在追庆余年"))
        bot.utility_llm.chat.assert_not_awaited()
        bot._group_open_topic[100] = (time.monotonic() - 30, "你们最近在追什么剧？")
        bot.utility_llm.chat = AsyncMock(return_value='{"refers":false}')
        self.assertFalse(await bot._refers_to_bot_words(100, 7, "有人会修电脑吗"))


if __name__ == "__main__":
    unittest.main()
