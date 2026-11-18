"""含“记住”的话要结合上文判断是不是新的记忆请求，不靠过滤词（“记住了吧”是在确认，不是要她记“了吧”）。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot


def bot_with(answer=None, error=False):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    chat = AsyncMock(side_effect=RuntimeError("down")) if error else AsyncMock(return_value=answer)
    bot.utility_llm = SimpleNamespace(chat=chat)
    bot.llm = bot.utility_llm
    bot.profiles_enabled = True
    return bot


class IsNewMemoryRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmation_with_context_is_not_a_memory_request(self):
        bot = bot_with('{"remember":false}')
        self.assertFalse(await bot._is_new_memory_request("记住了吧", "桃夭：原来是吃瓜呀\n用户：这个是吃瓜的表情包"))
        sent = bot.utility_llm.chat.await_args.args[0][1]["content"]
        self.assertIn("吃瓜的表情包", sent)          # 判断时带上了上文
        self.assertIn("记住了吧", sent)

    async def test_a_real_request_stays_a_memory_request(self):
        self.assertTrue(await bot_with('{"remember":true}')._is_new_memory_request("记住我不吃香菜", ""))

    async def test_unusable_judgement_keeps_the_old_behaviour(self):
        self.assertTrue(await bot_with("乱码")._is_new_memory_request("记住了吧", ""))
        self.assertTrue(await bot_with(error=True)._is_new_memory_request("记住了吧", ""))


class RecentChatTextTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_and_group_context_are_formatted_for_the_judge(self):
        bot = bot_with()
        bot.profiles = SimpleNamespace(
            private_context=AsyncMock(return_value=[{"role": "user", "content": "这个是吃瓜"}, {"role": "assistant", "content": "哦"}]),
            group_context=AsyncMock(return_value=[{"role": "user", "content": "群里的话"}]))
        self.assertEqual(await bot._recent_chat_text(1, False, 0), "用户：这个是吃瓜\n桃夭：哦")
        self.assertEqual(await bot._recent_chat_text(1, True, 9), "用户：群里的话")

    async def test_failure_gives_empty_context(self):
        bot = bot_with()
        bot.profiles = SimpleNamespace(private_context=AsyncMock(side_effect=RuntimeError("x")))
        self.assertEqual(await bot._recent_chat_text(1, False, 0), "")


class MemoryPromptContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_memory_prompt_gets_the_context_to_resolve_this_or_that(self):
        bot = bot_with(json.dumps({"accept": True, "summary": "喜欢吃瓜表情包"}))
        bot.config = {"memory_requests": {}}
        bot._memory_locks = {}
        bot._memory_forbidden = lambda text, owner: False
        bot.profiles = SimpleNamespace(get_user_memory=AsyncMock(return_value=""), set_user_memory=AsyncMock())
        bot._send_segments = AsyncMock(return_value=True)
        await bot._update_user_memory(1, "上面那个", "owner", "private", 1, {}, "用户：我喜欢吃瓜表情包")
        prompt = bot.utility_llm.chat.await_args.args[0][1]["content"]
        self.assertIn("我喜欢吃瓜表情包", prompt)
        self.assertIn("用户新要求：\n上面那个", prompt)
        bot.profiles.set_user_memory.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
