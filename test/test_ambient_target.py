"""没人叫桃夭时，她想插话前先按上下文判断这句是不是对别人说的。"""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot

ROWS = [
    {"role": "user", "user_id": 9, "nickname": "守夜", "content": "[群友] 我就是舍不得让话落到地上"},
    {"role": "assistant", "user_id": 7, "nickname": "", "content": "哼，谁接他话啦"},
    {"role": "user", "user_id": 7, "nickname": "哈哩哈哩", "content": "[群友] 变成讨好型鱼格了"},
]


def make(answer=None, error=False, route=None):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.profiles_enabled = True
    bot.profiles = SimpleNamespace(group_context=AsyncMock(return_value=ROWS))
    chat = AsyncMock(side_effect=RuntimeError("down")) if error else AsyncMock(return_value=answer)
    bot.utility_llm = SimpleNamespace(chat=chat)
    bot.llm = bot.utility_llm
    bot._group_routes = {(1, "m"): route} if route else {}
    return bot


class AmbientTargetTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_remark_about_someone_else_cancels_the_interjection(self):
        bot = make('{"target":"守夜"}')
        self.assertEqual(await bot._ambient_directed_elsewhere(1, "哈哩哈哩", "变成讨好型鱼格了", "m"), "守夜")
        prompt = bot.utility_llm.chat.await_args.args[0][1]["content"]
        self.assertIn("守夜：我就是舍不得让话落到地上", prompt)       # 带发言者
        self.assertIn("桃夭：哼，谁接他话啦", prompt)

    async def test_to_her_or_no_target_keeps_the_interjection(self):
        for answer in ('{"target":"桃夭"}', '{"target":"无"}', '{"target":"哈哩哈哩"}', "不是json"):
            self.assertEqual(await make(answer)._ambient_directed_elsewhere(1, "哈哩哈哩", "x", "m"), "", answer)

    async def test_model_failure_does_not_block_and_clear_routes_skip_the_judgement(self):
        self.assertEqual(await make(error=True)._ambient_directed_elsewhere(1, "哈哩哈哩", "x", "m"), "")
        bot = make('{"target":"守夜"}', route={"state": "bot"})
        self.assertEqual(await bot._ambient_directed_elsewhere(1, "哈哩哈哩", "x", "m"), "")
        bot.utility_llm.chat.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
