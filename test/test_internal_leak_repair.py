"""第五步（发送前检查）：回复里混进内部内容时，被拦下后带着修正说明只重新生成一次；
整段都是给模型自己的内部输出则不重新生成、也不发；内部内容不会发给用户。
"""
import re
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.output_guard import internal_notice_reason

LEAK = re.compile(r"\[语音\]|\[文件内容\]|\[图片内容\]|【[^】]*】|形象：|神态：|\"decision\"|\[\[|\]\]|quote")


class LeakRepairTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, outputs):
        """outputs：每次调用模型依次返回的分块列表。"""
        calls = []

        async def stream(messages):
            calls.append(messages)
            for chunk in outputs[min(len(calls), len(outputs)) - 1]:
                yield chunk

        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = load_config()
        bot.config["response_speed"]["two_part_reply"] = True
        bot.config["reply"].update(segment_gap_min_seconds=0, segment_gap_max_seconds=0)
        bot.llm = SimpleNamespace(stream_chat=stream)
        bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value='{"allowed":true}'))
        bot.privacy_index = None
        bot._clean_output_symbols = AsyncMock(side_effect=lambda text: text)
        bot.server = SimpleNamespace(send_text=AsyncMock(return_value={}))
        bot._maybe_send_meme = AsyncMock(return_value=False)
        bot.calls = calls
        return bot

    async def run_reply(self, bot, target="private"):
        return await bot._stream_reply(
            [{"role": "system", "content": "桃夭人设"}, {"role": "user", "content": "继续说吧"}],
            target, 2, 2, 70, role="user", review_policy={"identity": "user"}, viewer_id=2)

    def sent(self, bot):
        return [call.args[2] for call in bot.server.send_text.await_args_list]

    async def test_a_leak_in_the_first_sentence_is_caught_and_repaired_once_before_anything_is_sent(self):
        bot = self.make_bot([["[语音] 我听到你说今天很累。"], ["我听到啦，今天辛苦了。"]])
        result = await self.run_reply(bot)
        self.assertEqual(len(bot.calls), 2)                                   # 只重新生成一次
        self.assertIn("本轮只修正明确违规", bot.calls[1][0]["content"])         # 带着修正说明
        self.assertIn("internal_marker", bot.calls[1][0]["content"])
        sent = "".join(self.sent(bot))
        self.assertIn("今天辛苦了", sent)
        self.assertIsNone(LEAK.search(sent), sent)
        self.assertIsNone(LEAK.search(result[0]), result[0])

    async def test_a_leak_in_the_tail_keeps_the_already_sent_first_sentence_and_repairs_the_rest(self):
        bot = self.make_bot([["好呀，我看看。", "我读了一下。[文件内容] 作业.txt（文本）\n周五交作业"],
                             ["放心，我会按时提醒你。"]])
        await self.run_reply(bot)
        self.assertEqual(len(bot.calls), 2)
        self.assertIn("首句已经发出", bot.calls[1][0]["content"])               # 不重复、不改写已发出的首句
        sent = "".join(self.sent(bot))
        self.assertIn("好呀", sent)
        self.assertIn("按时提醒你", sent)
        self.assertIsNone(LEAK.search(sent), sent)

    async def test_if_the_repair_leaks_too_nothing_unchecked_is_sent_and_no_fixed_text_replaces_it(self):
        bot = self.make_bot([["[文件内容] 作业.txt（文本）\n周五交作业"], ["{\"decision\":\"approve\",\"evidence\":\"同意\"}"]])
        result = await self.run_reply(bot)
        self.assertEqual(len(bot.calls), 2)                                   # 有上限，不会无限重试
        self.assertEqual(self.sent(bot), [])
        self.assertEqual(result[0], "")

    async def test_wholly_self_directed_output_is_neither_repaired_nor_sent(self):
        bot = self.make_bot([["【内部状态】只给模型自己的备注，不要回复用户"]])
        result = await self.run_reply(bot)
        self.assertEqual(len(bot.calls), 1)
        self.assertEqual(self.sent(bot), [])
        self.assertEqual(result[0], "")

    async def test_the_same_holds_in_group_chat_and_tags_are_never_visible(self):
        bot = self.make_bot([["[[quote:123]] [群友] 你们好呀。"], ["[[quote]] 这个问题我来答。"]])
        await self.run_reply(bot, target="group")
        sent = "".join(self.sent(bot))
        self.assertIn("这个问题我来答", sent)
        self.assertIsNone(LEAK.search(sent), sent)

    def test_the_new_category_is_reported_for_marker_leaks(self):
        self.assertEqual(internal_notice_reason("[语音] 我听到了"), "internal_marker")
        self.assertEqual(internal_notice_reason("形象：猫；神态：哭；可能想表达：委屈"), "internal_marker")
        self.assertEqual(internal_notice_reason("【聊天规则】中文QQ短句"), "internal_prompt_or_routing")
        self.assertEqual(internal_notice_reason("哼，才不是呢。"), "")


if __name__ == "__main__":
    unittest.main()
