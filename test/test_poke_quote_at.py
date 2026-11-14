"""戳一戳、引用回复、@ 三条路径的端到端回归：收到戳、回戳、用 @ 提醒、[[at]] 点名、[[quote]] 引用。"""
import re
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.onebot import OneBotServer
from taoyao.turn_timing import set_value, turn_timing_context

OWNER = 900001


def poke_bot():
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {"owner_qq": OWNER, "poke": {"enabled": True, "cooldown_seconds": 5},
                  "poke_reply": {"enabled": True, "cooldown_seconds": 60, "max_per_hour": 4}}
    bot._group_reply_budget = SimpleNamespace(paused=lambda *a, **k: False)
    bot._roles = lambda: {"allow_all_groups": False}
    bot._allowed_groups = lambda: {100}
    bot._incoming_pokes, bot._poke_cooldowns = {}, {}
    bot._outgoing_poke_cooldowns, bot._outgoing_poke_times, bot._chat_activity = {}, {}, {}
    bot._handle_text = AsyncMock()
    return bot


class IncomingPokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_poke_at_the_bot_becomes_a_normal_reply_turn_in_private_and_group(self):
        bot = poke_bot()
        await bot._handle_notice({"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
                                  "user_id": 5, "target_id": 9, "self_id": 9})
        kwargs = bot._handle_text.await_args.kwargs
        self.assertEqual((kwargs["text"], kwargs["target"], kwargs["target_id"], kwargs["is_group"]),
                         ("（戳了戳你）", "private", 5, False))
        await bot._handle_notice({"post_type": "notice", "notice_type": "notify", "sub_type": "poke",
                                  "group_id": 100, "user_id": 6, "target_id": 9, "self_id": 9})
        kwargs = bot._handle_text.await_args.kwargs
        self.assertEqual((kwargs["text"], kwargs["target"], kwargs["target_id"], kwargs["is_group"]),
                         ("（戳了戳你）", "group", 100, True))
        self.assertEqual(bot._handle_text.await_count, 2)

    async def test_pokes_at_someone_else_unregistered_groups_and_rapid_repeats_are_ignored(self):
        bot = poke_bot()
        await bot._handle_notice({"notice_type": "notify", "sub_type": "poke", "user_id": 5, "target_id": 77, "self_id": 9})
        await bot._handle_notice({"notice_type": "notify", "sub_type": "poke", "group_id": 999, "user_id": 5,
                                  "target_id": 9, "self_id": 9})
        await bot._handle_notice({"notice_type": "notify", "sub_type": "poke", "user_id": 9, "target_id": 9, "self_id": 9})
        bot._handle_text.assert_not_awaited()
        for _ in range(2):                                                   # 冷却内的重复戳只回应一次
            await bot._handle_notice({"notice_type": "notify", "sub_type": "poke", "user_id": 5, "target_id": 9, "self_id": 9})
        self.assertEqual(bot._handle_text.await_count, 1)


class OutgoingPokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_poke_calls_onebot_with_the_right_target(self):
        bot = poke_bot()
        bot.server = SimpleNamespace(call=AsyncMock(return_value={"status": "ok"}))
        self.assertTrue(await bot._send_poke("group", 100, 5))
        self.assertEqual(bot.server.call.await_args.args[:2], ("send_poke", {"user_id": 5, "group_id": 100}))
        self.assertTrue(await bot._send_poke("private", 5, 5))
        self.assertEqual(bot.server.call.await_args.args[1], {"user_id": 5})

    async def test_a_reply_that_must_be_noticed_in_a_group_uses_a_real_at_not_a_quote(self):
        """“提醒你留意刚才那句话”走统一发送流程，@ 的是对方，不会变成引用回复。"""
        from taoyao import markers
        bot = poke_bot()
        sent = []

        async def fake_send(target, target_id, text, **kwargs):
            sent.append((target, target_id, text, kwargs))
            return True
        bot._send_segments = fake_send
        bot._emotion_poke_worthy = AsyncMock(return_value=False)
        await bot._maybe_send_reply_poke("group", 100, 5, "我提醒你，记得交作业。", "在吗", True, 100, False)
        self.assertEqual(len(sent), 1)
        self.assertEqual((sent[0][0], sent[0][1], sent[0][3]["at_user_id"]), ("group", 100, 5))
        self.assertNotIn(markers.QUOTE, sent[0][2])


class AtAndQuotePipelineTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, text):
        async def stream(messages):
            yield text
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = load_config()
        bot.config["response_speed"]["two_part_reply"] = True
        bot.config["reply"].update(segment_gap_min_seconds=0, segment_gap_max_seconds=0)
        bot.llm = SimpleNamespace(stream_chat=stream)
        bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value='{"allowed":true}'))
        bot.privacy_index = None
        bot._clean_output_symbols = AsyncMock(side_effect=lambda value: value)
        bot.server = SimpleNamespace(send_text=AsyncMock(return_value={}))
        bot._maybe_send_meme = AsyncMock(return_value=False)
        return bot

    async def run_group_reply(self, text, message_id="555"):
        bot = self.make_bot(text)

        @turn_timing_context
        async def run():
            set_value("quote_id", message_id)
            await bot._stream_reply([{"role": "system", "content": "桃夭人设"}, {"role": "user", "content": "继续"}],
                                    "group", 100, 2, 70, role="user", at_user_id=9000715, viewer_id=9000715,
                                    review_policy={"identity": "user"})
        await run()
        return bot.server.send_text.await_args_list

    async def test_at_tag_mentions_the_person_without_quoting(self):
        calls = await self.run_group_reply("[[at]] 这眼睛好圆。你再说说。")
        self.assertEqual(calls[0].kwargs["at_user_id"], 9000715)
        self.assertNotIn("quote_id", calls[0].kwargs)
        self.assertTrue(all(c.kwargs["at_user_id"] is None for c in calls[1:]))

    async def test_quote_tag_quotes_the_triggering_message_instead_of_at(self):
        calls = await self.run_group_reply("[[quote]] 这个问题我来答。你再说说。")
        self.assertEqual(calls[0].kwargs["quote_id"], "555")
        self.assertIsNone(calls[0].kwargs["at_user_id"])
        for call in calls:
            self.assertNotIn("[[", call.args[2])

    async def test_no_tag_means_neither_at_nor_quote(self):
        calls = await self.run_group_reply("好呀，我听着。")
        self.assertIsNone(calls[0].kwargs["at_user_id"])
        self.assertNotIn("quote_id", calls[0].kwargs)

    async def test_the_real_onebot_message_has_a_reply_segment_for_quote_and_an_at_segment_for_at(self):
        server = OneBotServer.__new__(OneBotServer)
        server.send_group = AsyncMock(return_value={})
        await server.send_text("group", 100, "好呀", at_user_id=None, quote_id="555")
        quoted = server.send_group.await_args.args[1]
        self.assertEqual(quoted[0], {"type": "reply", "data": {"id": "555"}})
        self.assertEqual(quoted[-1]["data"]["text"], "好呀")
        await server.send_text("group", 100, "好呀", at_user_id=9000715)
        mentioned = server.send_group.await_args.args[1]
        self.assertEqual(mentioned[0], {"type": "at", "data": {"qq": "9000715"}})
        self.assertFalse(any(seg["type"] == "reply" for seg in mentioned))


if __name__ == "__main__":
    unittest.main()
