import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import ReplyIntent, TaoYaoBot
from taoyao.request_inbox import RequestInbox

OWNER = 900001


class InboxTests(unittest.TestCase):
    def test_add_dedupes_by_flag_persists_and_finishes_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "r.json"
            inbox = RequestInbox(path)
            first = inbox.add(kind="friend", user_id=5, flag="F1", comment="你好")
            again = inbox.add(kind="friend", user_id=5, flag="F1", comment="你好")
            self.assertEqual(first["id"], again["id"])
            self.assertEqual(len(inbox.pending()), 1)
            reloaded = RequestInbox(path)                           # 重启后还在
            self.assertEqual(reloaded.latest_pending()["flag"], "F1")
            self.assertIsNotNone(reloaded.finish(first["id"], True))
            self.assertIsNone(reloaded.finish(first["id"], True))      # 不会处理两次
            self.assertEqual(reloaded.pending(), [])
            self.assertEqual(reloaded.add(kind="friend", user_id=6, flag="F2")["id"], first["id"] + 1)

    def test_old_unanswered_requests_expire(self):
        with tempfile.TemporaryDirectory() as folder:
            inbox = RequestInbox(Path(folder) / "r.json", ttl_days=1)
            item = inbox.add(kind="friend", user_id=5, flag="F1")
            item["at"] = time.time() - 2 * 86400
            self.assertEqual(inbox.pending(), [])


class RequestFlowTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, **cfg):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {"owner_qq": OWNER, "requests": {"enabled": True, **cfg}}
        bot.requests = RequestInbox(Path(folder.name) / "r.json")
        bot.calls = []

        async def call(action, params, timeout=30):
            bot.calls.append((action, params))
            return {"status": "ok", "retcode": 0, "data": {"nickname": "路人"}}
        bot.server = SimpleNamespace(call=call)
        bot.sent = []

        async def send(target, target_id, text, **kwargs):
            bot.sent.append((target, target_id, text))
            return True
        bot._send_segments = send
        return bot

    async def test_friend_request_is_stored_and_the_owner_is_asked_not_auto_accepted(self):
        bot = self.make_bot()
        await bot._handle_request({"post_type": "request", "request_type": "friend", "user_id": 55,
                                   "comment": "我是群友", "flag": "FL1"})
        self.assertEqual(len(bot.requests.pending()), 1)
        self.assertFalse([c for c in bot.calls if c[0].startswith("set_")])        # 没有擅自同意
        target, target_id, intent = bot.sent[0]
        self.assertEqual((target, target_id), ("private", OWNER))
        self.assertIsInstance(intent, ReplyIntent)
        self.assertIn("我是群友", intent.facts)
        self.assertIn("路人", intent.facts)
        self.assertIn("不要替哥哥做决定", intent.situation)

    async def test_owner_approves_with_command_and_the_onebot_call_is_made(self):
        bot = self.make_bot()
        await bot._handle_request({"request_type": "friend", "user_id": 55, "flag": "FL1"})
        handled = await bot._handle_request_command("/同意", "private", OWNER)
        self.assertTrue(handled)
        action, params = [c for c in bot.calls if c[0].startswith("set_")][0]
        self.assertEqual((action, params["flag"], params["approve"]), ("set_friend_add_request", "FL1", True))
        self.assertEqual(bot.requests.pending(), [])
        self.assertIn("成功", bot.sent[-1][2].facts)

    async def test_reject_by_number_and_unknown_number_is_reported_honestly(self):
        bot = self.make_bot()
        await bot._handle_request({"request_type": "friend", "user_id": 55, "flag": "FL1"})
        await bot._handle_request({"request_type": "friend", "user_id": 56, "flag": "FL2"})
        await bot._handle_request_command("/拒绝 1", "private", OWNER)
        self.assertEqual([i["flag"] for i in bot.requests.pending()], ["FL2"])
        self.assertFalse([c for c in bot.calls if c[0] == "set_friend_add_request" and c[1]["flag"] == "FL2"])
        await bot._handle_request_command("/同意 99", "private", OWNER)
        self.assertIn("没有找到", bot.sent[-1][2].facts)

    async def test_failed_submission_keeps_the_request_pending(self):
        bot = self.make_bot()
        await bot._handle_request({"request_type": "friend", "user_id": 55, "flag": "FL1"})

        async def failing(action, params, timeout=30):
            return {"status": "failed", "retcode": 100}
        bot.server.call = failing
        await bot._handle_request_command("/同意", "private", OWNER)
        self.assertEqual(len(bot.requests.pending()), 1)
        self.assertIn("失败", bot.sent[-1][2].facts)

    async def test_group_invite_from_the_owner_is_accepted_automatically_others_are_asked(self):
        bot = self.make_bot()
        await bot._handle_request({"request_type": "group", "sub_type": "invite", "user_id": OWNER,
                                   "group_id": 77, "flag": "G1"})
        action, params = [c for c in bot.calls if c[0].startswith("set_")][0]
        self.assertEqual((action, params["sub_type"], params["approve"]), ("set_group_add_request", "invite", True))
        self.assertEqual(bot.sent, [])                                 # 哥哥自己拉的，不用再通知
        bot.calls.clear()
        await bot._handle_request({"request_type": "group", "sub_type": "invite", "user_id": 77777,
                                   "group_id": 88, "flag": "G2"})
        self.assertFalse([c for c in bot.calls if c[0].startswith("set_")])   # 陌生人拉群：问哥哥
        self.assertIn("群号 88", bot.sent[-1][2].facts)

    async def test_ignore_mode_and_disabled_do_nothing(self):
        bot = self.make_bot(friend="ignore")
        await bot._handle_request({"request_type": "friend", "user_id": 55, "flag": "FL1"})
        self.assertEqual((bot.requests.pending(), bot.sent), ([], []))
        off = self.make_bot(enabled=False)
        await off._handle_request({"request_type": "friend", "user_id": 55, "flag": "FL1"})
        self.assertEqual(off.requests.pending(), [])

    async def test_non_request_commands_are_not_swallowed(self):
        bot = self.make_bot()
        self.assertFalse(await bot._handle_request_command("/状态", "private", OWNER))


class SpokenReplyTests(unittest.IsolatedAsyncioTestCase):
    async def make_bot(self, verdict, count=1):
        bot = RequestFlowTests.make_bot(self)
        for index in range(count):
            await bot._handle_request({"request_type": "friend", "user_id": 100 + index, "flag": f"FL{index + 1}",
                                       "comment": f"我是{index + 1}号"})
        bot.sent.clear()
        bot.calls.clear()
        bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value=verdict))
        bot.llm = bot.utility_llm
        return bot

    async def say(self, bot, text, target_id=OWNER):
        return await bot._handle_owner_request_intent("private", target_id, text)

    def approvals(self, bot):
        return [c for c in bot.calls if c[0].startswith("set_")]

    async def test_a_plain_spoken_yes_approves_the_only_pending_request(self):
        bot = await self.make_bot('{"decision":"approve","request_id":null,"evidence":"同意"}')
        self.assertTrue(await self.say(bot, "同意吧，加上"))
        self.assertEqual(self.approvals(bot)[0][1]["approve"], True)
        self.assertEqual(bot.requests.pending(), [])
        self.assertIsInstance(bot.sent[-1][2], ReplyIntent)                 # 结果仍走统一回复流程
        self.assertIn("成功", bot.sent[-1][2].facts)

    async def test_a_spoken_no_rejects(self):
        bot = await self.make_bot('{"decision":"reject","request_id":1,"evidence":"别加"}')
        self.assertTrue(await self.say(bot, "这个别加了"))
        self.assertEqual(self.approvals(bot)[0][1]["approve"], False)

    async def test_without_a_verbatim_evidence_or_a_clear_decision_nothing_happens(self):
        invented = await self.make_bot('{"decision":"approve","request_id":null,"evidence":"我批准了"}')
        self.assertFalse(await self.say(invented, "同意吧"))               # 证据不是原话里的
        self.assertEqual(self.approvals(invented), [])
        unsure = await self.make_bot('{"decision":"none","request_id":null,"evidence":""}')
        self.assertFalse(await self.say(unsure, "要不要通过呢？可以吗"))
        self.assertEqual(self.approvals(unsure), [])
        self.assertEqual(len(unsure.requests.pending()), 1)

    async def test_with_several_pending_the_number_picks_one_and_no_match_asks_instead_of_guessing(self):
        picked = await self.make_bot('{"decision":"approve","request_id":2,"evidence":"第二个通过"}', count=2)
        self.assertTrue(await self.say(picked, "第二个通过"))
        self.assertEqual([c[1]["flag"] for c in self.approvals(picked)], ["FL2"])
        self.assertEqual([i["flag"] for i in picked.requests.pending()], ["FL1"])
        unclear = await self.make_bot('{"decision":"approve","request_id":null,"evidence":"同意"}', count=2)
        self.assertTrue(await self.say(unclear, "同意吧"))
        self.assertEqual(self.approvals(unclear), [])                        # 没有擅自挑一个
        self.assertEqual(len(unclear.requests.pending()), 2)
        self.assertIn("问他是哪个", unclear.sent[-1][2].situation)

    async def test_no_model_call_without_pending_requests_cues_or_from_anyone_but_the_owner(self):
        none_pending = await self.make_bot('{"decision":"approve","request_id":null,"evidence":"同意"}', count=0)
        self.assertFalse(await self.say(none_pending, "同意吧"))
        none_pending.utility_llm.chat.assert_not_awaited()
        chatty = await self.make_bot('{"decision":"approve","request_id":null,"evidence":"同意"}')
        self.assertFalse(await self.say(chatty, "今天天气真不错呀"))          # 没有任何决定的线索
        chatty.utility_llm.chat.assert_not_awaited()
        self.assertFalse(await self.say(chatty, "同意吧", target_id=12345))    # 不是哥哥
        self.assertFalse(await chatty._handle_owner_request_intent("group", OWNER, "同意吧"))
        chatty.utility_llm.chat.assert_not_awaited()

    async def test_a_broken_model_answer_changes_nothing(self):
        bot = await self.make_bot("不是 JSON")
        self.assertFalse(await self.say(bot, "同意吧"))
        self.assertEqual(self.approvals(bot), [])


if __name__ == "__main__":
    unittest.main()
