import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from taoyao.app import PendingMessage, TaoYaoBot
from taoyao.config import load_config
from taoyao.personality import DIMENSIONS, ClassificationResult


def item(user_id, text, *, group=True, mid=""):
    return PendingMessage(target="group" if group else "private", target_id=100 if group else user_id,
                          session="group:100" if group else f"user_private:{user_id}", user_id=user_id,
                          nickname=f"u{user_id}", text=text, role="user", is_group=group,
                          group_id=100 if group else 0, scene="group" if group else "private",
                          message_id=mid or text)


class WorkerFlagTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot._queues, bot._locks, bot._batch_tasks = {}, {}, {}
        bot.batch_enabled, bot.batch_max = True, 6
        bot.guard = SimpleNamespace(allow_rate=lambda *a, **k: True)
        bot._note_group_arrival = Mock()
        bot.flags = {}

        async def handle(*args, **kwargs):
            bot.flags[kwargs["message_id"]] = kwargs.get("superseded")
        bot._handle_text = handle
        return bot

    async def test_only_earlier_messages_of_the_same_sender_are_superseded(self):
        bot = self.make_bot()
        for it in (item(1, "a1"), item(2, "b1"), item(1, "a2"), item(1, "a3")):
            await bot._enqueue(it)
        await asyncio.gather(*list(bot._batch_tasks.values()))
        self.assertEqual(bot.flags, {"a1": True, "b1": False, "a2": True, "a3": False})

    async def test_private_chat_is_never_superseded(self):
        bot = self.make_bot()
        for it in (item(1, "p1", group=False), item(1, "p2", group=False)):
            await bot._enqueue(it)
        await asyncio.gather(*list(bot._batch_tasks.values()))
        self.assertEqual(bot.flags, {"p1": False, "p2": False})

    async def test_a_lone_message_is_never_skipped(self):
        bot = self.make_bot()
        await bot._enqueue(item(1, "only"))
        await asyncio.gather(*list(bot._batch_tasks.values()))
        self.assertEqual(bot.flags, {"only": False})


class SupersededTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_superseded_message_is_recorded_and_counted_but_gets_no_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config["response_speed"].update(two_part_reply=False, speculative_reply=False, defer_six_axis=True,
                                            six_axis_settle_seconds=0.05)
            config["memory"]["db"] = str(Path(directory) / "memory.db")
            config["profile"].update(db=str(Path(directory) / "profiles.db"), llm_extract=False)
            config["memes"]["enabled"] = config["user_memes"]["enabled"] = False
            config["intro"]["enabled"] = False
            config["privacy_index"]["enabled"] = False
            bot = TaoYaoBot(config, mock=True)
            try:
                generated = []

                async def stream(messages):
                    generated.append(messages)
                    yield "不该生成"

                bot.llm = bot.utility_llm = SimpleNamespace(stream_chat=stream, chat=AsyncMock(return_value='{"allowed":true}'))
                labels = {d: ClassificationResult("中性", .9, applied=False) for d in DIMENSIONS}
                analyzed = []

                async def analyze(**kwargs):
                    analyzed.append(kwargs["text"])
                    return labels, "TEST"
                bot.analysis_pipeline = SimpleNamespace(config={"context_messages": 6}, analyze=analyze)
                bot._profile_identity_allowed = AsyncMock(return_value=True)
                bot._search_context_for = AsyncMock(return_value="")
                bot.server.send_text = AsyncMock(return_value={"status": "ok"})
                await bot._handle_text("group", 100, "group:100", 900001, "测试", "我先说第一句", "user", True, 100,
                                       "group", from_batch=True, message_id="m1", superseded=True)
                self.assertEqual(generated, [])                        # 没有为这条单独生成回复
                bot.server.send_text.assert_not_awaited()
                recent = await bot.memory.recent("group:100", 10)
                self.assertTrue(any("我先说第一句" in str(m.get("content")) for m in recent))   # 消息还在上下文里
                rows = await bot.profiles.group_context(100, 10)
                self.assertTrue(any("我先说第一句" in str(r.get("content")) for r in rows))       # 群记录里也有
                await asyncio.wait_for(bot._six_axis.flush(), 10)
                self.assertTrue(any("我先说第一句" in text for text in analyzed))                 # 六维照样记这条
            finally:
                await bot._six_axis.flush()
                bot.memory.close()
                bot.profiles.close()


if __name__ == "__main__":
    unittest.main()
