import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from taoyao.chat_history import ChatHistory
from taoyao.onebot import OneBotServer


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_failure_concurrency_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatHistory(directory)
            async def send(i):
                await store.record("private", 123, f"句子{i}",
                                   {"status": "ok", "data": {"message_id": i}}, 1700000000)
            await asyncio.gather(*(send(i) for i in range(1, 11)))
            await store.record("private", 123, "失败", {"status": "failed"}, 1700000000)
            await ChatHistory(directory).record("private", 123, "重复",
                {"status": "ok", "data": {"message_id": 1}}, 1700000000)
            await store.record("group", 456, [{"type": "image", "data": {"file": "file:///x.png"}}],
                {"status": "ok", "data": {"message_id": 20, "time": 1700000001}}, 1700000000)
            private = json.loads((Path(directory) / "private.json").read_text(encoding="utf-8"))
            group = json.loads((Path(directory) / "group.json").read_text(encoding="utf-8"))
            self.assertEqual(len(private["123"]), 10)
            self.assertTrue(all(row["sent_at"].endswith("+08:00") for row in private["123"]))
            self.assertEqual(private["123"][0]["time_source"], "local_send")
            self.assertEqual(private["123"][0]["timezone"], "Asia/Shanghai")
            self.assertEqual(private["123"][0]["sent_at_utc"], "2023-11-14T22:13:20.000+00:00")
            self.assertEqual(group["456"][0]["timestamp"], 1700000001)
            self.assertEqual(group["456"][0]["content"], "[image]")

    async def test_send_boundary(self):
        async def event(data):
            pass
        server = OneBotServer("127.0.0.1", 0, "", event)
        class FakeSocket:
            async def send(self, payload):
                action = json.loads(payload)
                await server._dispatch(json.dumps({"echo": action["echo"], "status": "ok",
                    "retcode": 0, "data": {"message_id": action["echo"]}}))
        with tempfile.TemporaryDirectory() as directory:
            server.chat_history = ChatHistory(directory)
            server.ws = FakeSocket()
            await server.send_text("private", 123, "你好")
            await server.send_text("group", 456, "大家好", at_user_id=789)
            await server.call("get_login_info", {})
            private = json.loads((Path(directory) / "private.json").read_text(encoding="utf-8"))
            group = json.loads((Path(directory) / "group.json").read_text(encoding="utf-8"))
            self.assertEqual(private["123"][0]["content"], "你好")
            self.assertEqual(group["456"][0]["message"][0]["type"], "at")


if __name__ == "__main__":
    unittest.main()
