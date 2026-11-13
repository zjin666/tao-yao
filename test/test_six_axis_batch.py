import asyncio
import time
import unittest

from taoyao.deferred_profile import SixAxisBatcher


class SixAxisBatcherTests(unittest.IsolatedAsyncioTestCase):
    def make(self, settle=0.1, max_wait=1.0, hold=0.0, fail=False):
        calls = []

        async def run(key, merged):
            calls.append((key, merged, time.monotonic()))
            if hold:
                await asyncio.sleep(hold)
            if fail:
                raise RuntimeError("boom")

        return SixAxisBatcher(run, settle=settle, max_wait=max_wait), calls

    def item(self, text, mid="m", raw=None, is_new=False, context=None):
        return {"text": text, "raw_text": raw, "message_id": mid, "sent_at": 1.0,
                "analysis_context": context, "profile": {"user_id": "1", "is_new": is_new}}

    async def test_consecutive_messages_are_merged_into_one_update(self):
        batcher, calls = self.make(settle=0.15)
        for i, text in enumerate(("第一句", "第二句", "第三句")):
            batcher.add(1, "private", 0, self.item(text, mid=f"m{i}"))
            await asyncio.sleep(0.05)
        await batcher.flush()
        self.assertEqual(len(calls), 1)
        merged = calls[0][1]
        self.assertEqual(merged["text"], "第一句\n第二句\n第三句")
        self.assertEqual(merged["count"], 3)
        self.assertEqual(merged["message_id"], "m2")          # 用最后一条的标识

    async def test_separate_users_and_scenes_do_not_mix(self):
        batcher, calls = self.make(settle=0.05)
        batcher.add(1, "private", 0, self.item("甲"))
        batcher.add(2, "private", 0, self.item("乙"))
        batcher.add(1, "group", 9, self.item("丙"))
        await batcher.flush()
        self.assertEqual(sorted(c[1]["text"] for c in calls), ["丙", "乙", "甲"])

    async def test_quiet_period_splits_into_separate_updates(self):
        batcher, calls = self.make(settle=0.05)
        batcher.add(1, "private", 0, self.item("早先的话"))
        await asyncio.sleep(0.2)
        batcher.add(1, "private", 0, self.item("很久以后的话"))
        await batcher.flush()
        self.assertEqual([c[1]["text"] for c in calls], ["早先的话", "很久以后的话"])

    async def test_non_stop_chat_is_still_updated_by_max_wait(self):
        batcher, calls = self.make(settle=0.2, max_wait=0.3)
        started = time.monotonic()
        for i in range(6):
            batcher.add(1, "private", 0, self.item(f"话{i}"))
            await asyncio.sleep(0.1)                     # 一直有新消息，安静期永远等不到
        await batcher.flush()
        self.assertGreaterEqual(len(calls), 2)
        self.assertLess(calls[0][2] - started, 0.5)

    async def test_updates_of_one_user_run_strictly_in_order(self):
        batcher, calls = self.make(settle=0.0, hold=0.15)
        batcher.add(1, "private", 0, self.item("先"))
        await asyncio.sleep(0.05)                          # 第一次正在做
        batcher.add(1, "private", 0, self.item("后"))
        await batcher.flush()
        self.assertEqual([c[1]["text"] for c in calls], ["先", "后"])
        self.assertGreaterEqual(calls[1][2] - calls[0][2], 0.14)   # 后一次等前一次做完

    async def test_adding_never_blocks_and_failures_are_contained(self):
        batcher, calls = self.make(settle=0.05, fail=True)
        started = time.monotonic()
        batcher.add(1, "private", 0, self.item("一"))
        self.assertLess(time.monotonic() - started, 0.02)  # 回合只排队，不等
        with self.assertLogs("taoyao.deferred_profile", level="ERROR"):
            await batcher.flush()
        batcher.add(1, "private", 0, self.item("二"))       # 上次失败不影响之后
        await batcher.flush()
        self.assertEqual(len(calls), 2)

    def test_merge_keeps_new_user_flag_and_latest_context_and_bounds_length(self):
        merged = SixAxisBatcher.merge([
            self.item("a" * 400, is_new=True, context=[{"c": 1}]),
            self.item("b" * 400, raw="原句B", context=None)], max_chars=600)
        self.assertTrue(merged["profile"]["is_new"])
        self.assertEqual(merged["analysis_context"], [{"c": 1}])
        self.assertLessEqual(len(merged["text"]), 600)
        self.assertTrue(merged["text"].endswith("b" * 400))
        self.assertIn("原句B", merged["raw_text"])


if __name__ == "__main__":
    unittest.main()
