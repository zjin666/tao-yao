"""私聊总围着同一件事转：回复提示要求不反复提旧事、能换话题；长期记忆整理要丢掉已完成/被叫停的事。"""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.memory import MemoryStore
from taoyao.persona import build_system_prompt


class TopicRotationTests(unittest.TestCase):
    def test_prompt_stops_reusing_the_same_matter_and_allows_new_topics(self):
        for kwargs in ({"role": "owner"}, {"role": "user", "is_group": True}):
            prompt = build_system_prompt("人设", [], **kwargs)
            self.assertIn("不反复拿同一件事当关心、催促或调侃的抓手", prompt)
            self.assertIn("对方说过“不用了、不要再催”的事不再提", prompt)
            self.assertIn("接不上时换个新话题", prompt)


class MemorySummaryRuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_prompt_drops_finished_matters_and_limits_one_topic(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "m.db")
            try:
                llm = SimpleNamespace(chat=AsyncMock(return_value="新摘要"))
                store._set_summary("owner_private:1", "旧摘要")
                prepared = {"transcript": "哥哥：不要再催了", "max_chars": 700, "epoch": 0}
                await store.summarize_prepared("owner_private:1", llm, prepared)
                prompt = llm.chat.await_args.args[0][1]["content"]
                self.assertIn("新聊天与旧记忆冲突时以新聊天为准", prompt)
                self.assertIn("不继续写成待办", prompt)
                self.assertIn("每件事或话题最多一两句", prompt)
                self.assertIn("口头禅和固定调侃套路不当作事实记录", prompt)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
