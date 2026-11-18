"""群聊：前面的图片还在识别时，后面的评论要等它入库；辅助工作（整理记忆等）走辅助模型，不用 Qwen。"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.persona import build_system_prompt


def bot_for_gate(timeout=20):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot._ambient_config = lambda: {"media_wait_seconds": timeout}
    return bot


IMAGE = {"message": [{"type": "image", "data": {}}]}
TEXT = {"message": [{"type": "text", "data": {"text": "诶，这张看着像……"}}]}


class MediaGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_later_comment_waits_for_the_earlier_image_to_be_stored(self):
        bot = bot_for_gate()
        image_gate, earlier_for_image = bot._group_media_gate(IMAGE, 1)
        self.assertEqual(earlier_for_image, [])
        _, earlier_for_comment = bot._group_media_gate(TEXT, 1)
        waiter = asyncio.create_task(bot._wait_media_gate(1, earlier_for_comment))
        await asyncio.sleep(0.05)
        self.assertFalse(waiter.done())
        image_gate.set()
        await asyncio.wait_for(waiter, 1)

    async def test_plain_text_never_blocks_anyone(self):
        bot = bot_for_gate()
        own, earlier = bot._group_media_gate(TEXT, 1)
        self.assertIsNone(own)
        _, later = bot._group_media_gate(TEXT, 1)
        await asyncio.wait_for(bot._wait_media_gate(1, later), 0.2)

    async def test_two_images_do_not_wait_on_each_other_in_a_cycle(self):
        bot = bot_for_gate()
        first, _ = bot._group_media_gate(IMAGE, 1)
        second, earlier = bot._group_media_gate(IMAGE, 1)
        self.assertEqual(len(earlier), 1)
        first.set()
        await asyncio.wait_for(bot._wait_media_gate(1, earlier), 0.5)

    async def test_wait_gives_up_after_the_timeout(self):
        bot = bot_for_gate(timeout=0.1)
        bot._group_media_gate(IMAGE, 1)
        _, earlier = bot._group_media_gate(TEXT, 1)
        await asyncio.wait_for(bot._wait_media_gate(1, earlier), 1)

    async def test_other_groups_are_not_blocked(self):
        bot = bot_for_gate()
        bot._group_media_gate(IMAGE, 1)
        _, other = bot._group_media_gate(TEXT, 2)
        self.assertEqual(other, [])


class HelperModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_summary_uses_the_helper_model(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {"memory": {}}
        bot.llm = SimpleNamespace(name="qwen")
        bot.utility_llm = SimpleNamespace(name="deepseek")
        seen = []

        async def summarize(session, llm, prepared):
            seen.append(llm)
            return ""
        bot.memory = SimpleNamespace(prepare_summary=AsyncMock(return_value={"x": 1}), summarize_prepared=summarize)
        bot._owner_summary_tasks = {}
        bot._memory_tasks = set()
        await bot._schedule_owner_summary("s")
        await asyncio.gather(*bot._memory_tasks)
        self.assertEqual([item.name for item in seen], ["deepseek"])


class GroupPromptTests(unittest.TestCase):
    def test_group_prompt_tells_her_to_attribute_lines_to_the_right_speaker(self):
        prompt = build_system_prompt("人设", [], role="owner", is_group=True)
        self.assertIn("认准是谁说的", prompt)
        self.assertIn("被回应的那个人", prompt)

    def test_ambient_prompt_makes_her_a_bystander_between_two_others(self):
        prompt = build_system_prompt("人设", [], role="owner", is_group=True, ambient=True)
        self.assertIn("你是旁观者", prompt)


if __name__ == "__main__":
    unittest.main()
