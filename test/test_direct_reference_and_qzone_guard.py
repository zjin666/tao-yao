"""有人明确回应桃夭时不受插话冷却限制；只发表情包/图片不能触发或确认说说操作；群聊的事不是说说评论。"""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot


def ambient_bot(chain=3, last_ago=88.0):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot._ambient_config = lambda: {"enabled": True, "topic_max_chain": 3, "topic_min_interval_seconds": 20,
                                   "cooldown_seconds": 300, "max_per_hour": 3}
    bot._group_reply_budget = SimpleNamespace(paused=lambda *a, **k: False)
    bot._group_msg_counts = {}
    bot._group_ambient_times = {}
    bot._group_topic_chain = {1: chain}
    bot._group_last_ambient = {1: time.monotonic() - last_ago}
    bot._group_last_reply_at = {}
    bot._locks = {}
    bot._ambient_directed_elsewhere = AsyncMock(return_value="")
    bot._note_group_arrival = lambda *a, **k: None
    bot._handle_text = AsyncMock()
    return bot


async def call(bot, **kwargs):
    return await bot._maybe_ambient(1, 7, "哥哥", "真不帮忙吗,我倒不觉得哦", "owner", "group:1",
                                    high_priority=True, same_topic=True, message_id="m", **kwargs)


class DirectReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_reply_to_her_words_is_not_blocked_by_the_chain_limit_and_cooldown(self):
        bot = ambient_bot(chain=3, last_ago=88)
        self.assertTrue(await call(bot, direct_reference=True))
        bot._handle_text.assert_awaited_once()

    async def test_without_a_direct_reference_the_old_limits_still_apply(self):
        bot = ambient_bot(chain=3, last_ago=88)
        self.assertFalse(await call(bot))
        bot._handle_text.assert_not_awaited()

    async def test_direct_references_still_stop_at_a_higher_chain_limit_and_keep_the_minimum_gap(self):
        self.assertFalse(await call(ambient_bot(chain=8, last_ago=88), direct_reference=True))   # 防和别的机器人互相接话
        self.assertFalse(await call(ambient_bot(chain=3, last_ago=5), direct_reference=True))    # 最短间隔


class TypedPartTests(unittest.TestCase):
    def test_typed_text_drops_the_attached_media_descriptions(self):
        self.assertEqual(TaoYaoBot._typed_part("发吧\n[图片内容] 一张樱花"), "发吧")
        self.assertEqual(TaoYaoBot._typed_part("[表情包内容] 形象：黄色圆脸；可能想表达：无奈"), "")
        self.assertEqual(TaoYaoBot._typed_part("你不去回复一下？"), "你不去回复一下？")


class QzoneGuardTests(unittest.IsolatedAsyncioTestCase):
    def bot(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {}
        bot._owner_qq = lambda: 9
        bot.memory = SimpleNamespace(recent=AsyncMock(return_value=[]))
        bot._qzone_pending_reply = {9: {"at": time.monotonic(), "item": {"name": "某人", "content": "评论"},
                                         "content": "拟回复", "view": {}}}
        bot._infer_owner_qq_intent = AsyncMock(return_value={"action": "confirm_reply", "content": ""})
        return bot

    async def test_a_sticker_alone_cannot_confirm_a_pending_public_reply(self):
        bot = self.bot()
        handled = await bot._handle_owner_qq_intent("private", 9, "[表情包内容] 形象：黄色圆脸；神态：斜眼微笑")
        self.assertFalse(handled)
        bot._infer_owner_qq_intent.assert_not_awaited()
        self.assertIn(9, bot._qzone_pending_reply)          # 待确认的回复原样保留，没有被发出

    async def test_the_classifier_is_told_group_matters_are_not_qzone_comments(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.llm = bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value='{"action":"none"}'))
        await bot._infer_owner_qq_intent("你不去回复一下？", [{"role": "user", "content": "群里再说你口是心非呢"}])
        system = bot.utility_llm.chat.await_args.args[0][0]["content"]
        self.assertIn("说的是群聊里的事", system)
        self.assertIn("表情包、图片、换了话题、反问、提问都不是确认", system)


if __name__ == "__main__":
    unittest.main()
