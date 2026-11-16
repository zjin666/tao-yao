"""owner 说“清空守夜的画像吧”：交给模型判断，找到唯一的人后先确认，确认后才删。"""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import DELETE_HINT_RE, TaoYaoBot

OWNER = 900001


def make_bot(answer, people=None):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {"owner_qq": OWNER}
    bot._pending_delete = {}
    bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value=answer))
    bot.sent = []

    async def send(target, target_id, text, **kwargs):
        bot.sent.append(str(text))
        return True
    bot._send_segments = send
    bot.profiles = SimpleNamespace(
        all_profiles=AsyncMock(return_value=people if people is not None else
                               [{"user_id": 3975938552, "nickname": "蓝色大肥鱼守夜"}, {"user_id": OWNER, "nickname": "哥哥"}]),
        delete_user=AsyncMock())
    bot.memory = SimpleNamespace(forget_session=AsyncMock())
    bot.impressions = SimpleNamespace(forget=AsyncMock())
    bot.privacy_index = None
    return bot


async def say(bot, text, user=OWNER, is_group=False):
    return await bot._handle_delete_request("private", user, user, text, is_group, None)


class DeleteOtherTests(unittest.IsolatedAsyncioTestCase):
    def test_the_prefilter_needs_an_action_and_an_object(self):
        for text in ("清空守夜的画像吧", "把守夜的记录删掉", "重置一下他的六维", "他的画像清空"):
            self.assertTrue(DELETE_HINT_RE.search(text), text)
        for text in ("你好", "守夜今天说了什么", "清空对话", "我把记录发给你"):
            self.assertIsNone(DELETE_HINT_RE.search(text), text)

    async def test_owner_names_someone_then_confirms_and_only_then_it_is_deleted(self):
        bot = make_bot('{"action":"other","target":"守夜"}')
        self.assertTrue(await say(bot, "清空守夜的画像吧"))
        bot.profiles.delete_user.assert_not_awaited()
        self.assertIn("蓝色大肥鱼守夜", bot.sent[-1])
        self.assertIn("确认删除", bot.sent[-1])
        self.assertTrue(await say(bot, "确认删除"))
        bot.profiles.delete_user.assert_awaited_once_with(3975938552)
        self.assertIn("已经删掉了", bot.sent[-1])

    async def test_cancel_and_unrelated_messages_do_not_delete(self):
        bot = make_bot('{"action":"other","target":"守夜"}')
        await say(bot, "清空守夜的画像吧")
        self.assertTrue(await say(bot, "算了不删了"))
        bot.profiles.delete_user.assert_not_awaited()
        self.assertFalse(await say(bot, "好呀"))                          # 取消后，别的话不会被当成确认

    async def test_unknown_name_and_ambiguous_names_ask_for_more(self):
        bot = make_bot('{"action":"other","target":"路人甲"}')
        await say(bot, "清空路人甲的画像")
        self.assertIn("没找到", bot.sent[-1])
        bot = make_bot('{"action":"other","target":"守夜"}', people=[
            {"user_id": 11111, "nickname": "守夜人"}, {"user_id": 22222, "nickname": "小守夜"}])
        await say(bot, "清空守夜的画像")
        self.assertIn("不止一个", bot.sent[-1])
        bot.profiles.delete_user.assert_not_awaited()

    async def test_a_non_owner_cannot_clear_someone_else_and_groups_are_refused(self):
        bot = make_bot('{"action":"other","target":"守夜"}')
        await say(bot, "清空守夜的画像", user=555)
        self.assertIn("只有哥哥", bot.sent[-1])
        await say(bot, "清空守夜的画像", is_group=True)
        self.assertIn("私聊", bot.sent[-1])
        bot.profiles.delete_user.assert_not_awaited()

    async def test_the_target_must_appear_verbatim_in_the_message(self):
        bot = make_bot('{"action":"other","target":"某个编出来的人"}')
        self.assertFalse(await say(bot, "清空画像吧"))
        bot.profiles.delete_user.assert_not_awaited()

    async def test_model_self_delete_goes_through_the_old_confirmation(self):
        bot = make_bot('{"action":"self","target":""}')
        self.assertTrue(await say(bot, "帮我重置一下我的画像", user=555))
        self.assertIn("确认删除我的记录", bot.sent[-1])
        bot.profiles.delete_user.assert_not_awaited()

    async def test_model_none_means_normal_chat(self):
        bot = make_bot('{"action":"none","target":""}')
        self.assertFalse(await say(bot, "我昨天把记录清空了"))


if __name__ == "__main__":
    unittest.main()
