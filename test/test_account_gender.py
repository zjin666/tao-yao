"""资料里的性别：默认取 QQ 账号上的性别，用户自己说过的优先，账号上没设置就当没有。"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.profiles import ProfileStore


class DefaultGenderStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = ProfileStore(str(Path(self.folder.name) / "p.db"))

    async def asyncTearDown(self):
        self.store.close()
        self.folder.cleanup()

    async def test_account_gender_fills_an_empty_profile(self):
        await self.store.ensure(7, "甲")
        self.assertEqual(await self.store.set_default_gender(7, "女"), "set")
        self.assertEqual((await self.store.get(7))["basic_info"]["gender"], "女")

    async def test_gender_the_user_already_stated_is_never_replaced(self):
        await self.store.ensure(7, "甲")
        await self.store.update_basic_info(7, {"gender": "男"}, scene="group", group_id=1)   # 只在某个群里说过
        self.assertEqual(await self.store.set_default_gender(7, "女"), "kept")
        self.assertEqual((await self.store.get(7))["basic_info"]["gender"], "男")      # 仍是用户自己说的
        await self.store.ensure(8, "乙")
        await self.store.update_basic_info(8, {"gender": "男"}, scene="private")
        self.assertEqual(await self.store.set_default_gender(8, "女"), "kept")

    async def test_missing_profile_and_unknown_gender_do_nothing(self):
        self.assertEqual(await self.store.set_default_gender(99, "女"), "missing")
        await self.store.ensure(7, "甲")
        self.assertEqual(await self.store.set_default_gender(7, ""), "kept")


def bot_with(store, reply=None):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.profiles_enabled = True
    bot.profiles = store
    bot.server = SimpleNamespace(call=AsyncMock(return_value=reply or {"data": {}}))
    return bot


class AccountGenderFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_sex_on_the_message_is_used_without_calling_the_api(self):
        store = SimpleNamespace(set_default_gender=AsyncMock(return_value="set"))
        bot = bot_with(store)
        bot._note_account_gender({"user_id": 7, "self_id": 1, "sender": {"sex": "female"}})
        await asyncio.gather(*bot._gender_tasks)
        store.set_default_gender.assert_awaited_once_with(7, "女")
        bot.server.call.assert_not_awaited()

    async def test_unknown_sex_asks_the_account_once_and_no_setting_means_none(self):
        store = SimpleNamespace(set_default_gender=AsyncMock(return_value="set"))
        bot = bot_with(store, {"data": {"sex": "unknown"}})
        for _ in range(2):
            bot._note_account_gender({"user_id": 7, "self_id": 1, "sender": {"sex": "unknown"}})
            await asyncio.gather(*bot._gender_tasks)
        self.assertEqual(bot.server.call.await_count, 1)           # 只问一次
        store.set_default_gender.assert_not_awaited()               # 没设置就当没有

    async def test_api_value_is_used_and_missing_profile_is_retried_on_the_next_message(self):
        outcomes = ["missing", "set"]
        store = SimpleNamespace(set_default_gender=AsyncMock(side_effect=outcomes))
        bot = bot_with(store, {"data": {"sex": "male"}})
        for _ in range(3):
            bot._note_account_gender({"user_id": 7, "self_id": 1, "sender": {}})
            await asyncio.gather(*bot._gender_tasks)
        self.assertEqual(store.set_default_gender.await_count, 2)   # 第二次成功后不再处理
        self.assertEqual(bot.server.call.await_count, 1)            # 接口只查一次
        self.assertEqual(store.set_default_gender.await_args.args, (7, "男"))

    async def test_the_bot_itself_is_skipped(self):
        bot = bot_with(SimpleNamespace(set_default_gender=AsyncMock()))
        bot._note_account_gender({"user_id": 1, "self_id": 1, "sender": {"sex": "female"}})
        self.assertFalse(getattr(bot, "_gender_tasks", set()))


class ProfileTextTests(unittest.TestCase):
    def test_gender_is_shown_as_an_addressing_reference(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        text = bot._profile_info_text({"gender": "女"})
        self.assertIn("性别 女", text)
        self.assertIn("称呼参考", text)


if __name__ == "__main__":
    unittest.main()
