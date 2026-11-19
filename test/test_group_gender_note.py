"""群里已确认性别的人：桃夭称呼他们要用对“他/她”，群聊里也看得到。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from taoyao.app import TaoYaoBot
from taoyao.profiles import ProfileStore


async def _empty():
    return []


class GenderTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_members_with_known_gender_are_listed_with_the_right_pronoun(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(str(Path(tmp) / "p.db"))
            try:
                rows = ((9, "守夜", {}, {"1": {"gender": "女"}}),      # 只在本群资料里
                        (7, "哈哩", {"gender": "男"}, {}),              # 在共用资料里
                        (8, "路人", {}, {}))                           # 没有性别
                for uid, name, shared, group_info in rows:
                    store._conn.execute(
                        "INSERT OR REPLACE INTO profiles(user_id, nickname, role, basic_info, basic_info_group,"
                        " created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
                        (str(uid), name, "user", json.dumps(shared), json.dumps(group_info), "t", "t"))
                    store._conn.execute(
                        "INSERT INTO user_messages(user_id, scene, group_id, role, content, created_at, message_id, routing)"
                        " VALUES(?,?,?,?,?,?,?,?)",
                        (str(uid), "group", 1, "user", "hi", "2026-11-19T00:00:00+00:00", str(uid), "{}"))
                store._conn.commit()
                self.assertEqual(dict(await store.group_genders(1)), {"守夜": "女", "哈哩": "男"})
                self.assertEqual(await store.group_genders(2), [])        # 别的群里没发过言的不列
                bot = TaoYaoBot.__new__(TaoYaoBot)
                bot.profiles_enabled = True
                bot.profiles = store
                note = await bot._group_gender_note(1)
                self.assertIn("【群友称谓】", note)
                self.assertIn("守夜：女，称“她”", note)
                self.assertIn("哈哩：男，称“他”", note)
                self.assertNotIn("路人", note)
            finally:
                store.close()

    async def test_nothing_is_added_when_no_gender_is_known_or_profiles_are_off(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.profiles_enabled = True
        bot.profiles = SimpleNamespace(group_genders=lambda *a, **k: _empty())
        self.assertEqual(await bot._group_gender_note(1), "")
        bot.profiles_enabled = False
        self.assertEqual(await bot._group_gender_note(1), "")


if __name__ == "__main__":
    unittest.main()
