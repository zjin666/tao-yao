"""用户自己先说了早安/晚安，当天对应的主动问候跳过；“[回复]”这种平台引用样式不能出现在桃夭的话里。"""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

from taoyao.app import TaoYaoBot
from taoyao.output_guard import guard_outgoing, internal_notice_reason


def greeting_bot(greeted=None):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {"profile": {"default_timezone": "Asia/Shanghai"}}
    marks = greeted if greeted is not None else []

    async def mark(uid, kind, date):
        marks.append((uid, kind, date))
    bot.profiles = SimpleNamespace(mark_greeted=mark)
    return bot, marks


class UserGreetingTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_greetings_are_recorded_for_the_users_local_day(self):
        today = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
        for text, kind in (("早安", "morning"), ("早上好～", "morning"), ("Good morning", "morning"),
                           ("晚安", "night"), ("晚安！", "night"), ("夜安", "night")):
            bot, marks = greeting_bot()
            await bot._note_user_greeting(5, text, {})
            self.assertEqual(marks, [(5, kind, today)], text)

    async def test_longer_or_unrelated_messages_do_not_count(self):
        for text in ("早安我的天今天好累", "我昨晚没说晚安就睡了", "你好", "", "晚安之后的配图怎么做"):
            bot, marks = greeting_bot()
            await bot._note_user_greeting(5, text, {})
            self.assertEqual(marks, [], text)

    async def test_a_greeted_day_is_skipped_by_the_scheduler(self):
        sent = []
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {"greetings": {"enabled": True, "morning_hour": 8, "night_hour": 23, "window_minutes": 60},
                      "profile": {"default_timezone": "Asia/Shanghai"}}
        bot.profiles_enabled = True
        now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai"))
        bot.config["greetings"]["morning_hour"] = now.hour
        bot.config["greetings"]["night_hour"] = (now.hour + 5) % 24
        done = {(5, "morning", now.date().isoformat())}
        bot.profiles = SimpleNamespace(
            all_profiles=AsyncMock(return_value=[{"user_id": 5}, {"user_id": 6}]),
            was_greeted=AsyncMock(side_effect=lambda uid, kind, date: (int(uid), kind, date) in done),
            mark_greeted=AsyncMock())
        bot._friend_ids = AsyncMock(return_value=[5, 6])
        bot._timezone_for = lambda profile: "Asia/Shanghai"

        async def fake_send(target, uid, intent, **kwargs):
            sent.append(uid)
            return True
        bot._send_segments = fake_send
        await bot._send_due_greetings()
        self.assertEqual(sent, [6])


class QuoteDisplayLabelTests(unittest.TestCase):
    def test_platform_quote_style_at_line_start_is_blocked(self):
        for text in ("[回复] 哼，问得这么正经。", "[回复 桃夭]: 好的", "好的。\n[回复] 你说呢", "[回复：小明] 好"):
            self.assertEqual(internal_notice_reason(text), "internal_marker", text)
            self.assertEqual(guard_outgoing(text), "", text)

    def test_ordinary_text_about_replying_is_fine(self):
        for text in ("我回复你了。", "这条[回复率]很高。"):
            self.assertNotEqual(guard_outgoing(text), "", text)


if __name__ == "__main__":
    unittest.main()
