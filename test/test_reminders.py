import asyncio
import json
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

from taoyao import reminders as rl
from taoyao.app import ReplyIntent, TaoYaoBot
from taoyao.profiles import ProfileStore
from taoyao.turn_timing import set_value, turn_timing_context

TZ = 'Asia/Shanghai'
ZONE = ZoneInfo(TZ)
OWNER = 111
USER = 222


def epoch(text):
    return datetime.strptime(text, '%Y-%m-%d %H:%M').replace(tzinfo=ZONE).timestamp()


def local(value):
    return datetime.fromtimestamp(value, ZONE).strftime('%Y-%m-%d %H:%M')


class LogicTests(unittest.TestCase):
    def test_once_daily_weekly_monthly_yearly(self):
        now = epoch('2026-11-11 10:00')                               # a Wednesday
        self.assertEqual(local(rl.next_occurrence('none', {'at': '2026-11-12 08:00'}, TZ, now)), '2026-11-12 08:00')
        self.assertIsNone(rl.next_occurrence('none', {'at': '2026-11-10 08:00'}, TZ, now))        # already past
        self.assertEqual(local(rl.next_occurrence('daily', {'hour': 8, 'minute': 0}, TZ, now)), '2026-11-12 08:00')
        self.assertEqual(local(rl.next_occurrence('daily', {'hour': 18, 'minute': 30}, TZ, now)), '2026-11-11 18:30')
        self.assertEqual(local(rl.next_occurrence('weekly', {'weekday': 4, 'hour': 9, 'minute': 0}, TZ, now)), '2026-11-13 09:00')
        self.assertEqual(local(rl.next_occurrence('weekly', {'weekday': 2, 'hour': 9, 'minute': 0}, TZ, now)), '2026-11-18 09:00')
        self.assertEqual(local(rl.next_occurrence('monthly', {'day': 5, 'hour': 9, 'minute': 0}, TZ, now)), '2026-12-05 09:00')
        self.assertEqual(local(rl.next_occurrence('yearly', {'month': 3, 'day': 5, 'hour': 9, 'minute': 0}, TZ, now)), '2027-03-05 09:00')

    def test_month_end_and_leap_day_are_clamped(self):
        self.assertEqual(local(rl.next_occurrence('monthly', {'day': 31, 'hour': 9, 'minute': 0}, TZ, epoch('2026-02-10 10:00'))), '2026-02-28 09:00')
        self.assertEqual(local(rl.next_occurrence('yearly', {'month': 2, 'day': 29, 'hour': 9, 'minute': 0}, TZ, epoch('2026-03-01 00:00'))), '2027-02-28 09:00')
        self.assertEqual(local(rl.next_occurrence('yearly', {'month': 2, 'day': 29, 'hour': 9, 'minute': 0}, TZ, epoch('2027-03-01 00:00'))), '2028-02-29 09:00')

    def test_each_person_is_scheduled_in_their_own_timezone(self):
        now = epoch('2026-11-11 10:00')
        east = rl.next_occurrence('daily', {'hour': 8, 'minute': 0}, 'Asia/Shanghai', now)
        west = rl.next_occurrence('daily', {'hour': 8, 'minute': 0}, 'America/New_York', now)
        self.assertNotEqual(east, west)

    def test_validate_accepts_good_input_and_explains_bad_input(self):
        now = epoch('2026-11-11 10:00')
        item, why = rl.validate({'kind': 'reminder', 'content': '交周报', 'repeat': 'weekly', 'weekday': 4, 'hour': 17}, now, TZ)
        self.assertEqual((why, local(item['due_at'])), ('', '2026-11-13 17:00'))
        for bad, text in (({'content': '', 'repeat': 'daily', 'hour': 8}, '提醒什么'),
                          ({'content': '吃药', 'repeat': 'none', 'at': '明天'}, '什么时候'),
                          ({'content': '吃药', 'repeat': 'none', 'at': '2026-11-11 09:00'}, '已经过去'),
                          ({'content': '交周报', 'repeat': 'weekly'}, '每周几'),
                          ({'kind': 'birthday', 'subject': '我的生日', 'month': 2, 'day': 30}, '不存在'),
                          ({'kind': 'birthday', 'subject': '我的生日'}, '几月几号')):
            item, why = rl.validate(bad, now, TZ)
            self.assertIsNone(item)
            self.assertIn(text, why)

    def test_birthdays_and_dates_repeat_yearly_and_only_others_dates_have_a_lead(self):
        now = epoch('2026-11-11 10:00')
        mine, _ = rl.validate({'kind': 'birthday', 'subject': '我的生日', 'month': 3, 'day': 5}, now, TZ)
        self.assertEqual((mine['repeat'], mine['lead_days'], local(mine['due_at'])), ('yearly', 0, '2027-03-05 09:00'))
        mom, _ = rl.validate({'kind': 'date', 'subject': '妈妈的生日', 'month': 11, 'day': 20}, now, TZ)
        self.assertEqual(mom['lead_days'], 3)
        self.assertTrue(rl.self_birthday({'subject': '我的生日'}))
        self.assertFalse(rl.self_birthday({'subject': '妈妈的生日'}))

    def test_describe_is_readable(self):
        item, _ = rl.validate({'kind': 'reminder', 'content': '交周报', 'repeat': 'weekly', 'weekday': 4, 'hour': 17}, epoch('2026-11-11 10:00'), TZ)
        self.assertIn('每周五 17:00', rl.describe(item))


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def chat(self, messages, *a, **k):
        self.calls.append(messages)
        reply = self.reply
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


class StoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStore(Path(self.tmp.name) / 'p.db')

    async def asyncTearDown(self):
        self.store._conn.close()
        self.tmp.cleanup()

    def item(self, due, **extra):
        return {'kind': 'reminder', 'content': '吃药', 'subject': '', 'repeat': 'none', 'spec': {'at': 'x'},
                'tz': TZ, 'due_at': due, 'lead_days': 0, **extra}

    async def test_add_list_due_update_cancel(self):
        now = time.time()
        a = await self.store.add_reminder(USER, 'private', 0, self.item(now + 3600))
        b = await self.store.add_reminder(USER, 'group', 55, self.item(now + 60, content='群里的'))
        self.assertEqual(len(await self.store.active_reminders(USER)), 2)
        self.assertEqual([r['content'] for r in await self.store.active_reminders(USER, 'group', 55)], ['群里的'])   # group view
        self.assertEqual(await self.store.active_reminders(USER, 'group', 66), [])
        self.assertEqual(await self.store.due_reminders(now), [])
        self.assertEqual([r['id'] for r in await self.store.due_reminders(now + 120)], [b])
        await self.store.update_reminder(b, status='done')
        self.assertEqual(await self.store.due_reminders(now + 120), [])
        self.assertFalse(await self.store.cancel_reminder(a, OWNER))             # someone else cannot cancel it
        self.assertTrue(await self.store.cancel_reminder(a, USER))
        self.assertEqual(await self.store.count_reminders(USER), 0)

    async def test_a_lead_reminder_is_due_before_the_day(self):
        now = time.time()
        rid = await self.store.add_reminder(USER, 'private', 0, self.item(now + 2 * 86400, kind='date', lead_days=3))
        self.assertEqual([r['id'] for r in await self.store.due_reminders(now)], [rid])
        await self.store.update_reminder(rid, lead_sent=now + 2 * 86400)
        self.assertEqual(await self.store.due_reminders(now), [])

    async def test_a_recalled_message_is_removed_only_from_its_own_chat(self):
        await self.store.ensure(USER, '小明', 'user')
        await self.store.add_message(USER, 'private', 0, 'user', '撤回的话', message_id='m1')
        await self.store.add_message(USER, 'private', 0, 'user', '留下的话', message_id='m2')
        await self.store.add_message(USER, 'group', 55, 'user', '群里同id', message_id='m1')
        removed = await self.store.remove_message_by_id('m1', 'private', 0, USER)
        self.assertEqual(removed, ['撤回的话'])
        left = [r[0] for r in self.store._conn.execute("SELECT content FROM user_messages ORDER BY id").fetchall()]
        self.assertEqual(left, ['留下的话', '群里同id'])


def make_bot(reply, tmp, store):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = {'owner_qq': OWNER, 'profile': {'default_timezone': TZ}}
    bot.utility_llm = FakeLLM(reply)
    bot.llm = bot.utility_llm
    bot.profiles = store
    bot.profiles_enabled = True
    bot.memory = SimpleNamespace(add_message=AsyncMock(), remove_message=AsyncMock())
    bot._owner_qq = lambda: OWNER
    bot._send_segments = AsyncMock(return_value=True)
    bot._recalled = {}
    return bot


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStore(Path(self.tmp.name) / 'p.db')
        await self.store.ensure(USER, '小明', 'user')

    async def asyncTearDown(self):
        self.store._conn.close()
        self.tmp.cleanup()

    async def say(self, bot, text, is_group=False, group_id=0):
        profile = await self.store.get(USER)
        return await bot._handle_reminder_intent('group' if is_group else 'private', group_id if is_group else USER,
                                                 USER, text, profile, is_group, group_id)

    def sent(self, bot):
        return bot._send_segments.await_args.args[2]

    async def test_chatting_that_only_mentions_the_word_is_not_a_reminder(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'none'}, tmp, self.store)
            self.assertFalse(await self.say(bot, '你提醒得对，谢谢'))
            bot._send_segments.assert_not_awaited()
            self.assertEqual(await self.store.count_reminders(USER), 0)
            quiet = make_bot({'action': 'create'}, tmp, self.store)
            self.assertFalse(await self.say(quiet, '今天天气不错'))
            self.assertEqual(quiet.utility_llm.calls, [])                  # no hint word -> no model call

    async def test_a_reminder_is_saved_and_confirmed_through_the_unified_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            at = local(time.time() + 7200)
            bot = make_bot({'action': 'create', 'kind': 'reminder', 'content': '吃药', 'repeat': 'none', 'at': at}, tmp, self.store)
            self.assertTrue(await self.say(bot, '两小时后提醒我吃药'))
            saved = await self.store.active_reminders(USER)
            self.assertEqual([r['content'] for r in saved], ['吃药'])
            intent = self.sent(bot)
            self.assertIsInstance(intent, ReplyIntent)
            self.assertIn('吃药', intent.facts)
            self.assertIn(at, intent.facts)
            bot.memory.add_message.assert_awaited()                         # the request stays in the conversation

    async def test_a_failed_parse_is_not_reported_as_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'create', 'kind': 'reminder', 'content': '吃药', 'repeat': 'none', 'at': '稍后'}, tmp, self.store)
            self.assertTrue(await self.say(bot, '提醒我吃药'))
            self.assertEqual(await self.store.count_reminders(USER), 0)
            self.assertIn('不要说设好了', self.sent(bot).situation)

    async def test_list_and_cancel_by_number_only_touch_the_askers_reminders(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = time.time()
            for text in ('吃药', '交周报'):
                await self.store.add_reminder(USER, 'private', 0, {'kind': 'reminder', 'content': text, 'repeat': 'none',
                                                                    'spec': {'at': 'x'}, 'tz': TZ, 'due_at': now + 3600, 'lead_days': 0})
            await self.store.add_reminder(OWNER, 'private', 0, {'kind': 'reminder', 'content': '别人的', 'repeat': 'none',
                                                                  'spec': {'at': 'x'}, 'tz': TZ, 'due_at': now + 3600, 'lead_days': 0})
            bot = make_bot({'action': 'list'}, tmp, self.store)
            await self.say(bot, '我有哪些提醒')
            self.assertNotIn('别人的', self.sent(bot).facts)
            self.assertIn('吃药', self.sent(bot).facts)
            bot = make_bot({'action': 'cancel', 'index': 2}, tmp, self.store)
            await self.say(bot, '取消第2个提醒')
            self.assertEqual([r['content'] for r in await self.store.active_reminders(USER)], ['吃药'])
            bot = make_bot({'action': 'cancel', 'index': 9}, tmp, self.store)
            await self.say(bot, '取消那个提醒')
            self.assertEqual(await self.store.count_reminders(USER), 1)     # unknown number cancels nothing

    async def test_the_active_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'create', 'kind': 'reminder', 'content': 'x', 'repeat': 'daily', 'hour': 8}, tmp, self.store)
            bot.config['reminders'] = {'max_active': 1}
            await self.say(bot, '每天8点提醒我x')
            await self.say(bot, '每天8点提醒我x')
            self.assertEqual(await self.store.count_reminders(USER), 1)
            self.assertIn('没有设', self.sent(bot).situation)

    async def test_birthdays_are_always_delivered_privately_even_when_asked_in_a_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = make_bot({'action': 'create', 'kind': 'birthday', 'subject': '我的生日', 'month': 3, 'day': 5}, tmp, self.store)
            await self.say(bot, '我的生日是3月5日', is_group=True, group_id=55)
            row = (await self.store.active_reminders(USER))[0]
            self.assertEqual((row['scene'], row['group_id'], row['kind'], row['repeat']), ('private', 0, 'birthday', 'yearly'))
            self.assertIn('私聊', self.sent(bot).facts)
            self.assertEqual(await self.store.active_reminders(USER, 'group', 55), [])    # not visible from the group

    async def test_other_peoples_dates_get_a_lead_and_skip_it_when_already_inside_the_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            soon = datetime.fromtimestamp(time.time() + 86400, ZONE)
            bot = make_bot({'action': 'create', 'kind': 'birthday', 'subject': '妈妈的生日', 'month': soon.month, 'day': soon.day,
                            'hour': soon.hour}, tmp, self.store)
            await self.say(bot, '妈妈的生日是明天')
            row = (await self.store.active_reminders(USER))[0]
            self.assertEqual((row['kind'], row['lead_days']), ('date', 3))
            self.assertEqual(row['lead_sent'], row['due_at'])               # one day away, the 3-day lead is already over


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStore(Path(self.tmp.name) / 'p.db')
        await self.store.ensure(USER, '小明', 'user')
        self.bot = make_bot({'action': 'none'}, self.tmp.name, self.store)

    async def asyncTearDown(self):
        self.store._conn.close()
        self.tmp.cleanup()

    async def add(self, **extra):
        base = {'kind': 'reminder', 'content': '吃药', 'subject': '', 'repeat': 'none', 'spec': {'at': 'x'}, 'tz': TZ,
                'due_at': time.time() - 5, 'lead_days': 0}
        return await self.store.add_reminder(USER, extra.pop('scene', 'private'), extra.pop('group_id', 0), {**base, **extra},
                                             extra.pop('lead_sent', 0.0))

    async def test_a_due_reminder_goes_out_through_the_unified_flow_and_is_marked_done(self):
        rid = await self.add()
        await self.bot._reminder_tick()
        args = self.bot._send_segments.await_args
        self.assertEqual(args.args[:2], ('private', USER))
        self.assertIsInstance(args.args[2], ReplyIntent)
        self.assertIn('吃药', args.args[2].facts)
        self.assertEqual(await self.store.due_reminders(time.time()), [])
        self.assertEqual(await self.store.count_reminders(USER), 0)
        await self.bot._reminder_tick()
        self.assertEqual(self.bot._send_segments.await_count, 1)             # never twice

    async def test_a_group_reminder_goes_to_the_group_mentioning_the_person(self):
        await self.add(scene='group', group_id=55)
        await self.bot._reminder_tick()
        args = self.bot._send_segments.await_args
        self.assertEqual(args.args[:2], ('group', 55))
        self.assertEqual(args.kwargs['at_user_id'], USER)

    async def test_a_failed_send_is_retried_and_eventually_given_up_not_marked_done(self):
        rid = await self.add()
        self.bot._send_segments = AsyncMock(return_value=False)
        await self.bot._reminder_tick()
        self.assertEqual(await self.store.count_reminders(USER), 1)         # still active
        for _ in range(5):
            await self.bot._reminder_tick()
        row = self.store._conn.execute('SELECT status, attempts FROM reminders WHERE id = ?', (rid,)).fetchone()
        self.assertEqual(row['status'], 'failed')

    async def test_a_repeating_reminder_moves_to_its_next_time(self):
        rid = await self.add(repeat='daily', spec={'hour': 8, 'minute': 0})
        await self.bot._reminder_tick()
        row = self.store._conn.execute('SELECT status, due_at FROM reminders WHERE id = ?', (rid,)).fetchone()
        self.assertEqual(row['status'], 'active')
        self.assertGreater(row['due_at'], time.time())
        self.assertLess(row['due_at'] - time.time(), 86400 + 5)

    async def test_a_late_reminder_says_so_but_does_not_invent_a_reason(self):
        await self.add(due_at=time.time() - 3600)
        await self.bot._reminder_tick()
        self.assertIn('晚了点', self.bot._send_segments.await_args.args[2].facts)

    async def test_a_birthday_gets_a_greeting_and_comes_back_next_year(self):
        rid = await self.add(kind='birthday', subject='我的生日', repeat='yearly', spec={'month': 3, 'day': 5, 'hour': 9, 'minute': 0})
        await self.bot._reminder_tick()
        intent = self.bot._send_segments.await_args.args[2]
        self.assertIn('生日', intent.situation)
        row = self.store._conn.execute('SELECT status, due_at FROM reminders WHERE id = ?', (rid,)).fetchone()
        self.assertEqual(row['status'], 'active')
        self.assertGreater(row['due_at'], time.time() + 20 * 86400)

    async def test_a_lead_notice_comes_first_then_the_day_itself(self):
        due = time.time() + 2 * 86400
        rid = await self.add(kind='date', subject='妈妈的生日', repeat='yearly', due_at=due, lead_days=3,
                             spec={'month': 1, 'day': 1, 'hour': 9, 'minute': 0})
        await self.bot._reminder_tick()
        self.assertIn('提前提醒', self.bot._send_segments.await_args.args[2].situation)
        self.assertEqual(await self.store.due_reminders(time.time()), [])    # lead sent, day not yet
        await self.bot._reminder_tick()
        self.assertEqual(self.bot._send_segments.await_count, 1)             # not repeated


class RecallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStore(Path(self.tmp.name) / 'p.db')
        await self.store.ensure(USER, '小明', 'user')
        self.bot = make_bot({'action': 'none'}, self.tmp.name, self.store)

    async def asyncTearDown(self):
        self.store._conn.close()
        self.tmp.cleanup()

    async def test_a_recalled_private_message_disappears_from_history_and_context(self):
        await self.store.add_message(USER, 'private', 0, 'user', '说错了的话', message_id='m1')
        await self.bot._handle_recall({'notice_type': 'friend_recall', 'user_id': USER, 'message_id': 'm1', 'self_id': 999})
        self.assertEqual(self.store._conn.execute('SELECT COUNT(*) FROM user_messages').fetchone()[0], 0)
        self.bot.memory.remove_message.assert_awaited_with(f'user_private:{USER}', 'user', '说错了的话')
        self.assertIn(f'private:{USER}:m1', self.bot._recalled)

    async def test_a_recalled_group_message_is_removed_for_that_group_only(self):
        await self.store.add_message(USER, 'group', 55, 'user', '群里撤回的', message_id='g1')
        await self.store.add_message(USER, 'group', 66, 'user', '另一个群', message_id='g1')
        await self.bot._handle_recall({'notice_type': 'group_recall', 'group_id': 55, 'user_id': USER, 'message_id': 'g1', 'self_id': 999})
        left = [r[0] for r in self.store._conn.execute('SELECT content FROM user_messages').fetchall()]
        self.assertEqual(left, ['另一个群'])

    async def test_recalling_the_bots_own_message_or_without_an_id_does_nothing(self):
        await self.bot._handle_recall({'notice_type': 'group_recall', 'group_id': 55, 'user_id': 999, 'message_id': 'x', 'self_id': 999})
        await self.bot._handle_recall({'notice_type': 'friend_recall', 'user_id': USER, 'self_id': 999})
        self.assertEqual(self.bot._recalled, {})

    async def test_a_reply_in_progress_for_a_recalled_message_is_not_sent(self):
        @turn_timing_context
        async def turn(recalled):
            set_value('message_key', f'private:{USER}:m9')
            if recalled:
                await self.bot._handle_recall({'notice_type': 'friend_recall', 'user_id': USER, 'message_id': 'm9', 'self_id': 999})
            return self.bot._turn_recalled()
        self.assertFalse(await turn(False))
        self.assertTrue(await turn(True))

    async def test_the_real_send_step_refuses_to_answer_a_recalled_message(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot._recalled = {f'private:{USER}:m5': time.monotonic()}

        @turn_timing_context
        async def turn():
            set_value('message_key', f'private:{USER}:m5')
            return await TaoYaoBot._send_segments(bot, 'private', USER, ReplyIntent('', '回答', '事实'))
        self.assertFalse(await turn())                                       # returns before touching the model or QQ

    async def test_the_recall_check_is_off_outside_a_turn(self):
        self.assertFalse(self.bot._turn_recalled())                          # proactive sends (reminders etc.) are never blocked


class GroupJoinGreetingTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_greeting_after_joining_a_group_goes_through_the_unified_flow(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {'group_join_greeting': {'enabled': True, 'text': '旧的固定文字'}}
        bot._roles = lambda: {'auto_whitelist_joined_groups': True}
        bot._add_allowed_group = lambda gid: True
        bot._known_groups, bot._group_last_activity = set(), {}
        bot._send_segments = AsyncMock(return_value=True)
        await bot._handle_notice({'notice_type': 'group_increase', 'group_id': 55, 'user_id': 999, 'self_id': 999})
        args = bot._send_segments.await_args.args
        self.assertEqual(args[:2], ('group', 55))
        self.assertIsInstance(args[2], ReplyIntent)                          # no fixed text any more
        self.assertNotIn('旧的固定文字', args[2].situation + args[2].facts)


class ContextTests(unittest.IsolatedAsyncioTestCase):
    def bot(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {'owner_qq': OWNER, 'profile': {'default_timezone': 'America/New_York'}}
        bot._owner_qq = lambda: OWNER
        bot.profiles_enabled = True
        return bot

    def iso(self, hours_ago):
        from datetime import timezone, timedelta
        return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(timespec='seconds')

    def test_every_message_is_tagged_with_its_exact_date_weekday_and_time_in_the_persons_timezone(self):
        bot = self.bot()
        self.assertRegex(bot._time_tag(self.iso(0.01), None), r'^\[\d\d-\d\d 周[一二三四五六日] \d\d:\d\d\] $')   # even a message from seconds ago
        self.assertRegex(bot._time_tag(self.iso(9), None), r'^\[\d\d-\d\d 周[一二三四五六日] \d\d:\d\d\] $')
        self.assertRegex(bot._time_tag(self.iso(24 * 400), None), r'^\[\d{4}-\d\d-\d\d 周[一二三四五六日] \d\d:\d\d\] $')   # other years carry the year
        self.assertEqual(bot._time_tag('不是时间', None), '')                    # bad data never breaks a reply
        self.assertEqual(bot._time_tag(None, None), '')

    def test_the_tag_uses_the_persons_timezone_and_the_right_weekday(self):
        bot = self.bot()                                                        # default zone America/New_York
        tag = bot._time_tag('2026-11-11T04:02:00+00:00', None)                  # Wednesday 04:02 UTC = Tuesday 23:02 in New York
        self.assertEqual(tag, '[11-10 周二 23:02] ')
        bot.config['profile']['default_timezone'] = 'Asia/Shanghai'
        self.assertEqual(bot._time_tag('2026-11-11T04:02:00+00:00', None), '[11-11 周三 12:02] ')

    async def test_a_proactive_message_sees_what_the_person_just_said_about_their_plans(self):
        bot = self.bot()
        rows = [{'role': 'assistant', 'content': '今天的棋下到哪儿了，还是在写代码？', 'created_at': self.iso(9.1)},
                {'role': 'user', 'content': '看小说去了，代码估计大后天才会写了', 'created_at': self.iso(9.0)}]
        bot.memory = SimpleNamespace(recent=AsyncMock(return_value=rows))
        block = await bot._recent_chat_block('private', USER, USER, None)
        self.assertIn('代码估计大后天才会写了', block)
        self.assertIn('以它为准', block)                                         # plans said in chat beat profile habits
        self.assertRegex(block, r'\[\d\d-\d\d 周[一二三四五六日] \d\d:\d\d\] 对方：看小说')
        bot.memory.recent.assert_awaited_with(f'user_private:{USER}', 14)       # that person's own private session only

    async def test_public_drafts_and_unknown_targets_never_get_any_chat(self):
        bot = self.bot()
        bot.memory = SimpleNamespace(recent=AsyncMock(return_value=[{'role': 'user', 'content': '私密', 'created_at': self.iso(1)}]))
        bot.profiles = SimpleNamespace(group_context=AsyncMock(return_value=[]))
        self.assertEqual(await bot._recent_chat_block('group', 0, None, None), '')   # qzone posts/comments are generated as ('group', 0)
        self.assertEqual(await bot._recent_chat_block('private', 0, 0, None), '')
        bot.memory.recent.assert_not_awaited()
        bot.profiles.group_context.assert_not_awaited()

    async def test_a_group_proactive_message_only_sees_the_last_hour_of_that_group(self):
        bot = self.bot()
        bot.profiles = SimpleNamespace(group_context=AsyncMock(return_value=[
            {'role': 'user', 'content': '刚才聊的', 'nickname': '小明', 'created_at': self.iso(0.2)}]))
        bot.memory = SimpleNamespace(recent=AsyncMock())
        block = await bot._recent_chat_block('group', 55, USER, None)
        self.assertIn('小明：刚才聊的', block)
        bot.profiles.group_context.assert_awaited_with(55, 10, 3600)
        bot.memory.recent.assert_not_awaited()

    async def test_a_broken_history_read_does_not_stop_the_message(self):
        bot = self.bot()
        bot.memory = SimpleNamespace(recent=AsyncMock(side_effect=RuntimeError('db')))
        self.assertEqual(await bot._recent_chat_block('private', USER, USER, None), '')


if __name__ == '__main__':
    unittest.main()
