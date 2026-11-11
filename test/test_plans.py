import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

from taoyao import plans as pl
from taoyao.app import TaoYaoBot
from taoyao.profiles import ProfileStore

NY = 'America/New_York'
USER = 222
OWNER = 111
TODAY = date(2026, 11, 10)


class LogicTests(unittest.TestCase):
    def test_only_messages_with_a_time_or_schedule_trace_are_worth_a_model_call(self):
        for text in ('看小说去了，代码估计大后天才会写了', '下周都在出差', '周五考试', '明天去医院', '这几天没空'):
            self.assertTrue(pl.worth_extracting(text), text)
        for text in ('哈哈哈', '你好呀', '/状态', '好', '今天吃什么好呢这个问题真难'[:3]):
            self.assertFalse(pl.worth_extracting(text), text)
        self.assertFalse(pl.worth_extracting('明天' * 200))                           # absurdly long
        self.assertFalse(pl.worth_extracting('好看\n[图片内容] 明天的画面'))           # a time word inside an image description does not count

    def test_validate_keeps_good_plans_and_drops_bad_dates(self):
        good = {'text': '写代码要到周六才开始', 'until': '2026-11-14', 'hedged': True}
        parsed = {'plans': [good,
                            {'text': '过期的', 'until': '2026-11-09'},                  # already past
                            {'text': '太远', 'until': '2027-06-01'},                    # beyond three months
                            {'text': '', 'until': '2026-11-12'},
                            {'text': '没日期'},
                            {'text': '日期乱写', 'until': '后天'}, 'junk']}
        self.assertEqual(pl.validate(parsed, TODAY), [{'text': '写代码要到周六才开始', 'until': '2026-11-14', 'hedged': True}])
        self.assertEqual(pl.validate({}, TODAY), [])
        self.assertEqual(pl.cancel_numbers({'cancel': [1, '2', 'x', None]}), [1, 2])

    def test_render_shows_active_plans_and_flags_the_ones_whose_date_just_passed(self):
        said = datetime(2026, 11, 10, 4, 2, tzinfo=timezone.utc).isoformat()
        plans = [{'text': '写代码要到周六才开始', 'until': '2026-11-14', 'hedged': 1, 'created_at': said},
                 {'text': '周一去医院', 'until': '2026-11-09', 'hedged': 0, 'created_at': said},    # one day ago
                 {'text': '很久以前的', 'until': '2026-10-01', 'hedged': 0, 'created_at': said}]   # beyond the grace days
        block = pl.render_block(plans, TODAY, NY)
        self.assertIn('写代码要到周六才开始', block)
        self.assertIn('“估计”', block)
        self.assertIn('11-14 周六', block)
        self.assertIn('问问后来怎么样了', block)                                       # past date: invite a natural follow-up
        self.assertNotIn('很久以前的', block)
        self.assertIn('优先于画像里的习惯', block)
        self.assertEqual(pl.render_block([], TODAY), '')
        self.assertEqual(pl.render_block([{'text': 'x', 'until': '坏'}], TODAY), '')   # corrupt rows are ignored, not fatal


class StoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStore(Path(self.tmp.name) / 'p.db')

    async def asyncTearDown(self):
        self.store._conn.close()
        self.tmp.cleanup()

    async def test_plans_are_scoped_to_the_chat_they_were_said_in(self):
        item = [{'text': '下周出差', 'until': '2026-11-20', 'hedged': False}]
        await self.store.add_plans(USER, 'private', 0, item, [], 10, '2026-11-01')
        await self.store.add_plans(USER, 'group', 55, [{'text': '群里说的', 'until': '2026-11-20', 'hedged': False}], [], 10, '2026-11-01')
        self.assertEqual([p['text'] for p in await self.store.active_plans(USER, 'private', 0, '2026-11-10')], ['下周出差'])
        self.assertEqual([p['text'] for p in await self.store.active_plans(USER, 'group', 55, '2026-11-10')], ['群里说的'])
        self.assertEqual(await self.store.active_plans(USER, 'group', 66, '2026-11-10'), [])      # other group: nothing
        self.assertEqual(await self.store.active_plans(OWNER, 'private', 0, '2026-11-10'), [])    # other person: nothing

    async def test_duplicates_cancellation_expiry_and_the_cap(self):
        item = {'text': '写代码要到周六才开始', 'until': '2026-11-14', 'hedged': True}
        self.assertEqual(await self.store.add_plans(USER, 'private', 0, [item], [], 10, '2026-11-01'), 1)
        self.assertEqual(await self.store.add_plans(USER, 'private', 0, [item], [], 10, '2026-11-01'), 0)   # same plan twice
        old = (await self.store.active_plans(USER, 'private', 0, '2026-11-10'))[0]['id']
        new = {'text': '明天就写代码', 'until': '2026-11-11', 'hedged': False}
        await self.store.add_plans(USER, 'private', 0, [new], [old], 10, '2026-11-01')                  # the user changed their mind
        self.assertEqual([p['text'] for p in await self.store.active_plans(USER, 'private', 0, '2026-11-10')], ['明天就写代码'])
        self.assertEqual(await self.store.active_plans(USER, 'private', 0, '2026-11-12'), [])           # expired by date
        for i in range(12):
            await self.store.add_plans(USER, 'private', 0, [{'text': f'计划{i}', 'until': '2026-12-01', 'hedged': False}], [], 10, '2026-11-01')
        self.assertLessEqual(len(await self.store.active_plans(USER, 'private', 0, '2026-11-10')), 10)

    async def test_deleting_a_users_data_also_deletes_their_plans_and_reminders(self):
        await self.store.ensure(USER, '小明', 'user')
        await self.store.add_plans(USER, 'private', 0, [{'text': '下周出差', 'until': '2026-11-20', 'hedged': False}], [], 10, '2026-11-01')
        await self.store.add_reminder(USER, 'private', 0, {'kind': 'reminder', 'content': 'x', 'repeat': 'none', 'spec': {},
                                                           'tz': 'Asia/Shanghai', 'due_at': 9e9, 'lead_days': 0})
        await self.store.delete_user(USER)
        self.assertEqual(self.store._conn.execute('SELECT COUNT(*) FROM plans').fetchone()[0], 0)
        self.assertEqual(self.store._conn.execute('SELECT COUNT(*) FROM reminders').fetchone()[0], 0)


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def chat(self, messages, *a, **k):
        self.calls.append(messages)
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStore(Path(self.tmp.name) / 'p.db')
        await self.store.ensure(USER, '小明', 'user')

    async def asyncTearDown(self):
        self.store._conn.close()
        self.tmp.cleanup()

    def bot(self, reply):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {'owner_qq': OWNER, 'profile': {'default_timezone': NY}}
        bot.utility_llm = FakeLLM(reply)
        bot.llm = bot.utility_llm
        bot.profiles, bot.profiles_enabled = self.store, True
        bot._owner_qq = lambda: OWNER
        bot._memory_tasks = set()
        return bot

    def friday(self):
        return (datetime.now(ZoneInfo(NY)).date() + timedelta(days=3)).isoformat()

    async def test_the_scenario_code_will_only_be_written_the_day_after_tomorrow(self):
        until = self.friday()
        bot = self.bot({'plans': [{'text': '写代码要到大后天才开始，在那之前不写', 'until': until, 'hedged': True}], 'cancel': []})
        bot._schedule_plan_extraction(USER, 'private', 0, '看小说去了，代码估计大后天才会写了', None)
        await asyncio.gather(*list(bot._memory_tasks))
        block = await bot._plans_block(USER, 'private', 0, None)
        self.assertIn('写代码要到大后天才开始', block)
        self.assertIn('优先于画像里的习惯', block)                                     # so a morning greeting no longer says "别又对着代码坐一上午"
        # and a proactive message built a few days later still carries it, even if the chat history has long scrolled away
        bot.memory = SimpleNamespace(recent=AsyncMock(return_value=[]))
        proactive = await bot._recent_chat_block('private', USER, USER, None)
        self.assertIn('写代码要到大后天才开始', proactive)

    async def test_chit_chat_costs_no_model_call_and_a_failed_model_never_breaks_anything(self):
        bot = self.bot({'plans': []})
        bot._schedule_plan_extraction(USER, 'private', 0, '哈哈哈哈', None)
        self.assertEqual(bot._memory_tasks, set())
        self.assertEqual(bot.utility_llm.calls, [])
        broken = self.bot('不是json')
        broken._schedule_plan_extraction(USER, 'private', 0, '明天去医院', None)
        await asyncio.gather(*list(broken._memory_tasks))
        self.assertEqual(await broken._plans_block(USER, 'private', 0, None), '')

    async def test_the_extractor_sees_existing_plans_so_a_changed_mind_replaces_the_old_one(self):
        until = self.friday()
        await self.store.add_plans(USER, 'private', 0, [{'text': '写代码要到周六才开始', 'until': until, 'hedged': False}], [], 10, '2026-01-01')
        bot = self.bot({'plans': [{'text': '明天就写代码', 'until': (datetime.now(ZoneInfo(NY)).date() + timedelta(days=1)).isoformat(), 'hedged': False}],
                        'cancel': [1]})
        bot._schedule_plan_extraction(USER, 'private', 0, '算了明天就开始写代码吧', None)
        await asyncio.gather(*list(bot._memory_tasks))
        self.assertIn('写代码要到周六才开始', bot.utility_llm.calls[0][0]['content'])           # shown to the model as plan 1
        texts = [p['text'] for p in await self.store.active_plans(USER, 'private', 0, '2000-01-01')]
        self.assertEqual(texts, ['明天就写代码'])

    async def test_it_can_be_switched_off_and_never_leaks_across_chats(self):
        bot = self.bot({'plans': []})
        bot.config['plans'] = {'enabled': False}
        bot._schedule_plan_extraction(USER, 'private', 0, '明天去医院', None)
        self.assertEqual(bot._memory_tasks, set())
        await self.store.add_plans(USER, 'private', 0, [{'text': '私聊里说的', 'until': self.friday(), 'hedged': False}], [], 10, '2026-01-01')
        self.assertEqual(await bot._plans_block(USER, 'group', 55, None), '')                 # a private plan is never shown in a group


import asyncio

if __name__ == '__main__':
    unittest.main()
