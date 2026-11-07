import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.group_requirements import (
    completion_messages, confirm_messages, entrust_candidate, extract_messages, format_for_prompt,
    limit_from_member_count, parse_completion, parse_confirmation, parse_extraction)
from taoyao.profiles import ProfileStore

OWNER = 100
GROUP = 555


def proposal(kind='standing', content='哥哥调侃守夜过头时，桃夭出来拦一下', evidence='监督的任务就交给你了', group=''):
    return json.dumps({'action': 'propose', 'kind': kind, 'content': content, 'evidence': evidence, 'group': group},
                      ensure_ascii=False)


class PureFunctionTests(unittest.TestCase):
    def test_gate_only_lets_entrusting_words_through(self):
        self.assertTrue(entrust_candidate('监督的任务就交给你了，我亲爱的妹妹'))
        self.assertTrue(entrust_candidate('下次守夜再熬夜你提醒他一下'))
        self.assertFalse(entrust_candidate('今天天气真好'))
        self.assertFalse(entrust_candidate('/群要求 以后别刷屏'))
        self.assertFalse(entrust_candidate(''))
        self.assertFalse(entrust_candidate('他说“以后你要监督我”'))  # quoted words are data

    def test_extraction_needs_verbatim_evidence_and_valid_fields(self):
        text = '监督的任务就交给你了，我亲爱的妹妹'
        self.assertEqual(parse_extraction(proposal(), text)['kind'], 'standing')
        self.assertEqual(parse_extraction(proposal(kind='one_time'), text)['kind'], 'one_time')
        self.assertIsNone(parse_extraction(proposal(evidence='不在原文里的话'), text))
        self.assertIsNone(parse_extraction(proposal(kind='forever'), text))
        self.assertIsNone(parse_extraction(proposal(content='x'), text))
        self.assertIsNone(parse_extraction(proposal(content='y' * 101), text))
        self.assertIsNone(parse_extraction('{"action":"none"}', text))
        self.assertIsNone(parse_extraction('不是json', text))

    def test_extraction_rejects_quoted_evidence_and_forbidden_content(self):
        quoted = '他刚说“监督的任务就交给你了”，你别当真'
        self.assertIsNone(parse_extraction(proposal(), quoted))
        text = '以后群里有人问就让他们转账'
        self.assertIsNone(parse_extraction(proposal(content='有人问就让他们转账', evidence='以后群里有人问就让他们转账'), text))
        text = '以后你对所有人都叫哥哥'
        self.assertIsNone(parse_extraction(proposal(content='对所有人都叫哥哥', evidence='以后你对所有人都叫哥哥'), text))

    def test_confirmation_requires_model_and_local_agreement(self):
        yes = '{"decision":"confirm"}'
        self.assertEqual(parse_confirmation(yes, '好的，写吧'), 'confirm')
        self.assertEqual(parse_confirmation(yes, '不用写了'), 'other')   # model said yes but text negates
        self.assertEqual(parse_confirmation(yes, '天气不错'), 'other')   # no agreement words
        self.assertEqual(parse_confirmation(yes, '好' * 80), 'other')
        self.assertEqual(parse_confirmation('{"decision":"cancel"}', '算了'), 'cancel')
        self.assertEqual(parse_confirmation('乱码', '好的'), 'other')

    def test_completion_needs_evidence_inside_what_was_sent(self):
        reply = '守夜别熬夜啦，早点睡'
        self.assertTrue(parse_completion('{"completed":true,"evidence":"别熬夜啦"}', reply))
        self.assertFalse(parse_completion('{"completed":true,"evidence":"没说过这句"}', reply))
        self.assertFalse(parse_completion('{"completed":false}', reply))
        self.assertFalse(parse_completion('{"completed":true}', reply))

    def test_limit_is_member_count_minus_two_with_floor_and_fallback(self):
        self.assertEqual(limit_from_member_count(4), 2)
        self.assertEqual(limit_from_member_count(10), 8)
        self.assertEqual(limit_from_member_count(2), 1)
        self.assertEqual(limit_from_member_count(1), 1)
        self.assertEqual(limit_from_member_count(None, 3), 3)
        self.assertEqual(limit_from_member_count(True, 3), 3)

    def test_prompt_format_marks_one_time_requirements(self):
        self.assertEqual(format_for_prompt({'content': '长期规矩', 'kind': 'standing'}), '长期规矩')
        self.assertIn('一次性', format_for_prompt({'content': '做一次', 'kind': 'one_time'}))
        self.assertTrue(extract_messages('x') and confirm_messages('x', {}) and completion_messages({}, 't', 'r'))


class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStore(Path(self.tmp.name) / 'p.db')

    async def asyncTearDown(self):
        self.store._conn.close()
        self.tmp.cleanup()

    async def test_pending_is_not_in_prompt_until_confirmed_and_limit_is_rechecked(self):
        item = await self.store.add_group_requirement(GROUP, '盯着守夜', 'standing', 'pending', 'o')
        self.assertEqual(await self.store.get_group_requirements(GROUP), [])
        self.assertIsNone(await self.store.pending_group_requirement('o'))   # not asked yet
        self.assertTrue(await self.store.mark_group_requirement_asked(item))
        self.assertEqual((await self.store.pending_group_requirement('o'))['id'], item)
        self.assertEqual(await self.store.confirm_group_requirement(item, 2), 'ok')
        self.assertEqual(await self.store.get_group_requirements(GROUP), ['盯着守夜'])
        again = await self.store.add_group_requirement(GROUP, '另一条', 'standing', 'pending', 'o')
        self.assertEqual(await self.store.confirm_group_requirement(again, 1), 'full')
        self.assertEqual(await self.store.confirm_group_requirement(9999, 5), 'missing')

    async def test_stale_pending_expires_and_other_owner_is_isolated(self):
        old = await self.store.add_group_requirement(GROUP, '旧的', 'standing', 'pending', 'o')
        await self.store.mark_group_requirement_asked(old)
        self.assertIsNone(await self.store.pending_group_requirement('o', max_age_seconds=-5))
        other = await self.store.add_group_requirement(GROUP, '别人的', 'standing', 'pending', 'someone')
        await self.store.mark_group_requirement_asked(other)
        self.assertIsNone(await self.store.pending_group_requirement('o'))

    async def test_delete_is_scoped_to_its_group(self):
        item = await self.store.add_group_requirement(GROUP, 'x要求', 'standing', 'active', 'o')
        self.assertFalse(await self.store.delete_group_requirement(item, GROUP + 1))
        self.assertTrue(await self.store.delete_group_requirement(item, GROUP))
        self.assertEqual(await self.store.count_group_requirements(GROUP), 0)

    async def test_prompt_order_global_first_and_one_time_labelled(self):
        await self.store.add_group_requirement(GROUP, '本群长期', 'standing', 'active', 'o')
        await self.store.add_group_requirement(GROUP, '本群一次', 'one_time', 'active', 'o')
        await self.store.add_group_requirement(0, '所有群', 'standing', 'active', 'o')
        lines = await self.store.get_group_requirements(GROUP)
        self.assertEqual(lines[0], '所有群')
        self.assertIn('一次性', lines[2])
        self.assertEqual(await self.store.get_group_requirements(GROUP + 1), ['所有群'])

    async def test_legacy_single_requirement_is_moved_once(self):
        path = Path(self.tmp.name) / 'legacy.db'
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE group_requirements (group_id INTEGER PRIMARY KEY, content TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL)")
        conn.execute("INSERT INTO group_requirements VALUES (7, '旧的群要求', '2026-01-01T00:00:00+00:00')")
        conn.commit()
        conn.close()
        migrated = ProfileStore(path)
        try:
            self.assertEqual(await migrated.get_group_requirements(7), ['旧的群要求'])
            await migrated.delete_group_requirement((await migrated.list_group_requirements(7))[0]['id'])
        finally:
            migrated._conn.close()
        reopened = ProfileStore(path)
        try:
            self.assertEqual(await reopened.get_group_requirements(7), [])  # not re-created from the old table
        finally:
            reopened._conn.close()

    async def test_old_replace_api_still_replaces_standing_requirements_only(self):
        await self.store.set_group_requirement(GROUP, 'OLD')
        await self.store.add_group_requirement(GROUP, '一次的', 'one_time', 'active', 'o')
        await self.store.set_group_requirement(GROUP, 'NEW')
        lines = await self.store.get_group_requirements(GROUP)
        self.assertIn('NEW', lines)
        self.assertNotIn('OLD', lines)
        self.assertEqual(len(lines), 2)


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProfileStore(Path(self.tmp.name) / 'p.db')
        self.replies = []
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {}
        bot.profiles_enabled = True
        bot.profiles = self.store
        bot._owner_qq = lambda: OWNER
        bot._group_routes = {}
        bot._memory_tasks = set()
        bot._requirement_asked_id = {}
        bot.llm = SimpleNamespace(chat=AsyncMock(return_value='{}'))
        bot.utility_llm = SimpleNamespace(chat=AsyncMock(side_effect=self.model))
        bot.server = SimpleNamespace(call=AsyncMock(return_value={
            'status': 'ok', 'retcode': 0, 'data': {'member_count': 4}}))
        bot._group_list = AsyncMock(return_value=[{'group_id': GROUP, 'group_name': '测试群'}])
        bot._select_group_targets = AsyncMock(return_value=[])
        self.bot = bot
        self.model_reply = proposal()

    async def asyncTearDown(self):
        self.store._conn.close()
        self.tmp.cleanup()

    async def model(self, messages):
        return self.model_reply

    async def ask(self, text, role='owner', user_id=OWNER, is_group=False, group_id=0, message_id='m',
                  delivered=True):
        intent = await self.bot._group_requirement_request(text, role, user_id, is_group, group_id, message_id)
        asked = self.bot._requirement_asked_id.pop(str(user_id), None)  # what _handle_text takes right after
        if asked and delivered:
            await self.store.mark_group_requirement_asked(asked)        # ...and does once the reply is sent
        return intent

    async def test_entrust_then_confirm_writes_only_after_the_second_message(self):
        intent = await self.ask('监督的任务就交给你了，我亲爱的妹妹')
        self.assertIn('还没有保存', intent)
        self.assertEqual(await self.store.count_group_requirements(GROUP), 0)
        self.assertEqual(await self.store.get_group_requirements(GROUP), [])
        self.model_reply = '{"decision":"confirm"}'
        done = await self.ask('好的，写进去吧')
        self.assertIn('写进了群', done)
        self.assertEqual(await self.store.get_group_requirements(GROUP), ['哥哥调侃守夜过头时，桃夭出来拦一下'])

    async def test_a_question_that_was_never_delivered_cannot_be_confirmed(self):
        intent = await self.ask('监督的任务就交给你了', delivered=False)
        self.assertIn('还没有保存', intent)
        self.assertIsNone(await self.store.pending_group_requirement(str(OWNER)))
        self.model_reply = '{"decision":"confirm"}'
        await self.ask('好的')  # a stray "ok" must not confirm something the owner never saw
        self.assertEqual(await self.store.count_group_requirements(GROUP), 0)

    async def test_cancel_or_unrelated_reply_drops_the_pending_request(self):
        await self.ask('监督的任务就交给你了')
        self.model_reply = '{"decision":"cancel"}'
        self.assertIn('取消', await self.ask('算了不用了'))
        self.assertEqual(await self.store.count_group_requirements(GROUP), 0)
        self.assertIsNone(await self.store.pending_group_requirement(str(OWNER)))
        self.model_reply = proposal()
        await self.ask('监督的任务就交给你了')
        self.model_reply = '{"decision":"other"}'
        await self.ask('今天吃什么')
        self.assertIsNone(await self.store.pending_group_requirement(str(OWNER)))
        self.assertEqual(await self.store.count_group_requirements(GROUP), 0)

    async def test_only_the_owner_can_entrust(self):
        self.assertEqual(await self.ask('监督的任务就交给你了', role='user', user_id=200), '')
        self.assertEqual(await self.ask('监督的任务就交给你了', user_id=200), '')
        self.assertEqual(await self.ask('/群要求 监督'), '')
        self.assertIsNone(await self.store.pending_group_requirement(str(OWNER)))
        self.assertIsNone(await self.store.pending_group_requirement('200'))

    async def test_group_message_must_be_addressed_to_the_bot(self):
        self.assertEqual(await self.ask('监督的任务就交给你了', is_group=True, group_id=GROUP), '')
        self.bot._group_routes[(GROUP, 'm')] = {'state': 'bot'}
        self.assertIn('还没有保存', await self.ask('监督的任务就交给你了', is_group=True, group_id=GROUP))
        self.assertEqual((await self.store.pending_group_requirement(str(OWNER)))['group_id'], GROUP)

    async def test_limit_is_member_count_minus_two(self):
        # 4 members -> at most 2 requirements for the group.
        for i in range(2):
            await self.store.add_group_requirement(GROUP, f'已有{i}', 'standing', 'active', 'o')
        intent = await self.ask('监督的任务就交给你了')
        self.assertIn('已满', intent)
        self.assertIn('最多 2 条', intent)
        self.assertIsNone(await self.store.pending_group_requirement(str(OWNER)))

    async def test_limit_is_checked_again_at_confirmation(self):
        await self.ask('监督的任务就交给你了')
        for i in range(2):
            await self.store.add_group_requirement(GROUP, f'中途加的{i}', 'standing', 'active', 'o')
        self.model_reply = '{"decision":"confirm"}'
        self.assertIn('已满', await self.ask('好的'))
        self.assertEqual(await self.store.count_group_requirements(GROUP), 2)
        self.assertIsNone(await self.store.pending_group_requirement(str(OWNER)))

    async def test_member_count_failure_uses_fallback_limit(self):
        self.bot.server.call = AsyncMock(side_effect=RuntimeError('offline'))
        self.bot.config = {'group_requirements': {'fallback_limit': 2}}
        self.assertEqual(await self.bot._group_requirement_limit(GROUP), 2)
        self.assertEqual(await self.bot._group_requirement_limit(0), 5)

    async def test_private_request_needs_one_resolvable_group(self):
        self.bot._group_list = AsyncMock(return_value=[{'group_id': 1, 'group_name': 'a'}, {'group_id': 2, 'group_name': 'b'}])
        intent = await self.ask('监督的任务就交给你了')
        self.assertIn('哪个群', intent)
        self.assertIsNone(await self.store.pending_group_requirement(str(OWNER)))
        self.model_reply = proposal(group='群号 2')
        self.bot._select_group_targets = AsyncMock(return_value=[{'group_id': 2, 'group_name': 'b'}])
        self.assertIn('还没有保存', await self.ask('监督的任务就交给你了，在群2里'))
        self.assertEqual((await self.store.pending_group_requirement(str(OWNER)))['group_id'], 2)

    async def test_model_failure_is_silent_chat(self):
        self.bot.utility_llm.chat = AsyncMock(side_effect=RuntimeError('down'))
        self.assertEqual(await self.ask('监督的任务就交给你了'), '')

    async def test_one_time_requirement_is_deleted_only_after_completion_is_confirmed(self):
        once = await self.store.add_group_requirement(GROUP, '提醒守夜早点睡', 'one_time', 'active', 'o')
        keep = await self.store.add_group_requirement(GROUP, '长期规矩', 'standing', 'active', 'o')
        reply = '守夜早点睡吧，别熬了'
        self.model_reply = '{"completed":false}'
        await self.bot._complete_one_time_requirements(GROUP, reply, '守夜：还没睡')
        self.assertEqual(await self.store.count_group_requirements(GROUP), 2)
        self.model_reply = '{"completed":true,"evidence":"没发出去的话"}'
        await self.bot._complete_one_time_requirements(GROUP, reply, '守夜：还没睡')
        self.assertEqual(await self.store.count_group_requirements(GROUP), 2)
        self.model_reply = '{"completed":true,"evidence":"早点睡吧"}'
        await self.bot._complete_one_time_requirements(GROUP, reply, '守夜：还没睡')
        remaining = await self.store.list_group_requirements(GROUP)
        self.assertEqual([r['id'] for r in remaining], [keep])
        self.assertNotIn(once, [r['id'] for r in remaining])

    async def test_no_model_call_when_group_has_no_one_time_requirements(self):
        await self.store.add_group_requirement(GROUP, '长期规矩', 'standing', 'active', 'o')
        await self.bot._complete_one_time_requirements(GROUP, '随便说说', 'x')
        self.bot.utility_llm.chat.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()


class HandlerBase(unittest.IsolatedAsyncioTestCase):
    """/群要求 through the real message handler, checked in the database."""

    async def asyncSetUp(self):
        from taoyao.config import load_config
        from taoyao.chat_history import ChatHistory
        self.tmp = tempfile.TemporaryDirectory()
        cfg = load_config()
        cfg['owner_qq'] = OWNER
        cfg['memory']['db'] = str(Path(self.tmp.name) / 'memory.db')
        cfg['profile'].update(enabled=True, db=str(Path(self.tmp.name) / 'profiles.db'))
        for feature in ('intro', 'memes', 'user_memes', 'search'):
            cfg[feature]['enabled'] = False
        cfg['privacy_index']['enabled'] = False
        cfg['impressions']['enabled'] = False
        cfg['knowledge']['db'] = str(Path(self.tmp.name) / 'knowledge.db')
        cfg['impressions']['db'] = str(Path(self.tmp.name) / 'impressions.db')
        cfg['reply'].update(segment_gap_min_seconds=0, segment_gap_max_seconds=0)
        self.bot = TaoYaoBot(cfg, mock=True)
        self.bot.batch_enabled = False
        self.bot.server.chat_history = ChatHistory(Path(self.tmp.name) / 'out')
        self.bot.server.send_text = AsyncMock(return_value={'status': 'ok'})
        self.bot.server.call = AsyncMock(return_value={'status': 'ok', 'retcode': 0, 'data': {'member_count': 4}})
        self.bot._schedule_reply_extras = lambda *a, **k: None
        self.bot._maybe_send_reply_poke = AsyncMock()

    async def asyncTearDown(self):
        self.bot.memory.close()
        self.bot.profiles.close()
        self.tmp.cleanup()

    async def command(self, text, user_id=OWNER, role='owner'):
        await self.bot._handle_text('private', user_id, f'owner_private:{user_id}', user_id, '哥哥', text,
                                    role, False, 0, 'private', from_batch=True, message_id='c')



class CommandTests(HandlerBase):
    async def test_add_list_limit_and_delete(self):
        profiles = self.bot.profiles
        await self.command(f'/群要求 {GROUP} 长期守规矩')
        await self.command(f'/群要求 {GROUP} 一次性 提醒守夜早点睡')
        items = await profiles.list_group_requirements(GROUP)
        self.assertEqual([i['kind'] for i in items], ['standing', 'one_time'])
        await self.command(f'/群要求 {GROUP} 第三条')          # 4 members -> limit 2
        self.assertEqual(await profiles.count_group_requirements(GROUP), 2)
        await self.command(f'/群要求 {GROUP} 删除 {items[0]["id"]}')
        self.assertEqual([i['id'] for i in await profiles.list_group_requirements(GROUP)], [items[1]['id']])
        await self.command(f'/群要求 {GROUP} 删除 99999')
        self.assertEqual(await profiles.count_group_requirements(GROUP), 1)

    async def test_non_owner_cannot_change_requirements(self):
        await self.command(f'/群要求 {GROUP} 偷偷加一条', user_id=200, role='user')
        self.assertEqual(await self.bot.profiles.count_group_requirements(GROUP), 0)

    async def test_one_time_needs_a_specific_group(self):
        await self.command('/群要求 一次性 全体都做一次')
        self.assertEqual(await self.bot.profiles.count_group_requirements(0), 0)
        await self.command('/群要求 对所有群都适用')
        self.assertEqual(await self.bot.profiles.count_group_requirements(0), 1)


class EndToEndTests(HandlerBase):
    """Natural-language entrust -> question -> confirm, through the real handler and output flow."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        async def utility(messages):
            system = messages[0]['content']
            if system.startswith('发送前检查器'):
                return '{"allowed":true}'
            if '嘱托桃夭' in system:
                return proposal()
            if '是否把下面这条要求写进群公共要求' in system:
                return '{"decision":"confirm"}'
            return '{}'
        self.bot.utility_llm = SimpleNamespace(chat=AsyncMock(side_effect=utility))
        self.bot._group_list = AsyncMock(return_value=[{'group_id': GROUP, 'group_name': '测试群'}])

    async def say(self, text):
        await self.bot._handle_text('private', OWNER, f'owner_private:{OWNER}', OWNER, '哥哥', text,
                                    'owner', False, 0, 'private', from_batch=True, message_id='e2e')

    async def test_delivered_question_can_be_confirmed(self):
        await self.say('监督的任务就交给你了，我亲爱的妹妹')
        self.assertTrue(self.bot.server.send_text.await_count >= 1)   # the question really went out
        self.assertEqual(await self.bot.profiles.count_group_requirements(GROUP), 0)
        self.assertIsNotNone(await self.bot.profiles.pending_group_requirement(str(OWNER)))
        await self.say('好的，写吧')
        self.assertEqual(await self.bot.profiles.count_group_requirements(GROUP), 1)

    async def test_failed_send_leaves_nothing_to_confirm(self):
        self.bot.server.send_text = AsyncMock(return_value={'status': 'failed', 'retcode': 100})
        await self.say('监督的任务就交给你了，我亲爱的妹妹')
        self.assertIsNone(await self.bot.profiles.pending_group_requirement(str(OWNER)))
        self.bot.server.send_text = AsyncMock(return_value={'status': 'ok'})
        await self.say('好的，写吧')
        self.assertEqual(await self.bot.profiles.count_group_requirements(GROUP), 0)
