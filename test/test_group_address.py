import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.group_address import message_route, route_note
from taoyao.group_budget import GroupReplyBudget
from taoyao.profiles import ProfileStore


def event(*segments):
    return {'self_id': 100, 'message': list(segments)}


def at(uid):
    return {'type': 'at', 'data': {'qq': uid}}


def reply(mid):
    return {'type': 'reply', 'data': {'id': mid}}


class GroupAddressTests(unittest.TestCase):
    def test_explicit_address_precedes_topic_and_quote(self):
        self.assertEqual(message_route(event(at(200)), '100')['state'], 'other')
        self.assertEqual(message_route(event(at(100), at(200)))['state'], 'bot')
        self.assertEqual(message_route(event(reply('1')), '200')['state'], 'other')
        self.assertEqual(message_route(event(reply('1')), '100')['state'], 'bot')
        self.assertEqual(message_route(event(reply('1')))['state'], 'unknown')
        self.assertEqual(message_route(event(at('all')))['state'], 'broad')
        self.assertEqual(message_route(event())['state'], 'none')
        self.assertIn('旧记录未标注', route_note(None))


class GroupAddressAsyncTests(unittest.IsolatedAsyncioTestCase):
    def bot(self):
        bot = object.__new__(TaoYaoBot)
        bot.config = {'group_followup_use_llm': True}
        bot.profiles_enabled = True
        bot.profiles = SimpleNamespace(find_message=AsyncMock(return_value=None), group_context=AsyncMock(return_value=[]))
        bot.server = SimpleNamespace(call=AsyncMock(return_value={'status': 'ok', 'retcode': 0, 'data': {'group_id': 10, 'sender': {'user_id': 200}}}))
        bot._group_reply_budget = GroupReplyBudget()
        bot._group_last_bot_reply = {10: (time.monotonic(), 300, '你在做什么？', '在家')}
        bot.llm = SimpleNamespace(chat=AsyncMock(return_value='{"followup":false}'))
        return bot

    async def test_quote_lookup_fallback_and_cross_group_rejection(self):
        bot = self.bot()
        data = event(reply('12'))
        self.assertEqual((await bot._group_message_route(data, 10, 300))['state'], 'other')
        bot.server.call.return_value['data']['sender']['user_id'] = 100
        self.assertEqual((await bot._group_message_route(data, 10, 300))['state'], 'bot')
        bot.server.call.return_value['data']['group_id'] = 99
        self.assertEqual((await bot._group_message_route(data, 10, 300))['state'], 'unknown')

    async def test_recent_bot_name_does_not_make_other_quote_target_bot(self):
        bot = self.bot()
        bot.profiles.find_message.return_value = {'role': 'user', 'user_id': '200'}
        bot.profiles.group_context.return_value = [{'role': 'assistant', 'content': '桃夭刚刚说了话'}]
        self.assertFalse(await bot._context_targets_bot(event(reply('12')), '你刚才说什么', 10, 300, ''))

    async def test_known_name_at_start_is_addressee_not_topic(self):
        bot = self.bot()
        bot.profiles.group_context.return_value = [{'role': 'user', 'nickname': '小雨', 'user_id': '200'}]
        data = event({'type': 'text', 'data': {'text': '小雨，你刚才说什么'}})
        self.assertEqual((await bot._group_message_route(data, 10, 300))['state'], 'other')
        data = event({'type': 'text', 'data': {'text': '我刚才看到小雨在说话'}})
        self.assertEqual((await bot._group_message_route(data, 10, 300))['state'], 'none')

    async def test_short_same_user_answer_requires_address_confirmation(self):
        bot = self.bot()
        self.assertFalse(await bot._is_group_followup(10, 300, '我在家'))
        bot.llm.chat.assert_awaited_once()
        bot.llm.chat.return_value = '{"followup":true}'
        self.assertTrue(await bot._is_group_followup(10, 300, '桃夭刚才问的事，我在家'))
        bot.llm.chat.return_value = '{"followup":"false"}'
        self.assertFalse(await bot._is_group_followup(10, 300, '我在家'))

    async def test_routing_survives_storage_archive_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profiles.db'
            store = ProfileStore(path)
            await store.ensure(300, '群友', 'user')
            route = message_route(event(at(200)))
            await store.add_message(300, 'group', 10, 'user', '你刚才说什么', '12', routing=route)
            await store.add_message(300, 'group', 10, 'user', '另一句', '13')
            rows = await store.group_context(10)
            self.assertEqual(json.loads(rows[0]['routing']), route)
            await store._run(store._prune_messages, '300', 999, 1)
            archived = store._conn.execute('SELECT routing FROM user_messages_archive').fetchone()
            self.assertEqual(json.loads(archived['routing']), route)
            store.close()
            reopened = ProfileStore(path)
            self.assertEqual(len(await reopened.group_context(10)), 1)
            reopened.close()
