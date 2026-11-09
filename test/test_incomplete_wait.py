import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import PendingMessage, TaoYaoBot
from taoyao.config import load_config
from taoyao.turn_timing import set_value, turn_timing_context
from taoyao.utterance import JUDGE_PROMPT_HEAD, judge_messages, parse_incomplete, worth_judging
from test_typing_hold import make_bot, stream

USER = 2
GROUP = 555
GKEY = ('g', GROUP, USER)


def done_future(value):
    fut = asyncio.get_running_loop().create_future()
    fut.set_result(value)
    return fut


class PureTests(unittest.TestCase):
    def test_what_is_worth_judging(self):
        self.assertTrue(worth_judging('我觉得吧，'))
        self.assertTrue(worth_judging('今天去了超市然后'))
        self.assertFalse(worth_judging(''))
        self.assertFalse(worth_judging('   '))
        self.assertFalse(worth_judging('/状态'))
        self.assertFalse(worth_judging('？？？'))
        self.assertFalse(worth_judging('很长的话' * 40))                       # long messages are usually whole
        self.assertTrue(worth_judging('好' + '\n[最近图片记忆] 一大段图片描述' * 20))   # appended image notes are ignored

    def test_prompt_contains_the_text_and_is_conservative(self):
        messages = judge_messages('我昨天去了')
        self.assertTrue(messages[0]['content'].startswith(JUDGE_PROMPT_HEAD))
        self.assertIn('拿不准一律输出', messages[0]['content'])
        self.assertEqual(messages[1]['content'], '我昨天去了')

    def test_only_an_explicit_true_means_unfinished(self):
        self.assertTrue(parse_incomplete('{"incomplete":true}'))
        self.assertTrue(parse_incomplete('好的 {"incomplete": true} 谢谢'))
        for raw in ('{"incomplete":false}', '{"incomplete":"true"}', '{"incomplete":1}', '{"allowed":true}',
                    '不是json', '', '[]', None):
            self.assertFalse(parse_incomplete(raw), raw)


class WaitDecisionTests(unittest.IsolatedAsyncioTestCase):
    HOLD = {'first_sentence_grace_seconds': 0, 'incomplete_wait_seconds': 0.5, 'incomplete_wait_group_seconds': 0.4,
            'burst_window_seconds': 15}

    async def check(self, bot, verdict, key=USER, first=True, private=True, budget=None, during=None):
        """Run the decision with a finished (or unfinished) parallel judge; `during` may act while it waits."""
        @turn_timing_context
        async def run():
            if verdict is not None:
                set_value('incomplete_task', done_future(verdict))
            else:
                set_value('incomplete_task', asyncio.get_running_loop().create_future())   # still running
            started = time.monotonic()
            task = asyncio.create_task(bot._hold_while_typing(
                key, dict(budget or {'seq': bot.typing.message_seq(key)}), first=first, private=private))
            if during is not None:
                await during()
            return await task, time.monotonic() - started
        return await run()

    async def test_unfinished_sentence_waits_then_sends_when_nothing_follows(self):
        bot = make_bot(stream, hold=self.HOLD)
        result, waited = await self.check(bot, True)
        self.assertEqual(result, 'send')
        self.assertGreater(waited, 0.4)
        self.assertLess(waited, 1.0)

    async def test_the_next_message_during_the_wait_hands_the_reply_off(self):
        bot = make_bot(stream, hold=self.HOLD)

        async def during():
            await asyncio.sleep(0.15)
            bot.typing.message_received(USER, 'm2')
        result, waited = await self.check(bot, True, during=during)
        self.assertEqual(result, 'handoff')
        self.assertLess(waited, 0.4)                                # did not wait out the full time

    async def test_finished_sentence_or_unknown_verdict_never_waits(self):
        bot = make_bot(stream, hold=self.HOLD)
        for verdict in (False, None):                              # None = the judge has not answered yet
            result, waited = await self.check(bot, verdict)
            self.assertEqual(result, 'send')
            self.assertLess(waited, 0.1)

    async def test_only_the_first_sentence_waits_not_the_tail(self):
        bot = make_bot(stream, hold=self.HOLD)
        result, waited = await self.check(bot, True, first=False)
        self.assertEqual(result, 'send')
        self.assertLess(waited, 0.1)

    async def test_typing_detected_during_the_wait_turns_into_a_normal_hold(self):
        bot = make_bot(stream, hold=self.HOLD, ttl=30)

        async def during():
            await asyncio.sleep(0.15)
            bot.typing.note(USER, 1)
            await asyncio.sleep(0.5)                               # past the unfinished-sentence wait
            bot.typing.message_received(USER, 'm2')
        result, waited = await self.check(bot, True, during=during)
        self.assertEqual(result, 'handoff')
        self.assertGreater(waited, 0.6)                            # held while typing, beyond the plain wait

    async def test_group_waits_for_the_same_person_with_its_own_limit(self):
        bot = make_bot(stream, hold=self.HOLD)
        result, waited = await self.check(bot, True, key=GKEY, private=False)
        self.assertEqual(result, 'send')
        self.assertGreater(waited, 0.3)
        self.assertLess(waited, 0.8)                                # 0.4 s group limit, not the 0.5 s private one

    async def test_group_handoff_follows_only_this_persons_messages(self):
        bot = make_bot(stream, hold=self.HOLD)

        async def other_person():
            await asyncio.sleep(0.1)
            bot.typing.message_received(('g', GROUP, 99), 'x1')       # someone else speaks
        result, _ = await self.check(bot, True, key=GKEY, private=False, during=other_person)
        self.assertEqual(result, 'send')

        async def same_person():
            await asyncio.sleep(0.1)
            bot.typing.message_received(GKEY, 'x2')
        result, _ = await self.check(bot, True, key=GKEY, private=False, during=same_person)
        self.assertEqual(result, 'handoff')

    async def test_group_has_no_tail_grace_and_can_be_switched_off(self):
        bot = make_bot(stream, hold={**self.HOLD, 'tail_grace_burst_seconds': 1.0})

        @turn_timing_context
        async def tail():
            set_value('arrival_gap', 2.0)                              # bursting
            started = time.monotonic()
            result = await bot._hold_while_typing(GKEY, {'anchor': time.monotonic()}, first=False, private=False)
            return result, time.monotonic() - started
        result, waited = await tail()
        self.assertEqual(result, 'send')
        self.assertLess(waited, 0.1)                                   # the tail grace is private-only
        bot.config['typing_hold']['group_enabled'] = False
        result, waited = await self.check(bot, True, key=GKEY, private=False)
        self.assertLess(waited, 0.1)


class ArrivalTrackingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = load_config()
        cfg['memory']['db'] = str(Path(self.tmp.name) / 'memory.db')
        cfg['profile'].update(enabled=False, db=str(Path(self.tmp.name) / 'profiles.db'))
        cfg['privacy_index']['enabled'] = False
        cfg['impressions'].update(enabled=False, db=str(Path(self.tmp.name) / 'impressions.db'))
        cfg['knowledge']['db'] = str(Path(self.tmp.name) / 'knowledge.db')
        self.bot = TaoYaoBot(cfg, mock=True)
        self.bot._handle_text = AsyncMock()

    async def asyncTearDown(self):
        self.bot.memory.close()
        self.bot.profiles.close()
        self.tmp.cleanup()

    async def test_group_arrival_is_noted_per_person_and_commands_do_not_count(self):
        bot = self.bot
        bot._note_group_arrival(GROUP, USER, 'a', '你好')
        bot._note_group_arrival(GROUP, USER, 'b', '/状态')
        bot._note_group_arrival(GROUP, 99, 'c', '别人说话')
        bot._note_group_arrival(GROUP + 1, USER, 'd', '别的群')
        self.assertEqual(bot.typing.message_seq(GKEY), 1)
        self.assertEqual(bot.typing.message_seq(('g', GROUP, 99)), 1)
        self.assertEqual(bot.typing.message_seq(('g', GROUP + 1, USER)), 1)
        self.assertEqual(bot.typing.seq_of(f'g{GROUP}:{USER}:a'), 1)
        self.assertEqual(bot.typing.message_seq(USER), 0)              # the private key is untouched

    async def test_group_arrival_can_be_switched_off(self):
        self.bot.config['typing_hold'] = {'enabled': True, 'group_enabled': False}
        self.bot._note_group_arrival(GROUP, USER, 'a', '你好')
        self.assertEqual(self.bot.typing.message_seq(GKEY), 0)

    async def test_a_group_message_is_noted_when_it_is_queued_for_a_reply(self):
        bot = self.bot
        await bot._enqueue(PendingMessage(
            target='group', target_id=GROUP, session=f'group:{GROUP}', user_id=USER, nickname='群友',
            text='你好桃夭', role='user', is_group=True, group_id=GROUP, scene='group', message_id='q1'))
        self.assertEqual(bot.typing.message_seq(GKEY), 1)
        self.assertEqual(bot.typing.seq_of(f'g{GROUP}:{USER}:q1'), 1)
        for task in list(bot._batch_tasks.values()):
            await asyncio.gather(task, return_exceptions=True)

    def test_unsent_reply_is_only_merged_into_the_same_persons_next_turn(self):
        bot = self.bot
        bot._held_tails[f'group:{GROUP}'] = {'text': '没发出的回复', 'at': time.monotonic(),
                                              'first_sent': False, 'user_id': USER}
        self.assertEqual(bot._take_held_reply(f'group:{GROUP}', 99), '')          # someone else: untouched
        self.assertIn(f'group:{GROUP}', bot._held_tails)
        self.assertIn('没发出的回复', bot._take_held_reply(f'group:{GROUP}', USER))
        self.assertNotIn(f'group:{GROUP}', bot._held_tails)


class ParallelJudgeEndToEnd(unittest.IsolatedAsyncioTestCase):
    """The judge runs next to generation: it never slows a finished message, and it merges an unfinished one."""

    async def asyncSetUp(self):
        from taoyao.chat_history import ChatHistory
        self.tmp = tempfile.TemporaryDirectory()
        cfg = load_config()
        cfg['owner_qq'] = 100
        cfg['memory']['db'] = str(Path(self.tmp.name) / 'memory.db')
        cfg['profile'].update(enabled=True, db=str(Path(self.tmp.name) / 'profiles.db'))
        for feature in ('intro', 'memes', 'user_memes', 'search'):
            cfg[feature]['enabled'] = False
        cfg['privacy_index']['enabled'] = False
        cfg['impressions'].update(enabled=False, db=str(Path(self.tmp.name) / 'impressions.db'))
        cfg['knowledge']['db'] = str(Path(self.tmp.name) / 'knowledge.db')
        cfg['reply'].update(segment_gap_min_seconds=0, segment_gap_max_seconds=0)
        cfg['typing_hold'] = {'enabled': True, 'poll_seconds': 0.02, 'first_sentence_grace_seconds': 0,
                              'incomplete_wait_seconds': 2.0, 'log_events': False}
        self.bot = TaoYaoBot(cfg, mock=True)
        self.bot.batch_enabled = False
        self.bot.server.chat_history = ChatHistory(Path(self.tmp.name) / 'out')
        self.bot.server.send_text = AsyncMock(return_value={'status': 'ok'})
        self.bot._schedule_reply_extras = lambda *a, **k: None
        self.bot._maybe_send_reply_poke = AsyncMock()
        self.requests = []
        self.verdict = '{"incomplete":false}'
        self.judge_delay = 0.0

        async def llm_stream(messages):
            self.requests.append(messages)
            yield '收到。那就这样。'

        async def utility(messages):
            system = messages[0]['content']
            if system.startswith(JUDGE_PROMPT_HEAD):
                await asyncio.sleep(self.judge_delay)
                return self.verdict
            return '{"allowed":true}' if system.startswith('发送前检查器') else '{}'
        self.bot.llm = SimpleNamespace(stream_chat=llm_stream, chat=AsyncMock(return_value='{}'),
                                       provider_name='test', model='test')
        self.bot.utility_llm = SimpleNamespace(chat=AsyncMock(side_effect=utility))
        # The first full reply in a process loads local models (seconds); do it before any timing is measured.
        await self.say('预热一下', 'warm')
        self.bot.server.send_text.reset_mock()
        self.requests.clear()

    async def asyncTearDown(self):
        self.bot.memory.close()
        self.bot.profiles.close()
        self.tmp.cleanup()

    async def say(self, text, message_id):
        await self.bot._handle_text('private', 100, 'owner_private:100', 100, '哥哥', text,
                                    'owner', False, 0, 'private', from_batch=True, message_id=message_id)

    async def test_a_slow_judge_never_delays_a_reply(self):
        self.verdict = '{"incomplete":true}'
        self.judge_delay = 2.5                                   # answers long after the reply is ready
        self.bot.typing.message_received(100, 'm1')
        started = time.monotonic()
        await self.say('我觉得吧，', 'm1')
        self.assertLess(time.monotonic() - started, 1.5)          # did not wait for the judge or the 2 s wait
        self.assertGreaterEqual(self.bot.server.send_text.await_count, 1)

    async def test_a_judge_that_says_finished_adds_no_wait(self):
        self.bot.typing.message_received(100, 'm1')
        started = time.monotonic()
        await self.say('今天天气不错。', 'm1')
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertGreaterEqual(self.bot.server.send_text.await_count, 1)

    async def test_an_unfinished_message_waits_and_merges_with_the_next_one(self):
        self.verdict = '{"incomplete":true}'
        bot = self.bot
        bot.typing.message_received(100, 'm1')
        task = asyncio.create_task(self.say('我觉得吧，', 'm1'))
        await asyncio.sleep(0.8)                                   # the reply is ready; the judge said unfinished
        self.assertEqual(bot.server.send_text.await_count, 0)      # held back
        bot.typing.message_received(100, 'm2')                      # the second half arrives
        await asyncio.wait_for(task, 3)
        self.assertEqual(bot.server.send_text.await_count, 0)
        held = bot._held_tails['owner_private:100']
        self.assertFalse(held['first_sent'])
        await self.say('这个方案不太行', 'm2')                        # m2's turn merges the unsent reply
        system = ' '.join(m['content'] for r in self.requests[1:] for m in r if m['role'] == 'system')
        self.assertIn('【未发出的话】', system)
        self.assertGreaterEqual(bot.server.send_text.await_count, 1)

    async def test_nothing_follows_so_the_unfinished_message_is_answered_after_the_wait(self):
        self.verdict = '{"incomplete":true}'
        self.bot.config['typing_hold']['incomplete_wait_seconds'] = 0.6
        self.bot.typing.message_received(100, 'm1')
        started = time.monotonic()
        await self.say('我觉得吧，', 'm1')
        self.assertGreaterEqual(self.bot.server.send_text.await_count, 1)
        self.assertGreater(time.monotonic() - started, 0.55)       # it did wait out the 0.6 s


if __name__ == '__main__':
    unittest.main()
