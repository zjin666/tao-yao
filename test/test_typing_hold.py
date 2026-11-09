import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.turn_timing import get_value, set_value, turn_timing_context
from taoyao.typing_state import TypingTracker

USER = 2
SESSION = 'user_private:2'


class TrackerTests(unittest.TestCase):
    def test_typing_lasts_until_a_stop_event(self):
        tracker = TypingTracker(ttl_seconds=30)
        self.assertFalse(tracker.is_typing(USER, now=100))
        self.assertIsNone(tracker.note(USER, 1, now=100))
        self.assertTrue(tracker.is_typing(USER, now=112))          # QQ sends one start event, not periodic pings
        self.assertAlmostEqual(tracker.note(USER, 2, now=115), 15)  # event_type 2 = stopped typing
        self.assertFalse(tracker.is_typing(USER, now=115.1))

    def test_safety_timeout_when_the_stop_event_is_lost(self):
        tracker = TypingTracker(ttl_seconds=30)
        tracker.note(USER, 1, now=100)
        self.assertTrue(tracker.is_typing(USER, now=129))
        self.assertFalse(tracker.is_typing(USER, now=131))

    def test_voice_and_unknown_event_types_count_as_typing(self):
        tracker = TypingTracker()
        tracker.note(USER, 3)
        self.assertTrue(tracker.is_typing(USER))
        tracker = TypingTracker()
        tracker.note(USER, None)
        self.assertTrue(tracker.is_typing(USER))

    def test_a_message_clears_earlier_typing_and_bumps_the_sequence(self):
        tracker = TypingTracker()
        tracker.note(USER, 1, now=100)                 # typing the message that is now being sent
        self.assertEqual(tracker.message_seq(USER), 0)
        self.assertEqual(tracker.message_received(USER), 1)
        self.assertFalse(tracker.is_typing(USER, now=100.5))   # not "still typing" after sending
        tracker.note(USER, 1, now=101)                 # a new event after the message: typing again
        self.assertTrue(tracker.is_typing(USER, now=102))

    def test_replaying_the_events_seen_in_the_real_log(self):
        """06:14:38 .. 06:15:03 of the first live test: the 6 s guess released the reply while the user still typed."""
        tracker = TypingTracker(ttl_seconds=30)
        tracker.note(USER, 1, now=38.020)
        tracker.message_received(USER)                 # message 1 arrives at 38.642
        self.assertFalse(tracker.is_typing(USER, now=38.7))
        tracker.note(USER, 2, now=38.911)
        tracker.note(USER, 1, now=40.145)              # starts typing message 2
        self.assertTrue(tracker.is_typing(USER, now=46.3))     # was wrongly False with a 6 s window
        self.assertTrue(tracker.is_typing(USER, now=52.0))
        tracker.note(USER, 2, now=52.584)              # stopped
        self.assertFalse(tracker.is_typing(USER, now=52.7))
        tracker.note(USER, 1, now=57.575)
        tracker.note(USER, 2, now=57.857)
        self.assertFalse(tracker.is_typing(USER, now=58.0))

    def test_stopped_within_covers_the_gap_before_the_message(self):
        tracker = TypingTracker()
        self.assertFalse(tracker.stopped_within(USER, 1.0, now=100))
        tracker.note(USER, 1, now=100)
        self.assertFalse(tracker.stopped_within(USER, 1.0, now=100.1))   # typing, not stopped
        tracker.note(USER, 2, now=101)
        self.assertTrue(tracker.stopped_within(USER, 1.0, now=101.29))   # the message comes ~0.29 s later
        self.assertFalse(tracker.stopped_within(USER, 1.0, now=102.5))
        tracker.note(USER, 1, now=103)                                    # typing again clears the stop
        self.assertFalse(tracker.stopped_within(USER, 1.0, now=103.1))
        tracker.note(USER, 2, now=104)
        tracker.message_received(USER)                                    # a message clears it too
        self.assertFalse(tracker.stopped_within(USER, 1.0, now=104.1))

    def test_each_message_remembers_its_own_sequence_and_commands_do_not_count(self):
        tracker = TypingTracker()
        tracker.message_received(USER, 'm1')
        tracker.message_received(USER, 'm2')
        self.assertEqual((tracker.seq_of('m1'), tracker.seq_of('m2')), (1, 2))
        tracker.note(USER, 1)
        tracker.message_received(USER, 'cmd', bump=False)       # "/状态" is not a new message to merge with
        self.assertEqual(tracker.message_seq(USER), 2)
        self.assertEqual(tracker.seq_of('cmd'), 2)
        self.assertFalse(tracker.is_typing(USER))               # but it still ends the typing state
        self.assertIsNone(tracker.seq_of('never-seen'))

    def test_gap_before_a_message_measures_how_fast_the_user_is_sending(self):
        tracker = TypingTracker()
        tracker.message_received(USER, 'a', now=100)
        tracker.message_received(USER, 'b', now=104)
        self.assertIsNone(tracker.gap_before('a'))              # nothing before the first message
        self.assertAlmostEqual(tracker.gap_before('b'), 4.0)
        tracker.message_received(USER, 'cmd', bump=False, now=105)
        self.assertAlmostEqual(tracker.gap_before('cmd'), 1.0)
        tracker.message_received(USER, 'd', now=110)            # the command did not reset the reference
        self.assertAlmostEqual(tracker.gap_before('d'), 6.0)
        self.assertIsNone(tracker.gap_before('unknown'))

    def test_users_are_independent(self):
        tracker = TypingTracker()
        tracker.note(1)
        self.assertTrue(tracker.is_typing(1))
        self.assertFalse(tracker.is_typing(2))


def make_bot(stream, hold=None, ttl=0.4):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = load_config()
    bot.config['response_speed']['two_part_reply'] = True
    bot.config['reply'].update(segment_gap_min_seconds=0, segment_gap_max_seconds=0)
    bot.config['typing_hold'] = {'enabled': True, 'max_hold_seconds': 3, 'poll_seconds': 0.02,
                                 'stop_grace_seconds': 0.2, 'first_sentence_grace_seconds': 0.3, **(hold or {})}
    bot.typing = TypingTracker(ttl)
    bot._held_tails = {}
    bot.llm = SimpleNamespace(stream_chat=stream)
    bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value='{"allowed":true}'))
    bot.privacy_index = None
    bot._clean_output_symbols = AsyncMock(side_effect=lambda text: text)
    bot.server = SimpleNamespace(send_text=AsyncMock(return_value={}))
    bot._maybe_send_meme = AsyncMock(return_value=False)
    return bot


async def stream(messages):
    yield '我在听。'
    yield '慢慢说。别着急。'


class HoldDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_not_typing_sends_immediately(self):
        bot = make_bot(stream)
        started = time.monotonic()
        self.assertEqual(await bot._hold_while_typing(USER, {}), 'send')
        self.assertLess(time.monotonic() - started, 0.05)

    async def test_waits_while_typing_then_sends_when_typing_stops(self):
        bot = make_bot(stream, ttl=0.3)
        bot.typing.note(USER)
        started = time.monotonic()
        self.assertEqual(await bot._hold_while_typing(USER, {}), 'send')
        waited = time.monotonic() - started
        self.assertGreater(waited, 0.15)
        self.assertLess(waited, 1.0)

    async def test_waits_through_long_typing_and_releases_on_the_stop_event(self):
        bot = make_bot(stream, ttl=30)
        bot.typing.note(USER, 1)
        task = asyncio.create_task(bot._hold_while_typing(USER, {}))
        await asyncio.sleep(0.4)
        self.assertFalse(task.done())                   # a single start event keeps the hold going
        bot.typing.note(USER, 2)                        # event_type 2: stopped
        self.assertEqual(await asyncio.wait_for(task, 1), 'send')

    async def test_stop_event_is_followed_by_a_grace_wait_for_the_message(self):
        """Live log 06:25:40: the stop event arrived 0.29 s before the message and the tail was released too early."""
        bot = make_bot(stream, hold={'stop_grace_seconds': 1.0}, ttl=30)
        bot.typing.note(USER, 1)
        task = asyncio.create_task(bot._hold_while_typing(USER, {}))
        await asyncio.sleep(0.1)
        bot.typing.note(USER, 2)                       # "stopped" -- really the send button
        await asyncio.sleep(0.29)
        self.assertFalse(task.done())                  # still waiting for the message
        bot.typing.message_received(USER)              # the message shows up
        self.assertEqual(await asyncio.wait_for(task, 1), 'handoff')

    async def test_really_stopped_releases_after_the_grace_period(self):
        bot = make_bot(stream, hold={'stop_grace_seconds': 0.4}, ttl=30)
        bot.typing.note(USER, 1)
        task = asyncio.create_task(bot._hold_while_typing(USER, {}))
        await asyncio.sleep(0.1)
        bot.typing.note(USER, 2)
        stopped_at = time.monotonic()
        self.assertEqual(await asyncio.wait_for(task, 2), 'send')
        waited = time.monotonic() - stopped_at
        self.assertGreater(waited, 0.3)                # waited out the grace period
        self.assertLess(waited, 1.0)

    async def test_typing_again_during_the_grace_period_keeps_waiting(self):
        bot = make_bot(stream, hold={'stop_grace_seconds': 0.3}, ttl=30)
        bot.typing.note(USER, 1)
        task = asyncio.create_task(bot._hold_while_typing(USER, {}))
        await asyncio.sleep(0.1)
        bot.typing.note(USER, 2)
        await asyncio.sleep(0.15)
        bot.typing.note(USER, 1)                       # started typing again before the grace ran out
        await asyncio.sleep(0.5)
        self.assertFalse(task.done())
        bot.typing.message_received(USER)
        self.assertEqual(await asyncio.wait_for(task, 1), 'handoff')

    async def test_nothing_detected_at_the_first_check_means_send_immediately(self):
        """The first check happens when the sentence is ready; no typing then = assume the user stopped."""
        bot = make_bot(stream, hold={'stop_grace_seconds': 1.0}, ttl=30)
        started = time.monotonic()
        self.assertEqual(await bot._hold_while_typing(USER, {}), 'send')
        self.assertLess(time.monotonic() - started, 0.05)
        bot.typing.note(USER, 1)
        bot.typing.note(USER, 2)                       # stopped just before the check: still "not typing"
        started = time.monotonic()
        self.assertEqual(await bot._hold_while_typing(USER, {}), 'send')
        self.assertLess(time.monotonic() - started, 0.05)    # no grace wait when nothing is being held

    async def test_new_message_already_here_hands_off_immediately(self):
        bot = make_bot(stream)
        budget = {'seq': bot.typing.message_seq(USER)}
        bot.typing.message_received(USER, 'm2')            # arrived while the reply was being generated
        started = time.monotonic()
        self.assertEqual(await bot._hold_while_typing(USER, budget, first=True), 'handoff')
        self.assertLess(time.monotonic() - started, 0.05)  # zero added latency

    async def test_a_command_is_not_a_new_message_for_handoff(self):
        bot = make_bot(stream)
        budget = {'seq': bot.typing.message_seq(USER)}
        bot.typing.message_received(USER, 'cmd', bump=False)
        self.assertEqual(await bot._hold_while_typing(USER, budget), 'send')

    async def test_first_sentence_gets_a_short_detection_grace(self):
        bot = make_bot(stream)
        started = time.monotonic()
        self.assertEqual(await bot._hold_while_typing(USER, {}, first=True), 'send')
        waited = time.monotonic() - started
        self.assertGreater(waited, 0.25)                   # waited out the 0.3 s grace
        self.assertLess(waited, 0.8)

    async def _first_check(self, bot, gap):
        @turn_timing_context
        async def run():
            set_value('arrival_gap', gap)
            started = time.monotonic()
            result = await bot._hold_while_typing(USER, {}, first=True)
            return result, time.monotonic() - started
        return await run()

    async def test_sending_in_a_burst_waits_longer_before_the_first_sentence(self):
        hold = {'first_sentence_grace_seconds': 0.1, 'first_sentence_grace_burst_seconds': 0.5,
                'burst_window_seconds': 5}
        bot = make_bot(stream, hold=hold)
        result, waited = await self._first_check(bot, 2.0)          # previous message 2 s ago: bursting
        self.assertEqual(result, 'send')
        self.assertGreater(waited, 0.4)
        self.assertLess(waited, 0.9)
        result, waited = await self._first_check(bot, 30.0)         # a calm one-question-one-answer chat
        self.assertEqual(result, 'send')
        self.assertLess(waited, 0.35)
        result, waited = await self._first_check(bot, None)         # first message we have seen
        self.assertLess(waited, 0.35)

    async def test_burst_grace_catches_the_late_typing_event_seen_in_the_live_log(self):
        """Live log 07:06:31: the typing event came 0.86 s after the first sentence; a 0.3 s grace missed it."""
        hold = {'first_sentence_grace_seconds': 0.1, 'first_sentence_grace_burst_seconds': 1.0,
                'burst_window_seconds': 15}
        bot = make_bot(stream, hold=hold, ttl=30)

        @turn_timing_context
        async def run():
            set_value('arrival_gap', 4.2)
            return await bot._hold_while_typing(USER, {}, first=True)
        task = asyncio.create_task(run())
        await asyncio.sleep(0.5)                                    # past the normal grace, inside the burst grace
        bot.typing.note(USER, 1)
        await asyncio.sleep(0.3)
        self.assertFalse(task.done())                                # now holding instead of having sent already
        bot.typing.message_received(USER, 'm2')
        self.assertEqual(await asyncio.wait_for(task, 1), 'handoff')

    async def _tail_check(self, bot, gap, anchor_age, budget=None):
        @turn_timing_context
        async def run():
            set_value('arrival_gap', gap)
            state = budget if budget is not None else {}
            state['anchor'] = time.monotonic() - anchor_age        # when the first sentence went out
            started = time.monotonic()
            result = await bot._hold_while_typing(USER, state, first=False)
            return result, time.monotonic() - started
        return await run()

    async def test_tail_waits_until_a_grace_after_the_first_sentence_when_bursting(self):
        bot = make_bot(stream, hold={'tail_grace_burst_seconds': 0.6, 'burst_window_seconds': 5})
        result, waited = await self._tail_check(bot, 2.0, anchor_age=0.2)   # first sentence went out 0.2 s ago
        self.assertEqual(result, 'send')
        self.assertGreater(waited, 0.3)                    # waited the remaining ~0.4 s, not a full 0.6 s
        self.assertLess(waited, 0.7)

    async def test_tail_check_time_counts_towards_the_grace(self):
        bot = make_bot(stream, hold={'tail_grace_burst_seconds': 0.6, 'burst_window_seconds': 5})
        result, waited = await self._tail_check(bot, 2.0, anchor_age=1.0)   # the tail check already took longer
        self.assertEqual(result, 'send')
        self.assertLess(waited, 0.05)                      # zero extra delay

    async def test_tail_does_not_wait_when_not_bursting(self):
        bot = make_bot(stream, hold={'tail_grace_burst_seconds': 0.6, 'burst_window_seconds': 5})
        for gap in (30.0, None):
            result, waited = await self._tail_check(bot, gap, anchor_age=0.0)
            self.assertEqual(result, 'send')
            self.assertLess(waited, 0.05)

    async def test_tail_grace_default_is_two_seconds_after_the_first_sentence(self):
        bot = make_bot(stream)
        bot.config['typing_hold'] = {'enabled': True, 'poll_seconds': 0.02}     # no grace keys at all
        result, waited = await self._tail_check(bot, 3.0, anchor_age=1.8)       # 2.0 s default -> ~0.2 s left
        self.assertEqual(result, 'send')
        self.assertGreater(waited, 0.1)
        self.assertLess(waited, 0.6)
        result, waited = await self._tail_check(bot, 3.0, anchor_age=2.1)       # nothing left
        self.assertLess(waited, 0.05)

    async def test_late_typing_event_inside_the_tail_grace_holds_the_tail(self):
        """Typing events reach us 0.6-1.1 s after the first sentence; the tail grace is what catches them."""
        bot = make_bot(stream, hold={'tail_grace_burst_seconds': 1.0, 'burst_window_seconds': 15}, ttl=30)

        @turn_timing_context
        async def run():
            set_value('arrival_gap', 4.0)
            return await bot._hold_while_typing(USER, {'anchor': time.monotonic()}, first=False)
        task = asyncio.create_task(run())
        await asyncio.sleep(0.3)
        bot.typing.note(USER, 1)                           # late "started typing"
        await asyncio.sleep(0.3)
        self.assertFalse(task.done())
        bot.typing.message_received(USER, 'm2')
        self.assertEqual(await asyncio.wait_for(task, 1), 'handoff')

    async def test_default_config_releases_the_first_sentence_directly(self):
        """Live log 07:22: 4 of 5 first-sentence graces just delayed replies, so the default is no grace."""
        bot = make_bot(stream)
        bot.config['typing_hold'] = {'enabled': True, 'poll_seconds': 0.02}     # no grace keys at all
        for gap in (2.0, 30.0, None):
            result, waited = await self._first_check(bot, gap)                  # bursting or not
            self.assertEqual(result, 'send')
            self.assertLess(waited, 0.05)

    async def test_the_tail_gets_no_detection_grace(self):
        bot = make_bot(stream)
        started = time.monotonic()
        self.assertEqual(await bot._hold_while_typing(USER, {}, first=False), 'send')
        self.assertLess(time.monotonic() - started, 0.05)

    async def test_grace_can_be_turned_off(self):
        bot = make_bot(stream, hold={'first_sentence_grace_seconds': 0})
        started = time.monotonic()
        self.assertEqual(await bot._hold_while_typing(USER, {}, first=True), 'send')
        self.assertLess(time.monotonic() - started, 0.05)

    async def test_typing_detected_during_the_grace_turns_into_a_normal_hold(self):
        """Live log 06:16:36 / 06:25:37: the typing event arrived 0.0-0.5 s after the first sentence was ready."""
        bot = make_bot(stream, hold={'first_sentence_grace_seconds': 0.5}, ttl=30)
        task = asyncio.create_task(bot._hold_while_typing(USER, {}, first=True))
        await asyncio.sleep(0.15)
        bot.typing.note(USER, 1)                           # late typing event inside the grace window
        await asyncio.sleep(0.4)
        self.assertFalse(task.done())                      # now holding, not sending
        bot.typing.message_received(USER, 'm2')
        self.assertEqual(await asyncio.wait_for(task, 1), 'handoff')

    async def test_new_message_during_the_grace_hands_off(self):
        bot = make_bot(stream, hold={'first_sentence_grace_seconds': 0.5})
        budget = {'seq': bot.typing.message_seq(USER)}
        task = asyncio.create_task(bot._hold_while_typing(USER, budget, first=True))
        await asyncio.sleep(0.1)
        bot.typing.message_received(USER, 'm2')
        self.assertEqual(await asyncio.wait_for(task, 1), 'handoff')

    async def test_new_message_while_typing_means_handoff(self):
        bot = make_bot(stream, ttl=5)
        bot.typing.note(USER)
        task = asyncio.create_task(bot._hold_while_typing(USER, {}))
        await asyncio.sleep(0.1)
        self.assertFalse(task.done())
        bot.typing.message_received(USER)
        self.assertEqual(await asyncio.wait_for(task, 1), 'handoff')

    async def test_wait_has_an_upper_limit_shared_by_the_whole_reply(self):
        bot = make_bot(stream, hold={'max_hold_seconds': 0.3}, ttl=60)
        bot.typing.note(USER)
        budget = {}
        started = time.monotonic()
        self.assertEqual(await bot._hold_while_typing(USER, budget), 'send')
        self.assertLess(time.monotonic() - started, 1.0)
        again = time.monotonic()                      # the tail would wait again, but the budget is spent
        self.assertEqual(await bot._hold_while_typing(USER, budget), 'send')
        self.assertLess(time.monotonic() - again, 0.05)

    async def test_disabled_never_waits(self):
        bot = make_bot(stream, hold={'enabled': False})
        bot.typing.note(USER)
        self.assertEqual(await bot._hold_while_typing(USER, {}), 'send')


class ReplyFlowTests(unittest.IsolatedAsyncioTestCase):
    async def reply(self, bot):
        return await bot._stream_reply(
            [{'role': 'system', 'content': '桃夭人设'}, {'role': 'user', 'content': '继续说吧'}],
            'private', USER, 4, 70, role='user', review_policy={'identity': 'user'},
            viewer_id=USER, reply_window_key=(SESSION, USER))

    async def test_nothing_is_sent_while_typing_then_first_sentence_goes_first(self):
        bot = make_bot(stream, ttl=0.4)
        bot.typing.note(USER)
        task = asyncio.create_task(self.reply(bot))
        await asyncio.sleep(0.15)
        self.assertEqual(bot.server.send_text.await_count, 0)        # generated and checked, but held
        text, category, *_ = await asyncio.wait_for(task, 3)
        sent = [call.args[2] for call in bot.server.send_text.await_args_list]
        self.assertEqual(sent[0], '我在听')                         # first sentence first, as soon as typing stopped
        self.assertGreaterEqual(len(sent), 2)                         # then the tail, sentence by sentence
        self.assertEqual(bot._held_tails, {})

    async def test_new_message_while_typing_keeps_the_whole_reply_for_merging(self):
        bot = make_bot(stream, ttl=30)
        bot.typing.note(USER)
        task = asyncio.create_task(self.reply(bot))
        await asyncio.sleep(0.2)
        bot.typing.message_received(USER)
        text, category, *_ = await asyncio.wait_for(task, 3)
        self.assertEqual(bot.server.send_text.await_count, 0)        # nothing went out
        self.assertEqual((text, category), ('', ''))
        held = bot._held_tails[SESSION]
        self.assertFalse(held['first_sent'])
        self.assertIn('我在听', held['text'])
        self.assertIn('别着急', held['text'])                        # the tail was collected too

    async def test_first_sentence_goes_out_at_once_but_the_tail_waits_in_a_burst(self):
        bot = make_bot(stream, hold={'first_sentence_grace_seconds': 0, 'tail_grace_burst_seconds': 0.6,
                                     'burst_window_seconds': 15})

        @turn_timing_context
        async def run():
            set_value('arrival_gap', 3.0)
            return await self.reply(bot)
        started = time.monotonic()
        task = asyncio.create_task(run())
        await asyncio.sleep(0.2)
        self.assertEqual(bot.server.send_text.await_count, 1)          # first sentence: no added delay
        self.assertEqual(bot.server.send_text.await_args_list[0].args[2], '我在听')
        await asyncio.wait_for(task, 3)
        self.assertGreaterEqual(bot.server.send_text.await_count, 2)   # the tail followed
        self.assertGreater(time.monotonic() - started, 0.5)            # ...only after the grace had passed

    async def test_typing_detected_late_hands_the_tail_to_the_next_message(self):
        bot = make_bot(stream, hold={'first_sentence_grace_seconds': 0, 'tail_grace_burst_seconds': 1.0,
                                     'burst_window_seconds': 15}, ttl=30)

        @turn_timing_context
        async def run():
            set_value('arrival_gap', 3.0)
            return await self.reply(bot)
        task = asyncio.create_task(run())
        await asyncio.sleep(0.25)
        bot.typing.note(USER, 1)                                       # the late typing event
        await asyncio.sleep(0.2)
        bot.typing.message_received(USER, 'm2')                        # then the next message
        text, *_ = await asyncio.wait_for(task, 3)
        self.assertEqual(bot.server.send_text.await_count, 1)          # only the first sentence went out
        held = bot._held_tails[SESSION]
        self.assertTrue(held['first_sent'])
        self.assertIn('别着急', held['text'])

    async def test_no_tail_wait_outside_a_burst(self):
        bot = make_bot(stream, hold={'first_sentence_grace_seconds': 0, 'tail_grace_burst_seconds': 1.0,
                                     'burst_window_seconds': 15})
        started = time.monotonic()
        await self.reply(bot)                                          # no turn context = not bursting
        self.assertLess(time.monotonic() - started, 0.6)
        self.assertGreaterEqual(bot.server.send_text.await_count, 2)

    async def test_group_replies_are_never_held(self):
        bot = make_bot(stream)
        bot.typing.note(USER)
        await bot._stream_reply([{'role': 'system', 'content': 'x'}, {'role': 'user', 'content': 'y'}],
                                'group', 555, 4, 70, role='user', review_policy={'identity': 'user'},
                                viewer_id=USER, reply_window_key=('group:555', USER))
        self.assertGreaterEqual(bot.server.send_text.await_count, 1)


class MergeNoteTests(unittest.TestCase):
    def bot(self):
        bot = make_bot(stream)
        return bot

    def test_held_text_becomes_a_note_once(self):
        bot = self.bot()
        bot._held_tails[SESSION] = {'text': '我在听\n慢慢说[[meme:shy]]', 'at': time.monotonic(), 'first_sent': False}
        note = bot._take_held_reply(SESSION)
        self.assertIn('慢慢说', note)
        self.assertIn('一句都没有发出', note)
        self.assertNotIn('meme', note)
        self.assertEqual(bot._take_held_reply(SESSION), '')           # consumed

    def test_note_says_when_the_first_sentence_already_went_out(self):
        bot = self.bot()
        bot._held_tails[SESSION] = {'text': '后面的话', 'at': time.monotonic(), 'first_sent': True}
        self.assertIn('首句已经发出去了', bot._take_held_reply(SESSION))

    def test_stale_or_other_sessions_are_ignored(self):
        bot = self.bot()
        bot._held_tails[SESSION] = {'text': '太久以前的', 'at': time.monotonic() - 500, 'first_sent': False}
        self.assertEqual(bot._take_held_reply(SESSION), '')
        bot._held_tails[SESSION] = {'text': '别的会话', 'at': time.monotonic(), 'first_sent': False}
        self.assertEqual(bot._take_held_reply('user_private:3'), '')


class EventTests(unittest.IsolatedAsyncioTestCase):
    async def test_input_status_notice_marks_typing_and_a_message_clears_it(self):
        bot = make_bot(stream, ttl=30)
        bot.config['typing_hold']['log_events'] = False
        await bot._handle_notice({'post_type': 'notice', 'notice_type': 'notify', 'sub_type': 'input_status',
                                  'user_id': USER, 'event_type': 1, 'self_id': 100})
        self.assertTrue(bot.typing.is_typing(USER))
        await bot._handle_notice({'post_type': 'notice', 'notice_type': 'notify', 'sub_type': 'input_status',
                                  'user_id': USER, 'event_type': 2, 'self_id': 100})
        self.assertFalse(bot.typing.is_typing(USER))
        bot.typing.note(USER, 1)
        bot.typing.message_received(USER)
        self.assertFalse(bot.typing.is_typing(USER))

    async def test_own_input_status_is_ignored(self):
        bot = make_bot(stream)
        await bot._handle_notice({'post_type': 'notice', 'notice_type': 'notify', 'sub_type': 'input_status',
                                  'user_id': 100, 'self_id': 100})
        self.assertFalse(bot.typing.is_typing(100))


if __name__ == '__main__':
    unittest.main()


class MergeEndToEnd(unittest.IsolatedAsyncioTestCase):
    """The next message's real generation request carries the unsent reply."""

    async def asyncSetUp(self):
        import tempfile
        from pathlib import Path
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
        self.bot = TaoYaoBot(cfg, mock=True)
        self.bot.batch_enabled = False
        self.bot.server.chat_history = ChatHistory(Path(self.tmp.name) / 'out')
        self.bot.server.send_text = AsyncMock(return_value={'status': 'ok'})
        self.bot._schedule_reply_extras = lambda *a, **k: None
        self.bot._maybe_send_reply_poke = AsyncMock()
        self.requests = []

        async def llm_stream(messages):
            self.requests.append(messages)
            yield '收到。那就这样。'
        async def utility(messages):
            return '{"allowed":true}' if messages[0]['content'].startswith('发送前检查器') else '{}'
        self.bot.llm = SimpleNamespace(stream_chat=llm_stream, chat=AsyncMock(return_value='{}'),
                                       provider_name='test', model='test')
        self.bot.utility_llm = SimpleNamespace(chat=AsyncMock(side_effect=utility))

    async def asyncTearDown(self):
        self.bot.memory.close()
        self.bot.profiles.close()
        self.tmp.cleanup()

    async def say(self, text, message_id='m'):
        await self.bot._handle_text('private', 100, 'owner_private:100', 100, '哥哥', text,
                                    'owner', False, 0, 'private', from_batch=True, message_id=message_id)

    async def test_unsent_reply_reaches_the_next_generation_once(self):
        self.bot._held_tails['owner_private:100'] = {
            'text': '这是上一条没发出去的回复', 'at': time.monotonic(), 'first_sent': False}
        await self.say('我又补了一句')
        system = ' '.join(m['content'] for r in self.requests for m in r if m['role'] == 'system')
        self.assertIn('【未发出的话】', system)
        self.assertIn('这是上一条没发出去的回复', system)
        self.assertEqual(self.bot._held_tails, {})
        self.requests.clear()
        await self.say('再说一句')
        system = ' '.join(m['content'] for r in self.requests for m in r if m['role'] == 'system')
        self.assertNotIn('【未发出的话】', system)                    # consumed, never repeated

    async def test_replies_go_out_normally_when_nothing_is_held(self):
        await self.say('你好')
        self.assertGreaterEqual(self.bot.server.send_text.await_count, 1)

    async def test_message_arriving_while_generating_hands_the_whole_reply_to_the_next_turn(self):
        """No typing involved: m2 lands after m1 was received but before m1's first sentence is sent."""
        bot = self.bot
        bot.config['typing_hold'] = {'enabled': True, 'poll_seconds': 0.02, 'first_sentence_grace_seconds': 0.3,
                                     'stop_grace_seconds': 0.2, 'log_events': False}
        bot.typing.message_received(100, 'm1')                 # what on_event does when m1 arrives

        async def slow_stream(messages):
            self.requests.append(messages)
            if len(self.requests) == 1:
                await asyncio.sleep(0.2)
                bot.typing.message_received(100, 'm2')       # the second message lands mid-generation
            yield '好的。后面还有一句。'
        bot.llm = SimpleNamespace(stream_chat=slow_stream, chat=AsyncMock(return_value='{}'),
                                  provider_name='test', model='test')

        await self.say('第一条', 'm1')
        self.assertEqual(bot.server.send_text.await_count, 0)    # nothing of m1's reply went out
        held = bot._held_tails['owner_private:100']
        self.assertFalse(held['first_sent'])
        self.assertIn('后面还有一句', held['text'])

        await self.say('第二条', 'm2')                            # m2's turn merges it
        system = ' '.join(m['content'] for r in self.requests[1:] for m in r if m['role'] == 'system')
        self.assertIn('【未发出的话】', system)
        self.assertIn('后面还有一句', system)
        self.assertGreaterEqual(bot.server.send_text.await_count, 1)
        self.assertEqual(bot._held_tails, {})

    async def test_a_command_arriving_mid_generation_does_not_swallow_the_reply(self):
        bot = self.bot
        bot.config['typing_hold'] = {'enabled': True, 'poll_seconds': 0.02, 'first_sentence_grace_seconds': 0,
                                     'log_events': False}
        bot.typing.message_received(100, 'm1')

        async def slow_stream(messages):
            await asyncio.sleep(0.1)
            bot.typing.message_received(100, 'cmd', bump=False)
            yield '好的。'
        bot.llm = SimpleNamespace(stream_chat=slow_stream, chat=AsyncMock(return_value='{}'),
                                  provider_name='test', model='test')
        await self.say('第一条', 'm1')
        self.assertGreaterEqual(bot.server.send_text.await_count, 1)   # sent normally
        self.assertEqual(bot._held_tails, {})

    async def test_on_event_records_arrival_and_ignores_commands(self):
        bot = self.bot
        bot._enqueue = AsyncMock()

        def private(message_id, text):
            return {'post_type': 'message', 'message_type': 'private', 'user_id': 100, 'self_id': 999,
                    'message_id': message_id, 'time': int(time.time()), 'raw_message': text,
                    'sender': {'nickname': '哥哥'}, 'message': [{'type': 'text', 'data': {'text': text}}]}
        await bot.on_event(private('a1', '你好'))
        self.assertEqual(bot.typing.seq_of('a1'), 1)
        self.assertEqual(bot.typing.message_seq(100), 1)
        await bot.on_event(private('a2', '/状态'))
        self.assertEqual(bot.typing.message_seq(100), 1)      # the command did not count as a new message
        self.assertEqual(bot.typing.seq_of('a2'), 1)
        await bot.on_event(private('a3', '再说一句'))
        self.assertEqual(bot.typing.message_seq(100), 2)

    async def test_handler_passes_the_gap_since_the_previous_message_to_the_hold(self):
        bot = self.bot
        seen = []

        async def spy(user_id, budget, first=False, private=True):
            seen.append((first, get_value('arrival_gap')))
            return 'send'
        bot._hold_while_typing = spy
        now = time.monotonic()
        bot.typing.message_received(100, 'm0', now=now - 5)
        bot.typing.message_received(100, 'm1', now=now)
        await self.say('第二条', 'm1')
        self.assertTrue(seen)
        self.assertTrue(seen[0][0])                                  # the first-sentence check
        self.assertAlmostEqual(seen[0][1], 5.0, places=1)           # previous message was 5 s earlier

