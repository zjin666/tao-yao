import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from taoyao.profiles import ProfileStore
from taoyao.turn_timing import set_value, turn_timing_context
from taoyao.wait_habits import PAUSE_COUNT, WaitHabits
from test_typing_hold import make_bot, stream

USER = 2
GKEY = ('g', 555, USER)


def done_future(value):
    fut = asyncio.get_running_loop().create_future()
    fut.set_result(value)
    return fut


class LearnerTests(unittest.TestCase):
    def test_unseen_person_uses_the_default(self):
        self.assertEqual(WaitHabits().wait_for('p:1', 4.0), 4.0)

    def test_quick_follow_ups_shorten_the_wait(self):
        habits = WaitHabits()
        for delay in (1.0, 1.2, 0.9, 1.1):
            habits.record('p:1', hit=True, delay=delay, default=4.0)
        wait = habits.wait_for('p:1', 4.0)
        self.assertLess(wait, 3.0)
        self.assertGreater(wait, 1.5)                  # moves toward ~1.5*delay+0.5 gradually, not in one jump

    def test_late_follow_ups_lengthen_the_wait(self):
        habits = WaitHabits()
        for delay in (3.7, 3.8, 3.9, 3.6):
            habits.record('p:1', hit=True, delay=delay, default=4.0)
        self.assertGreater(habits.wait_for('p:1', 4.0), 5.0)

    def test_waiting_in_vain_shortens_the_wait_down_to_the_floor(self):
        habits = WaitHabits()
        for _ in range(5):
            habits.record('p:1', hit=False, default=4.0)
        self.assertLess(habits.wait_for('p:1', 4.0), 2.5)
        for _ in range(3):
            habits.record('p:1', hit=True, default=4.0)           # a hit without a delay does not move the wait
        self.assertGreaterEqual(habits.wait_for('p:1', 4.0, floor=1.0), 1.0)

    def test_the_wait_stays_within_floor_and_ceiling(self):
        habits = WaitHabits()
        habits.record('p:1', hit=True, delay=100.0, default=4.0, floor=1.0, ceil=6.0)
        self.assertEqual(habits.wait_for('p:1', 4.0, 1.0, 6.0), 6.0)
        habits.record('p:2', hit=True, delay=0.0, default=1.2, floor=1.0, ceil=6.0)
        self.assertGreaterEqual(habits.wait_for('p:2', 1.2, 1.0, 6.0), 1.0)

    def test_someone_who_never_continues_is_paused_then_relearned(self):
        habits = WaitHabits()
        for _ in range(6):
            habits.record('p:1', hit=False, default=4.0)
        self.assertEqual(habits.snapshot('p:1')['pause'], PAUSE_COUNT)
        self.assertEqual([habits.wait_for('p:1', 4.0) for _ in range(PAUSE_COUNT)], [0.0] * PAUSE_COUNT)
        self.assertGreater(habits.wait_for('p:1', 4.0), 0.0)        # probation over: waits again, from the floor
        self.assertEqual(habits.snapshot('p:1')['n'], 0)

    def test_people_are_independent(self):
        habits = WaitHabits()
        for _ in range(5):
            habits.record('p:1', hit=False, default=4.0)
        self.assertEqual(habits.wait_for('p:2', 4.0), 4.0)
        self.assertEqual(habits.wait_for('g:555:1', 3.0), 3.0)

    def test_state_round_trips_and_bad_data_is_ignored(self):
        habits = WaitHabits()
        habits.record('p:1', hit=True, delay=1.0, default=4.0)
        saved = habits.to_dict('p:1')
        restored = WaitHabits()
        restored.load('p:1', saved)
        self.assertEqual(restored.snapshot('p:1'), saved)
        restored.load('p:2', {'wait': 'oops'})
        restored.load('p:3', None)
        self.assertEqual(restored.wait_for('p:2', 4.0), 4.0)         # ignored, falls back to the default
        self.assertEqual(restored.wait_for('p:3', 4.0), 4.0)


class PersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_habits_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(Path(tmp) / 'p.db')
            try:
                habits = WaitHabits()
                habits.record('p:1', hit=True, delay=1.0, default=4.0)
                await store.save_wait_habit('p:1', habits.to_dict('p:1'))
                habits.record('p:1', hit=False, default=4.0)
                await store.save_wait_habit('p:1', habits.to_dict('p:1'))   # an update replaces, not duplicates
                reopened = ProfileStore(Path(tmp) / 'p.db')
                try:
                    loaded = await reopened.load_wait_habits()
                finally:
                    reopened._conn.close()
            finally:
                store._conn.close()
            self.assertEqual(list(loaded), ['p:1'])
            restored = WaitHabits()
            restored.load('p:1', loaded['p:1'])
            self.assertEqual(restored.snapshot('p:1'), habits.snapshot('p:1'))


class AdaptiveWaitTests(unittest.IsolatedAsyncioTestCase):
    HOLD = {'first_sentence_grace_seconds': 0, 'incomplete_wait_seconds': 0.6, 'incomplete_wait_group_seconds': 0.5,
            'incomplete_wait_min_seconds': 0.2, 'burst_window_seconds': 15}

    def bot(self, hold=None):
        bot = make_bot(stream, hold={**self.HOLD, **(hold or {})})
        bot.wait_habits = WaitHabits()
        return bot

    async def run_wait(self, bot, key=USER, private=True, during=None):
        @turn_timing_context
        async def run():
            set_value('incomplete_task', done_future(True))
            started = time.monotonic()
            task = asyncio.create_task(bot._hold_while_typing(
                key, {'seq': bot.typing.message_seq(key)}, first=True, private=private))
            if during is not None:
                await during()
            return await task, time.monotonic() - started
        return await run()

    async def test_a_hit_is_recorded_and_the_wait_moves_toward_the_persons_pace(self):
        bot = self.bot({'incomplete_wait_seconds': 2.0})                      # a 0.2 s follow-up -> target ~0.8 s

        async def follow_up():
            await asyncio.sleep(0.2)
            bot.typing.message_received(USER, 'm2')
        result, _ = await self.run_wait(bot, during=follow_up)
        self.assertEqual(result, 'handoff')
        state = bot.wait_habits.snapshot('p:2')
        self.assertEqual(state['n'], 1)
        self.assertGreater(state['p'], 0.5)                                   # hit rate went up
        self.assertLess(state['wait'], 2.0)                                   # a quick follow-up -> shorter wait
        self.assertGreater(state['wait'], 0.8)                                # ...gradually, not in one jump

    async def test_waiting_in_vain_is_recorded_and_shortens_the_next_wait(self):
        bot = self.bot()
        first_wait = None
        for i in range(3):
            result, waited = await self.run_wait(bot)
            self.assertEqual(result, 'send')
            first_wait = first_wait if first_wait is not None else waited
        self.assertLess(waited, first_wait - 0.05)                           # each fruitless wait is shorter
        self.assertLess(bot.wait_habits.snapshot('p:2')['p'], 0.5)

    async def test_a_paused_person_is_not_made_to_wait(self):
        bot = self.bot()
        bot.wait_habits.load('p:2', {'wait': 0.2, 'p': 0.5, 'n': 0, 'pause': 3})
        result, waited = await self.run_wait(bot)
        self.assertEqual(result, 'send')
        self.assertLess(waited, 0.1)

    async def test_a_learned_wait_is_used_instead_of_the_default(self):
        bot = self.bot()
        bot.wait_habits.load('p:2', {'wait': 0.25, 'p': 0.6, 'n': 4, 'pause': 0})
        result, waited = await self.run_wait(bot)
        self.assertEqual(result, 'send')
        self.assertLess(waited, 0.5)                                          # the default would have been 0.6 s

    async def test_group_people_learn_separately_from_private_chat(self):
        bot = self.bot()
        bot.wait_habits.load('p:2', {'wait': 0.2, 'p': 0.6, 'n': 4, 'pause': 0})
        result, waited = await self.run_wait(bot, key=GKEY, private=False)
        self.assertGreater(waited, 0.35)                                      # uses its own 0.5 s group default
        self.assertIsNotNone(bot.wait_habits.snapshot('g:555:2'))

    async def test_a_message_already_here_counts_as_a_hit_without_moving_the_wait(self):
        bot = self.bot()
        bot.wait_habits.load('p:2', {'wait': 0.5, 'p': 0.5, 'n': 0, 'pause': 0})

        @turn_timing_context
        async def run():
            set_value('incomplete_task', done_future(True))
            budget = {'seq': bot.typing.message_seq(USER)}
            bot.typing.message_received(USER, 'm2')                          # arrived before the reply was ready
            return await bot._hold_while_typing(USER, budget, first=True, private=True)
        self.assertEqual(await run(), 'handoff')
        state = bot.wait_habits.snapshot('p:2')
        self.assertEqual(state['wait'], 0.5)
        self.assertGreater(state['p'], 0.5)

    async def test_adaptation_can_be_switched_off(self):
        bot = self.bot({'incomplete_adapt': False})
        bot.wait_habits.load('p:2', {'wait': 0.2, 'p': 0.5, 'n': 0, 'pause': 5})
        result, waited = await self.run_wait(bot)
        self.assertGreater(waited, 0.5)                                       # fixed default, pause ignored
        self.assertEqual(bot.wait_habits.snapshot('p:2')['n'], 0)             # nothing recorded

    async def test_outcomes_are_saved_in_the_background(self):
        bot = self.bot()
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(Path(tmp) / 'p.db')
            try:
                bot.profiles, bot.profiles_enabled, bot._memory_tasks = store, True, set()
                await self.run_wait(bot)
                await asyncio.gather(*list(bot._memory_tasks))
                saved = await store.load_wait_habits()
                self.assertIn('p:2', saved)
                self.assertEqual(saved['p:2']['n'], 1)
            finally:
                store._conn.close()


if __name__ == '__main__':
    unittest.main()
