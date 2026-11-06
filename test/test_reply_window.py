import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from taoyao.app import TaoYaoBot, PendingMessage
from taoyao.reply_window import ReplyWindow, reply_window_context, conversation_session


class ReplyWindowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = {'mono': 10.0, 'wall': 100.0}
        self.patches = [patch('taoyao.reply_window.time.monotonic', side_effect=lambda: self.clock['mono']),
                        patch('taoyao.reply_window.time.time', side_effect=lambda: self.clock['wall'])]
        for handle in self.patches:
            handle.start()
            self.addCleanup(handle.stop)

    async def test_only_remaining_time_is_waited(self):
        window = ReplyWindow()
        key = ('group:1', 2)
        window.note(key, 100.0, .8)
        self.clock['mono'] = 10.3
        async def sleep(seconds):
            self.clock['mono'] += seconds
        with patch('taoyao.reply_window.asyncio.sleep', new=AsyncMock(side_effect=sleep)) as sleeper:
            await window.wait(key)
        sleeper.assert_awaited_once()
        self.assertAlmostEqual(sleeper.call_args.args[0], .5)

    async def test_finished_after_window_sends_without_sleep(self):
        window = ReplyWindow()
        key = ('private:2', 2)
        window.note(key, 100.0, .8)
        self.clock['mono'] = 11.0
        with patch('taoyao.reply_window.asyncio.sleep', new=AsyncMock()) as sleeper:
            await window.wait(key)
        sleeper.assert_not_awaited()

    async def test_new_same_user_message_extends_wait_but_other_user_does_not(self):
        for same_user in (False, True):
            self.clock.update(mono=10., wall=100.)
            window = ReplyWindow()
            key = ('group:1', 2)
            window.note(key, 100.0, .8)
            waits = []
            async def sleep(seconds):
                waits.append(seconds)
                if len(waits) == 1:
                    self.clock.update(mono=10.6, wall=100.6)
                    window.note(key if same_user else ('group:1', 3), 100.6, .8)
                    self.clock['mono'] = 10.8
                else:
                    self.clock['mono'] += seconds
            with patch('taoyao.reply_window.asyncio.sleep', new=AsyncMock(side_effect=sleep)):
                await window.wait(key)
            self.assertEqual(len(waits), 2 if same_user else 1)
            if same_user:
                self.assertAlmostEqual(waits[1], .6)

    async def test_gate_happens_after_polish_and_cleanup(self):
        bot = object.__new__(TaoYaoBot)
        bot.config = {'reply': {}}
        bot.profiles_enabled = False
        bot._reply_window = ReplyWindow()
        key = ('private:2', 2)
        bot._reply_window.note(key, 100.0, .8)
        bot.server = SimpleNamespace(send_text=AsyncMock())
        bot._clean_output_symbols = AsyncMock(side_effect=lambda text: text)
        async def polish(text, profile):
            self.clock['mono'] = 11.2
            return text
        bot._polish_text = AsyncMock(side_effect=polish)
        with patch('taoyao.reply_window.asyncio.sleep', new=AsyncMock()) as sleeper:
            await bot._send_segments('private', 2, '你好。', polish=True, reply_window_key=key)
        sleeper.assert_not_awaited()
        bot.server.send_text.assert_awaited_once()

    async def test_following_bubbles_do_not_wait_again(self):
        window = ReplyWindow()
        key = ('group:1', 2)
        window.note(key, 100.0, .8)
        async def sleep(seconds):
            self.clock['mono'] += seconds
        @reply_window_context
        async def response(session, user_id):
            await window.before_send()
            self.clock['wall'] = 100.8
            window.note(key, 100.8, .8)
            await window.before_send()
        with patch('taoyao.reply_window.asyncio.sleep', new=AsyncMock(side_effect=sleep)) as sleeper:
            await response('group:1', 2)
        sleeper.assert_awaited_once()

    def test_old_event_and_invalid_timestamp(self):
        window = ReplyWindow()
        window.note(('private:1', 1), 95.0, .8)
        self.assertEqual(window.deadlines[('private:1', 1)], 10.0)
        window.note(('private:2', 2), None, .8)
        self.assertAlmostEqual(window.deadlines[('private:2', 2)], 10.8)

    async def test_private_session_key_matches_dispatch_and_first_delivery_metrics(self):
        window = ReplyWindow()
        for user_id, expected in ((1, 'owner_private:1'), (2, 'user_private:2')):
            session = conversation_session(0, user_id, 1)
            self.assertEqual(session, expected)
            key = (session, user_id)
            self.clock.update(mono=10., wall=100.)
            window.note(key, 100., .8)
            self.clock['mono'] = 14.
            await window.before_send(key)
            self.assertAlmostEqual(window.first_delivery_elapsed(key), 4.)
        self.assertEqual(conversation_session(100, 2, 1), 'group:100')

    async def test_queue_starts_generation_without_pre_window_sleep(self):
        bot = object.__new__(TaoYaoBot)
        key = ('group:1', 0)
        item = PendingMessage(target='group', target_id=1, session='group:1', user_id=2,
                              nickname='群友', text='你好', role='user', is_group=True,
                              group_id=1, scene='group')
        bot._queues = {key: [item]}
        bot.batch_window = .8
        bot.batch_max = 6
        bot._batch_tasks = {}
        bot._locks = {}
        bot._handle_text = AsyncMock()
        with patch('taoyao.app.asyncio.sleep', new=AsyncMock()) as sleeper:
            await bot._batch_worker(key)
        bot._handle_text.assert_awaited_once()
        sleeper.assert_not_awaited()

    async def test_segment_gap_only_applies_to_following_bubbles(self):
        bot = object.__new__(TaoYaoBot)
        bot.config = {'reply': {'segment_gap_min_seconds': 1.0, 'segment_gap_max_seconds': 1.6}}
        bot.profiles_enabled = False
        bot._clean_output_symbols = AsyncMock(side_effect=lambda text: text)
        bot.server = SimpleNamespace(send_text=AsyncMock())
        with patch('taoyao.app.asyncio.sleep', new=AsyncMock()) as sleeper, patch('taoyao.app.random.uniform', return_value=1.3) as random_gap:
            await bot._send_segments('group', 1, '第一句。', polish=False, delay=False)
            sleeper.assert_not_awaited()
            random_gap.assert_not_called()
            await bot._send_segments('group', 1, '第二句。', polish=False, delay=True)
            sleeper.assert_awaited_once_with(1.3)
            random_gap.assert_called_once_with(1.0, 1.6)
        self.assertEqual(bot.server.send_text.await_count, 2)
