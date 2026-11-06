import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.personality import DIMENSIONS, ClassificationResult, describe_scores
from taoyao.search_gate import clearly_daily_message


class InstalledDeferredTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_gate_skips_only_clear_daily_messages(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = load_config()
        bot.llm = bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value='{"search":false}'))
        self.assertEqual(await bot._search_query_for('我有点累了', 'user：吃饭了吗'), '')
        bot.utility_llm.chat.assert_not_awaited()
        await bot._search_query_for('蓝瘦香菇')
        self.assertEqual(bot.utility_llm.chat.await_count, 1)
        await bot._search_query_for('我有点累了', 'user：这句话是什么梗')
        self.assertEqual(bot.utility_llm.chat.await_count, 2)

    def test_daily_gate_keeps_unknown_meme_fact_and_context_judgment(self):
        for value in ('我有点累了', '我刚忙完了', '陪我聊聊', '晚上好！'):
            self.assertTrue(clearly_daily_message(value, 'user：吃饭了吗'))
        for value in ('鸡你太美', '蓝瘦香菇', '今晚月色真美', '我困了是什么意思', '最新天气', '请查一下价格'):
            self.assertFalse(clearly_daily_message(value))
        self.assertFalse(clearly_daily_message('我有点累了', 'user：这句台词是什么出处'))

    async def test_two_turns_use_previous_scores_and_update_only_after_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config()
            config['response_speed']['two_part_reply'] = False
            config['response_speed'].update(speculative_reply=True, defer_six_axis=True)
            config['memory']['db'] = str(Path(directory)/'memory.db')
            config['profile'].update(db=str(Path(directory)/'profiles.db'), llm_extract=False)
            config['memes']['enabled'] = config['user_memes']['enabled'] = False
            config['intro']['enabled'] = False
            bot = TaoYaoBot(config, mock=True)
            original_tone = bot._tone_hint
            def observable_tone(profile, compact=False):
                return original_tone(profile, compact) + f" [TEST HUM {float(profile['scores']['HUM']):.1f}]"
            bot._tone_hint = observable_tone
            try:
                await bot.profiles.ensure(900001, '测试用户', 'user')
                scores = dict.fromkeys(DIMENSIONS, 50.)
                await bot.profiles.update_scores(900001, scores, describe_scores(scores))
                generations, order = [], []
                analysis_started, release = asyncio.Event(), asyncio.Event()
                async def stream(messages):
                    generations.append(messages)
                    yield '我在听，接着说。'
                bot.llm = bot.utility_llm = SimpleNamespace(stream_chat=stream,
                    chat=AsyncMock(return_value='{"allowed":true}'))
                labels = {d: ClassificationResult('中性', .9, applied=False) for d in DIMENSIONS}
                labels['HUM'] = ClassificationResult('轻松玩笑', .9)
                async def analyze(**kwargs):
                    order.append('analysis')
                    analysis_started.set()
                    await release.wait()
                    return labels, 'TEST'
                bot.analysis_pipeline = SimpleNamespace(config={'context_messages': 6}, analyze=analyze)
                bot._profile_identity_allowed = AsyncMock(return_value=True)
                bot._maybe_send_reply_poke = AsyncMock()
                bot._search_context_for = AsyncMock(return_value='')
                bot._schedule_profile_summary = lambda p: None
                async def send(*args, **kwargs):
                    order.append('sent')
                    return {'status': 'ok'}
                bot.server.send_text = AsyncMock(side_effect=send)
                async def turn(session, mid):
                    await bot._handle_text('private', 900001, session, 900001, '测试用户',
                        '我有点累了', 'user', False, 0, 'private', from_batch=True, message_id=mid)
                one = asyncio.create_task(turn('user_private:900001', 'ONE'))
                await asyncio.wait_for(analysis_started.wait(), 5)
                self.assertEqual(order, ['sent', 'analysis'])
                self.assertIn('HUM 50.0', generations[0][0]['content'])
                two = asyncio.create_task(turn('another_session:900001', 'TWO'))
                await asyncio.sleep(0)
                self.assertEqual(len(generations), 1)
                release.set()
                await asyncio.wait_for(asyncio.gather(one, two), 10)
                self.assertEqual(order, ['sent', 'analysis', 'sent', 'analysis'])
                self.assertEqual(len(generations), 2)
                self.assertNotIn('HUM 50.0', generations[1][0]['content'])
            finally:
                bot.memory.close()
                bot.profiles.close()
