import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from taoyao import memes as meme_module
from taoyao.app import TaoYaoBot
from taoyao.memes import MIN_SCORE, MemeLibrary

ROOT = Path(__file__).resolve().parents[1] / 'memes'
CONFIG = {'memes': {'enabled': True, 'probability': 1.0, 'min_interval_seconds': 120, 'max_per_window': 6,
                    'window_seconds': 900, 'skip_long_chars': 100}}


class FlatEmbedder:
    """Every text is equally unrelated to every category: any score above zero must come from elsewhere."""

    def embed_texts(self, texts):
        return np.zeros((len(texts), 4))


class ScoreGateTests(unittest.TestCase):
    def test_a_bare_keyword_can_no_longer_carry_a_meme_when_the_semantic_model_is_unsure(self):
        library = MemeLibrary(ROOT)
        library.set_embedder(FlatEmbedder())
        ranked, chosen = library.match_candidates('一个人扛确实累，卡住了也别硬熬到烦', None)
        self.assertIsNone(chosen)
        self.assertLess(ranked[0][1], MIN_SCORE)

    def test_without_a_semantic_model_keywords_still_decide_as_before(self):
        library = MemeLibrary(ROOT)
        self.assertIsNotNone(library.match('这下真的有点手足无措。')[0])


class Gate(unittest.TestCase):
    def bot(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = CONFIG
        return bot

    def test_long_informational_replies_and_links_get_no_untagged_meme(self):
        bot = self.bot()
        self.assertTrue(bot._meme_allowed('c', '这个好笑呀。'))
        self.assertFalse(bot._meme_allowed('c', '攻略很长' * 30))
        self.assertFalse(bot._meme_allowed('c', '看这个 https://example.com/a'))

    def test_two_minute_spacing_and_the_fifteen_minute_cap(self):
        bot = self.bot()
        clock = [1000.0]
        with patch.object(meme_module.time, 'monotonic', lambda: clock[0]), patch('taoyao.app.time.monotonic', lambda: clock[0]):
            self.assertTrue(bot._meme_allowed('c', '好'))
            bot._note_meme_sent('c')
            clock[0] += 119
            self.assertFalse(bot._meme_allowed('c', '好'))                 # not yet two minutes
            clock[0] += 2
            self.assertTrue(bot._meme_allowed('c', '好'))                  # two minutes passed
            self.assertTrue(bot._meme_allowed('other chat', '好'))         # chats are independent
            for _ in range(5):
                bot._note_meme_sent('c')
                clock[0] += 121
            self.assertEqual(len(bot._meme_times['c']), 6)
            clock[0] -= 121
            self.assertFalse(bot._meme_allowed('c', '好'))                 # six within fifteen minutes: that is the cap
            clock[0] += 400
            self.assertTrue(bot._meme_allowed('c', '好'))                  # the window moved on


class SendTests(unittest.IsolatedAsyncioTestCase):
    def bot(self, tagged_ok=True):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {'memes': {**CONFIG['memes'], 'llm_judge': False}}
        bot.memes = SimpleNamespace(categories={'happy': '开心'}, match_candidates=lambda s, p: ([('happy', 0.9)], ('happy', 0.9)),
                                    _last_sentence=lambda s: s, infer_intensity=lambda s: 0.5, pick=lambda *a, **k: Path('x.png'))
        bot.user_memes = SimpleNamespace(use_probability=0.0)
        bot._meme_probability = lambda role: 1.0
        bot._send_image = AsyncMock(return_value=True)
        return bot

    async def test_a_tag_from_the_model_ignores_spacing_and_length_but_still_counts_toward_it(self):
        bot = self.bot()
        self.assertTrue(await bot._maybe_send_meme('private', 1, '好开心呀', 'owner', preferred='happy'))
        self.assertTrue(await bot._maybe_send_meme('private', 1, '又开心了' * 40, 'owner', preferred='happy'))    # right after, and long
        self.assertEqual(bot._send_image.await_count, 2)
        self.assertEqual(len(bot._meme_times['private:1']), 2)
        # an untagged one right after is held back by the spacing
        self.assertFalse(await bot._maybe_send_meme('private', 1, '这个好笑呀。', 'owner'))
        self.assertEqual(bot._send_image.await_count, 2)

    async def test_an_untagged_meme_is_sent_when_allowed_and_recorded(self):
        bot = self.bot()
        self.assertTrue(await bot._maybe_send_meme('group', 5, '这个好笑呀。', 'owner'))
        self.assertEqual(len(bot._meme_times['group:5']), 1)
        self.assertFalse(await bot._maybe_send_meme('group', 5, '这个好笑呀。', 'owner'))     # within two minutes


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def chat(self, messages, *a, **k):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)


CANDIDATES = [('lonely', 0.74), ('tired', 0.65), ('plain', 0.50), ('happy', 0.30)]


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    def bot(self, reply):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = CONFIG
        bot.utility_llm = FakeLLM(reply)
        bot.llm = bot.utility_llm
        bot.memes = SimpleNamespace(categories={'lonely': '孤单', 'tired': '疲惫', 'plain': '淡定', 'happy': '开心'})
        bot._recent_chat_lines = AsyncMock(return_value=['[11-10 周二 23:02] 对方：我一个人扛着好累', '[11-10 周二 23:03] 桃夭：我陪你'])
        return bot

    async def judge(self, bot, local=('lonely', 0.74), candidates=CANDIDATES):
        return await bot._meme_judge('private', 1, '一个人扛确实累，我陪哥哥一块找哪儿漏了', candidates, local)

    async def test_the_model_picks_among_the_candidates_using_the_dated_context(self):
        bot = self.bot({'category': 'tired'})
        self.assertEqual((await self.judge(bot))[0], 'tired')
        prompt = bot.utility_llm.calls[0][1]['content']
        self.assertIn('[11-10 周二 23:02] 对方：我一个人扛着好累', prompt)          # the dated context is what it judges with
        self.assertIn('一个人扛确实累', prompt)
        self.assertIn('- lonely：孤单', prompt)
        self.assertIn('其他分类', prompt)                                         # the rest of the categories are offered too
        self.assertIn('happy：', prompt)

    async def test_none_an_unknown_category_garbage_or_an_error_all_mean_no_meme(self):
        for reply in ({'category': 'none'}, {'category': 'angry'}, '不是json', RuntimeError('down')):
            self.assertIsNone(await self.judge(self.bot(reply)), reply)

    async def test_a_very_confident_local_match_skips_the_model_and_a_hopeless_one_never_asks(self):
        bot = self.bot({'category': 'tired'})
        self.assertEqual(await self.judge(bot, local=('sleepy', 0.94), candidates=[('sleepy', 0.94)]), ('sleepy', 0.94))
        self.assertIsNone(await self.judge(bot, local=None, candidates=[('plain', 0.40), ('happy', 0.30)]))
        self.assertEqual(bot.utility_llm.calls, [])

    async def test_the_model_may_pick_a_better_category_that_the_local_model_did_not_rank(self):
        bot = self.bot({'category': 'happy'})                                       # not among the top candidates, but a known category
        self.assertEqual((await self.judge(bot))[0], 'happy')

    async def test_a_weak_local_pick_can_be_rescued_by_the_model(self):
        bot = self.bot({'category': 'plain'})
        self.assertEqual((await self.judge(bot, local=None))[0], 'plain')           # local said "not sure", the model decides

    async def test_the_judge_can_be_switched_off(self):
        bot = self.bot({'category': 'tired'})
        bot.config = {'memes': {**CONFIG['memes'], 'llm_judge': False}}
        self.assertEqual(await self.judge(bot), ('lonely', 0.74))
        self.assertEqual(bot.utility_llm.calls, [])


class JudgeInTheSendStepTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_untagged_meme_is_only_sent_after_the_model_agrees_and_a_tagged_one_never_asks(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = CONFIG
        picked = []
        bot.memes = SimpleNamespace(categories={'tired': '疲惫', 'lonely': '孤单'},
                                    match_candidates=lambda s, p: ([('lonely', 0.74), ('tired', 0.65)], ('lonely', 0.74)),
                                    _last_sentence=lambda s: s, infer_intensity=lambda s: 0.5,
                                    pick=lambda cat, **k: picked.append(cat) or Path('x.png'))
        bot.user_memes = SimpleNamespace(use_probability=0.0)
        bot._meme_probability = lambda role: 1.0
        bot._send_image = AsyncMock(return_value=True)
        bot._recent_chat_lines = AsyncMock(return_value=[])
        bot.utility_llm = FakeLLM({'category': 'none'})
        bot.llm = bot.utility_llm
        self.assertFalse(await bot._maybe_send_meme('private', 1, '这个好累呀。', 'owner'))         # the model said no
        self.assertEqual(picked, [])
        bot.utility_llm.reply = {'category': 'tired'}
        self.assertTrue(await bot._maybe_send_meme('group', 2, '这个好累呀。', 'owner'))
        self.assertEqual(picked, ['tired'])                                                         # the model's choice, not the local top
        calls = len(bot.utility_llm.calls)
        self.assertTrue(await bot._maybe_send_meme('group', 3, '好累啊。', 'owner', preferred='tired'))
        self.assertEqual(len(bot.utility_llm.calls), calls)                                         # a tag from the reply's author is never second-guessed


if __name__ == '__main__':
    unittest.main()
