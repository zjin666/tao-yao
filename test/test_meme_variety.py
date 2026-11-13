import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot

CONFIG = {'memes': {'enabled': True, 'probability': 1.0, 'min_interval_seconds': 0, 'max_per_window': 99,
                    'window_seconds': 900, 'skip_long_chars': 100, 'llm_judge': True, 'direct_score': 0.9,
                    'judge_min_score': 0.45}}


def make_bot(candidates=None):
    bot = TaoYaoBot.__new__(TaoYaoBot)
    bot.config = CONFIG
    bot.llm = SimpleNamespace()
    sent = []
    bot.memes = SimpleNamespace(
        categories={'teasing': '调侃得意', 'plain': '淡定', 'happy': '开心'},
        match_candidates=lambda sentence, preferred: (list(candidates or []), (candidates or [(None, 0)])[0] if candidates else None),
        pick=lambda category, intensity=0.5, conversation='': Path(f'{category}.png'),
        infer_intensity=lambda text: 0.5, _last_sentence=lambda text: text,
        can_use=lambda *a, **k: True)
    bot.user_memes = SimpleNamespace(use_probability=0.0, match=lambda *a, **k: None)
    bot._send_image = AsyncMock(side_effect=lambda target, target_id, path: sent.append(Path(path).stem) or True)
    bot._meme_probability = lambda role: 1.0
    bot._recent_chat_lines = AsyncMock(return_value=[])
    bot.sent = sent
    return bot


class MemeVarietyTests(unittest.IsolatedAsyncioTestCase):
    async def send(self, bot, tag):
        return await bot._maybe_send_meme('private', 1, '哼，才不是呢。', 'owner', preferred=tag)

    async def test_a_category_used_twice_in_the_last_four_is_set_aside(self):
        bot = make_bot(candidates=[('plain', 0.92), ('happy', 0.5)])
        for _ in range(2):
            self.assertTrue(await self.send(bot, 'teasing'))
        self.assertEqual(bot.sent, ['teasing', 'teasing'])
        self.assertEqual(bot._tired_meme_categories('private:1'), {'teasing'})
        # 第三次模型还是标 teasing：不再沿用，改用本地语义里下一个够格的分类
        self.assertTrue(await self.send(bot, 'teasing'))
        self.assertEqual(bot.sent[-1], 'plain')

    async def test_when_nothing_else_fits_it_prefers_no_meme_over_repeating(self):
        bot = make_bot(candidates=[])
        for _ in range(2):
            await self.send(bot, 'teasing')
        self.assertFalse(await self.send(bot, 'teasing'))
        self.assertEqual(bot.sent, ['teasing', 'teasing'])

    async def test_conversations_are_independent_and_variety_recovers(self):
        bot = make_bot(candidates=[('plain', 0.92)])
        await self.send(bot, 'teasing')
        await self.send(bot, 'teasing')
        other = await bot._maybe_send_meme('private', 2, '哼，才不是呢。', 'owner', preferred='teasing')
        self.assertTrue(other)
        self.assertEqual(bot.sent[-1], 'teasing')              # 别的聊天不受影响
        for _ in range(3):                                     # 最近 4 个里换成别的分类，teasing 又能用了
            bot._note_meme_sent('private:1', 'happy')
        self.assertEqual(bot._tired_meme_categories('private:1'), {'happy'})
        self.assertTrue(await self.send(bot, 'teasing'))

    async def test_judge_is_not_offered_the_tired_category(self):
        bot = make_bot()
        bot.utility_llm = SimpleNamespace(chat=AsyncMock(return_value='{"category":"none"}'))
        await bot._meme_judge('private', 1, '哼。', [('plain', 0.6)], ('plain', 0.6), avoid={'teasing'})
        prompt = bot.utility_llm.chat.await_args.args[0][1]['content']
        self.assertNotIn('teasing', prompt)
        self.assertIn('happy', prompt)


if __name__ == '__main__':
    unittest.main()
