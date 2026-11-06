import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.memes import MemeLibrary


class MemeReplyEndingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.library = MemeLibrary(Path(__file__).resolve().parents[1]/'memes')

    def test_actual_reply_and_opposite_ending(self):
        cases = [
            ('行行行，都怪我可可爱爱，让哥哥忍不住笑。\n\n再说下去我就要把脸埋枕头里了。', 'shy'),
            ('刚才确实有些脸红。现在终于成功了，好开心！', 'happy'),
            ('刚才还挺开心。现在只觉得难过。', 'sad'),
        ]
        for text, expected in cases:
            self.assertEqual(self.library.match(text, preferred='happy')[0], expected)

    def test_caring_reminder_wins_over_plain_or_food(self):
        cases=[
            '一年，那这血条确实不是纸糊的了\n好吧，刚才那句算我小看你了\n练完记得补点能量，别空腹撑着',
            '嘴上说不管你。记得吃饭，别饿着。',
            '今天风大，记得保暖，别冻着。',
            '你可别逞强，累了就休息。',
        ]
        for text in cases:
            self.assertEqual(self.library.match(text,preferred='plain')[0],'gentle')
        self.assertEqual(self.library.match('今天这顿饭真好吃')[0],'food')
        self.assertEqual(self.library.match('我饿了，想吃点东西')[0],'food')
        self.assertEqual(self.library.match('刚才说记得吃饭。再说我就脸红了。')[0],'shy')
        self.assertEqual(self.library.match('你还好吗？我担心你。')[0],'worried')

    def test_care_intent_beats_neutral_similarity_but_keeps_explicit_worry(self):
        self.library._semantic_scores=lambda text:{'plain':.94,'food':.91,'gentle':.36}
        self.assertEqual(self.library.match('练完记得补点能量，别空腹撑着',preferred='plain')[0],'gentle')
        self.library._semantic_scores=lambda text:{}
        self.assertEqual(self.library.match('我担心你，别硬撑了')[0],'worried')

    def test_last_sentence_semantics_weigh_more_than_full_text(self):
        self.library._semantic_scores = lambda text: ({'happy': .9, 'shy': .5}
            if '。' in text else {'happy': .4, 'shy': .85})
        self.assertEqual(self.library.match('前面表达轻快。结尾含蓄地躲开视线')[0], 'shy')

    async def test_meme_selected_once_after_all_text_with_full_reply(self):
        bot = TaoYaoBot.__new__(TaoYaoBot)
        events = []
        reply = '开心！\n\n再说下去我就要把脸埋枕头里了。'
        async def stream(messages):
            yield reply + '[[meme:happy]]'
        bot.llm = SimpleNamespace(stream_chat=stream)
        bot._review_reply_boundary = AsyncMock(return_value=('', ''))
        async def send(*args, **kwargs):
            events.append('text')
            return True
        async def meme(*args, **kwargs):
            events.append('meme')
            self.assertIn('脸埋枕头', args[2])
            self.assertIn('开心', args[2])
            return True
        bot._send_segments = AsyncMock(side_effect=send)
        bot._maybe_send_meme = AsyncMock(side_effect=meme)
        result = await bot._stream_reply([{'role': 'user', 'content': '继续聊'}],
            'private', 1, 2, 20, role='owner', review_policy={})
        self.assertTrue(result[3])
        bot._maybe_send_meme.assert_awaited_once()
        self.assertEqual(events[-1], 'meme')
        self.assertGreater(events.count('text'), 1)
