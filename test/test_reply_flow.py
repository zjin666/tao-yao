import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.persona import PROMPT_SECTIONS, build_system_prompt, load_persona
from taoyao.personality import describe_tone
from taoyao.reply_flow import reply_boundary_issue


class ReplyFlowTests(unittest.TestCase):
    def test_sections_are_ordered_and_omit_empty_data(self):
        persona = load_persona()
        tone = describe_tone({'HUM': 80})
        for role in ('owner', 'user'):
            for group in (False, True):
                prompt = build_system_prompt(persona, ['happy'], role=role, is_group=group,
                    tone_hint=tone, search_context='REFERENCE', media_present=True,
                    relationship={'label': '朋友', 'behavior': '既有关系说明'},
                    message_routing={'state': 'bot', 'evidence': 'explicit_at'})
                stages = ['【步骤一：理解意图与对象】', '【步骤二：确定关系边界】',
                          '【步骤三：选择六维行为倾向】', '【步骤四：以桃夭人设一次生成】',
                          '【步骤五：发送前检查】']
                positions = [prompt.index(stage) for stage in stages]
                self.assertEqual(positions, sorted(positions))
                self.assertEqual(prompt.count(persona), 1)
                self.assertEqual(prompt.count(tone), 1)
                self.assertLess(prompt.index('【联网搜索结果】'), positions[1])
                self.assertLess(prompt.index('【当前对话对象】'), positions[2])
                self.assertLess(prompt.index('【称呼身份准则】'), positions[2])
                self.assertGreater(prompt.index(persona), positions[3])
                for name, text in PROMPT_SECTIONS.items():
                    if name in ('长期记忆摘要', '关于哥哥的事实', '用户要求记住的事', '群公共要求', '当前用户画像'):
                        self.assertNotIn('【'+name+'】', prompt)
                    else:
                        self.assertEqual(prompt.count('【'+name+'】'), 1)
                        if '{' not in text:
                            self.assertIn(text, prompt)

    def test_review_contract_is_an_identity_checklist_not_generation_rules(self):
        policy = {}
        build_system_prompt('PERSONA_ONLY', [], role='user', is_group=True,
                            relationship={'label': '朋友', 'behavior': 'EXISTING_BOUNDARY'},
                            review_policy=policy)
        self.assertNotIn('rules', policy)
        self.assertNotIn('EXISTING_BOUNDARY', str(policy))
        self.assertNotIn('PERSONA_ONLY', str(policy))
        self.assertEqual(policy['identity'], 'user')
        self.assertEqual(policy['scene'], 'group')
        self.assertEqual(set(policy['check_only']), {'identity', 'privacy', 'intimacy', 'internal_message'})

    def test_non_owner_vocative_does_not_block_discussing_owner(self):
        for text in ('哥哥，你看这个。', '嗯，哥哥，我知道啦。', '[[at]] 哥，你先试试。'):
            self.assertEqual(reply_boundary_issue(text, 'user'), 'non_owner_vocative')
            self.assertEqual(reply_boundary_issue(text, 'owner'), '')
        for text in ('哥哥刚才说过这个。', '哥哥今天在忙。', '你说的是哥哥的意思吗？', '你先试试。'):
            self.assertEqual(reply_boundary_issue(text, 'user'), '')


class ReplyFlowStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_reviewer_uses_single_contract_and_never_passes_unknown_decisions(self):
        bot = object.__new__(TaoYaoBot)
        bot.llm = SimpleNamespace(chat=AsyncMock())
        reply = '你这脑洞也太会拐弯了吧。'
        for verdict in (
            '{"allowed":false,"reason":"style","evidence":"脑洞"}',
            '{"allowed":false,"reason":"unsupported_action","evidence":"脑洞"}',
        ):
            bot.llm.chat.return_value = verdict
            self.assertEqual(await bot._review_reply_boundary(reply, [], {}), ('boundary_review_unavailable', ''))
        for verdict in ('{"allowed":false,"reason":"intimacy","evidence":"不存在的内容"}',
                        '{"allowed":false,"reason":"intimacy"}'):
            bot.llm.chat.return_value=verdict
            self.assertEqual(await bot._review_reply_boundary(reply, [], {}),('intimacy',''))
        bot.llm.chat.return_value = '{"allowed":false,"reason":"privacy","evidence":"秘密原文"}'
        self.assertEqual(await bot._review_reply_boundary('秘密原文', [], {}), ('privacy', '秘密原文'))

    async def test_boundary_violation_is_repaired_before_any_send(self):
        bot = object.__new__(TaoYaoBot)
        calls = []
        async def stream(messages):
            calls.append(messages)
            bot._send_segments.assert_not_awaited()
            yield '哥哥，你看这个。' if len(calls) == 1 else '你看这个。'
        bot.llm = SimpleNamespace(stream_chat=stream)
        bot._send_segments = AsyncMock()
        bot._maybe_send_meme = AsyncMock(return_value=False)
        original = [{'role': 'system', 'content': '既有人设、边界与六维'}]
        with self.assertLogs('taoyao.app', level='WARNING'):
            result = await bot._stream_reply(original, 'group', 1, 2, 70, role='user')
        self.assertEqual(len(calls), 2)
        self.assertEqual(result[0], '你看这个。')
        self.assertIn('当前对象不是 owner', calls[1][0]['content'])
        self.assertEqual(original[0]['content'], '既有人设、边界与六维')

    async def test_valid_reply_uses_one_generation(self):
        bot = object.__new__(TaoYaoBot)
        calls = []
        async def stream(messages):
            calls.append(messages)
            yield '这下接上啦。'
        bot.llm = SimpleNamespace(stream_chat=stream)
        bot._send_segments = AsyncMock()
        bot._maybe_send_meme = AsyncMock(return_value=False)
        result = await bot._stream_reply([], 'group', 1, 2, 70, role='user')
        self.assertEqual(result[0], '这下接上啦。')
        self.assertEqual(len(calls), 1)

    async def test_semantic_check_is_not_a_second_polish(self):
        bot = object.__new__(TaoYaoBot)
        generated = []
        async def stream(messages):
            generated.append(messages)
            yield '这下接上啦。'
        bot.llm = SimpleNamespace(stream_chat=stream, chat=AsyncMock(return_value='{"allowed":true}'))
        bot._send_segments = AsyncMock()
        bot._maybe_send_meme = AsyncMock(return_value=False)
        result = await bot._stream_reply([], 'group', 1, 2, 70, role='user',
                                        review_policy={'identity': 'user', 'scene': 'group'})
        self.assertEqual(len(generated), 1)
        self.assertEqual(result[0], '这下接上啦。')
        bot.llm.chat.assert_awaited_once()
        self.assertIn('不润色', bot.llm.chat.call_args.args[0][0]['content'])

    async def test_unavailable_review_has_no_outward_side_effects(self):
        bot = object.__new__(TaoYaoBot)
        async def stream(messages):
            yield '准备回复的内容。'
        bot.llm = SimpleNamespace(stream_chat=stream, chat=AsyncMock(return_value='not json'))
        bot._send_segments = AsyncMock()
        bot._maybe_send_meme = AsyncMock(return_value=False)
        with self.assertLogs('taoyao.app', level='WARNING'):
            result = await bot._stream_reply([], 'group', 1, 2, 70, role='user', review_policy={})
        self.assertEqual(result[0], '')
        bot._send_segments.assert_not_awaited()
        bot._maybe_send_meme.assert_not_awaited()
