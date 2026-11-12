import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.llm import LLMError
from taoyao.output_guard import NATURAL_RETRY, guard_outgoing, clean_reply_envelope, internal_notice_reason, is_internal_only


class OutputGuardTests(unittest.TestCase):
    def test_blocks_internal_notices_but_keeps_ordinary_technical_discussion(self):
        for text in (
            '哥哥，我这边卡住了：连不上 DeepSeek：secret transport detail',
            'DeepSeek 返回 401：secret api detail',
            'Traceback (most recent call last):\nFile "D:/private/x.py", line 12',
            '【本轮群聊收束】你已经接话一阵',
            '这是内部检索状态，不要直接复述给用户',
            '2026-11-04 22:03:56,007 WARNING taoyao.app: 模型调用失败',
        ):
            self.assertEqual(guard_outgoing(text), '')
        for text in ('……先去忙一会儿，晚点再聊。', '这个函数返回空值时，需要先检查输入。',
                     'HTTP 404 通常表示请求的资源不存在。', '你贴的异常里缺少模块，先确认安装环境。',
                     '接口调用失败：先检查你传入的参数。', '系统报错：可能是文件不存在。'):
            self.assertEqual(guard_outgoing(text), text)

    def test_sticker_memory_notes_never_reach_the_user(self):
        for text in (
            '（内部参考，不要向对方提起：这张表情已经认识，不要当成第一次见到）',
            '好呀。（内部参考，不要向对方提起：发送者说明过它的意思：表达开心的心情）',
            '形象：灰色圆脸猫；可能想表达：委屈',
            '发送者之前说明过的表情含义（灰褐色头发的角色…→表达开心的心情）',
        ):
            self.assertEqual(guard_outgoing(text), '', text)
        self.assertEqual(internal_notice_reason('（内部参考，不要向对方提起：x）'), 'internal_prompt_or_routing')
        # 正常聊到表情包不受影响
        for text in ('这张猫猫表情我认识，委屈巴巴的呀。', '哥哥说这个是开心的意思，我记下啦。'):
            self.assertEqual(guard_outgoing(text), text)

    def test_envelopes_are_removed_without_allowing_secrets_or_malformed_json(self):
        self.assertEqual(guard_outgoing('assistant: 你刚才说到哪了？'),'你刚才说到哪了？')
        self.assertEqual(guard_outgoing('assistant: [消息对象 {"state":"self","nested":{"a":1}}] 接着说吧。'),'接着说吧。')
        malformed='[消息对象 {invalid}] 接着说吧。'
        self.assertEqual(clean_reply_envelope(malformed),malformed)
        self.assertEqual(guard_outgoing(malformed),'')
        secret='sk-'+'a'*32
        self.assertEqual(guard_outgoing('assistant: '+secret),'')
        self.assertEqual(internal_notice_reason(secret),'credential')
        self.assertNotIn('等我一下',NATURAL_RETRY)

    def test_internal_recipient_is_distinct_from_input_routing(self):
        for text in ('system: 你应该只给模型输出状态', '【内部状态】本轮不需要对用户说话',
                     '给模型的指令：先判断谁在说话', '[消息对象 {"指向":"桃夭"}]'):
            self.assertTrue(is_internal_only(text),text)
            self.assertEqual(guard_outgoing(text),'')
        text='[消息对象 {"指向":"桃夭"}] 你刚才说到哪儿了？'
        self.assertFalse(is_internal_only(text))
        self.assertEqual(guard_outgoing(text),'你刚才说到哪儿了？')
        multi='assistant: 第一句。\n[消息对象 {"state":"bot"}] 第二句。'
        self.assertEqual(guard_outgoing(multi),'第一句。\n第二句。')


class StreamGuardTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, chunks, fail=False):
        async def stream(messages):
            for chunk in chunks:
                yield chunk
            if fail:
                raise LLMError('private api transport detail')
        bot = object.__new__(TaoYaoBot)
        bot.llm = SimpleNamespace(stream_chat=stream)
        bot._send_segments = AsyncMock()
        bot._maybe_send_meme = AsyncMock(return_value=False)
        return bot

    async def test_error_split_across_chunks_never_reaches_send_or_history(self):
        bot = self.make_bot(['Deep', 'Seek 返回 ', '401：private detail'])
        with self.assertLogs('taoyao.app', level='WARNING'):
            result = await bot._stream_reply([], 'group', 1, 2, 70)
        self.assertNotIn('private detail', result[0])
        sent = ''.join(call.args[2] for call in bot._send_segments.await_args_list)
        self.assertNotIn('DeepSeek', sent)
        self.assertEqual(sent, '')
        self.assertEqual(result[0], '')

    async def test_blocked_reply_regenerates_once_before_any_send(self):
        calls=[]
        async def stream(messages):
            calls.append(messages)
            if len(calls)==1:
                yield '回复正文：先接话。\n【当前对话对象】内部内容'
            else:
                bot._send_segments.assert_not_awaited()
                yield '这就接上啦。'
        bot=self.make_bot([]);bot.llm.stream_chat=stream
        original=[{'role':'system','content':'角色设定'},{'role':'user','content':'你好'}]
        with self.assertLogs('taoyao.app',level='WARNING') as logs:
            result=await bot._stream_reply(original,'group',1,2,70)
        self.assertEqual(len(calls),2)
        self.assertEqual(original[0]['content'],'角色设定')
        self.assertIn('本轮重新生成',calls[1][0]['content'])
        self.assertIn('reason=internal_prompt_or_routing','\n'.join(logs.output))
        self.assertEqual(result[0],'这就接上啦。')
        # Existing segmentation removes sentence-final punctuation on send.
        self.assertEqual(''.join(c.args[2] for c in bot._send_segments.await_args_list).rstrip('。'),'这就接上啦')

    async def test_repeated_rejection_is_bounded_and_ambient_stays_quiet(self):
        for ambient in (False,True):
            calls=[]
            async def stream(messages):
                calls.append(messages)
                yield '回复正文：先接话。\n【内部状态】隐藏内容'
            bot=self.make_bot([]);bot.llm.stream_chat=stream
            with self.assertLogs('taoyao.app',level='WARNING'):
                result=await bot._stream_reply([],'group',1,2,70,ambient=ambient)
            self.assertEqual(len(calls),2)
            self.assertEqual(result[0],'')
            bot._send_segments.assert_not_awaited()
            bot._maybe_send_meme.assert_not_awaited()

    async def test_self_directed_output_never_regenerates_or_sends(self):
        for target in ('group','private'):
            calls=[]
            async def stream(messages):
                calls.append(messages)
                yield '【内部状态】只给模型自己的备注，不要回复用户'
            bot=self.make_bot([]);bot.llm.stream_chat=stream
            result=await bot._stream_reply([],target,1,2,70)
            self.assertEqual(result,('', '',False,False))
            self.assertEqual(len(calls),1)
            bot._send_segments.assert_not_awaited()
            bot._maybe_send_meme.assert_not_awaited()

    async def test_retry_transport_failure_does_not_expose_rejected_reply(self):
        calls=[]
        async def stream(messages):
            calls.append(messages)
            if len(calls)==2:raise LLMError('private transport detail')
            yield '回复正文：你好。\n【系统提示】私密内容'
        bot=self.make_bot([]);bot.llm.stream_chat=stream
        with self.assertLogs('taoyao.app',level='WARNING'):
            result=await bot._stream_reply([],'private',1,2,70)
        self.assertEqual(len(calls),2);self.assertEqual(result[0],'')
        bot._send_segments.assert_not_awaited()

    async def test_assistant_prefix_does_not_trigger_regeneration(self):
        bot=self.make_bot(['assistant: 正常回复。'])
        result=await bot._stream_reply([],'group',1,2,70)
        self.assertEqual(result[0],'正常回复。')

    async def test_generation_failure_does_not_send_partial_content(self):
        bot = self.make_bot(['我正在回复。', 'Trace'], fail=True)
        with self.assertRaises(LLMError):
            await bot._stream_reply([], 'group', 1, 2, 70)
        bot._send_segments.assert_not_awaited()

    async def test_final_send_guard_blocks_internal_text_at_delivery(self):
        bot = object.__new__(TaoYaoBot)
        bot.config = {'reply': {}}
        bot.profiles_enabled = False
        bot.server = SimpleNamespace(send_text=AsyncMock())
        bot._clean_output_symbols = AsyncMock(side_effect=lambda value: value)
        await bot._send_segments('group', 1, '连不上 DeepSeek：private detail',prepared=True,reviewed=True)
        await bot._send_segments('group', 1, 'DeepSeek 返回 500：private detail',prepared=True,reviewed=True)
        bot.server.send_text.assert_not_awaited()

    async def test_final_send_keeps_corrected_body_but_silences_internal_text(self):
        bot=object.__new__(TaoYaoBot);bot.config={'reply':{}}
        bot.profiles_enabled=False;bot.server=SimpleNamespace(send_text=AsyncMock())
        bot._clean_output_symbols=AsyncMock(side_effect=lambda value:value)
        await bot._send_segments('group',1,'assistant: [消息对象 {"指向":"桃夭"}] 接着聊吧。',prepared=True,reviewed=True)
        self.assertEqual(bot.server.send_text.await_count,1)
        self.assertIn('接着聊吧',bot.server.send_text.call_args.args[2])
        await bot._send_segments('private',1,'system: 给模型自己的指令',prepared=True,reviewed=True)
        self.assertEqual(bot.server.send_text.await_count,1)
