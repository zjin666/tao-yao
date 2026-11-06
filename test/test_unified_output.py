import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot, ReplyIntent, _draft_output
from taoyao.config import load_config
from taoyao.persona import load_persona
from taoyao.llm import LLMError
from taoyao.reply_window import reply_window_context


class UnifiedOutputTests(unittest.IsolatedAsyncioTestCase):
    def bot(self,body='我记下了，哥哥。'):
        bot=TaoYaoBot.__new__(TaoYaoBot)
        bot.config=load_config();bot.config['owner_qq']=1
        bot.config['response_speed']['two_part_reply']=False
        bot.config['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
        bot.persona=load_persona();bot.profiles_enabled=False
        bot.inputs=[]
        async def stream(messages):
            bot.inputs.append(messages)
            yield body
        bot.llm=SimpleNamespace(stream_chat=stream,chat=AsyncMock(side_effect=AssertionError('No independent text generation')))
        bot.utility_llm=SimpleNamespace(chat=AsyncMock(return_value='{"allowed":true}'))
        bot.server=SimpleNamespace(send_text=AsyncMock(return_value={'status':'ok'}))
        bot._maybe_send_meme=AsyncMock(return_value=False)
        bot._clean_output_symbols=AsyncMock(side_effect=lambda text:text)
        bot._tone_hint=lambda *args,**kwargs:'SIX_AXIS_REFERENCE'
        return bot

    async def test_fixed_result_with_polish_false_still_uses_original_flow_once(self):
        bot=self.bot()
        self.assertTrue(await bot._send_segments('private',1,'知识已保存。',polish=False))
        self.assertEqual(len(bot.inputs),1)
        system=bot.inputs[0][0]['content']
        stages=['【步骤二：确定关系边界】','【步骤三：选择六维行为倾向】','【步骤四：以桃夭人设一次生成】','【步骤五：发送前检查】']
        self.assertEqual([system.index(s) for s in stages],sorted(system.index(s) for s in stages))
        self.assertIn('SIX_AXIS_REFERENCE',system)
        self.assertIn(bot.persona,system)
        self.assertIn('知识已保存',bot.inputs[0][-1]['content'])
        bot.llm.chat.assert_not_awaited()
        bot.utility_llm.chat.assert_awaited_once()
        bot.server.send_text.assert_awaited_once()
        self.assertIn('我记下了',bot.server.send_text.await_args.args[2])

    async def test_feature_intent_is_not_generated_until_normal_flow(self):
        bot=self.bot()
        intent=ReplyIntent('记住这个','知识写入成功，需要确认。','已保存。')
        self.assertIsInstance(intent,ReplyIntent)
        self.assertEqual(bot.inputs,[])
        bot.server.send_text.assert_not_awaited()
        self.assertTrue(await bot._send_segments('private',1,intent,polish=False))
        self.assertIn('知识写入成功',bot.inputs[0][0]['content'])
        self.assertTrue(intent.delivered_text)

    async def test_owner_notice_and_group_task_enter_same_stream(self):
        bot=self.bot('大家晚安。')
        await bot._notify_owner('已加入一个群。')
        bot.profiles=SimpleNamespace(group_context=AsyncMock(return_value=[]))
        task=await bot._group_task_text('去测试群说晚安',[{'group_id':10}])
        self.assertIsInstance(task,ReplyIntent)
        self.assertEqual(len(bot.inputs),1)
        await bot._send_segments('group',10,task)
        self.assertEqual(len(bot.inputs),2)
        self.assertIn('【步骤二：确定关系边界】',bot.inputs[-1][0]['content'])
        self.assertIn('写发到群里的消息',bot.inputs[-1][0]['content'])
        bot.llm.chat.assert_not_awaited()

    async def test_failed_generation_never_sends_fixed_fallback(self):
        bot=self.bot()
        async def fail(messages):
            raise LLMError('synthetic failure')
            yield ''
        bot.llm.stream_chat=fail
        with self.assertLogs('taoyao.app',level='ERROR'):
            self.assertFalse(await bot._send_segments('private',1,'已保存。',polish=False))
        bot.server.send_text.assert_not_awaited()

    async def test_generated_internal_text_never_leaks_through_function_result(self):
        bot=self.bot('DeepSeek 返回 500：private transport detail')
        with self.assertLogs('taoyao.app',level='WARNING'):
            self.assertFalse(await bot._send_segments('private',1,'普通功能已完成。'))
        bot.server.send_text.assert_not_awaited()

    async def test_public_space_intro_is_only_intent_and_draft_uses_same_pipeline(self):
        bot=self.bot('我是桃夭，喜欢看花。')
        intent=await bot._generate_self_intro()
        self.assertIsInstance(intent,ReplyIntent)
        self.assertEqual(bot.inputs,[])
        capture=[]
        token=_draft_output.set(capture)
        try:self.assertTrue(await bot._reply_from_intent('group',0,intent))
        finally:_draft_output.reset(token)
        self.assertIn('【步骤二：确定关系边界】',bot.inputs[0][0]['content'])
        self.assertTrue(capture)
        bot.server.send_text.assert_not_awaited()

    async def test_failed_onebot_result_is_not_success(self):
        bot=self.bot()
        bot.server.send_text.return_value={'status':'failed','retcode':100}
        with self.assertLogs('taoyao.app',level='WARNING'):
            self.assertFalse(await bot._send_segments('private',1,'已保存。'))

    async def test_group_function_inherits_current_persons_axes_without_guessing_at(self):
        bot=self.bot('好，记下了。')
        bot.profiles_enabled=True
        profile={'user_id':'2','role':'user','scores':{'HUM':80}}
        bot.profiles=SimpleNamespace(get=AsyncMock(return_value=profile))
        bot._tone_hint=lambda p,**kw:'ACTOR_HUM_'+str(p['scores']['HUM']) if p else ''
        @reply_window_context
        async def function(target,target_id,session,user_id,role):
            return await bot._send_segments(target,target_id,'设置已完成。')
        self.assertTrue(await function('group',10,'group:10',2,'user'))
        bot.profiles.get.assert_awaited_once_with(2)
        self.assertIn('ACTOR_HUM_80',bot.inputs[0][0]['content'])
        self.assertIsNone(bot.server.send_text.await_args.kwargs['at_user_id'])

    def test_single_physical_text_send_and_business_cannot_bypass_pipeline(self):
        tree=ast.parse(Path('taoyao/app.py').read_text(encoding='utf-8-sig'))
        sends=[]
        for fn in ast.walk(tree):
            if not isinstance(fn,(ast.AsyncFunctionDef,ast.FunctionDef)):continue
            for node in ast.walk(fn):
                if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):continue
                if ast.unparse(node.func) in ('self.server.send_text','self.server.send_private','self.server.send_group'):
                    sends.append(fn.name)
                if ast.unparse(node.func)=='self._send_segments' and any(k.arg=='prepared' and isinstance(k.value,ast.Constant) and k.value.value is True for k in node.keywords):
                    self.assertIn(fn.name,('_stream_reply','_two_part_reply','deliver'))
        self.assertEqual(sends,['_send_segments'])
        methods={node.name for node in ast.walk(tree) if isinstance(node,ast.AsyncFunctionDef)}
        self.assertNotIn('_persona_reply',methods)
        self.assertNotIn('_polish_text',methods)
