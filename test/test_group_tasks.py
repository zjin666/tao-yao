import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.group_tasks import send_request_candidate


class GroupTaskTests(unittest.IsolatedAsyncioTestCase):
    def bot(self,decision=None):
        bot=TaoYaoBot.__new__(TaoYaoBot)
        bot.config={'owner_qq':1,'group_tasks':{'enabled':True}}
        bot._pending_group_tasks={}
        bot._group_list=AsyncMock(return_value=[{'group_id':900001,'group_name':'测试群'}])
        bot.llm=SimpleNamespace(chat=AsyncMock(return_value=json.dumps(decision or {'action':'none'},ensure_ascii=False)))
        bot.utility_llm=bot.llm
        bot._group_task_text=AsyncMock(return_value='大家晚安。')
        bot._send_segments=AsyncMock(return_value=True)
        bot.profiles=SimpleNamespace(add_group_task=AsyncMock(return_value=1),delete_group_task=AsyncMock())
        return bot

    async def call(self,bot,text):
        return await bot._handle_owner_group_task('private',1,1,text,None)

    async def test_retrospective_regression_never_becomes_send_or_pending(self):
        for text in ('对啊，之前你在群里，不是不小心说了玫瑰嘛',
                     '你在群里说过这句话，我是在提醒你。',
                     '之前在群里发的那句话是什么意思？',
                     '去群里看看最近的消息',
                     '不要去群里说这件事',
                     '如果有人问，去群里说晚安',
                     '他说“去群里发晚安”，我没有让你发'):
            with self.subTest(text=text):
                bot=self.bot()
                self.assertFalse(await self.call(bot,text))
                bot.llm.chat.assert_not_awaited()
                bot._group_task_text.assert_not_awaited()
                bot._send_segments.assert_not_awaited()
                self.assertEqual(bot._pending_group_tasks,{})

    async def test_keyword_hit_still_requires_verified_model_intent(self):
        text='帮我去测试群发一句晚安'
        self.assertTrue(send_request_candidate(text))
        for decision in ({'action':'none'},{'task':True},
                {'action':'send','request_evidence':'捏造的要求','destination_evidence':'测试群'},
                {'action':'send','request_evidence':text,'destination_evidence':'另一个群'}):
            bot=self.bot(decision)
            self.assertFalse(await self.call(bot,text))
            bot._send_segments.assert_not_awaited()

    async def test_explicit_request_sends_only_to_stated_destination(self):
        text='帮我去测试群发一句晚安'
        bot=self.bot({'action':'send','request_evidence':text,'destination_evidence':'测试群'})
        self.assertTrue(await self.call(bot,text))
        self.assertEqual(bot._send_segments.await_args_list[0].args[:2],('group',900001))
        bot.profiles.add_group_task.assert_awaited_once()

    async def test_unclear_target_is_not_guessed_from_topic(self):
        text='帮我去某个群发一句晚安'
        bot=self.bot({'action':'send','request_evidence':text,'destination_evidence':'某个群'})
        self.assertTrue(await self.call(bot,text))
        self.assertIn(1,bot._pending_group_tasks)
        self.assertTrue(all(c.args[0]=='private' for c in bot._send_segments.await_args_list))

    async def test_pending_needs_target_only_reply_and_unrelated_turn_cancels_it(self):
        for text in ('对啊，之前你在测试群说错了','测试群先别发了','好的','不要发了'):
            bot=self.bot()
            bot._pending_group_tasks[1]=('帮我去群里说晚安',time.monotonic())
            self.assertFalse(await self.call(bot,text))
            bot._send_segments.assert_not_awaited()
            self.assertEqual(bot._pending_group_tasks,{})
        bot=self.bot()
        bot._pending_group_tasks[1]=('帮我去群里说晚安',time.monotonic())
        self.assertTrue(await self.call(bot,'测试群'))
        self.assertEqual(bot._send_segments.await_args_list[0].args[0],'group')
        bot.llm.chat.assert_not_awaited()

    async def test_expired_pending_and_plain_group_mention_do_not_send(self):
        bot=self.bot()
        bot._pending_group_tasks[1]=('帮我去群里说晚安',time.monotonic()-121)
        self.assertFalse(await self.call(bot,'测试群'))
        bot._send_segments.assert_not_awaited()
        self.assertEqual(await bot._select_group_targets('刚才那个话题'),[])

    async def test_unsent_bubble_is_not_reported_as_completed(self):
        text='帮我去测试群发一句晚安'
        bot=self.bot({'action':'send','request_evidence':text,'destination_evidence':'测试群'})
        bot._send_segments.side_effect=[False,True]
        await self.call(bot,text)
        self.assertIn('发送失败',bot._send_segments.await_args.args[2].situation)

    async def test_real_private_handler_keeps_retrospective_in_private_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config();cfg['owner_qq']=1
            cfg['memory']['db']=str(Path(tmp)/'memory.db')
            cfg['knowledge']['db']=str(Path(tmp)/'knowledge.db')
            cfg['profile'].update(enabled=False,db=str(Path(tmp)/'profiles.db'))
            cfg['intro']['enabled']=cfg['memes']['enabled']=cfg['user_memes']['enabled']=False
            cfg['response_speed'].update(speculative_reply=False,two_part_reply=False,defer_six_axis=False)
            cfg['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
            bot=TaoYaoBot(cfg,mock=True)
            inputs=[]
            async def stream(messages):
                inputs.append(messages)
                yield '嗯，那次是我说错了。'
            bot.llm=bot.utility_llm=SimpleNamespace(stream_chat=stream,
                chat=AsyncMock(return_value='{"allowed":true}'))
            bot._search_context_for=AsyncMock(return_value='')
            bot._schedule_reply_extras=lambda *a,**kw:None
            bot._schedule_owner_summary=AsyncMock()
            bot._group_task_text=AsyncMock()
            bot.server.send_text=AsyncMock(return_value={'status':'ok'})
            statement='对啊，之前你在群里，不是不小心说了玫瑰嘛'
            try:
                await bot._handle_text('private',1,'owner_private:1',1,'测试owner',statement,
                    'owner',False,0,'private',from_batch=True)
                self.assertTrue(inputs)
                self.assertIn(statement,inputs[0][-1]['content'])
                self.assertTrue(all(c.args[0]=='private' for c in bot.server.send_text.await_args_list))
                bot._group_task_text.assert_not_awaited()
                self.assertIn(statement,str(await bot.memory.recent('owner_private:1')))
                bot._pending_group_tasks[1]=('去群里说晚安',time.monotonic())
                bot._group_list=AsyncMock(return_value=[{'group_id':900001,'group_name':'测试群'}])
                bot._knowledge_request=AsyncMock(return_value='知识已实际保存，本轮自然确认。')
                await bot._handle_text('private',1,'owner_private:1',1,'测试owner','学习一下这个知识。',
                    'owner',False,0,'private',from_batch=True)
                self.assertEqual(bot._pending_group_tasks,{})
                self.assertTrue(all(c.args[0]=='private' for c in bot.server.send_text.await_args_list))
            finally:bot.memory.close();bot.profiles.close()
