import json
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.knowledge import KnowledgeStore, extract_lesson
from taoyao.memory import MemoryStore


def lesson(privacy='public',content='鲸鱼是哺乳动物，不是鱼类。',replace_id=None):
    return {'accept':True,'kind':'reusable_knowledge','evidence':'鲸鱼','topic':'鲸鱼分类','content':content,
        'keywords':['鲸鱼','哺乳动物'],'replace_id':replace_id,'privacy':privacy}


class KnowledgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=KnowledgeStore(Path(self.tmp.name)/'knowledge.db')
        self.source={'owner_id':1,'scene':'private','message_id':'teaching','text':'owner纠正原话'}

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_persistence_scope_relevance_and_timestamp(self):
        await self.store.save(lesson(),'public',self.source)
        private=lesson(content='私人鲸鱼计划，周六讨论。')
        private['topic']='鲸鱼计划'
        await self.store.save(private,'owner:1',self.source)
        reopened=KnowledgeStore(self.store.path)
        public=await reopened.search('鲸鱼是什么动物',['public'])
        self.assertEqual(len(public),1)
        self.assertEqual(public[0]['content'],lesson()['content'])
        self.assertTrue(public[0]['updated_at'].endswith('+08:00'))
        self.assertEqual(len(await reopened.search('鲸鱼',['public','owner:1'])),2)
        self.assertEqual(await reopened.search('今天晚饭吃什么',['public']),[])

    async def test_correction_replaces_visible_old_version_and_keeps_revision(self):
        old=await self.store.save(lesson(content='旧的错误知识'),'public',self.source)
        await self.store.save(lesson(),'public',self.source,old)
        result=await self.store.search('鲸鱼',['public'])
        self.assertEqual([r['content'] for r in result],[lesson()['content']])
        with closing(sqlite3.connect(self.store.path)) as conn:
            self.assertIn('旧的错误知识',conn.execute('SELECT previous FROM revisions').fetchone()[0])
        await self.store.forget(old)
        self.assertEqual(await self.store.search('鲸鱼',['public']),[])

    async def test_private_correction_removes_previous_public_version(self):
        old=await self.store.save(lesson(),'public',self.source)
        await self.store.save(lesson('possible_private'),'owner:1',self.source,old)
        self.assertEqual(await self.store.rows(['public']),[])
        self.assertEqual(len(await self.store.rows(['owner:1'])),1)

    async def test_publication_needs_correct_owner_and_survives_reopen(self):
        item=lesson('possible_private');item['awaiting_public']=True
        item_id=await self.store.save(item,'owner:1',self.source)
        self.assertFalse(await self.store.confirm(item_id,2,True))
        self.assertEqual(await self.store.rows(['public']),[])
        self.assertTrue(await self.store.confirm(item_id,1,True))
        reopened=KnowledgeStore(self.store.path)
        rows=await reopened.rows(['public'])
        self.assertEqual(rows[0]['awaiting_public'],0)

    async def test_missing_privacy_is_never_assumed_public(self):
        item=lesson();item.pop('privacy')
        llm=SimpleNamespace(chat=AsyncMock(return_value=json.dumps(item)))
        result=await extract_lesson(llm,'纠正鲸鱼分类',[],[])
        self.assertEqual(result['privacy'],'possible_private')

    async def test_semantic_retrieval_matches_paraphrase_without_shared_terms(self):
        await self.store.save(lesson(),'public',self.source)
        self.store.embedder=SimpleNamespace(embed_texts=lambda texts:[[1.0,0.0] for _ in texts])
        self.assertEqual(len(await self.store.search('海中巨兽靠肺呼吸吗',['public'])),1)

    async def test_invalid_or_truncated_judgment_cannot_save(self):
        for result in ('{"accept":true',json.dumps(lesson(replace_id=999))):
            llm=SimpleNamespace(chat=AsyncMock(return_value=result))
            if result.startswith('{"accept":true') and not result.endswith('}'):
                self.assertIsNone(await extract_lesson(llm,'纠正',[],[]))
            else:
                with self.assertRaises(ValueError):await extract_lesson(llm,'纠正',[],[])
        self.assertEqual(await self.store.rows(['public']),[])

    def bot(self,result):
        bot=TaoYaoBot.__new__(TaoYaoBot)
        bot.config={'owner_qq':1}
        bot.knowledge=self.store
        bot.memory=MemoryStore(Path(self.tmp.name)/'memory.db')
        bot.llm=SimpleNamespace(chat=AsyncMock(return_value=json.dumps(result)))
        bot.utility_llm=bot.llm
        return bot

    async def test_only_owner_direct_teaching_is_accepted(self):
        bot=self.bot(lesson())
        try:
            self.assertEqual(await bot._knowledge_request('纠正：鲸鱼是哺乳动物','user',2,'s',False,0,'m'),'')
            self.assertEqual(await bot._knowledge_request('纠正：鲸鱼是哺乳动物','owner',1,'s',True,10,'m'),'')
            self.assertEqual(await bot._knowledge_request('纠正：鲸鱼是哺乳动物','owner',1,'s',False,0,'m',ambient=True),'')
            bot.llm.chat.assert_not_awaited()
            intent=await bot._knowledge_request('/学习 鲸鱼是哺乳动物','owner',1,'s',False,0,'m')
            self.assertIn('已实际保存',intent)
            self.assertEqual(len(await self.store.rows(['public'])),1)
        finally:bot.memory.close()

    async def test_private_confirmation_is_intent_not_direct_message(self):
        bot=self.bot(lesson('possible_private'))
        bot._send_segments=AsyncMock()
        try:
            intent=await bot._knowledge_request('/学习 鲸鱼计划','owner',1,'s',False,0,'m')
            self.assertIn('询问',intent)
            self.assertIn('/知识公开',intent)
            bot._send_segments.assert_not_awaited()
            self.assertEqual(await self.store.rows(['public']),[])
            self.assertIn('鲸鱼',await bot._knowledge_context('鲸鱼','owner',1,False))
            self.assertEqual(await bot._knowledge_context('鲸鱼','owner',1,True),'')
            self.assertEqual(await bot._knowledge_context('鲸鱼','user',2,False),'')
            bot.llm.chat.return_value=json.dumps({'confirmed':True,'id':1,'public':True})
            intent=await bot._knowledge_request('不是私人信息，可以公开给其他人参考','owner',1,'s',False,0,'m')
            self.assertIn('现已实际设为公开',intent)
            self.assertIn('鲸鱼',await bot._knowledge_context('鲸鱼','user',2,False))
        finally:bot.memory.close()

    async def test_failed_save_never_claims_success(self):
        bot=self.bot(lesson())
        bot.knowledge.save=AsyncMock(side_effect=OSError('synthetic disk failure'))
        try:
            with self.assertLogs('taoyao.app',level='ERROR'):
                intent=await bot._knowledge_request('/学习 鲸鱼分类','owner',1,'s',False,0,'m')
            self.assertIn('未完成',intent)
            self.assertNotIn('已实际保存',intent)
            self.assertEqual(await self.store.rows(['public']),[])
        finally:bot.memory.close()


class KnowledgeReplyIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_and_privacy_question_use_normal_prompt_and_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config();cfg['owner_qq']=1
            cfg['memory']['db']=str(Path(tmp)/'memory.db')
            cfg['knowledge']['db']=str(Path(tmp)/'knowledge.db')
            cfg['profile'].update(enabled=False,db=str(Path(tmp)/'profiles.db'))
            cfg['intro']['enabled']=cfg['memes']['enabled']=cfg['user_memes']['enabled']=False
            cfg['response_speed'].update(speculative_reply=False,two_part_reply=False,defer_six_axis=False)
            cfg['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
            bot=TaoYaoBot(cfg,mock=True)
            bot.knowledge.embedder=None
            generated=[]
            async def stream(messages):
                generated.append(messages)
                yield '我记下了，这个能让别人也知道吗？'
            bot.llm=SimpleNamespace(stream_chat=stream,chat=AsyncMock())
            bot.utility_llm=SimpleNamespace(chat=AsyncMock(return_value='{"allowed":true}'))
            bot.knowledge_llm=SimpleNamespace(chat=AsyncMock(return_value=json.dumps(lesson('possible_private'))))
            bot._tone_hint=lambda *args,**kwargs:'SIX_AXIS_TEST_REFERENCE'
            bot._search_context_for=AsyncMock(return_value='')
            bot._schedule_owner_summary=AsyncMock()
            bot._schedule_reply_extras=lambda *args,**kwargs:None
            bot.server.send_text=AsyncMock(return_value={'status':'ok'})
            try:
                await bot._handle_text('private',1,'owner_private:1',1,'测试owner',
                    '/学习 鲸鱼知识','owner',False,0,'private',from_batch=True)
                self.assertTrue(generated)
                system=generated[0][0]['content']
                self.assertIn(bot.persona,system)
                self.assertIn('SIX_AXIS_TEST_REFERENCE',system)
                self.assertIn('询问',system)
                self.assertIn('知识已实际保存',system)
                self.assertIn('【步骤二：确定关系边界】',system)
                bot.server.send_text.assert_awaited()
                self.assertIn('能让别人也知道吗',bot.server.send_text.await_args.args[2])
                bot.utility_llm.chat.assert_awaited()  # Normal pre-send review.
                self.assertEqual(await bot.knowledge.rows(['public']),[])
                await bot._send_knowledge_confirmation(1,1,'鲸鱼分类')
                self.assertIn('SIX_AXIS_TEST_REFERENCE',generated[-1][0]['content'])
                self.assertIn('【步骤二：确定关系边界】',generated[-1][0]['content'])
                self.assertIn('现在在私聊自然询问',generated[-1][0]['content'])
                await bot._handle_text('private',1,'owner_private:1',1,'测试owner',
                    '/知识公开 1','owner',False,0,'private',from_batch=True)
                self.assertIn('已完成知识库操作',generated[-1][0]['content'])
                await bot._handle_text('private',2,'user_private:2',2,'测试用户',
                    '鲸鱼属于什么动物？','user',False,0,'private',from_batch=True)
                self.assertIn('【桃夭已学知识参考】',generated[-1][0]['content'])
                self.assertIn('鲸鱼是哺乳动物',generated[-1][0]['content'])
            finally:bot.memory.close();bot.profiles.close()
