import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.knowledge import KnowledgeStore


class KnowledgeSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=KnowledgeStore(Path(self.tmp.name)/'knowledge.db',{'summary_batch_size':2})
        self.source={'owner_id':1,'scene':'private','text':'教学原话'}

    async def asyncTearDown(self):self.tmp.cleanup()

    async def add(self,topic,content,scope='public'):
        return await self.store.save({'topic':topic,'content':content,'keywords':['鲸鱼',topic]},scope,self.source)

    def judge(self,ids=(1,2),summary='鲸鱼属于哺乳动物，用肺呼吸。'):
        return SimpleNamespace(chat=AsyncMock(return_value=json.dumps({'groups':[
            {'source_ids':list(ids),'topic':'鲸鱼知识','summary':summary}]},ensure_ascii=False)))

    async def seed(self,scope='public'):
        await self.add('鲸鱼分类','鲸鱼是哺乳动物，不是鱼类。',scope)
        await self.add('鲸鱼呼吸','鲸鱼通过肺呼吸，需要浮出水面换气。',scope)

    async def test_default_threshold_twenty_and_no_repeated_summarizing(self):
        self.store.config={}
        for i in range(19):await self.add(f'知识{i}',f'鲸鱼的测试知识{i}。')
        llm=self.judge(range(1,21))
        self.assertEqual(await self.store.compact_due(llm),0)
        llm.chat.assert_not_awaited()
        await self.add('知识19','鲸鱼的测试知识19。')
        self.assertEqual(await self.store.compact_due(llm),1)
        self.assertEqual(await self.store.compact_due(llm),0)
        llm.chat.assert_awaited_once()
        self.assertEqual(len(await self.store.rows(['public'])),20)

    async def test_summary_used_for_replies_but_originals_retained_for_learning(self):
        await self.seed()
        self.assertEqual(await self.store.compact_due(self.judge()),1)
        reopened=KnowledgeStore(self.store.path,self.store.config)
        refs=await reopened.references('鲸鱼',['public'])
        self.assertEqual(len(refs),1)
        self.assertEqual(refs[0]['source_ids'],[1,2])
        self.assertTrue(refs[0]['summary_created_at'].endswith('+08:00'))
        self.assertEqual(len(refs[0]['source_dates']),2)
        raw=await reopened.search('鲸鱼',['public'])
        self.assertEqual(len(raw),2)
        self.assertTrue(any('浮出水面' in row['content'] for row in raw))

    async def test_public_and_private_batches_are_never_mixed(self):
        await self.add('公开一','PUBLIC 鲸鱼知识。')
        await self.add('私有一','PRIVATE 鲸鱼计划。','owner:1')
        llm=self.judge()
        self.assertEqual(await self.store.compact_due(llm),0)
        await self.add('公开二','PUBLIC 鲸鱼呼吸。')
        llm=self.judge((1,3),'PUBLIC 鲸鱼知识与呼吸。')
        self.assertEqual(await self.store.compact_due(llm),1)
        self.assertNotIn('PRIVATE',llm.chat.await_args.args[0][-1]['content'])
        self.assertNotIn('PRIVATE',str(await self.store.references('鲸鱼',['public'])))

    async def test_correction_invalidates_summary_including_unchanged_peer(self):
        await self.seed()
        await self.store.compact_due(self.judge())
        await self.store.save({'topic':'鲸鱼分类','content':'修正后的鲸鱼分类知识。','keywords':['鲸鱼','分类']},
            'public',self.source,1)
        refs=await self.store.references('鲸鱼呼吸',['public'])
        self.assertTrue(refs)
        self.assertTrue(all('summary_created_at' not in r for r in refs))
        await self.store.compact_due(self.judge(summary='修订后的鲸鱼要点。'))
        self.assertTrue(any('summary_created_at' in r for r in await self.store.references('鲸鱼',['public'])))

    async def test_deletion_and_visibility_change_invalidate_old_summary(self):
        await self.seed()
        await self.store.compact_due(self.judge())
        await self.store.confirm(1,1,False)
        refs=await self.store.references('鲸鱼',['public'])
        self.assertEqual([r['id'] for r in refs],[2])
        self.assertNotIn('summary_created_at',refs[0])
        self.assertNotIn('不是鱼类',str(refs))
        await self.store.forget(2)
        self.assertEqual(await self.store.references('鲸鱼',['public']),[])

    async def test_invalid_summary_never_replaces_original_knowledge(self):
        await self.seed()
        for ids in ((1,),(1,1,2),(1,99)):
            with self.assertRaises(ValueError):await self.store.compact_due(self.judge(ids))
        self.assertEqual(len(await self.store.rows(['public'])),2)
        refs=await self.store.references('鲸鱼',['public'])
        self.assertTrue(all('summary_created_at' not in r for r in refs))

    async def test_slow_summary_does_not_block_read_or_commit_stale_snapshot(self):
        await self.seed()
        started=asyncio.Event();release=asyncio.Event()
        async def judge(messages):
            started.set();await release.wait()
            return self.judge().chat.return_value
        task=asyncio.create_task(self.store.compact_due(SimpleNamespace(chat=judge)))
        await asyncio.wait_for(started.wait(),2)
        try:
            self.assertEqual(len(await asyncio.wait_for(self.store.references('鲸鱼',['public']),2)),2)
            await self.store.forget(1)
        finally:release.set()
        self.assertEqual(await asyncio.wait_for(task,2),0)
        self.assertEqual([r['id'] for r in await self.store.references('鲸鱼',['public'])],[2])
