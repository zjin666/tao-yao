import tempfile
from pathlib import Path
import unittest
from unittest.mock import AsyncMock
from taoyao.memory import MemoryStore
from taoyao.profiles import ProfileStore
from taoyao.privacy_index import PrivacyIndex


class PrivacyIndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_deletion_removes_all_private_memory_and_index_copies(self):
        await self.memory.add_message('user_private:1','user','私聊内容。')
        await self.memory.add_fact('user_private:1','私聊事实。')
        await self.memory.set_summary('user_private:1','私聊摘要。')
        await self.index.sync(self.extract)
        self.assertTrue(self.index.snapshot('2').facts)
        await self.memory.forget_session('user_private:1')
        await self.index.forget_user('1')
        self.assertEqual(self.index.stats()['facts'],0)
        self.assertFalse(self.index.snapshot('2').ready)
        await self.index.sync(self.extract)
        self.assertEqual(self.index.snapshot('2').facts,())
        self.assertEqual(await self.memory.get_summary('user_private:1'),'')
        self.assertEqual(await self.memory.facts('user_private:1'),[])
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        root=Path(self.temp.name)
        self.profiles=ProfileStore(root/'profiles.db')
        self.memory=MemoryStore(root/'memory.db')
        self.index=PrivacyIndex(root/'private.db',self.profiles.path,self.memory.path)

    async def asyncTearDown(self):
        self.index.close();self.profiles.close();self.memory.close();self.temp.cleanup()

    async def extract(self,docs):
        return {d['id']:{'no_private_fact':False,'facts':[{'kind':'other_private',
            'text':d['text'],'values':[],'about':'self'}]} for d in docs}

    async def test_sources_coverage_deduplication_and_own_private_permissions(self):
        await self.profiles.ensure(1,'一号','owner')
        await self.profiles.add_message(1,'private',0,'user','我的测试口令是桃花五月。')
        await self.memory.add_message('owner_private:1','user','我的测试口令是桃花五月。')
        await self.profiles.add_message(1,'group',10,'user','这是公开消息，不应变成私聊来源。')
        await self.memory.add_message('group:10','user','群聊不索引。')
        await self.memory.add_fact('owner_private:1','这是私聊长期事实。')
        await self.memory.set_summary('owner_private:1','这是私聊摘要。')
        await self.profiles.set_user_memory(1,'用户要求记住的私聊信息。')
        # Archive-only sources must be covered even if no longer in current table.
        with self.profiles._conn:
            self.profiles._conn.execute("INSERT INTO user_messages_archive(id,user_id,scene,group_id,role,content,created_at,archived_at) VALUES(999,'1','private',0,'user','旧归档私聊内容。','old','now')")
        result=await self.index.sync(self.extract)
        self.assertEqual(result['pending'],0)
        snapshot=self.index.snapshot('2','10')
        self.assertTrue(snapshot.ready)
        payload=snapshot.checker_context()
        self.assertNotIn('source_refs',payload)
        self.assertNotIn('values_json',payload)
        self.assertNotIn(snapshot.facts[0]['id'],payload)
        texts=[x['text'] for x in snapshot.facts]
        self.assertEqual(texts.count('我的测试口令是桃花五月。'),1)
        self.assertIn('旧归档私聊内容。',texts)
        self.assertFalse(any('公开消息' in s or '群聊不索引' in s for s in texts))
        self.assertEqual(self.index.snapshot('1').facts,())
        self.assertTrue(self.index.snapshot('1','10').facts)

    async def test_incremental_updates_edits_deletes_and_reopen_require_resync(self):
        await self.memory.add_fact('user_private:1','第一条私聊信息。')
        extractor=AsyncMock(side_effect=self.extract)
        await self.index.sync(extractor)
        extractor.reset_mock()
        self.index._collect=unittest.mock.Mock(wraps=self.index._collect)
        await self.index.sync(extractor)
        extractor.assert_not_awaited()
        self.index._collect.assert_not_called()
        await self.memory.add_message('group:10','user','公开群聊新增，不应使隐私索引失效。')
        self.assertTrue(self.index.snapshot('2').ready)
        await self.memory.add_fact('user_private:1','第二条私聊信息。')
        self.assertFalse(self.index.snapshot('2').ready)
        await self.index.sync(extractor)
        self.assertEqual(len(extractor.await_args.args[0]),1)
        with self.memory._conn:
            self.memory._conn.execute("DELETE FROM facts WHERE fact='第一条私聊信息。'")
        self.assertFalse(self.index.snapshot('2').ready)
        await self.index.sync(extractor)
        self.assertEqual([f['text'] for f in self.index.snapshot('2').facts],['第二条私聊信息。'])
        path=self.index.path;self.index.close()
        self.index=PrivacyIndex(path,self.profiles.path,self.memory.path)
        self.assertFalse(self.index.snapshot('2').ready)
        await self.index.sync(extractor)
        self.assertTrue(self.index.snapshot('2').ready)

    async def test_invalid_evidence_and_failed_batches_stay_pending(self):
        await self.memory.add_fact('user_private:1','有证据的私聊事实。')
        await self.index.sync(AsyncMock(side_effect=RuntimeError('test')))
        self.assertEqual(self.index.snapshot('2').reason,'pending_extraction')
        async def invalid(docs):
            return {d['id']:{'no_private_fact':False,'facts':[{'kind':'health','text':'编造的病情',
                'values':[],'about':'self'}]} for d in docs}
        await self.index.sync(invalid)
        self.assertEqual(self.index.stats()['facts'],0)
        self.assertEqual(self.index.stats()['pending'],1)
        await self.index.sync(self.extract)
        self.assertTrue(self.index.snapshot('2').ready)

    async def test_third_party_is_protected_even_in_source_owners_private_chat(self):
        await self.memory.add_fact('user_private:1','朋友正在接受治疗。')
        async def third_party(docs):
            return {d['id']:{'no_private_fact':False,'facts':[{'kind':'health','text':d['text'],
                'values':[],'about':'third_party'}]} for d in docs}
        await self.index.sync(third_party)
        self.assertEqual(len(self.index.snapshot('1').facts),1)
        self.assertEqual(self.index.snapshot('1',max_chars=1).reason,'budget_exceeded')
        fact=self.index.snapshot('1','10').facts[0]
        self.index.mark_public(fact['id'],'10')
        self.assertEqual(self.index.snapshot('1','10').facts,())
        self.assertEqual(len(self.index.snapshot('1','11').facts),1)
