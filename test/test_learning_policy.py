import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.app import TaoYaoBot
from taoyao.knowledge import KnowledgeStore, extract_lesson, temporary_learning_instruction
from taoyao.memory import MemoryStore


class LearningPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=KnowledgeStore(Path(self.tmp.name)/'knowledge.db')
        self.memory=MemoryStore(Path(self.tmp.name)/'memory.db')
        self.bot=TaoYaoBot.__new__(TaoYaoBot)
        self.bot.config={'owner_qq':1};self.bot.knowledge=self.store;self.bot.memory=self.memory
        self.bot.llm=self.bot.utility_llm=SimpleNamespace(chat=AsyncMock())
        self.source={'owner_id':1,'scene':'private','text':'教学原文'}

    async def asyncTearDown(self):
        self.memory.close();self.tmp.cleanup()

    async def teach(self,text,**updates):
        item={'accept':True,'kind':'reusable_knowledge','evidence':text,'topic':'花的区分',
            'content':'樱花和桃花不同。','keywords':['樱花','桃花'],'privacy':'public','replace_id':None}
        item.update(updates)
        llm=SimpleNamespace(chat=AsyncMock(return_value=json.dumps(item,ensure_ascii=False)))
        return await extract_lesson(llm,text,[],[])

    async def test_temporary_instructions_never_enter_knowledge_even_when_marked_learning(self):
        for text in ('值班同学不用接着提了','记住，今天不聊这个。',
                     '/学习 这件事不用再提了','这个可以不学','这条不用记了'):
            with self.subTest(text=text):
                self.assertTrue(temporary_learning_instruction(text))
                self.assertIsNone(await self.teach(text))
                intent=await self.bot._knowledge_request(text,'owner',1,'s',False,0,'m')
                self.assertIn('没有写入',intent)
        self.assertEqual(await self.store.rows(['public','owner:1']),[])
        self.bot.llm.chat.assert_not_awaited()

    async def test_model_must_supply_eligible_kind_and_current_message_evidence(self):
        text='记住，樱花和桃花不是同一种花。'
        for patch in ({'kind':'temporary_instruction'},{'kind':'personal_memory'},{'kind':None},
                      {'evidence':'从旧聊天推断出的结论'},{'evidence':None}):
            self.assertIsNone(await self.teach(text,**patch))
        self.assertIsNotNone(await self.teach(text))
        rule='以后用户讲完一件事，要接着互动，不要再把这件事改写成旁白。'
        self.assertFalse(temporary_learning_instruction(rule))
        self.assertIsNotNone(await self.teach(rule,kind='expression_rule'))

    async def test_inventory_uses_database_not_chat_summary_or_assistant_claims(self):
        await self.store.save({'topic':'樱花与桃花','content':'樱花和桃花不是同一种花。',
            'keywords':['樱花','桃花']},'public',self.source)
        await self.memory.set_summary('owner_private:1','桃夭曾自称学会了不用接着提某位群友。')
        intent=await self.bot._knowledge_request('桃夭，你目前有学到哪些新内容？',
            'owner',1,'owner_private:1',False,0,'m')
        self.assertIn('实际知识总数为1',intent)
        self.assertIn('樱花和桃花不是同一种花',intent)
        self.assertNotIn('某位群友',intent)
        self.assertIn('以前自己声称学会',intent)
        self.bot.llm.chat.assert_not_awaited()
        stop=await self.bot._knowledge_request('这件事不用再提了，这个可以不学。',
            'owner',1,'owner_private:1',False,0,'m2')
        self.assertIn('没有删除知识操作',stop)
        self.assertEqual(len(await self.store.rows(['public'])),1)

    async def test_public_inventory_cannot_list_owner_private_items(self):
        await self.store.save({'topic':'私有测试项目','content':'PRIVATE TEST',
            'keywords':['私有','测试']},'owner:1',self.source)
        for role,uid,group in [('user',2,False),('owner',1,True)]:
            intent=await self.bot._knowledge_request('你学到了哪些知识？',role,uid,'s',group,10,'m')
            self.assertIn('实际知识总数为0',intent)
            self.assertNotIn('PRIVATE TEST',intent)

    async def test_evidence_and_category_are_persisted(self):
        text='学习：樱花和桃花不是同一种花。'
        item=await self.teach(text)
        await self.store.save(item,'public',self.source)
        row=(await self.store.rows(['public']))[0]
        self.assertEqual(row['lesson_kind'],'reusable_knowledge')
        self.assertEqual(row['teaching_evidence'],text)

    async def test_legacy_database_migration_preserves_original_knowledge(self):
        path=Path(self.tmp.name)/'legacy.db'
        conn=sqlite3.connect(path)
        try:
            conn.execute('CREATE TABLE knowledge(id INTEGER PRIMARY KEY,topic TEXT,content TEXT,keywords TEXT,scope TEXT,owner_id TEXT,source_scene TEXT,source_group TEXT,source_message TEXT,source_text TEXT,created_at TEXT,updated_at TEXT,active INTEGER,awaiting_public INTEGER)')
            conn.execute('INSERT INTO knowledge VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (1,'原知识','原结论','[]','public','1','private','','','原话','2026-11-06T21:32:52+08:00','2026-11-06T21:32:52+08:00',1,0))
            conn.commit()
        finally:conn.close()
        rows=await KnowledgeStore(path).rows(['public'])
        self.assertEqual(rows[0]['content'],'原结论')
        self.assertEqual(rows[0]['lesson_kind'],'')
