import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from taoyao.impressions import ImpressionStore, POLICY_VERSION, describe_public_impression


class ImpressionTraitTests(unittest.IsolatedAsyncioTestCase):
    def rows(self,*texts):
        return [{'id':i+1,'user_id':'2','content':text,'sent_at':'2026-11-07T02:00:00+08:00'} for i,text in enumerate(texts)]

    def trait(self,rows,word='幽默',basis='speaker_tone',**changes):
        return {'trait':word,'subject_user_id':'2','basis':basis,'confidence':.9,
            'evidence':[{'message_id':r['id'],'quote':r['content']} for r in rows[:2]],**changes}

    async def analyze(self,rows,*items):
        llm=SimpleNamespace(chat=AsyncMock(return_value=json.dumps({'traits':list(items)},ensure_ascii=False)))
        return await describe_public_impression(llm,rows)

    async def test_tone_can_support_adjective_without_literal_self_label(self):
        rows=self.rows('哈哈，这个梗我接住了，我再编个反转。','那我也开个玩笑，想象一下鱼会跑步。')
        result=await self.analyze(rows,self.trait(rows))
        self.assertEqual([r['trait'] for r in result],['幽默'])
        self.assertEqual(result[0]['subject_user_id'],'2')
        self.assertEqual(result[0]['basis'],'speaker_tone')

    async def test_evaluating_someone_else_is_not_speakers_personality(self):
        for texts in (('你很温柔。','我觉得他也挺温柔。'),
                      ('小林这个人很温柔。','桃夭太温柔了。'),
                      ('我的朋友很温柔。','他说自己很温柔。')):
            rows=self.rows(*texts)
            self.assertEqual(await self.analyze(rows,self.trait(rows,'温柔')),[])

    async def test_cannot_hide_other_subject_or_negation_by_shortening_quote(self):
        rows=self.rows('你真幽默。','她确实很幽默。')
        item=self.trait(rows,evidence=[{'message_id':1,'quote':'真幽默'},{'message_id':2,'quote':'很幽默'}])
        self.assertEqual(await self.analyze(rows,item),[])
        rows=self.rows('我不是很内向。','我并不内向。')
        self.assertEqual(await self.analyze(rows,self.trait(rows,'内向','self_description')),[])

    async def test_speakers_own_conduct_toward_others_is_distinguished(self):
        rows=self.rows('我对你一直很温柔。','我对朋友平时也挺温柔的。')
        result=await self.analyze(rows,self.trait(rows,'温柔','self_description'))
        self.assertEqual([r['trait'] for r in result],['温柔'])

    async def test_rejects_behavior_sentences_wrong_subject_and_invented_quotes(self):
        rows=self.rows('哈哈，我来编一个反转。','那我也讲个冷笑话吧。')
        for changes in ({'trait':'经常主动提问'},{'trait':'回复比较简短'},
                        {'subject_user_id':'3'},{'basis':'other_evaluation'},
                        {'evidence':[{'message_id':1,'quote':'捏造的形容词'},{'message_id':2,'quote':rows[1]['content']}]}):
            self.assertEqual(await self.analyze(rows,self.trait(rows,**changes)),[])

    async def test_contradictory_traits_use_stronger_recent_evidence(self):
        rows=self.rows('哈哈，我来接这个梗。','我再开个小玩笑。','我讨论正事时很严肃。','我向来严肃认真。')
        humorous=self.trait(rows,confidence=.75)
        serious=self.trait(rows[2:],'严肃','self_description',confidence=.95)
        result=await self.analyze(rows,humorous,serious)
        self.assertEqual([r['trait'] for r in result],['严肃'])

    async def test_duplicate_messages_do_not_make_multiple_trait_evidence(self):
        rows=self.rows('哈哈，这句话1。','哈哈，这句话2。')
        self.assertEqual(await self.analyze(rows,self.trait(rows)),[])

    async def test_old_behavior_cache_is_hidden_and_rebuilt_without_new_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ImpressionStore(Path(tmp)/'impressions.db')
            texts=['哈哈，我来编个反转。','我再开个小玩笑。']
            for i in range(8):
                await store.observe_interaction(2,10,['公开昵称'],texts[i%2],str(i),replied=True)
            async def judge(messages):
                payload=json.loads(messages[-1]['content']);rows=payload['messages']
                item=self.trait(rows)
                return json.dumps({'traits':[item]},ensure_ascii=False)
            llm=SimpleNamespace(chat=AsyncMock(side_effect=judge))
            self.assertEqual(await store.update_due(llm),1)
            c=store.connect()
            try:
                c.execute("UPDATE impressions SET policy_version='',traits=?",(json.dumps([{'description':'经常提问'}]),));c.commit()
            finally:c.close()

            self.assertEqual(await store.read(2,[10]),[])
            self.assertEqual(await store.update_due(llm),1)
            self.assertEqual((await store.read(2,[10]))[0]['traits'][0]['description'],'幽默')
            c=store.connect()
            try:self.assertEqual(c.execute('SELECT policy_version FROM impressions').fetchone()[0],POLICY_VERSION)
            finally:c.close()

    async def test_new_conflicting_trait_deletes_old_but_keeps_compatible_traits(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ImpressionStore(Path(tmp)/'impressions.db')
            texts=['我在不熟悉的人面前常常比较内向。','我平时偏内向，喜欢先熟悉再开口。',
                '哈哈，我来编个反转。','那我再开个冷笑话。','我通常先听别人讲完再回应。',
                '我更喜欢熟人间聊天。','我会慢慢熟悉新朋友。','我也偶尔主动讲些好笑的。']
            for gid in (10,20):
                for i,text in enumerate(texts):
                    await store.observe_interaction(2,gid,['公开昵称'],text,f'{gid}:{i}',replied=True)
            async def old_judge(messages):
                rows=json.loads(messages[-1]['content'])['messages']
                return json.dumps({'traits':[self.trait(rows,'内向','self_description'),
                    self.trait(rows[2:],'幽默','speaker_tone')]},ensure_ascii=False)
            self.assertEqual(await store.update_due(SimpleNamespace(chat=old_judge)),2)
            newer=['我最近在新人面前也很外向。','我如今更外向，常常先主动打招呼。',
                '我最近认识了不少新朋友。','我现在会先开口聊天。','我刚又主动参加了一个交流。']
            for i,text in enumerate(newer):
                await store.observe_interaction(2,10,[],text,f'new:{i}',replied=True)
            async def new_judge(messages):
                rows=json.loads(messages[-1]['content'])['messages']
                rows=[r for r in rows if '外向' in r['content']]
                return json.dumps({'traits':[self.trait(rows,'外向','self_description')]},ensure_ascii=False)
            self.assertEqual(await store.update_due(SimpleNamespace(chat=new_judge)),1)
            latest=(await store.read(2,[10]))[0]['traits']
            self.assertEqual({t['trait'] for t in latest},{'外向','幽默'})
            self.assertNotIn('内向',str(latest))
            other=(await store.read(2,[20]))[0]['traits']
            self.assertIn('内向',{t['trait'] for t in other})
