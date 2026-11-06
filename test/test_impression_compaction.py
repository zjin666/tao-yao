import json
import tempfile
import unittest
from pathlib import Path

from taoyao.impressions import ImpressionStore, POLICY_VERSION


class ImpressionCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_recent_window_and_aliases_remain_bounded_and_group_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ImpressionStore(Path(tmp)/'impressions.db')
            for i in range(110):
                await store.observe_interaction('2',10,[f'name-{i}'],f'本人互动内容{i}',f'msg-{i}',replied=True)
            await store.observe_interaction('2',20,['other-group'],'另一个群里的互动','other',replied=True)
            c=store.connect()
            try:
                rows=c.execute('SELECT message_key FROM observations WHERE group_id=10 ORDER BY id').fetchall()
                self.assertEqual(len(rows),80)
                self.assertEqual(rows[0][0],'msg-30')
                self.assertEqual(rows[-1][0],'msg-109')
                self.assertEqual(c.execute('SELECT COUNT(*) FROM observations WHERE group_id=20').fetchone()[0],1)
                names=json.loads(c.execute('SELECT names FROM people WHERE group_id=10').fetchone()[0])
                self.assertEqual(len(names),8)
                self.assertIn('name-109',names)
            finally:c.close()

    async def test_existing_data_compacts_without_changing_checkpoint_or_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ImpressionStore(Path(tmp)/'impressions.db')
            c=store.connect()
            words=['幽默','细心','温柔','理性','独立','耐心','好奇','热心','俏皮','有主见']
            traits=[{'trait':word,'description':word,'subject_user_id':'2','confidence':.9,
                'evidence':[{'message_id':i,'quote':f'原文证据{i}'} for i in (1,2,3)],
                'evidence_ids':[1,2,3]} for word in words]
            try:
                with c:
                    c.execute('INSERT INTO people VALUES (?,?,?)',('2',10,json.dumps([f'name{i}' for i in range(20)])))
                    for i in range(150):
                        c.execute('INSERT INTO observations(user_id,group_id,message_key,content,sent_at) VALUES (?,?,?,?,?)',
                            ('2',10,str(i),'旧互动内容','2026-11-07T03:00:00+08:00'))
                    c.execute('INSERT INTO impressions VALUES (?,?,?,?,?,?,?)',
                        ('2',10,json.dumps(traits),150,40,'2026-11-07T03:00:00+08:00',POLICY_VERSION))
            finally:c.close()
            await store.run(store._compact_existing)
            result=(await store.read('2',[10]))[0]
            self.assertEqual(len(result['traits']),8)
            for trait in result['traits']:
                self.assertEqual(len(trait['evidence']),2)
                self.assertEqual(trait['evidence'][0]['quote'],'原文证据1')
            self.assertEqual(await store.run(store._due),[])
            c=store.connect()
            try:
                self.assertEqual(c.execute('SELECT COUNT(*) FROM observations').fetchone()[0],80)
                self.assertEqual(c.execute('SELECT upto FROM impressions').fetchone()[0],150)
            finally:c.close()
