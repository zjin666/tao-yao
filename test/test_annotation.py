import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock
import httpx
from taoyao.annotation import AnalysisPipeline, DeepSeekJudge, SampleStore, contract, validate_labels
from taoyao.annotation_cli import export, queue, apply
from taoyao.personality import DIMENSIONS, ClassificationResult
from taoyao.profiles import ProfileStore

def payload():
    return {'dimensions':{d:dict(bucket='中性',confidence=.9,evidence='',scope='none',ambiguous=False) for d in DIMENSIONS}}

class FakeLocal:
    def classify(self,text):
        return {d:ClassificationResult('攻击挑衅' if d=='AGG' else '中性',.99) for d in DIMENSIONS}

class FakeJudge:
    def __init__(self,fail=False):self.calls=0;self.fail=fail
    async def analyze(self,text,context,spec):
        self.calls+=1
        if self.fail:raise RuntimeError('unavailable')
        labels,reasons=validate_labels(payload(),text,spec)
        return dict(status='ok',labels=labels,model_requested='fake'),reasons
    async def close(self):pass

class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_primary_wins_dedup_and_preserves_review(self):
        with tempfile.TemporaryDirectory() as d:
            judge=FakeJudge();pipe=AnalysisPipeline({'directory':d},FakeLocal(),judge)
            args=dict(text='你是不是有点累了',user_id=1,scene='private',group_id=0,message_id='m1',sent_at=1700000000)
            result,sid=await pipe.analyze(**args)
            self.assertEqual(result['AGG'].bucket,'中性');self.assertFalse(result['AGG'].applied)
            row=pipe.store.read(sid)
            self.assertIn('AGG:local_disagreement',row['review_reasons'])
            self.assertEqual(row['sent_at'],'2023-11-15T06:13:20.000+08:00')
            self.assertEqual(row['time_source'],'qq_event')
            labels={d:'中性' for d in DIMENSIONS};pipe.store.review(sid,labels,'tester')
            await pipe.analyze(**args)
            self.assertEqual(judge.calls,1)
            pipe.store.analysis(sid,{'status':'error','labels':{}},{},['primary_failed'])
            self.assertEqual(pipe.store.read(sid)['human_review']['labels'],labels)
            self.assertEqual(len(pipe.store.read(sid)['analysis_history']),1)

    async def test_primary_failure_never_uses_local_as_truth(self):
        with tempfile.TemporaryDirectory() as d:
            pipe=AnalysisPipeline({'directory':d},FakeLocal(),FakeJudge(True))
            results,sid=await pipe.analyze(text='关心你',user_id=1,scene='group',group_id=22)
            self.assertTrue(all(r.bucket=='unknown' and not r.applied for r in results.values()))
            self.assertIn('primary_failed',pipe.store.read(sid)['review_reasons'])

    async def test_http_json_contract(self):
        async def handler(request):
            body=json.loads(request.content)
            self.assertEqual(body['response_format'],{'type':'json_object'})
            self.assertEqual(body['thinking'],{'type':'disabled'})
            self.assertTrue(body['stream'])
            event=dict(model='resolved',id='r',choices=[dict(delta={'content':json.dumps(payload())},finish_reason='stop')])
            return httpx.Response(200,text='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n')
        with tempfile.TemporaryDirectory() as d:
            key=Path(d)/'key';key.write_text('test-only');judge=DeepSeekJudge({'key_file':str(key)})
            await judge.client.aclose();judge.client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            out,_=await judge.analyze('正常询问',[],contract()[1])
            self.assertEqual(out['model_returned'],'resolved');self.assertFalse(out['confidence_calibrated'])
            await judge.close()

    async def test_scene_isolation_and_current_message_cutoff(self):
        with tempfile.TemporaryDirectory() as d:
            store=ProfileStore(Path(d)/'profiles.db')
            try:
                for uid,scene,gid,text,mid in [(1,'private',0,'私密','a'),(1,'group',10,'群十','b'),(2,'group',20,'其他群','c'),(2,'group',10,'其他群友','d'),(1,'group',10,'本句','e'),(1,'group',10,'后句','f')]:
                    await store.add_message(uid,scene,gid,'user',text,mid)
                ctx=await store.annotation_context(1,'group',10,10,'e')
                self.assertEqual([r['text'] for r in ctx],['群十','其他群友'])
                self.assertEqual([r['speaker'] for r in ctx],['self','other_1'])
                self.assertEqual([r['text'] for r in await store.annotation_context(1,'private',0,10)],['私密'])
            finally:store.close()

    async def test_app_forwards_raw_text_and_time(self):
        from taoyao.app import TaoYaoBot
        self.assertIn('sent_at',inspect.signature(TaoYaoBot._handle_text).parameters)
        bot=TaoYaoBot.__new__(TaoYaoBot);bot.config={'profile':{'llm_extract':False}}
        bot.profiles=AsyncMock();bot.profiles.annotation_context.return_value=[]
        bot.profiles.get.return_value={'user_id':'1'};bot.analysis_pipeline=AsyncMock()
        bot.analysis_pipeline.config={'context_messages':6}
        bot.analysis_pipeline.analyze.return_value=({d:ClassificationResult('中性',.9,applied=False) for d in DIMENSIONS},'sample')
        await bot._update_profile({'user_id':'1'},'生成的媒体说明',raw_text='关心原句',message_id='m',sent_at=1700000000)
        args=bot.analysis_pipeline.analyze.call_args.kwargs
        self.assertEqual(args['text'],'关心原句');self.assertEqual(args['sent_at'],1700000000)
        self.assertEqual(bot.profiles.add_update.call_args.args[2]['annotation_sample_id'],'sample')

class StoreTests(unittest.TestCase):
    def test_validation(self):
        spec=contract()[1];p=payload();p['dimensions']['ACT'].update(bucket='明显疲惫',evidence='累',scope='other')
        self.assertEqual(validate_labels(p,'你累了吗',spec)[0]['ACT']['bucket'],'unknown')
        p['dimensions']['ACT'].update(scope='speaker',evidence='不存在的字')
        self.assertEqual(validate_labels(p,'你累了吗',spec)[0]['ACT']['bucket'],'unknown')
        p['dimensions']['ACT']['confidence']=float('nan')
        with self.assertRaises(ValueError):validate_labels(p,'句子',spec)
        with self.assertRaises(ValueError):validate_labels({'dimensions':{}},'句子',spec)

    def test_queue_review_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            store=SampleStore(d)
            rows=[store.create(text=t,raw_text=t,context=[],user_id=u,scene='private',group_id=0,message_id=str(i)) for i,(t,u) in enumerate([('相同的话',1),('相同的话',2),('另句话',2),('未确认',3)])]
            labels={dim:'中性' for dim in DIMENSIONS};labels['INT']='unknown'
            for r in rows[:3]:store.review(r['id'],labels,'test')
            out=Path(d)/'export.json';export(store,contract()[0],out)
            data=json.loads(out.read_text(encoding='utf-8'))['rows']
            self.assertEqual(len(data),10);self.assertEqual(len({r['split'] for r in data}),1)
            self.assertFalse(any(r['text']=='未确认' or r['dimension']=='INT' for r in data))
            queue(store,20);path=next((Path(d)/'review').glob('queue*.json'))
            review=json.loads(path.read_text(encoding='utf-8'));review['items'][0].update(decision='reject',reviewer='test')
            path.write_text(json.dumps(review),encoding='utf-8');apply(store,path)
            self.assertEqual(store.read(rows[-1]['id'])['human_review']['status'],'rejected')
            with self.assertRaises(ValueError):store.read('../escape')

class ReviewProvenanceTests(unittest.TestCase):
    def test_partial_chat_confirmation_exports_only_explicit_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            store=SampleStore(directory)
            row=store.create(text='只是玩笑',raw_text='只是玩笑',context=[],user_id=1,scene='private',group_id=0)
            labels={d:'中性' for d in DIMENSIONS};labels['HUM']='轻松玩笑'
            store.review(row['id'],labels,'User via chat',confirmed_axes=['AGG','HUM'])
            out=Path(directory)/'export.json';export(store,contract()[0],out)
            rows=json.loads(out.read_text(encoding='utf-8'))['rows']
            self.assertEqual({r['dimension'] for r in rows},{'AGG','HUM'})

    def test_assistant_review_is_not_human_gold_and_human_can_override(self):
        with tempfile.TemporaryDirectory() as directory:
            store=SampleStore(directory)
            row=store.create(text='你好',raw_text='你好',context=[],user_id=1,scene='private',group_id=0)
            labels={d:'中性' for d in DIMENSIONS};labels['INT']='社交礼仪'
            store.review(row['id'],labels,'Codex',reviewer_kind='assistant')
            self.assertEqual(store.read(row['id'])['human_review']['status'],'assistant_reviewed')
            out=Path(directory)/'export.json';export(store,contract()[0],out)
            self.assertEqual(json.loads(out.read_text(encoding='utf-8'))['rows'],[])
            store.review(row['id'],labels,'Human',reviewer_kind='human')
            human=store.read(row['id'])['human_review']
            self.assertEqual(human['status'],'confirmed')
            self.assertEqual(human['history'][0]['reviewer_kind'],'assistant')
            with self.assertRaises(ValueError):
                store.review(row['id'],labels,'Codex',reviewer_kind='assistant')
            export(store,contract()[0],out)
            self.assertEqual(len(json.loads(out.read_text(encoding='utf-8'))['rows']),6)

if __name__=='__main__':unittest.main()
