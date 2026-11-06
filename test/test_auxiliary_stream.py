import json
import unittest
import httpx
from taoyao.llm import collect_stream

class AuxiliaryStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, events, done=True):
        def handler(request):
            self.assertTrue(json.loads(request.content)['stream'])
            body=''.join('data: '+json.dumps(e)+'\n\n' for e in events)
            if done: body+='data: [DONE]\n\n'
            return httpx.Response(200,text=body)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await collect_stream(client,'https://synthetic.invalid/v1',{}, {})

    async def test_complete_json_chunks_ignore_reasoning_and_keep_metadata(self):
        result=await self.collect([
            {'model':'test','id':'r','choices':[{'delta':{'reasoning_content':'hidden','content':'{"allowed":'}}]},
            {'choices':[{'delta':{'content':'true}'}}]},
            {'choices':[{'delta':{},'finish_reason':'stop'}]},
            {'choices':[],'usage':{'total_tokens':5}}])
        self.assertEqual(json.loads(result['choices'][0]['message']['content']),{'allowed':True})
        self.assertEqual(result['usage']['total_tokens'],5)
        self.assertEqual(result['model'],'test')

    async def test_truncated_or_nonstop_results_rejected(self):
        for finish,done in [('stop',False),('length',True),('content_filter',True),(None,True)]:
            with self.subTest(finish=finish,done=done):
                with self.assertRaises(ValueError):
                    await self.collect([{'choices':[{'delta':{'content':'{"allowed":true}'},'finish_reason':finish}]}],done)

    async def test_error_and_empty_results_rejected(self):
        for events in ([{'error':{'message':'synthetic'}}], [{'choices':[{'delta':{},'finish_reason':'stop'}]}]):
            with self.assertRaises(ValueError):
                await self.collect(events)
