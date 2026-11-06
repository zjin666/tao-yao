import asyncio,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
from taoyao.image_order import ImageOrder,private_image_order
from taoyao.app import TaoYaoBot
from taoyao.config import load_config


class ImageOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_all_images_and_keeps_other_users_independent(self):
        order=ImageOrder();a=order.begin(1);b=order.begin(1)
        waiter=asyncio.create_task(order.wait(1))
        await order.wait(2)
        order.finish(a);await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        order.finish(b);await asyncio.wait_for(waiter,1)
        self.assertFalse(order.pending(1))

    async def test_cancelled_waiter_does_not_cancel_image(self):
        order=ImageOrder();token=order.begin(1)
        waiter=asyncio.create_task(order.wait(1));await asyncio.sleep(0)
        waiter.cancel();await asyncio.gather(waiter,return_exceptions=True)
        self.assertFalse(token[1].cancelled())
        order.finish(token)
        await order.wait(1)

    async def test_failed_image_handler_releases_followup_text(self):
        class Receiver:
            @private_image_order
            async def event(self,data):
                if data.get('message'):raise ValueError('synthetic failure')
                return 'text processed'
        receiver=Receiver()
        image={'post_type':'message','message_type':'private','user_id':1,'self_id':2,
               'message':[{'type':'image'}]}
        with self.assertRaises(ValueError):await receiver.event(image)
        self.assertEqual(await receiver.event({**image,'message':[]}),'text processed')

    async def test_late_image_stops_unsent_blind_reply_and_orders_followup(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config();cfg['profile'].update(enabled=False,db=str(Path(tmp)/'profiles.db'))
            cfg['memory']['db']=str(Path(tmp)/'memory.db')
            cfg['intro']['enabled']=cfg['privacy_index']['enabled']=False
            cfg['memes']['enabled']=cfg['user_memes']['enabled']=False
            cfg['reply'].update(segment_gap_min_seconds=0,segment_gap_max_seconds=0)
            bot=TaoYaoBot(cfg,mock=True);bot.batch_enabled=False
            bot.guard.allow_rate=Mock(return_value=True)
            image_started=asyncio.Event();release_image=asyncio.Event()
            draft_started=asyncio.Event();release_draft=asyncio.Event();requests=[];sent=[]
            async def stream(messages):
                requests.append(messages)
                if len(requests)==1:
                    draft_started.set();await release_draft.wait();yield '你还没发图。'
                else:yield '原来是贝果。'
            bot.llm=bot.utility_llm=SimpleNamespace(stream_chat=stream,chat=AsyncMock(return_value='{"allowed":true}'))
            bot._search_context_for=AsyncMock(return_value='');bot._maybe_send_reply_poke=AsyncMock()
            bot._forward_text=AsyncMock(return_value='')
            path=Path(tmp)/'image.png';path.write_bytes(b'synthetic')
            bot.vision.fetch_segment=AsyncMock(return_value=(path,False))
            async def describe(*a):image_started.set();await release_image.wait();return '图中是贝果。'
            bot.vision.describe=AsyncMock(side_effect=describe)
            bot.media_memory.add=Mock();bot.media_memory.recent=Mock(return_value=[])
            async def send(*a,**k):sent.append(a[2]);return {'status':'ok'}
            bot.server.send_text=AsyncMock(side_effect=send)
            def event(mid,segments):return {'post_type':'message','message_type':'private',
                'user_id':900123,'self_id':900999,'sender':{'nickname':'合成用户'},
                'message_id':mid,'message':segments}
            def text(value):return [{'type':'text','data':{'text':value}}]
            tasks=[]
            try:
                tasks.append(asyncio.create_task(bot.on_event(event('question',text('看看这是什么')))))
                await asyncio.wait_for(draft_started.wait(),2)
                tasks.append(asyncio.create_task(bot.on_event(event('image',[{'type':'image','data':{'file':str(path)}}]))))
                await asyncio.wait_for(image_started.wait(),2)
                tasks.append(asyncio.create_task(bot.on_event(event('followup',text('这是贝果')))))
                release_draft.set();await asyncio.wait_for(tasks[0],3)
                self.assertEqual(sent,[])
                self.assertFalse(tasks[2].done())
                release_image.set();await asyncio.wait_for(asyncio.gather(*tasks),5)
                self.assertEqual(len(sent),2)
                self.assertNotIn('你还没发图。',sent)
                self.assertIn('图中是贝果',requests[1][-1]['content'])
                self.assertIn('看看这是什么',str(requests[1]))
                self.assertEqual(requests[2][-1]['content'],'这是贝果')
            finally:
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                for task in list(bot._reply_extra_pending):task.cancel()
                await asyncio.gather(*list(bot._reply_extra_pending),return_exceptions=True)
                bot.memory.close();bot.profiles.close()
