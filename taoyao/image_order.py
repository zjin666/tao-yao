"""Keep private text behind images already received in the same conversation."""
import asyncio
import logging
import time
from functools import wraps

log=logging.getLogger(__name__)


class ImageOrder:
    def __init__(self):self.jobs={}

    def pending(self,user_id):return bool(self.jobs.get(str(user_id)))

    def begin(self,user_id):
        key=str(user_id);future=asyncio.get_running_loop().create_future()
        self.jobs.setdefault(key,set()).add(future)
        return key,future

    def finish(self,token):
        key,future=token
        if not future.done():future.set_result(None)
        jobs=self.jobs.get(key,set());jobs.discard(future)
        if not jobs:self.jobs.pop(key,None)

    async def wait(self,user_id):
        while self.pending(user_id):
            await asyncio.gather(*(asyncio.shield(job) for job in list(self.jobs[str(user_id)])))


def private_image_order(function):
    @wraps(function)
    async def wrapped(self,data):
        if (data.get('post_type')!='message' or data.get('message_type')!='private'
                or str(data.get('user_id'))==str(data.get('self_id'))):
            return await function(self,data)
        order=getattr(self,'_image_order',None)
        if order is None:order=self._image_order=ImageOrder()
        images=any(isinstance(s,dict) and s.get('type') in ('image','mface','video','record')
                   for s in data.get('message') or [])
        if images:
            token=order.begin(data.get('user_id'))
            try:return await function(self,data)
            finally:order.finish(token)
        started=time.monotonic()
        if order.pending(data.get('user_id')):
            await order.wait(data.get('user_id'))
            log.info('私聊文字等待图片处理：%.3fs',time.monotonic()-started)
        return await function(self,data)
    return wrapped
