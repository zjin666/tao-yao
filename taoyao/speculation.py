"""Per-turn speculative body generation. Never sends stale text; optional head checks are discarded with stale drafts."""
import asyncio
from copy import deepcopy
import json
import logging
import time

log = logging.getLogger(__name__)


def request_key(request):
    return json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


class SpeculativeDraft:
    def __init__(self, request, generate):
        snapshot = deepcopy(request)
        self.key = request_key(snapshot)
        self.task = asyncio.create_task(generate(snapshot['messages']))

    async def resolve(self, final_request):
        if self.key != request_key(final_request):
            await self.close()
            log.info('预生成作废：最终输入变化，按最新输入生成')
            return None
        try:
            text = await self.task
        except asyncio.CancelledError:
            raise
        except Exception:
            log.info('预生成失败，回退正常生成', exc_info=False)
            return None
        log.info('预生成复用：最终输入完全一致')
        return text

    async def close(self):
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


class StreamingDraft:
    """Buffer speculative chunks, expose them only after complete request validation."""
    def __init__(self, request, stream, preview=None, preview_guard=None):
        snapshot=deepcopy(request)
        self.key=request_key(snapshot)
        self.queue=asyncio.Queue()
        self.error=None
        self.preview_task=None
        self.preview_guard=preview_guard
        async def produce():
            started=time.monotonic();complete=False;buffer=''
            try:
                async for chunk in stream(snapshot['messages']):
                    if preview is not None and self.preview_task is None:
                        from .two_part import first_sentence
                        import re
                        buffer+=chunk
                        part,_=first_sentence(buffer)
                        if part is not None and not re.fullmatch(r'[A-Za-z0-9\s.,:;!?。！？-]{1,16}',part.strip()):
                            self.preview_task=asyncio.create_task(preview(part))
                    await self.queue.put(('text',chunk))
                if preview is not None and self.preview_task is None and buffer.strip():
                    self.preview_task=asyncio.create_task(preview(buffer))
                await self.queue.put(('end',None))
                complete=True
            except Exception as exc:
                self.error=exc
                await self.queue.put(('error',exc))
            finally:
                log.info('流式预生成正文耗时：%.3fs complete=%s',time.monotonic()-started,complete)
        self.task=asyncio.create_task(produce())

    async def resolve_stream(self, final_request):
        if self.key!=request_key(final_request):
            await self.close()
            log.info('流式预生成作废：最终输入变化')
            return None
        if self.error is not None:
            await self.close()
            log.info('流式预生成失败，回退正常生成')
            return None
        log.info('流式预生成复用：最终输入完全一致')
        async def consume():
            while True:
                kind,value=await self.queue.get()
                if kind=='end':break
                if kind=='error':raise value
                yield value
        return consume()

    async def close(self):
        if not self.task.done():self.task.cancel()
        await asyncio.gather(self.task,return_exceptions=True)
        if self.preview_task is not None:
            if not self.preview_task.done():self.preview_task.cancel()
            await asyncio.gather(self.preview_task,return_exceptions=True)
