"""Check a complete first sentence, then the entire remaining body. No unchecked sends."""
import asyncio
from dataclasses import dataclass
import json
import re
import logging
import time

from .output_guard import internal_notice_reason,is_internal_only
from .reply_flow import reply_boundary_issue
from .review_prompts import REVIEW_PROMPT
log=logging.getLogger(__name__)


class RawPartBlocked(Exception):
    pass


def first_sentence(buffer):
    pairs={'「':'」','『':'』','“':'”','（':'）','(':')','[':']','【':'】','{':'}'}
    stack=[]
    for i,char in enumerate(buffer):
        if char=='"':
            if stack and stack[-1]=='"':stack.pop()
            else:stack.append('"')
        elif char in pairs:stack.append(pairs[char])
        elif stack and char==stack[-1]:
            stack.pop()
            if not stack and i and buffer[i-1] in '。！？!?':return buffer[:i+1],buffer[i+1:]
        if char in '。！？!?' and not stack:return buffer[:i+1],buffer[i+1:]
        if char=='\n' and not stack:
            line=buffer[:i].strip()
            # A complete-looking prose line is a fallback boundary; never cut code,
            # ASCII/numeric fragments, or common unfinished clauses.
            if (bool(re.search(r'[\u4e00-\u9fff]',line))
                and not re.search(r'[,，、:：;；—…=+*/\-]$|(?:因为|如果|比如|例如|关于|以及|为了|只要|除非|虽然)$|^(?:所以|但是|包括|然后|正在|我觉得|需要|想要|应该|可以|必须|就是|把|被|让|给|和|与|或)$',line)
                and not re.search(r'```|https?://|\[\[|^\s*(?:[-*#]|\d+[.)、])',line)
                and buffer[i+1:].strip()):
                return buffer[:i],buffer[i+1:]

    return None,buffer



class ScopedReplyGuard:
    """Check scoped reply data only; never opens a global privacy index."""
    def __init__(self,llm,*,role,current,policy,hints=()):
        self.llm,self.role,self.current,self.policy=llm,role,current,policy
        self.hints=tuple(hints)
        self.ticket=None
        self.reviewed=None
        self.evidence=''

    def key(self,sent,pending):
        return (sent,pending,json.dumps(self.policy,ensure_ascii=False,sort_keys=True))

    def is_current(self,sent,pending):
        return self.ticket==self.key(sent,pending)

    async def check(self,sent,pending):
        started=time.monotonic();reason='review_unavailable';key=self.key(sent,pending);cancelled=False
        try:
            if self.reviewed is not None and self.reviewed[0]==key:
                reason=self.reviewed[1]
                log.info('首段检查复用：文本与身份清单一致，未重复调用模型')
                return reason
            self.evidence=''
            cumulative=sent+'\n'+pending if sent else pending
            reason=internal_notice_reason(cumulative) or reply_boundary_issue(cumulative,self.role)
            if reason:return reason
            if is_internal_only(cumulative):
                reason='internal_only';return reason
            key=self.key(sent,pending)
            if self.ticket==key:
                reason='';return reason
            result=await self.llm.chat([
                {'role':'system','content':REVIEW_PROMPT},
                {'role':'user','content':json.dumps({'policy':self.policy,'current':self.current,
                    'already_sent':sent,'pending':pending,'protected':[f['text'] for f in self.hints]},ensure_ascii=False,separators=(',',':'))}])
            verdict=json.loads(result.strip())
            if verdict.get('allowed') is True:
                self.ticket=key;reason='';return reason
            value=verdict.get('reason')
            reason=value if value in ('privacy','identity','intimacy','internal_message') else 'review_unavailable'
            evidence=verdict.get('evidence')
            if reason!='review_unavailable' and isinstance(evidence,str) and evidence.strip() and evidence in cumulative:
                self.evidence=evidence
            return reason
        except asyncio.CancelledError:
            cancelled=True
            raise
        except Exception:
            reason='review_unavailable';return reason
        finally:
            if not cancelled:
                self.reviewed=(key,reason)
            log.info('范围隔离检查耗时：%.3fs stage=%s result=%s',time.monotonic()-started,
                'tail' if sent else 'first',reason or 'allowed')


def split_approved_bubbles(text,limit):
    """Split already-approved text locally; keep overflow intact in the last bubble."""
    parts=[];remaining=text.strip();limit=max(1,int(limit))
    while remaining and len(parts)<limit-1:
        part,rest=first_sentence(remaining)
        if part is None:break
        if part.strip():parts.append(part.strip())
        remaining=rest.strip()
    if remaining:parts.append(remaining)
    return parts


@dataclass
class DeliveryResult:
    text: str
    complete: bool


@dataclass
class PartsResult:
    text: str
    approved_raw: str
    complete: bool
    issue: str
    failed_stage: str = ''


async def send_checked_parts(stream,guard,prepare,deliver,*,early=True,prefix=''):
    queue=asyncio.Queue();emitted=[];approved=[]
    async def produce():
        buffer='';first=False
        try:
            async for chunk in stream:
                buffer+=chunk
                if early and not prefix and not first:
                    part,rest=first_sentence(buffer)
                    if part is not None:
                        if re.fullmatch(r'[A-Za-z0-9\s.,:;!?。！？-]{1,16}',part.strip()):continue
                        await queue.put(('first',part));first=True;buffer=rest
            if buffer.strip():await queue.put(('tail' if first or prefix else 'first',buffer))
            await queue.put(('end',''))
        except Exception as exc:await queue.put(('error',exc))
    producer=asyncio.create_task(produce())
    try:
        while True:
            stage,raw=await queue.get()
            sent='\n'.join(([prefix] if prefix else [])+emitted)
            if stage=='end':return PartsResult(sent,'\n'.join(approved),True,'')
            if stage=='error':
                if not sent:raise raw
                return PartsResult(sent,'\n'.join(approved),False,'generation_unavailable','tail')
            try:pending=await prepare(raw)
            except RawPartBlocked as exc:
                return PartsResult(sent,'\n'.join(approved),False,str(exc),stage)
            if not pending:
                approved.append(raw)
                continue
            issue=await guard.check(sent,pending)
            if issue:return PartsResult(sent,'\n'.join(approved),False,issue,stage)
            ok=await deliver(pending,sent,guard)
            if isinstance(ok,DeliveryResult):
                if ok.text:
                    emitted.append(ok.text);approved.append(raw if ok.complete else ok.text)
                if not ok.complete:
                    actual='\n'.join(([prefix] if prefix else [])+emitted)
                    return PartsResult(actual,'\n'.join(approved),False,'delivery_unavailable',stage)
                continue
            if ok is False:return PartsResult(sent,'\n'.join(approved),False,'delivery_unavailable',stage)
            emitted.append(pending);approved.append(raw)
    finally:
        if not producer.done():producer.cancel()
        await asyncio.gather(producer,return_exceptions=True)
