"""只在回复准备完成后补足用户发言的空窗期。"""
import asyncio
import math
import time
from contextvars import ContextVar
from functools import wraps
from inspect import signature

_DELIVERY = ContextVar('taoyao_reply_delivery', default=None)


def conversation_session(group_id: int, user_id: int, owner_id: int) -> str:
    if group_id:
        return f'group:{group_id}'
    return f'{"owner" if owner_id and user_id == owner_id else "user"}_private:{user_id}'


def reply_window_context(function):
    parameters = signature(function)
    @wraps(function)
    async def wrapped(*args, **kwargs):
        values = parameters.bind_partial(*args, **kwargs).arguments
        state = {'key': (values['session'], values['user_id']), 'released': False,
                 'target':values.get('target'),'target_id':values.get('target_id'),
                 'user_id':values['user_id'],'role':values.get('role')}
        token = _DELIVERY.set(state)
        try:
            return await function(*args, **kwargs)
        finally:
            _DELIVERY.reset(token)
    return wrapped


def current_reply_actor(target, target_id):
    """Reuse the initiating person only inside the same destination context."""
    state=_DELIVERY.get()
    if state and state.get('target')==target and state.get('target_id')==target_id:
        return state.get('user_id'),state.get('role')
    return None,None


class ReplyWindow:
    def __init__(self):
        self.deadlines: dict[tuple[str, int], float] = {}
        self.sent_times: dict[tuple[str, int], float] = {}

    def note(self, key: tuple[str, int], sent_at, seconds: float) -> None:
        now = time.monotonic()
        elapsed = 0.0
        if isinstance(sent_at, (float, int)) and not isinstance(sent_at, bool) and math.isfinite(sent_at) and sent_at > 0:
            elapsed = max(0.0, time.time() - sent_at)
        deadline = now + max(0.0, seconds - elapsed)
        self.sent_times[key] = max(self.sent_times.get(key, float('-inf')), now - elapsed)
        self.deadlines[key] = max(self.deadlines.get(key, 0.0), deadline)
        if len(self.deadlines) > 2000:
            expired = [k for k, value in self.deadlines.items() if value <= now and k != key]
            for k in expired:
                self.deadlines.pop(k, None)
                self.sent_times.pop(k, None)

    async def wait(self, key: tuple[str, int]) -> None:
        while True:
            remaining = self.deadlines.get(key, 0.0) - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)
            # 睡眠期间同一用户可能又发言，重新计算；不再追加固定延迟。

    async def before_send(self, key: tuple[str, int] | None = None) -> None:
        state = _DELIVERY.get()
        if state is not None:
            if state['released']:
                return
            key = state['key']
        if key is not None:
            await self.wait(key)
            if state is not None:
                state['released'] = True

    def first_delivery_elapsed(self, key: tuple[str, int] | None = None) -> float | None:
        state = _DELIVERY.get()
        if state is not None:
            if state.get('first_logged'):
                return None
            key = state['key']
        if key not in self.sent_times:
            return None
        if state is not None:
            state['first_logged'] = True
        return max(0.0, time.monotonic() - self.sent_times[key])
