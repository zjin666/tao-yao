"""Serialize a user's reply turns, signal successful first delivery, and batch six-axis updates."""
import asyncio
import logging
import time
from contextvars import ContextVar
from functools import wraps
from weakref import WeakValueDictionary

log = logging.getLogger('taoyao.deferred_profile')
after_delivery = ContextVar('profile_after_delivery', default=None)


def turn_lock(bot, user_id):
    if not hasattr(bot, '_profile_turn_locks'):
        bot._profile_turn_locks = WeakValueDictionary()
    return bot._profile_turn_locks.setdefault(str(user_id), asyncio.Lock())


def update_lock(bot, user_id):
    """Serializes one user's six-axis updates only; reply turns never wait on it."""
    if not hasattr(bot, '_profile_update_locks'):
        bot._profile_update_locks = WeakValueDictionary()
    return bot._profile_update_locks.setdefault(str(user_id), asyncio.Lock())


def ordered_background_update(func):
    @wraps(func)
    async def wrapped(self, profile, *args, **kwargs):
        if not (profile and kwargs.get('background') and
                (self.config.get('response_speed') or {}).get('defer_six_axis', False)):
            return await func(self, profile, *args, **kwargs)
        async with turn_lock(self, profile['user_id']):
            async with update_lock(self, profile['user_id']):
                fresh = await self.profiles.get(profile['user_id'])
                return await func(self, fresh or profile, *args, **kwargs)
    return wrapped


def ordered_profile_turn(func):
    @wraps(func)
    async def wrapped(self, target, target_id, session, user_id, *args, **kwargs):
        if not (self.config.get('response_speed') or {}).get('defer_six_axis', False):
            return await func(self, target, target_id, session, user_id, *args, **kwargs)
        lock = turn_lock(self, user_id)
        async with lock:
            return await func(self, target, target_id, session, user_id, *args, **kwargs)
    return wrapped


class SixAxisBatcher:
    """Merge one user's consecutive messages into a single six-axis update.

    Replies never wait for this: a turn only calls add() and moves on. The merged
    update runs after the conversation has been quiet for `settle` seconds (or at
    the latest `max_wait` seconds after its first message) and updates of one user
    run strictly one after another, so scores always build on the previous result.
    """

    def __init__(self, run, bot=None, settle=4.0, max_wait=15.0, max_chars=600):
        self._run = run
        self._bot = bot
        self.settle = max(0.0, float(settle))
        self.max_wait = max(self.settle, float(max_wait))
        self.max_chars = max(50, int(max_chars))
        self._pending = {}
        self._tasks = set()

    @staticmethod
    def _key(user_id, scene, group_id):
        return (str(user_id), str(scene), int(group_id or 0))

    def add(self, user_id, scene, group_id, item):
        key = self._key(user_id, scene, group_id)
        now = time.monotonic()
        entry = self._pending.get(key)
        if entry is None:
            entry = {'items': [], 'first': now, 'last': now, 'task': None, 'wake': asyncio.Event()}
            self._pending[key] = entry
            entry['task'] = asyncio.create_task(self._wait_and_run(key))
            self._tasks.add(entry['task'])
            entry['task'].add_done_callback(self._tasks.discard)
        entry['items'].append(item)
        entry['last'] = now

    async def _wait_and_run(self, key):
        while True:
            entry = self._pending[key]
            now = time.monotonic()
            remaining = min(entry['last'] + self.settle - now, entry['first'] + self.max_wait - now)
            if remaining <= 0 or entry['wake'].is_set():
                break
            try:
                await asyncio.wait_for(entry['wake'].wait(), remaining)
            except asyncio.TimeoutError:
                pass
        entry = self._pending.pop(key)
        merged = self.merge(entry['items'], self.max_chars)
        lock = update_lock(self._bot if self._bot is not None else self, key[0])
        try:
            async with lock:
                await self._run(key, merged)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('合并六维更新失败')

    @staticmethod
    def merge(items, max_chars=600):
        texts = [str(item.get('text') or '').strip() for item in items]
        raws = [item.get('raw_text') for item in items]
        merged_text = '\n'.join(t for t in texts if t)[-max_chars:]
        has_raw = any(r is not None for r in raws)
        merged_raw = '\n'.join(
            (str(r) if r is not None else t).strip() for r, t in zip(raws, texts))[-max_chars:] if has_raw else None
        last = items[-1]
        profile = next((i.get('profile') for i in items if (i.get('profile') or {}).get('is_new')), items[0].get('profile'))
        return {
            'text': merged_text, 'raw_text': merged_raw, 'count': len(items), 'profile': profile,
            'message_id': last.get('message_id') or '', 'sent_at': last.get('sent_at'),
            'analysis_context': next((i.get('analysis_context') for i in reversed(items)
                                      if i.get('analysis_context') is not None), None),
        }

    async def flush(self, timeout=15.0):
        """Run whatever is still waiting (shutdown); never raises."""
        for entry in list(self._pending.values()):
            entry['wake'].set()
        tasks = list(self._tasks)          # 含已经开始、还没做完的更新
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
