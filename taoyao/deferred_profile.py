"""Serialize a user's reply turns and signal successful first delivery."""
import asyncio
from contextvars import ContextVar
from functools import wraps
from weakref import WeakValueDictionary

after_delivery = ContextVar('profile_after_delivery', default=None)


def turn_lock(bot, user_id):
    if not hasattr(bot, '_profile_turn_locks'):
        bot._profile_turn_locks = WeakValueDictionary()
    return bot._profile_turn_locks.setdefault(str(user_id), asyncio.Lock())


def ordered_background_update(func):
    @wraps(func)
    async def wrapped(self, profile, *args, **kwargs):
        if not (profile and kwargs.get('background') and
                (self.config.get('response_speed') or {}).get('defer_six_axis', False)):
            return await func(self, profile, *args, **kwargs)
        async with turn_lock(self, profile['user_id']):
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
