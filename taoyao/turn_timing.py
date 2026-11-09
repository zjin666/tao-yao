"""Per-turn timeline marks for the first reply. Logging only; never changes behavior."""
import time
from contextvars import ContextVar
from functools import wraps

_TIMING = ContextVar('taoyao_turn_timing', default=None)

# Order shown in the log line: (mark name, label).
_LABELS = (
    ('draft_start', '开始生成'),
    ('prep_done', '准备完成'),
    ('first_token', '首字'),
    ('first_sentence', '首句写完'),
    ('check_start', '检查开始'),
    ('check_end', '检查结束'),
    ('grace_start', '首句宽限开始'),
    ('grace_end', '首句宽限结束'),
    ('tail_grace_start', '尾部宽限开始'),
    ('tail_grace_end', '尾部宽限结束'),
    ('incomplete_wait_start', '未说完等待开始'),
    ('incomplete_wait_end', '未说完等待结束'),
    ('hold_start', '开始扣住'),
    ('hold_end', '扣住结束'),
    ('send_start', '开始发送'),
    ('send_end', '发送确认'),
)


def turn_timing_context(function):
    """Open a fresh timeline for one handled message; async tasks created inside share it."""
    @wraps(function)
    async def wrapped(*args, **kwargs):
        token = _TIMING.set({'t0': time.monotonic()})
        try:
            return await function(*args, **kwargs)
        finally:
            _TIMING.reset(token)
    return wrapped


def mark(name: str) -> None:
    """Record the first time a stage is reached in this turn."""
    state = _TIMING.get()
    if state is not None:
        state.setdefault(name, time.monotonic())


def set_value(name: str, value) -> None:
    """在本轮里存一个值（比如这条消息到达时的序号），同一轮里的并行任务也能读到。"""
    state = _TIMING.get()
    if state is not None:
        state[name] = value


def get_value(name: str, default=None):
    state = _TIMING.get()
    return default if state is None else state.get(name, default)


def timeline_once() -> str:
    """Return the formatted timeline the first time it is asked for in a turn, else ''."""
    state = _TIMING.get()
    if state is None or state.get('logged'):
        return ''
    state['logged'] = True
    t0 = state['t0']
    parts = [f'{label}={state[name] - t0:.2f}' for name, label in _LABELS if name in state]
    return '，'.join(parts)
