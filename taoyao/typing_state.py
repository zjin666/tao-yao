"""私聊“对方正在输入”的短时状态，只放内存，不存盘。

QQ 只在私聊里推送输入状态（SnowLuma 转成 notice/notify/input_status 事件）。实测的规律：
event_type=1 是开始输入，只在开始时推一次，不会每隔几秒重复；event_type=2 是停止输入
（输入框清空或消息发出时也会推）；event_type=3 是语音输入。所以：1 和 3 之后一直算“在输入”，
直到收到 2；为防止停止事件丢失，再设一个较长的保险超时 ttl。
注意：“发出消息”本身也会触发 2，而且 2 比消息本身早约 0.3 秒到达（实测每次都是）。所以收到 2 时对方可能
只是刚按了发送，不是真的停下；用 stopped_within 记住“刚刚收到过 2”，让上层再多等一小会儿看消息有没有跟上来。
对方发出一条消息时，这条消息之前的输入事件属于它自己，要清掉，之后新来的才算“发完还在输入”。
"""
from __future__ import annotations

import time
from collections import OrderedDict

STOP_EVENT = 2


class TypingTracker:
    def __init__(self, ttl_seconds: float = 30.0):
        self.ttl = max(0.1, float(ttl_seconds))      # 保险超时：停止事件丢了也不会一直当作在输入
        self._started: dict[int, float] = {}         # 最近一次“开始输入”的时间；已停止的会被移除
        self._last_event: dict[int, float] = {}
        self._stopped: dict[int, float] = {}         # 最近一次“停止输入”的时间；开始输入或收到消息后移除
        self._seq: dict[int, int] = {}
        self._own_seq: OrderedDict[str, int] = OrderedDict()   # 每条消息到达时的序号，供“这轮处理的是哪一条”查询
        self._last_msg_at: dict[int, float] = {}               # 上一条（算数的）消息的到达时间
        self._gap_of: OrderedDict[str, float | None] = OrderedDict()  # 每条消息到达时距上一条消息的秒数

    def note(self, user_id: int, event_type=1, now: float | None = None) -> float | None:
        """记录一次输入事件，返回距上一次输入事件的秒数（没有则 None），供日志观察事件节奏。"""
        now = time.monotonic() if now is None else now
        previous = self._last_event.get(user_id)
        self._last_event[user_id] = now
        if event_type == STOP_EVENT:
            self._started.pop(user_id, None)
            self._stopped[user_id] = now
        else:
            self._started[user_id] = now
            self._stopped.pop(user_id, None)
        return None if previous is None else now - previous

    def is_typing(self, user_id: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        started = self._started.get(user_id)
        return started is not None and now - started <= self.ttl

    def stopped_within(self, user_id: int, seconds: float, now: float | None = None) -> bool:
        """刚刚（seconds 秒内）收到过“停止输入”，且之后还没有新的开始输入或新消息。"""
        now = time.monotonic() if now is None else now
        stopped = self._stopped.get(user_id)
        return stopped is not None and now - stopped <= seconds

    def message_received(self, user_id: int, message_id: str = "", bump: bool = True,
                         now: float | None = None) -> int:
        """对方发来一条消息：清掉它之前的输入状态，并让新消息序号加一（用来发现“回复发出前又来了新消息”）。
        命令（/开头）不算新消息：bump=False 只清输入状态，不改序号，免得一条命令把上一条回复交接走。"""
        now = time.monotonic() if now is None else now
        self._started.pop(user_id, None)
        self._stopped.pop(user_id, None)
        previous = self._last_msg_at.get(user_id)
        if bump:
            self._seq[user_id] = self._seq.get(user_id, 0) + 1
            self._last_msg_at[user_id] = now
        if message_id:
            self._own_seq[message_id] = self._seq.get(user_id, 0)
            self._gap_of[message_id] = None if previous is None else now - previous
            while len(self._own_seq) > 500:
                self._own_seq.popitem(last=False)
            while len(self._gap_of) > 500:
                self._gap_of.popitem(last=False)
        return self._seq.get(user_id, 0)

    def gap_before(self, message_id: str) -> float | None:
        """这条消息到达时，距对方上一条消息过了多少秒；没有上一条或不认识这条消息则为 None。
        间隔很短说明对方正在连发，这时首句多等一会儿更可能赶上他的下一条。"""
        return self._gap_of.get(message_id)

    def seq_of(self, message_id: str) -> int | None:
        """这条消息到达时的序号；之后序号变大就说明又来了新消息。"""
        return self._own_seq.get(message_id)

    def message_seq(self, user_id: int) -> int:
        return self._seq.get(user_id, 0)
