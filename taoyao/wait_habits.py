"""每个人“没说完时要多等多久”的自适应：根据他实际多久发出下一条，慢慢调整。

纯逻辑，不碰数据库和网络；持久化由调用方用 to_dict / load 完成。每个人（群聊里是“群+人”）一份：

- 等到了下一条（命中）：等待时间向“他实际用的时间×1.5+0.5 秒”靠拢，发得快就缩短，发得慢就延长。
- 白等了（没等到下一条）：缩短 15%。
- 近期命中率太低（样本够了仍低于 15%）：这个人先暂停等待若干次，之后从下限重新学。
- 始终限制在 [floor, ceil] 内；没有记录的人用配置里的默认值。
"""
from __future__ import annotations

ALPHA = 0.25          # 命中率的滑动平均速度
BETA = 0.30           # 命中后等待时间向目标靠拢的速度
SHRINK = 0.85         # 每次白等缩短到原来的 85%
MARGIN_FACTOR = 1.5   # 目标 = 实际延迟 × 1.5 + 0.5
MARGIN_SECONDS = 0.5
MIN_SAMPLES = 6       # 样本少于这个数，不做“暂停”判断
PAUSE_RATE = 0.15     # 命中率低于这个值就暂停
PAUSE_COUNT = 8       # 暂停的次数（按“判为没说完的消息”计数）


class WaitHabits:
    def __init__(self):
        self._state: dict[str, dict] = {}

    def _get(self, key: str, default: float) -> dict:
        state = self._state.get(key)
        if state is None:
            state = self._state[key] = {"wait": float(default), "p": 0.5, "n": 0, "pause": 0}
        return state

    def wait_for(self, key: str, default: float, floor: float = 1.0, ceil: float = 8.0) -> float:
        """这次要等多久；暂停中返回 0（并消耗一次暂停）。"""
        state = self._get(key, default)
        if state["pause"] > 0:
            state["pause"] -= 1
            return 0.0
        return min(max(state["wait"], floor), ceil)

    def record(self, key: str, *, hit: bool, default: float, delay: float | None = None,
               floor: float = 1.0, ceil: float = 8.0) -> dict:
        """记录一次结果：hit=等到了下一条（delay 是从开始等到它到达的秒数），否则白等了。返回最新状态。"""
        state = self._get(key, default)
        state["p"] = (1 - ALPHA) * state["p"] + ALPHA * (1.0 if hit else 0.0)
        state["n"] += 1
        if hit:
            if delay is not None:
                target = max(0.0, delay) * MARGIN_FACTOR + MARGIN_SECONDS
                state["wait"] += BETA * (target - state["wait"])
        else:
            state["wait"] *= SHRINK
        state["wait"] = min(max(state["wait"], floor), ceil)
        if state["n"] >= MIN_SAMPLES and state["p"] < PAUSE_RATE:
            state.update(pause=PAUSE_COUNT, p=0.5, n=0, wait=floor)   # 暂停一阵，之后重新学
        return dict(state)

    def snapshot(self, key: str) -> dict | None:
        state = self._state.get(key)
        return dict(state) if state else None

    def to_dict(self, key: str) -> dict | None:
        return self.snapshot(key)

    def load(self, key: str, data: dict) -> None:
        """从持久化的数据恢复；字段缺失或不合法就忽略这个人。"""
        try:
            self._state[key] = {"wait": float(data["wait"]), "p": float(data["p"]),
                                "n": int(data["n"]), "pause": int(data.get("pause", 0))}
        except (KeyError, TypeError, ValueError):
            return
