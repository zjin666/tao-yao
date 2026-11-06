"""按群和消息发起人独立计数；每次完整回复算一轮。"""
from dataclasses import dataclass, field


@dataclass
class GroupReplyBudget:
    replies: dict[tuple[int, int | None], list[float]] = field(default_factory=dict)
    paused_until: dict[int, float] = field(default_factory=dict)
    user_paused_until: dict[tuple[int, int], float] = field(default_factory=dict)

    def paused(self, group_id: int, now: float, user_id: int | None = None) -> bool:
        return (now < self.paused_until.get(group_id, 0.0) or
                (user_id is not None and now < self.user_paused_until.get((group_id, user_id), 0.0)))

    def departure_due(self, group_id: int, now: float, limit: int, window: float, user_id: int | None = None) -> bool:
        key = (group_id, user_id)
        recent = [t for t in self.replies.get(key, []) if now - t < window]
        self.replies[key] = recent
        return len(recent) >= max(1, limit) - 1

    def record(self, group_id: int, now: float, departing: bool, pause: float, user_id: int | None = None) -> None:
        if departing:
            self.replies.pop((group_id, user_id), None)
            if user_id is None:
                self.paused_until[group_id] = now + max(0.0, pause)
            else:
                self.user_paused_until = {key: until for key, until in self.user_paused_until.items() if until > now}
                self.user_paused_until[(group_id, user_id)] = now + max(0.0, pause)
        else:
            self.replies.setdefault((group_id, user_id), []).append(now)
