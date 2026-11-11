"""主动评论好友说说的状态和筛选：每条说说只评论一次，记在文件里，重启也不重复。纯逻辑，不碰网络。"""
from __future__ import annotations

import json
import time
from pathlib import Path

KEEP_SECONDS = 3 * 86400      # 记录保留 3 天，足够覆盖 1 天内的说说和之后的跟进


class ProactiveState:
    """tid -> {"state": commented|skipped|asked, "uin": 好友QQ, "at": 时间, ...}"""

    def __init__(self, path):
        self.path = Path(path)
        self.posts: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.posts = {str(tid): dict(rec) for tid, rec in data.items() if isinstance(rec, dict)}
        except (OSError, ValueError, AttributeError):
            self.posts = {}

    def save(self) -> None:
        cutoff = time.time() - KEEP_SECONDS
        self.posts = {tid: rec for tid, rec in self.posts.items() if float(rec.get("at") or 0) >= cutoff}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.posts, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass

    def get(self, tid) -> dict | None:
        return self.posts.get(str(tid))

    def mark(self, tid, state: str, uin: int, **extra) -> None:
        self.posts[str(tid)] = {"state": state, "uin": int(uin), "at": time.time(), **extra}
        self.save()

    def commented_within(self, seconds: float) -> list[tuple[str, dict]]:
        now = time.time()
        return [(tid, rec) for tid, rec in self.posts.items()
                if rec.get("state") == "commented" and now - float(rec.get("at") or 0) <= seconds]

    def count_commented(self, seconds: float) -> int:
        return len(self.commented_within(seconds))


def recent_posts(posts: list[dict], max_age_hours: float = 24.0, now: float | None = None) -> list[dict]:
    """最近 max_age_hours 小时内、别人能看到的、有文字或图片的说说，新的在前。"""
    now = now if now is not None else time.time()
    found = []
    for post in posts or []:
        if not isinstance(post, dict) or post.get("is_private"):
            continue
        try:
            age = now - float(post.get("time") or 0)
        except (TypeError, ValueError):
            continue
        if not 0 <= age <= max_age_hours * 3600:
            continue
        if not str(post.get("content") or "").strip() and not post.get("images"):
            continue
        found.append(post)
    return sorted(found, key=lambda p: -float(p.get("time") or 0))


def decide_unsure(rng_value: float, skip_probability: float = 0.6) -> str:
    """拿不准怎么评论时：skip_probability 的概率不评论，其余私聊问主人。rng_value 是 0~1 的随机数。"""
    return "skip" if rng_value < skip_probability else "ask"
