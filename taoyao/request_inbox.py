"""好友申请、加群申请和邀请进群：先记下来，由哥哥决定同意还是拒绝。

只存必要的事实（谁、哪个群、验证信息、OneBot 的 flag），超过 ttl_days 天没人处理就作废，重启后还在。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RequestInbox:
    def __init__(self, path: str | Path, ttl_days: float = 7, max_items: int = 30):
        self.path = Path(path)
        self.ttl = float(ttl_days) * 86400
        self.max_items = max(5, int(max_items))
        self._items: list[dict[str, Any]] = []
        self._next_id = 1
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._items = [i for i in data.get("items", []) if isinstance(i, dict)]
            self._next_id = int(data.get("next_id") or 1)
        except Exception:
            self._items, self._next_id = [], 1
        self._expire()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps({"next_id": self._next_id, "items": self._items}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        temp.replace(self.path)

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl
        self._items = [i for i in self._items if i.get("status") != "pending" or float(i.get("at", 0)) >= cutoff]
        done = [i for i in self._items if i.get("status") != "pending"]
        if len(done) > self.max_items:                                  # 已处理的只留最近一些
            drop = {id(i) for i in done[: len(done) - self.max_items]}
            self._items = [i for i in self._items if id(i) not in drop]

    def add(self, *, kind: str, user_id: int, flag: str, group_id: int = 0, sub_type: str = "",
            comment: str = "", name: str = "") -> dict[str, Any]:
        """kind: friend / group_invite / group_add。同一个 flag 重复到达只记一次。"""
        self._expire()
        for item in self._items:
            if item.get("flag") == flag and item.get("status") == "pending":
                return item
        item = {"id": self._next_id, "kind": kind, "user_id": int(user_id or 0), "group_id": int(group_id or 0),
                "sub_type": sub_type, "comment": str(comment or "")[:200], "name": str(name or "")[:40],
                "flag": str(flag), "status": "pending", "at": time.time()}
        self._next_id += 1
        self._items.append(item)
        self._save()
        return item

    def pending(self) -> list[dict[str, Any]]:
        self._expire()
        return [i for i in self._items if i.get("status") == "pending"]

    def get(self, request_id: int) -> dict[str, Any] | None:
        return next((i for i in self._items if i.get("id") == int(request_id)), None)

    def latest_pending(self) -> dict[str, Any] | None:
        pending = self.pending()
        return pending[-1] if pending else None

    def finish(self, request_id: int, approved: bool) -> dict[str, Any] | None:
        item = self.get(request_id)
        if item is None or item.get("status") != "pending":
            return None
        item["status"] = "approved" if approved else "rejected"
        item["done_at"] = time.time()
        self._save()
        return item
