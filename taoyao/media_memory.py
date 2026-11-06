from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("taoyao.media_memory")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MediaMemory:
    """按群/私聊保存最近的图片描述，供后续“刚才那张图”追问使用。"""

    def __init__(self, config: dict[str, Any] | None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.ttl_seconds = float(self.config.get("ttl_seconds", 86400))
        self.max_items = max(1, int(self.config.get("max_items", 8)))
        raw_dir = Path(str(self.config.get("dir") or "data/media_memory"))
        self.dir = raw_dir if raw_dir.is_absolute() else PROJECT_ROOT / raw_dir
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _filename(chat_key: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", chat_key)
        return f"{safe}.json"

    def _load(self, chat_key: str) -> list[dict]:
        path = self.dir / self._filename(chat_key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, chat_key: str, entries: list[dict]) -> None:
        path = self.dir / self._filename(chat_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def _trim(self, entries: list[dict]) -> list[dict]:
        cutoff = time.time() - self.ttl_seconds
        alive = [
            item for item in entries
            if isinstance(item, dict) and float(item.get("created_at") or 0) >= cutoff
        ]
        return alive[-self.max_items:]

    def add(self, chat_key: str, user_id: int, description: str) -> None:
        if not self.enabled or not description.strip():
            return
        try:
            entries = self._trim(self._load(chat_key))
            entries.append({
                "chat_key": chat_key,
                "user_id": str(user_id),
                "description": description.strip()[:2500],
                "created_at": time.time(),
            })
            self._save(chat_key, self._trim(entries))
        except Exception:
            log.debug("保存图片记忆失败", exc_info=True)

    def recent(self, chat_key: str, limit: int = 3) -> list[dict]:
        if not self.enabled:
            return []
        try:
            entries = self._trim(self._load(chat_key))
            return entries[-max(1, int(limit)):]
        except Exception:
            return []
