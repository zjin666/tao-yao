from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("taoyao.user_memes")


class UserMemeLibrary:
    """用户发来的表情包，按桃夭的理解命名；默认低概率收录、低概率使用。"""

    def __init__(self, config: dict[str, Any] | None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.dir = Path(str(cfg.get("dir") or "user_memes"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.store_probability = max(0.0, min(1.0, float(cfg.get("store_probability", 0.08))))
        self.use_probability = max(0.0, min(1.0, float(cfg.get("use_probability", 0.15))))
        self.min_description_chars = max(1, int(cfg.get("min_description_chars", 4)))
        self.max_files = max(1, int(cfg.get("max_files", 200)))
        self.min_use_score = float(cfg.get("min_use_score", 0.42))
        self._entries: list[dict] | None = None
        self._vectors = None
        self._vector_key: tuple = ()

    def _load(self) -> list[dict]:
        if self._entries is not None:
            return self._entries
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._entries = [item for item in raw if isinstance(item, dict) and item.get("file")]
        except Exception:
            self._entries = []
        return self._entries

    def _save(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self._entries or [], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._vectors = None
        self._vector_key = ()

    @staticmethod
    def _label(description: str, category: str) -> str:
        text = re.sub(r"[\s\r\n]+", "", description or "")
        text = re.sub(r"[\\/:*?\"<>|]", "", text).strip("。！？!?，,；;：:")
        if category:
            text = f"{category}_{text}"
        return (text or "表情")[:24]

    @staticmethod
    def _digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 128), b""):
                h.update(chunk)
        return h.hexdigest()[:12]

    def _prune(self) -> None:
        entries = self._entries or []
        while len(entries) > self.max_files:
            old = entries.pop(0)
            try:
                (self.dir / str(old.get("file") or "")).unlink()
            except OSError:
                pass

    def maybe_store(self, path: Path, description: str, category: str = "") -> bool:
        if not self.enabled or not path.exists():
            return False
        description = (description or "").strip()
        if len(description) < self.min_description_chars:
            return False
        if random.random() > self.store_probability:
            return False
        entries = self._load()
        digest = self._digest(path)
        if any(item.get("hash") == digest for item in entries):
            return False
        suffix = path.suffix.lower() if path.suffix else ".png"
        if suffix not in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            suffix = ".png"
        label = self._label(description, category)
        name = f"{label}_{digest}{suffix}"
        dest = self.dir / name
        try:
            shutil.copy2(path, dest)
        except Exception as exc:
            log.warning("保存用户表情失败：%s", exc)
            return False
        entries.append({
            "file": name,
            "label": label,
            "description": description[:500],
            "category": category,
            "hash": digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        self._prune()
        self._save()
        log.info("收录用户表情：%s", name)
        return True

    def match(self, sentence: str, embedder: Any) -> dict | None:
        if not self.enabled or not sentence or embedder is None:
            return None
        entries = self._load()
        if not entries:
            return None
        key = tuple(item.get("hash") for item in entries)
        if self._vectors is None or self._vector_key != key:
            texts = [str(item.get("description") or item.get("label") or "") for item in entries]
            self._vectors = embedder.embed_texts(texts)
            self._vector_key = key
        vec = embedder.embed_texts([sentence])[0]
        sims = self._vectors @ vec
        idx = int(max(range(len(sims)), key=lambda i: float(sims[i])))
        score = float(sims[idx])
        if score < self.min_use_score:
            return None
        entry = dict(entries[idx])
        entry["path"] = str(self.dir / str(entry.get("file") or ""))
        entry["score"] = score
        return entry
