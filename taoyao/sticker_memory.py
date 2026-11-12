"""表情包记忆：按群/私聊记住见过的表情包，以及发送者亲口说明过的含义。

- 用 8x8 差值哈希给图片做指纹，同一张表情（含轻微压缩差异）再次出现时直接用已存描述，
  不再重新识图，也不再当作第一次见到。
- 含义只来自发送者本人的原话（带原文证据），按“群/私聊”隔离，私聊里的说明不会带进群聊。
- 数量有界：每个范围最多 max_per_scope 条，按最近出现淘汰。
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

log = logging.getLogger("taoyao.sticker_memory")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 只是便宜的前置过滤，真正判断由模型做；命中面宁宽勿窄。
EXPLAIN_HINT_RE = re.compile(
    r"表情|表达|意思|代表|象征|用来|当(?:作|成|做)|只是|心情|这(?:个|张)?图|这是|就是")


class StickerMemory:
    def __init__(self, config: dict[str, Any] | None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.max_per_scope = max(10, int(cfg.get("max_per_scope", 200)))
        self.near_distance = max(0, int(cfg.get("near_distance", 6)))
        self.pending_seconds = float(cfg.get("pending_seconds", 600))
        raw = Path(str(cfg.get("db") or "data/stickers.db"))
        self.path = raw if raw.is_absolute() else PROJECT_ROOT / raw
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, int], tuple[int, float, int]] = {}
        self._recent: dict[str, deque] = {}
        self._awaiting: dict[str, tuple[int, str, float]] = {}
        self.context_seconds = float(cfg.get("context_seconds", 120))
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS stickers("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, fp TEXT NOT NULL,"
                "description TEXT NOT NULL, meaning TEXT NOT NULL DEFAULT '', meaning_by INTEGER,"
                "evidence TEXT NOT NULL DEFAULT '', seen_count INTEGER NOT NULL DEFAULT 1,"
                "sender INTEGER, first_seen REAL, last_seen REAL)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sticker_scope ON stickers(scope, last_seen)")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ---- 指纹 ----
    @staticmethod
    def fingerprint(path: Path) -> str:
        try:
            from PIL import Image
            with Image.open(path) as img:
                img.seek(0)
                small = img.convert("RGBA")
                background = Image.new("RGBA", small.size, (255, 255, 255, 255))
                gray = Image.alpha_composite(background, small).convert("L").resize((9, 8))
            pixels = gray.tobytes()
        except Exception:
            return ""
        bits = 0
        for row in range(8):
            for col in range(8):
                bits = (bits << 1) | (1 if pixels[row * 9 + col] > pixels[row * 9 + col + 1] else 0)
        return f"{bits:016x}"

    @staticmethod
    def _distance(left: str, right: str) -> int:
        try:
            return bin(int(left, 16) ^ int(right, 16)).count("1")
        except ValueError:
            return 64

    # ---- 查找与记录 ----
    @staticmethod
    def _row(row) -> dict:
        keys = ("id", "scope", "fp", "description", "meaning", "meaning_by", "evidence",
                "seen_count", "sender", "first_seen", "last_seen")
        return dict(zip(keys, row))

    def lookup(self, scope: str, fp: str) -> dict | None:
        """找到近似的旧表情就记一次出现并返回；没有返回 None。"""
        if not self.enabled or not fp or self._conn is None:
            return None
        with self._lock:
            rows = self._conn.execute("SELECT * FROM stickers WHERE scope=?", (scope,)).fetchall()
            best, best_distance = None, self.near_distance + 1
            for row in rows:
                distance = self._distance(fp, row[2])
                if distance < best_distance:
                    best, best_distance = row, distance
            if best is None:
                return None
            self._conn.execute(
                "UPDATE stickers SET seen_count=seen_count+1, last_seen=? WHERE id=?",
                (time.time(), best[0]))
            self._conn.commit()
            entry = self._row(best)
            entry["seen_count"] += 1
            return entry

    def remember(self, scope: str, sender: int, fp: str, description: str) -> int:
        description = (description or "").strip()[:500]
        if not self.enabled or not fp or not description or self._conn is None:
            return 0
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO stickers(scope,fp,description,sender,first_seen,last_seen)"
                " VALUES(?,?,?,?,?,?)", (scope, fp, description, int(sender or 0), now, now))
            self._conn.execute(
                "DELETE FROM stickers WHERE scope=? AND id NOT IN "
                "(SELECT id FROM stickers WHERE scope=? ORDER BY last_seen DESC LIMIT ?)",
                (scope, scope, self.max_per_scope))
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def set_meaning(self, sticker_id: int, sender: int, meaning: str, evidence: str) -> bool:
        meaning = re.sub(r"\s+", " ", meaning or "").strip()[:80]
        if not self.enabled or not sticker_id or not meaning or self._conn is None:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE stickers SET meaning=?, meaning_by=?, evidence=? WHERE id=?",
                (meaning, int(sender or 0), (evidence or "").strip()[:80], sticker_id))
            self._conn.commit()
            return cursor.rowcount > 0

    def taught_by(self, scope: str, sender: int, exclude_id: int = 0, limit: int = 3) -> list[dict]:
        """该发送者在这个范围里亲口说明过含义的表情，最近的几条。"""
        if not self.enabled or self._conn is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM stickers WHERE scope=? AND meaning_by=? AND meaning!='' AND id!=?"
                " ORDER BY last_seen DESC LIMIT ?",
                (scope, int(sender or 0), int(exclude_id or 0), limit)).fetchall()
        return [self._row(row) for row in rows]

    # ---- 刚发的表情 → 等发送者的说明 ----
    def mark_pending(self, scope: str, sender: int, sticker_id: int) -> None:
        if sticker_id:
            self._pending[(scope, int(sender or 0))] = (sticker_id, time.time(), 0)

    def take_pending(self, scope: str, sender: int, text: str) -> int:
        """发送者在时限内又说了话：疑似说明就返回表情 id（最多看 3 句），过期则清掉。"""
        key = (scope, int(sender or 0))
        item = self._pending.get(key)
        if not item:
            return 0
        sticker_id, created, tries = item
        if time.time() - created > self.pending_seconds or tries >= 3:
            self._pending.pop(key, None)
            return 0
        self._pending[key] = (sticker_id, created, tries + 1)
        return sticker_id if EXPLAIN_HINT_RE.search(text or "") else 0

    def clear_pending(self, scope: str, sender: int) -> None:
        self._pending.pop((scope, int(sender or 0)), None)

    # ---- 前后各一句参考 ----
    def note_text(self, scope: str, text: str) -> None:
        """记下这个范围里最近的一句话（只在内存里留一条，作为下一张表情的“上一句”）。"""
        text = re.sub(r"\s+", " ", text or "").strip()[:60]
        if text:
            self._recent[scope] = deque([(time.time(), text)], maxlen=1)

    def before_text(self, scope: str) -> str:
        item = (self._recent.get(scope) or [None])[-1]
        return item[1] if item and time.time() - item[0] <= self.context_seconds else ""

    def await_next(self, scope: str, sticker_id: int, before: str) -> None:
        if sticker_id:
            self._awaiting[scope] = (sticker_id, before, time.time())

    def take_next(self, scope: str, text: str) -> tuple[int, str, str] | None:
        """表情之后的下一句话：返回 (表情 id, 上一句, 下一句)，每张只用一次。"""
        item = self._awaiting.pop(scope, None)
        text = re.sub(r"\s+", " ", text or "").strip()[:60]
        if not item or not text or time.time() - item[2] > self.context_seconds:
            return None
        return item[0], item[1], text

    def refine_guess(self, sticker_id: int, guess: str) -> bool:
        """用上下文修正模型的猜想；发送者已经说明过含义的不改。"""
        guess = re.sub(r"\s+", " ", guess or "").strip("；;。 ")[:40]
        if not self.enabled or not sticker_id or not guess or self._conn is None:
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT description, meaning FROM stickers WHERE id=?", (sticker_id,)).fetchone()
            if not row or row[1]:
                return False
            description = str(row[0])
            if "可能想表达：" in description:
                description = re.sub(r"可能想表达：[^；\n]*", lambda _m: f"可能想表达：{guess}", description, count=1)
            else:
                description = f"{description}；可能想表达：{guess}"
            self._conn.execute("UPDATE stickers SET description=? WHERE id=?", (description[:500], sticker_id))
            self._conn.commit()
            return True

    def description_of(self, sticker_id: int) -> str:
        if self._conn is None:
            return ""
        with self._lock:
            row = self._conn.execute("SELECT description FROM stickers WHERE id=?", (sticker_id,)).fetchone()
        return str(row[0]) if row else ""

    # ---- 注入回复上下文的短注记 ----
    @staticmethod
    def note(entry: dict | None, taught: list[dict], sender: int = 0) -> str:
        parts: list[str] = []
        if entry and int(entry.get("seen_count") or 0) >= 2:
            parts.append("这张表情已经认识，不要当成第一次见到")
        if entry and entry.get("meaning"):
            who = "发送者" if entry.get("meaning_by") == int(sender or 0) else "有人"
            parts.append(f"{who}说明过它的意思：{entry['meaning']}")
        elif taught:
            sample = "；".join(f"{item['description'][:40]}…→{item['meaning']}" for item in taught[:3])
            parts.append(f"发送者之前说明过的表情含义（{sample}）。若这张是同一个角色或同一类表情，按这个含义理解，别当成新图乱猜")
        # 内部提示：只用来理解这张表情，不向对方提起“记得”“见过”“记忆”这类说法。
        return f"（内部参考，不要向对方提起：{'，'.join(parts)}）" if parts else ""
