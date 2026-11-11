from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .sqlite_tuning import tune


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        tune(self._conn)
        self._lock = asyncio.Lock()
        self._summary_epochs: dict[str,int] = {}
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session, id);
                CREATE TABLE IF NOT EXISTS summaries (
                    session TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_facts_session
                    ON facts(session, id);
                """
            )

    async def _run(self, fn, *args, **kwargs):
        async with self._lock:
            return await asyncio.to_thread(fn, *args, **kwargs)

    def _add_message(self, session: str, role: str, content: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO messages(session, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session, role, content, _now()),
            )

    async def add_message(self, session: str, role: str, content: str) -> None:
        await self._run(self._add_message, session, role, content)

    def _remove_message(self, session: str, role: str, content: str) -> bool:
        """删掉这个会话里最近一条完全相同的消息（用户撤回了它）。"""
        with self._conn:
            row = self._conn.execute(
                "SELECT id FROM messages WHERE session = ? AND role = ? AND content = ? ORDER BY id DESC LIMIT 1",
                (session, role, content)).fetchone()
            if row:
                self._conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
            return bool(row)

    async def remove_message(self, session: str, role: str, content: str) -> bool:
        return await self._run(self._remove_message, session, role, content)

    def _recent(self, session: str, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session = ? ORDER BY id DESC LIMIT ?",
            (session, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    async def recent(self, session: str, limit: int = 16) -> list[dict[str, Any]]:
        return await self._run(self._recent, session, limit)

    def _count(self, session: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session = ? AND role = 'user'",
            (session,),
        ).fetchone()
        return int(row["n"] if row else 0)

    async def user_count(self, session: str) -> int:
        return await self._run(self._count, session)

    def _clear(self, session: str) -> int:
        self._summary_epochs[session]=self._summary_epochs.get(session,0)+1
        with self._conn:
            cur = self._conn.execute("DELETE FROM messages WHERE session = ?", (session,))
        return int(cur.rowcount or 0)

    async def clear_session(self, session: str) -> int:
        return await self._run(self._clear, session)

    def _forget_session(self, session: str) -> None:
        self._summary_epochs[session]=self._summary_epochs.get(session,0)+1
        with self._conn:
            for table in ('messages','summaries','facts'):
                self._conn.execute(f'DELETE FROM {table} WHERE session=?',(session,))

    async def forget_session(self, session: str) -> None:
        """Explicit user-data deletion, including private long-term memory."""
        await self._run(self._forget_session,session)

    def _get_summary(self, session: str) -> str:
        row = self._conn.execute(
            "SELECT summary FROM summaries WHERE session = ?", (session,)
        ).fetchone()
        return str(row["summary"]) if row else ""

    async def get_summary(self, session: str) -> str:
        return await self._run(self._get_summary, session)

    def _reply_memory(self, session: str) -> tuple[str, list[str]]:
        return self._get_summary(session), self._facts(session)

    async def reply_memory(self, session: str) -> tuple[str, list[str]]:
        return await self._run(self._reply_memory, session)

    def _set_summary(self, session: str, summary: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO summaries(session, summary, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(session) DO UPDATE SET summary = excluded.summary, "
                "updated_at = excluded.updated_at",
                (session, summary, _now()),
            )

    async def set_summary(self, session: str, summary: str) -> None:
        await self._run(self._set_summary, session, summary)

    def _add_fact(self, session: str, fact: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO facts(session, fact, created_at) VALUES (?, ?, ?)",
                (session, fact, _now()),
            )

    async def add_fact(self, session: str, fact: str) -> None:
        await self._run(self._add_fact, session, fact)

    def _facts(self, session: str, limit: int = 20) -> list[str]:
        rows = self._conn.execute(
            "SELECT fact FROM facts WHERE session = ? ORDER BY id DESC LIMIT ?",
            (session, limit),
        ).fetchall()
        return [str(row["fact"]) for row in reversed(rows)]

    async def facts(self, session: str, limit: int = 20) -> list[str]:
        return await self._run(self._facts, session, limit)

    def _stats(self, session: str) -> dict[str, int]:
        msg = self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session = ?", (session,)
        ).fetchone()
        fact = self._conn.execute(
            "SELECT COUNT(*) AS n FROM facts WHERE session = ?", (session,)
        ).fetchone()
        return {"messages": int(msg["n"]), "facts": int(fact["n"])}

    async def stats(self, session: str) -> dict[str, int]:
        return await self._run(self._stats, session)

    async def maybe_summarize(self, session: str, llm, every: int = 20, max_chars: int = 700) -> str:
        prepared=await self.prepare_summary(session,every,max_chars)
        return await self.summarize_prepared(session,llm,prepared) if prepared else ''

    async def prepare_summary(self, session: str, every: int = 20, max_chars: int = 700) -> dict | None:
        """Snapshot completed messages before yielding the turn; no model work."""
        epoch=self._summary_epochs.get(session,0)
        if every <= 0:
            return None
        count = await self.user_count(session)
        if count == 0 or count % every != 0:
            return None
        recent = await self.recent(session, 60)
        transcript = "\n".join(
            f"{'哥哥' if m['role'] == 'user' else '桃夭'}：{m['content']}" for m in recent
        )
        return {'transcript':transcript,'max_chars':max_chars,'epoch':epoch}

    async def summarize_prepared(self, session: str, llm, prepared: dict) -> str:
        # Called in per-session order so older jobs cannot overwrite newer memory.
        if prepared['epoch']!=self._summary_epochs.get(session,0):return ''
        transcript=prepared['transcript'];max_chars=prepared['max_chars']
        old = await self.get_summary(session)
        prompt = (
            "你是记忆整理器。把下面这段聊天整理成一段简短的中文长期记忆，"
            f"不超过 {max_chars} 字。只保留关于哥哥、桃夭、约定、喜好和重要事件的事实，"
            "不要编造，不要写分析过程。\n\n"
            f"旧记忆：\n{old or '（无）'}\n\n新聊天：\n{transcript}"
        )
        summary = await llm.chat(
            [
                {"role": "system", "content": "你只负责压缩记忆，输出一段中文事实摘要。"},
                {"role": "user", "content": prompt},
            ]
        )
        summary = summary.strip()
        if summary:
            if not await self._run(self._commit_summary,session,summary,prepared['epoch']):return ''
        return summary

    def _commit_summary(self,session,summary,epoch):
        if self._summary_epochs.get(session,0)!=epoch:return False
        self._set_summary(session,summary)
        return True

    def close(self) -> None:
        self._conn.close()
