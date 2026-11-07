"""桃夭实际发送的消息；独立于画像和模型上下文。

每个场景一个只追加的 JSONL 文件（private.jsonl / group.jsonl），一行一条记录并带 target_id，
单次写入的耗时与历史长度无关。旧的 private.json / group.json 在首次访问时一次性转换，
原文件保留、不再修改。写入先 flush 进操作系统，再由后台任务 fsync，不拖慢发送。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


class ChatHistory:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._lock = asyncio.Lock()
        self._seen: dict[str, set[tuple[str, str]]] = {}
        self._torn: dict[str, bool] = {}
        self._dirty: set[str] = set()
        self._sync_task: asyncio.Task | None = None

    def _path(self, scene: str) -> Path:
        return self.directory / f"{scene}.jsonl"

    @staticmethod
    def _parse(line: str) -> dict | None:
        try:
            row = json.loads(line)
        except ValueError:
            return None
        return row if isinstance(row, dict) and "target_id" in row else None

    def _migrate(self, legacy: Path, path: Path) -> None:
        """旧版 {scene}.json（目标 -> 记录数组）一次性转成 JSONL；旧文件原样保留。"""
        try:
            history = json.loads(legacy.read_text(encoding="utf-8"))
            if not isinstance(history, dict):
                raise ValueError("legacy history is not an object")
        except (OSError, ValueError):
            log.exception("旧聊天记录无法转换，保留原文件并从空记录开始：%s", legacy)
            return
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for target_id, rows in history.items():
                for row in rows if isinstance(rows, list) else []:
                    if isinstance(row, dict):
                        stream.write(json.dumps({"target_id": str(target_id), **row},
                                                ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        log.info("聊天记录已转换为追加写入格式：%s", path.name)

    def _ensure(self, scene: str) -> None:
        """首次访问某个场景：必要时迁移旧文件，并读一遍建立去重表。"""
        if scene in self._seen:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path, legacy = self._path(scene), self.directory / f"{scene}.json"
        if not path.exists() and legacy.exists():
            self._migrate(legacy, path)
        seen: set[tuple[str, str]] = set()
        torn = False
        if path.exists():
            raw = path.read_bytes()
            torn = bool(raw) and not raw.endswith(b"\n")
            for line in raw.decode("utf-8", "replace").splitlines():
                row = self._parse(line)
                if row and row.get("message_id"):
                    seen.add((str(row["target_id"]), str(row["message_id"])))
        self._seen[scene], self._torn[scene] = seen, torn

    async def find_message(self, scene: str, target_id: int, message_id: str) -> dict | None:
        if scene not in ('private', 'group') or not message_id:
            return None
        async with self._lock:
            return await asyncio.to_thread(self._find_message, scene, target_id, message_id)

    def _find_message(self, scene, target_id, message_id):
        self._ensure(scene)
        path = self._path(scene)
        if not path.exists():
            return None
        wanted, mid = str(target_id), str(message_id)
        for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if mid not in line:
                continue
            row = self._parse(line)
            if row and str(row["target_id"]) == wanted and str(row.get("message_id") or "") == mid:
                row.pop("target_id", None)
                return row
        return None

    async def record(self, scene: str, target_id: int, message, result: dict,
                     sent_at: float) -> None:
        # 仅确认成功后写入，超时或失败不冒充已发送。
        if result.get("status") != "ok" or result.get("retcode", 0) != 0:
            return
        async with self._lock:
            wrote = await asyncio.to_thread(self._record, scene, target_id, message, result, sent_at)
        if wrote:
            self._schedule_sync(scene)

    def _record(self, scene, target_id, message, result, sent_at) -> bool:
        self._ensure(scene)
        data = result.get("data") or {}
        message_id = str(data.get("message_id") or "")
        key = (str(target_id), message_id)
        if message_id and key in self._seen[scene]:
            return False
        # QQ 返回时间优先；否则保存成功发送请求的本机发起时间，并注明来源。
        stamp = data.get("time")
        source = "qq" if isinstance(stamp, (int, float)) and stamp > 0 else "local_send"
        stamp = stamp if source == "qq" else sent_at
        if isinstance(message, str):
            content = message
        else:
            content = "".join(
                str(seg.get("data", {}).get("text", "")) if seg.get("type") == "text"
                else f"[{seg.get('type', 'unknown')}]" for seg in message
            )
        line = json.dumps({
            "target_id": str(target_id),
            "message_id": message_id,
            "sent_at": datetime.fromtimestamp(stamp, timezone(timedelta(hours=8))).isoformat(timespec="milliseconds"),
            "timezone": "Asia/Shanghai",
            "sent_at_utc": datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="milliseconds"),
            "timestamp": stamp,
            "time_source": source,
            "content": content,
            "message": message,
        }, ensure_ascii=False) + "\n"
        # 上次若在写到一半时退出，最后一行不完整：先补换行，避免和新记录粘在一起。
        prefix = "\n" if self._torn[scene] else ""
        with self._path(scene).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(prefix + line)
            stream.flush()
        self._torn[scene] = False
        if message_id:
            self._seen[scene].add(key)
        return True

    def _schedule_sync(self, scene: str) -> None:
        self._dirty.add(scene)
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.get_running_loop().create_task(self._sync_loop())

    async def _sync_loop(self) -> None:
        while self._dirty:
            scene = self._dirty.pop()
            try:
                await asyncio.to_thread(self._fsync, scene)
            except Exception:
                log.exception("聊天记录写入磁盘确认失败：%s", scene)

    def _fsync(self, scene: str) -> None:
        # Windows 上 fsync 需要可写句柄。
        with self._path(scene).open("ab") as stream:
            os.fsync(stream.fileno())

    async def flush(self) -> None:
        """等待后台 fsync 完成（测试和退出时使用）。"""
        task = self._sync_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
