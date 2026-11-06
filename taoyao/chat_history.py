"""桃夭实际发送的消息；独立于画像和模型上下文。"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path


class ChatHistory:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._lock = asyncio.Lock()

    async def find_message(self, scene: str, target_id: int, message_id: str) -> dict | None:
        if scene not in ('private','group') or not message_id:
            return None
        async with self._lock:
            return await asyncio.to_thread(self._find_message,scene,target_id,message_id)

    def _find_message(self, scene, target_id, message_id):
        path=self.directory/f'{scene}.json'
        if not path.exists():return None
        rows=json.loads(path.read_text(encoding='utf-8')).get(str(target_id),[])
        return next((dict(row) for row in reversed(rows)
                     if str(row.get('message_id') or '')==str(message_id)),None)

    async def record(self, scene: str, target_id: int, message, result: dict,
                     sent_at: float) -> None:
        # 仅确认成功后写入，超时或失败不冒充已发送。
        if result.get("status") != "ok" or result.get("retcode", 0) != 0:
            return
        async with self._lock:
            await asyncio.to_thread(self._record, scene, target_id, message, result, sent_at)

    def _record(self, scene, target_id, message, result, sent_at):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{scene}.json"
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        records = history.setdefault(str(target_id), [])
        data = result.get("data") or {}
        message_id = str(data.get("message_id") or "")
        if message_id and any(row.get("message_id") == message_id for row in records):
            return
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
        records.append({
            "message_id": message_id,
            "sent_at": datetime.fromtimestamp(stamp, timezone(timedelta(hours=8))).isoformat(timespec="milliseconds"),
            "timezone": "Asia/Shanghai",
            "sent_at_utc": datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="milliseconds"),
            "timestamp": stamp,
            "time_source": source,
            "content": content,
            "message": message,
        })
        # 同目录原子替换，避免写到一半退出破坏旧记录。
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
