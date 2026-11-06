"""测量单条消息从发送到第一条回复、以及全部回复的耗时。"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parent.parent


def owner_qq() -> int:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    return int(cfg.get("owner_qq") or 0)


def event(text: str, user_id: int, self_id: int = 10000) -> dict:
    return {
        "time": int(time.time()),
        "self_id": self_id,
        "post_type": "message",
        "message_type": "private",
        "sub_type": "friend",
        "message_id": int(time.time() * 1000) % 1000000,
        "user_id": user_id,
        "raw_message": text,
        "font": 0,
        "sender": {"user_id": user_id, "nickname": "延迟测试"},
        "message": [{"type": "text", "data": {"text": text}}],
    }


async def main() -> None:
    owner = owner_qq()
    uid = owner + 5
    async with connect("ws://127.0.0.1:6199") as ws:
        t0 = time.perf_counter()
        await ws.send(json.dumps(event("你好，今天过得怎么样？", uid), ensure_ascii=False))
        first = None
        actions = 0
        end = t0 + 30
        while time.perf_counter() < end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.2, end - time.perf_counter()))
            except asyncio.TimeoutError:
                break
            data = json.loads(raw)
            if data.get("action") != "send_private_msg":
                continue
            actions += 1
            now = time.perf_counter()
            if first is None:
                first = now - t0
            text = ""
            for seg in (data.get("params") or {}).get("message") or []:
                if seg.get("type") == "text":
                    text += (seg.get("data") or {}).get("text", "")
                elif seg.get("type") == "image":
                    text += "[图片]"
            print(f"第 {actions} 条 +{now - t0:.2f}s: {text}")
            await ws.send(json.dumps({
                "status": "ok", "retcode": 0, "data": {"message_id": 1},
                "echo": data.get("echo"),
            }, ensure_ascii=False))
        print(f"首条回复：{first:.2f}s" if first is not None else "没有收到回复")
        print(f"全部动作：{actions} 条")


if __name__ == "__main__":
    asyncio.run(main())
