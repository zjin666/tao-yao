"""连发 3 条消息，验证按用户批量收集、顺序回复。"""
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
        "sender": {"user_id": user_id, "nickname": "普通好友"},
        "message": [{"type": "text", "data": {"text": text}}],
    }


async def main() -> None:
    owner = owner_qq()
    if not owner:
        print("owner_qq 未设置")
        return
    uid = owner + 1
    texts = ["你好", "我今天很难过", "你能陪陪我吗？"]
    async with connect("ws://127.0.0.1:6199") as ws:
        for text in texts:
            print("发送：", text)
            await ws.send(json.dumps(event(text, uid), ensure_ascii=False))
            await asyncio.sleep(0.15)
        end = time.time() + 20
        while time.time() < end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.2, end - time.time()))
            except asyncio.TimeoutError:
                break
            data = json.loads(raw)
            if data.get("action") != "send_private_msg":
                continue
            msg = ""
            for seg in (data.get("params") or {}).get("message") or []:
                if seg.get("type") == "text":
                    msg += (seg.get("data") or {}).get("text", "")
                elif seg.get("type") == "image":
                    msg += "[图片]"
            print("回复：", msg)
            await ws.send(json.dumps({
                "status": "ok", "retcode": 0, "data": {"message_id": 1},
                "echo": data.get("echo"),
            }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
