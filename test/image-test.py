"""模拟一条带图片的 QQ 消息，并重复发送同一个 message_id。"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parent.parent


def main_owner() -> int:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    return int(cfg.get("owner_qq") or 0)


def event(user_id: int, text: str, image: str, mid: str) -> dict:
    return {
        "time": int(time.time()),
        "self_id": 10000,
        "post_type": "message",
        "message_type": "private",
        "sub_type": "friend",
        "message_id": mid,
        "user_id": user_id,
        "raw_message": text,
        "font": 0,
        "sender": {"user_id": user_id, "nickname": "图片测试"},
        "message": [
            {"type": "text", "data": {"text": text}},
            {"type": "image", "data": {"path": image}},
        ],
    }


async def main() -> None:
    owner = main_owner()
    uid = owner + 1
    image = str(Path(r"D:\tao-yao\data\media\ocr-test.png"))
    async with connect("ws://127.0.0.1:6199") as ws:
        payload = event(uid, "看看这张图", image, "img-dup-1")
        await ws.send(json.dumps(payload, ensure_ascii=False))
        await asyncio.sleep(0.2)
        await ws.send(json.dumps(payload, ensure_ascii=False))
        end = time.time() + 40
        replies = 0
        while time.time() < end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(0.2, end - time.time()))
            except asyncio.TimeoutError:
                break
            data = json.loads(raw)
            if data.get("action") != "send_private_msg":
                continue
            replies += 1
            text = ""
            for seg in (data.get("params") or {}).get("message") or []:
                if seg.get("type") == "text":
                    text += (seg.get("data") or {}).get("text", "")
                elif seg.get("type") == "image":
                    text += "[图片]"
            print(f"回复 {replies}: {text}")
            await ws.send(json.dumps({
                "status": "ok", "retcode": 0, "data": {"message_id": 1},
                "echo": data.get("echo"),
            }, ensure_ascii=False))
        print("总回复动作:", replies)


if __name__ == "__main__":
    asyncio.run(main())
