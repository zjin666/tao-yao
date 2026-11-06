"""本地交互聊天：在终端里手动输入消息，模拟 owner / 普通好友 / 群友。

用法（在项目根目录）：
  python test/chat-local.py                 # 以 owner 身份私聊
  python test/chat-local.py --role user     # 以普通好友身份私聊
  python test/chat-local.py --group-id 30001  # 以群友身份在群里 @ 桃夭

机器人要先在另一个窗口跑起来。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parent.parent


def load_owner() -> int:
    try:
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
        return int(cfg.get("owner_qq") or 0)
    except Exception:
        return 0


def event(text: str, user_id: int, self_id: int = 10000, group_id: int = 0, nickname: str = "") -> dict:
    item = {
        "time": int(time.time()),
        "self_id": self_id,
        "post_type": "message",
        "message_id": int(time.time() * 1000) % 1000000,
        "user_id": user_id,
        "font": 0,
        "sender": {"user_id": user_id, "nickname": nickname or "user"},
    }
    if group_id:
        item.update({
            "message_type": "group",
            "sub_type": "normal",
            "group_id": group_id,
            "raw_message": f"[CQ:at,qq={self_id}] {text}",
            "message": [
                {"type": "at", "data": {"qq": str(self_id)}},
                {"type": "text", "data": {"text": " " + text}},
            ],
        })
    else:
        item.update({
            "message_type": "private",
            "sub_type": "friend",
            "raw_message": text,
            "message": [{"type": "text", "data": {"text": text}}],
        })
    return item


def action_text(data: dict) -> str:
    params = data.get("params") or {}
    message = params.get("message")
    out = ""
    if isinstance(message, list):
        for seg in message:
            if seg.get("type") == "text":
                out += (seg.get("data") or {}).get("text", "")
            elif seg.get("type") == "image":
                out += "[图片]"
    return out or str(message)


async def receive_until_quiet(ws, seconds: float = 20.0, quiet: float = 1.5) -> None:
    end = time.time() + seconds
    last = time.time()
    got = False
    while time.time() < end:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.2, min(quiet, end - time.time())))
        except asyncio.TimeoutError:
            if got and time.time() - last >= quiet:
                break
            continue
        data = json.loads(raw)
        if not data.get("action"):
            continue
        print(f"桃夭> {action_text(data)}")
        got = True
        last = time.time()
        await ws.send(json.dumps({
            "status": "ok",
            "retcode": 0,
            "data": {"message_id": 1},
            "echo": data.get("echo"),
        }, ensure_ascii=False))


async def main() -> int:
    parser = argparse.ArgumentParser(description="本地交互聊天")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6199)
    parser.add_argument("--token", default="")
    parser.add_argument("--role", choices=["owner", "user"], default="owner")
    parser.add_argument("--user-id", type=int, default=0, help="普通好友测试号；默认 owner+1")
    parser.add_argument("--user-offset", type=int, default=1, help="普通好友测试号相对 owner 的偏移")
    parser.add_argument("--group-id", type=int, default=0)
    args = parser.parse_args()

    owner = load_owner()
    if not owner:
        print("owner_qq 还没设置，先双击 setup-owner.cmd。")
        return 1
    if args.role == "owner":
        user_id = owner
    else:
        user_id = args.user_id or (owner + args.user_offset)

    where = f"群 {args.group_id}" if args.group_id else "私聊"
    print(f"角色：{args.role}（不显示 QQ）  {where}")
    print("输入消息，回车发送；输入 /quit 退出。")
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    try:
        async with connect(f"ws://{args.host}:{args.port}", additional_headers=headers) as ws:
            while True:
                try:
                    text = await asyncio.to_thread(input, "你> ")
                except (EOFError, KeyboardInterrupt):
                    break
                text = text.strip()
                if not text:
                    continue
                if text in ("/quit", "quit", "exit"):
                    break
                nickname = "哥哥" if args.role == "owner" and not args.group_id else ("群友" if args.group_id else "普通好友")
                await ws.send(json.dumps(event(text, user_id, group_id=args.group_id, nickname=nickname), ensure_ascii=False))
                await receive_until_quiet(ws)
                # 普通用户有 2 秒频率限制，本地手动测试时自动错开。
                if args.role == "user" and not args.group_id:
                    await asyncio.sleep(0.8)
    except OSError as exc:
        print(f"连不上机器人：{exc}")
        print("先在另一个窗口启动 start-bot.cmd 或 start-bot-mock.cmd。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
