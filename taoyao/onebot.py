from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, urlparse

from websockets.asyncio.server import serve

from .chat_history import ChatHistory

log = logging.getLogger("taoyao.onebot")


class OneBotError(RuntimeError):
    pass


class OneBotServer:
    """Minimal OneBot 11 reverse-WebSocket server.

    SnowLuma/NapCat connects to us; we receive events and send actions.
    """

    def __init__(
        self,
        host: str,
        port: int,
        access_token: str,
        on_event: Callable[[dict[str, Any]], Awaitable[None]],
    ):
        self.host = host
        self.port = port
        self.access_token = access_token.strip()
        self.on_event = on_event
        self.ws = None
        self.connected = asyncio.Event()
        self._pending: dict[str, asyncio.Future] = {}
        self._tasks: set[asyncio.Task] = set()
        self._server = None
        self.chat_history = ChatHistory(Path(__file__).resolve().parent.parent / "data" / "taoyao_chat_history")

    async def start(self) -> None:
        self._server = await serve(
            self._handler,
            self.host,
            self.port,
            max_size=32 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        log.info("OneBot 反向 WS 已监听 ws://%s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for task in list(self._tasks):
            task.cancel()
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    def _authorized(self, ws) -> bool:
        if not self.access_token:
            return True
        header = ""
        request = getattr(ws, "request", None)
        headers = getattr(request, "headers", None) if request is not None else None
        if headers is not None:
            header = headers.get("Authorization") or headers.get("authorization") or ""
        if header.startswith("Bearer "):
            header = header[7:].strip()
        query = parse_qs(urlparse(getattr(request, "path", "") or "").query)
        token = header or (query.get("access_token") or [""])[0]
        return token == self.access_token

    async def _handler(self, ws) -> None:
        if not self._authorized(ws):
            log.warning("OneBot 连接被拒绝：token 不对")
            await ws.close(code=1008, reason="unauthorized")
            return
        self.ws = ws
        self.connected.set()
        log.info("OneBot 已连接：%s", getattr(ws, "remote_address", "?"))
        try:
            async for raw in ws:
                await self._dispatch(raw)
        except Exception as exc:
            log.warning("OneBot 连接断开：%s", exc)
        finally:
            if self.ws is ws:
                self.ws = None
                self.connected.clear()
            log.info("OneBot 已断开")

    async def _dispatch(self, raw: str | bytes) -> None:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            data = json.loads(raw)
        except Exception:
            log.warning("收到非 JSON 消息：%r", str(raw)[:200])
            return

        echo = data.get("echo")
        if echo is not None and echo in self._pending:
            fut = self._pending[echo]
            if not fut.done():
                fut.set_result(data)
            return

        if data.get("post_type"):
            task = asyncio.create_task(self._safe_event(data))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return

        log.debug("忽略未知消息：%s", str(data)[:200])

    async def _safe_event(self, data: dict[str, Any]) -> None:
        try:
            await self.on_event(data)
        except Exception:
            log.exception("处理 OneBot 事件出错")

    async def call(self, action: str, params: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
        if self.ws is None:
            raise OneBotError("OneBot 还没连接")
        echo = uuid.uuid4().hex
        payload = {"action": action, "params": params, "echo": echo}
        fut = asyncio.get_running_loop().create_future()
        self._pending[echo] = fut
        sent_at = time.time()
        try:
            await self.ws.send(json.dumps(payload, ensure_ascii=False))
            result = await asyncio.wait_for(fut, timeout=timeout)
            scene = {"send_private_msg": "private", "send_group_msg": "group"}.get(action)
            if action == "send_msg":
                scene = params.get("message_type") or ("group" if params.get("group_id") else "private")
            if scene in ("private", "group"):
                target_id = params.get("group_id" if scene == "group" else "user_id")
                try:
                    await self.chat_history.record(scene, int(target_id), params.get("message", []), result, sent_at)
                except Exception:
                    log.exception("桃夭消息已发出，但独立聊天记录写入失败")
            if isinstance(result, dict) and result.get("status") == "failed":
                log.warning("OneBot 动作失败：%s %s", action, result)
            return result
        finally:
            self._pending.pop(echo, None)

    async def send_private(self, user_id: int, message) -> dict[str, Any]:
        return await self.call(
            "send_private_msg",
            {"user_id": int(user_id), "message": message},
        )

    async def send_group(self, group_id: int, message) -> dict[str, Any]:
        return await self.call(
            "send_group_msg",
            {"group_id": int(group_id), "message": message},
        )

    async def send_text(
        self, target: str, target_id: int, text: str, at_user_id: int | None = None,
        quote_id: str | int | None = None,
    ) -> dict[str, Any]:
        message = []
        if target == "group" and quote_id:
            # QQ 的引用回复：点明回的是哪一条，对方也会收到提醒，所以不再另加 @。
            message.append({"type": "reply", "data": {"id": str(quote_id)}})
        elif target == "group" and at_user_id:
            message.append({"type": "at", "data": {"qq": str(int(at_user_id))}})
            message.append({"type": "text", "data": {"text": " "}})
        message.append({"type": "text", "data": {"text": text}})
        if target == "group":
            return await self.send_group(target_id, message)
        return await self.send_private(target_id, message)

    async def send_image(self, target: str, target_id: int, path: str | Path) -> dict[str, Any]:
        p = Path(path).resolve()
        uri = p.as_uri()
        message = [{"type": "image", "data": {"file": uri}}]
        result = (
            await self.send_group(target_id, message)
            if target == "group"
            else await self.send_private(target_id, message)
        )
        if isinstance(result, dict) and result.get("status") == "failed":
            # Some OneBot implementations do not accept file://; retry with base64.
            import base64

            encoded = base64.b64encode(p.read_bytes()).decode("ascii")
            message = [{"type": "image", "data": {"file": f"base64://{encoded}"}}]
            result = (
                await self.send_group(target_id, message)
                if target == "group"
                else await self.send_private(target_id, message)
            )
        return result
