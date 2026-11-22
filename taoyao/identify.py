"""被要求“再试一次”或第一次认错后，让 Qwen 直接看图认角色。

平时识图仍走 DeepSeek（只把图变成文字描述）；这里只在再次观察时用，因为把图变成文字会丢掉“她是谁”这类信息，
而 Qwen 直接看图、对动漫游戏角色了解更多。只把裁出来的角色区域发给它，不发整张聊天截图。
"""
from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from .llm import collect_stream

log = logging.getLogger("taoyao.identify")

PROMPT = (
    "这张图里的角色是谁？出自哪个作品或游戏？给出你认为最可能的角色（最多两个候选）和作品，并写明把握程度（高、中、低）"
    "以及依据的最主要的几个特征；它可能是同人画作，换了服装或画风也要考虑原角色。"
    "完全没有头绪才说认不出。不要编造，不要写成列表，控制在 140 字以内。"
)


class CharacterIdentifier:
    def __init__(self, config: dict[str, Any]):
        qwen = config.get("qwen") or {}
        relook = config.get("relook") or {}
        self.enabled = bool(relook.get("identify", True)) and bool(qwen)
        self.api_base = str(qwen.get("api_base") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        main_model = str(qwen.get("model") or "")
        # 先用配置的识图模型（更强、速度要靠关掉思考），调用失败再退回主文本模型。
        self.models = [m for m in dict.fromkeys([str(relook.get("identify_model") or ""), main_model]) if m]
        self.model = self.models[0] if self.models else ""
        self.timeout = float(relook.get("identify_timeout_seconds", 45))
        self._key = ""
        key_file = Path(str(qwen.get("key_file") or ""))
        try:
            if key_file.exists():
                self._key = key_file.read_text(encoding="utf-8").strip()
        except OSError:
            self._key = ""
        if not (self._key and self.models):
            self.enabled = False

    async def identify(self, image: Path, rejected: str = "") -> str:
        """返回 Qwen 的判断（一两句话）；没配置或失败返回空字符串。rejected 是已被对方否定的猜测，要求不再给。"""
        if not self.enabled or not image.exists():
            return ""
        prompt = PROMPT + (f"之前有人猜过“{rejected[:80]}”，对方说不对，不要再给这个答案。" if rejected else "")
        mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
        data = base64.b64encode(image.read_bytes()).decode("ascii")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for model in self.models:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]}],
                "temperature": 0.2,
                "max_tokens": 500,
                "enable_thinking": False,      # 大模型默认会先长时间思考（实测 200 秒），关掉后约 12 秒
            }
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=self.timeout)) as client:
                    result = await collect_stream(client, f"{self.api_base}/chat/completions", headers, payload)
                return str(result["choices"][0]["message"].get("content") or "").strip()
            except Exception as exc:
                log.warning("Qwen（%s）直接看图认角色失败：%s", model, exc)
        return ""
