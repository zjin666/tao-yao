from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, AsyncIterator

import httpx


class LLMError(RuntimeError):
    pass


async def collect_stream(client, url, headers, payload):
    """Collect a complete SSE response; partial/truncated decisions must never pass."""
    chunks = []
    metadata = {}
    finish = None
    done = False
    async with client.stream("POST", url, headers=headers, json={**payload, "stream": True}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise ValueError("Invalid streaming response")
            raw = line[5:].strip()
            if raw == "[DONE]":
                done = True
                break
            obj = json.loads(raw)
            if not isinstance(obj, dict) or obj.get("error"):
                raise ValueError("Invalid streaming response")
            for key in ("id", "model", "usage"):
                if obj.get(key) is not None:
                    metadata[key] = obj[key]
            for choice in obj.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                content = (choice.get("delta") or {}).get("content")
                if content is not None:
                    if not isinstance(content, str) or finish is not None:
                        raise ValueError("Invalid streaming content")
                    chunks.append(content)
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
    if not done or finish != "stop" or not "".join(chunks).strip():
        raise ValueError("Incomplete streaming response")
    return {**metadata, "choices": [{"finish_reason": finish,
        "message": {"content": "".join(chunks)}}]}


class DeepSeekLLM:
    provider_name = 'DeepSeek'
    default_api_base = 'https://api.deepseek.com/v1'
    default_model = 'deepseek-chat'

    def thinking_payload(self):
        return {'thinking': {'type': 'enabled' if self.thinking else 'disabled'}} if isinstance(self.thinking, bool) else {}

    def __init__(self, config: dict[str, Any]):
        self.api_base = str(config.get("api_base") or self.default_api_base).rstrip("/")
        self.model = str(config.get("model") or self.default_model)
        self.temperature = float(config.get("temperature", 1.0))
        self.max_tokens = int(config.get("max_tokens", 600))
        self.thinking = config.get("thinking")
        self.timeout = float(config.get("timeout", 60))
        key_file = Path(str(config.get("key_file") or ""))
        if not key_file.exists():
            raise LLMError(f"{self.provider_name} key 文件不存在：{key_file}")
        self.api_key = key_file.read_text(encoding="utf-8").strip()
        if not self.api_key:
            raise LLMError(f"{self.provider_name} key 文件是空的：{key_file}")
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload.update(self.thinking_payload())
        try:
            async with self._client.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code == 401:
                    raise LLMError(f"{self.provider_name} key 不对或过期了。")
                if response.status_code == 402:
                    raise LLMError(f"{self.provider_name} 余额不足或账户不可用。")
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "ignore")[:300]
                    raise LLMError(f"{self.provider_name} 返回 {response.status_code}：{detail}")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield str(content)
        except httpx.HTTPError as exc:
            raise LLMError(f"连不上 {self.provider_name}：{exc}") from exc

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            **self.thinking_payload(),
        }
        try:
            data = await collect_stream(self._client, f"{self.api_base}/chat/completions",
                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, payload)
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"{self.provider_name} 辅助调用失败或返回不完整") from exc
        return data["choices"][0]["message"]["content"].strip()

    async def close(self) -> None:
        await self._client.aclose()


class QwenLLM(DeepSeekLLM):
    """DashScope OpenAI-compatible text client; never yields reasoning_content."""
    provider_name = 'Qwen'
    default_api_base = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    default_model = 'qwen3.7-plus'

    def thinking_payload(self):
        return {'enable_thinking': self.thinking if isinstance(self.thinking, bool) else False}


def llm_request_settings(config):
    provider = str(config.get('text_provider', 'deepseek')).lower()
    if provider not in ('deepseek', 'qwen'):
        raise LLMError('text_provider 只能是 deepseek 或 qwen')
    return {'provider': provider, 'settings': dict(config.get(provider) or {})}


class MockLLM:
    """开发用：不花钱、不需要 key，专门验证 QQ 消息链路。"""

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        text = await self.chat(messages)
        for index in range(0, len(text), 4):
            yield text[index : index + 4]

    REPLIES = [
        "……嗯，我在听。哥哥今天怎么想起找我了？ [[meme:reply]]",
        "哼，才没有一直在等你。 [[meme:shy]]",
        "哥哥是笨蛋。……不过，这次就算了。 [[meme:affection]]",
    ]

    async def chat(self, messages: list[dict[str, str]]) -> str:
        text = ""
        system = ""
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
        for msg in reversed(messages):
            if msg.get("role") == "user":
                text = msg.get("content", "")
                break
        is_outsider = "普通 QQ 好友" in system or "群友" in system
        if is_outsider:
            if "你好" in text:
                return "你好，有什么事吗？ [[meme:reply]]"
            if "难过" in text or "累" in text:
                return "听起来你很难过，先歇一会儿吧。 [[meme:reply]]"
            if "陪" in text:
                return "这个……你找哥哥去吧。 [[meme:shy]]"
            if "在吗" in text:
                return "……你好。找我有事吗？ [[meme:reply]]"
            return random.choice([
                "嗯，我在听。 [[meme:reply]]",
                "这个……你问哥哥去吧。 [[meme:shy]]",
                "我不太清楚。 [[meme:confused]]",
            ])
        if "多说" in text or "长一点" in text:
            return "哥哥今天怎么了？其实我有点想你。不过只有一点点。"
        if "在吗" in text:
            return "我在呢，哥哥，怎么啦？ [[meme:淡定]]"
        if "累" in text:
            return "那就先靠一会儿。……我才不是在关心你。 [[meme:抱抱]]"
        if "开心" in text or "高兴" in text:
            return "哼，算你今天运气好。 [[meme:开心]]"
        if "睡" in text:
            return "再睡五分钟……就五分钟。 [[meme:困倦]]"
        return random.choice(self.REPLIES)

    async def close(self) -> None:
        return None


def create_llm(config: dict[str, Any], mock: bool = False, *, provider: str | None = None):
    if mock or config.get("mock_llm"):
        return MockLLM()
    selected = llm_request_settings({**config, **({'text_provider': provider} if provider else {})})
    client_class = QwenLLM if selected['provider'] == 'qwen' else DeepSeekLLM
    return client_class(selected['settings'])
