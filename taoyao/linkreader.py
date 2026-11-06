from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import logging
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

LINK_RE = re.compile(r"https?://[^\s<>\"'）)】,，。；;！!？?、]+", re.IGNORECASE)


_MEDIA_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg",
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".mp3", ".m4a", ".wav",
)


def extract_links(text: str) -> list[str]:
    links = []
    for match in LINK_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,;:!?，。；：！？、")
        if url.lower().split("?")[0].endswith(_MEDIA_SUFFIXES):
            continue
        if url not in links:
            links.append(url)
    return links


@dataclass
class LinkResult:
    url: str
    title: str = ""
    text: str = ""
    error: str = ""


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "title":
            self.in_title = True
        if tag in ("script", "style", "noscript", "svg"):
            self.skip_depth += 1

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self.in_title = False
        if tag in ("script", "style", "noscript", "svg") and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if self.skip_depth == 0 and data.strip():
            self.parts.append(data.strip())


class LinkReader:
    def __init__(self, config: dict[str, Any] | None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.timeout = float(cfg.get("timeout", 15))
        self.max_chars = int(cfg.get("max_chars", 2000))
        self.max_bytes = int(cfg.get("max_bytes", 1024 * 1024))
        self.max_urls = max(1, int(cfg.get("max_urls", 2)))
        self.user_agent = str(
            cfg.get("user_agent")
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
        )

    def _safe_host(self, url: str) -> tuple[bool, str]:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, "只支持 http/https"
        if not parsed.hostname:
            return False, "没有域名"
        if parsed.port not in (None, 80, 443):
            return False, "只允许 80/443 端口"
        try:
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        except Exception as exc:
            return False, f"域名解析失败：{exc}"
        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                return False, "地址不合法"
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                return False, "不允许访问内网地址"
        return True, ""

    async def fetch(self, url: str) -> LinkResult:
        if not self.enabled:
            return LinkResult(url, error="链接读取没开")
        ok, why = await asyncio.to_thread(self._safe_host, url)
        if not ok:
            return LinkResult(url, error=why)
        headers = {"User-Agent": self.user_agent, "Accept": "text/html,text/plain,*/*"}
        current = url
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False
            ) as client:
                response = None
                for _ in range(4):
                    response = await client.get(current, headers=headers)
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            break
                        current = urljoin(current, location)
                        ok, why = await asyncio.to_thread(self._safe_host, current)
                        if not ok:
                            return LinkResult(url, error=f"跳转被拒绝：{why}")
                        continue
                    break
                if response is None:
                    return LinkResult(url, error="没有响应")
                if response.status_code >= 400:
                    return LinkResult(url, error=f"HTTP {response.status_code}")
                body = response.content[: self.max_bytes]
                content_type = response.headers.get("content-type", "").lower()
                charset = response.encoding or "utf-8"
                try:
                    raw = body.decode(charset, errors="ignore")
                except LookupError:
                    raw = body.decode("utf-8", errors="ignore")
                if "html" in content_type or raw.lstrip().startswith("<"):
                    parser = _TextParser()
                    parser.feed(raw)
                    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()
                    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
                else:
                    title = ""
                    text = re.sub(r"\s+", " ", raw).strip()
                text = html.unescape(text)[: self.max_chars]
                return LinkResult(current, title=title[:200], text=text)
        except httpx.HTTPError as exc:
            return LinkResult(url, error=f"抓取失败：{exc}")
        except Exception as exc:
            return LinkResult(url, error=f"抓取失败：{exc}")

    async def fetch_text(self, text: str) -> str:
        links = extract_links(text)[: self.max_urls]
        if not links:
            return ""
        results = await asyncio.gather(*(self.fetch(url) for url in links))
        parts = []
        for result in results:
            if result.text:
                title = f"《{result.title}》" if result.title else ""
                parts.append(f"[链接内容 {result.url}] {title}\n{result.text}")
            else:
                logging.getLogger(__name__).debug("链接读取失败：%s", result.error or "无正文")
                parts.append(
                    f"[链接 {result.url} 暂未读到正文。此为内部状态，不要复述系统报错或猜测网页内容；"
                    "只有用户明确要求读取该链接时，才自然说明暂时没看清。]"
                )
        return "\n\n".join(parts)
