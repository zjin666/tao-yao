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

BILI_ID_RE = re.compile(r"/video/(BV[0-9A-Za-z]{10}|av\d+)", re.IGNORECASE)
BILI_HOSTS = ("bilibili.com", "b23.tv")
LINK_RE = re.compile(r"https?://[^\s<>\"'）)】,，。；;！？、]+", re.IGNORECASE)


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
        self.video_watcher = None          # 由应用设置：看视频画面（VideoWatcher.watch）
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

    @staticmethod
    def _is_bilibili(url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == h or host.endswith("." + h) for h in BILI_HOSTS)

    async def _fetch_bilibili(self, url: str) -> LinkResult | None:
        """B 站视频：普通网页抓取只会得到“视频去哪了”的空壳，改走官方接口读标题/简介/作者/分P。
        拿不到就返回 None，让调用方按普通网页处理。"""
        headers = {"User-Agent": self.user_agent, "Referer": "https://www.bilibili.com/"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=headers, follow_redirects=False) as client:
                current = url
                for _ in range(4):                       # b23.tv 短链：手动跟随跳转并逐跳检查地址
                    match = BILI_ID_RE.search(urlparse(current).path)
                    if match or not self._is_bilibili(current):
                        break
                    response = await client.get(current)
                    location = response.headers.get("location")
                    if response.status_code not in (301, 302, 303, 307, 308) or not location:
                        break
                    current = urljoin(current, location)
                    ok, _ = await asyncio.to_thread(self._safe_host, current)
                    if not ok or not self._is_bilibili(current):
                        return None
                match = BILI_ID_RE.search(urlparse(current).path)
                if not match:
                    return None
                vid = match.group(1)
                params = {"aid": vid[2:]} if vid.lower().startswith("av") else {"bvid": vid}
                await client.get("https://www.bilibili.com/")      # 取一次 cookie，否则接口会 412
                data = (await client.get("https://api.bilibili.com/x/web-interface/view", params=params)).json()
        except Exception as exc:
            logging.getLogger(__name__).debug("B站读取失败：%s", exc)
            return None
        info = data.get("data") if data.get("code") == 0 else None
        if not info:
            return LinkResult(url, error=f"B站视频不可读（code={data.get('code')}）")
        pages = [str(p.get("part", "")).strip() for p in (info.get("pages") or []) if p.get("part")]
        stat = info.get("stat") or {}
        lines = [
            f"B站视频，作者：{(info.get('owner') or {}).get('name', '')}，分区：{info.get('tname', '')}，"
            f"时长 {int(info.get('duration') or 0)} 秒，播放 {stat.get('view', 0)}、点赞 {stat.get('like', 0)}。",
        ]
        if str(info.get("desc", "")).strip() and info.get("desc") != "-":
            lines.append("简介：" + re.sub(r"\s+", " ", str(info["desc"])).strip())
        if len(pages) > 1:
            lines.append("分P：" + "；".join(pages[:8]))
        return LinkResult(current, title=str(info.get("title", ""))[:200], text="\n".join(lines)[: self.max_chars])

    async def fetch(self, url: str) -> LinkResult:
        if not self.enabled:
            return LinkResult(url, error="链接读取没开")
        if self._is_bilibili(url):
            ok, why = await asyncio.to_thread(self._safe_host, url)
            if not ok:
                return LinkResult(url, error=why)
            result = await self._fetch_bilibili(url)
            if result is not None:
                return result
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

    async def _watch(self, url: str) -> str:
        watcher = self.video_watcher
        # B站的标题、作者、简介已经由专用接口读到，视频部分只需要画面概括。
        return await watcher.watch(url, with_meta=not self._is_bilibili(url)) if watcher is not None else ""

    async def fetch_text(self, text: str) -> str:
        links = extract_links(text)[: self.max_urls]
        if not links:
            return ""
        pages, watched = await asyncio.gather(
            asyncio.gather(*(self.fetch(url) for url in links)),
            asyncio.gather(*(self._watch(url) for url in links)),
        )
        parts = []
        for result, video in zip(pages, watched):
            if result.text or video:
                title = f"《{result.title}》" if result.title else ""
                body = result.text
                if video:
                    body = f"{body}\n{video}".strip()
                elif self._is_bilibili(result.url) and result.text:
                    body += "\n（只读到标题、简介等文字信息，没有看视频画面和声音。）"
                parts.append(f"[链接内容 {result.url}] {title}\n{body}")
            else:
                logging.getLogger(__name__).debug("链接读取失败：%s", result.error or "无正文")
                parts.append(
                    f"[链接 {result.url} 暂未读到正文。此为内部状态，不要复述系统报错或猜测网页内容；"
                    "只有用户明确要求读取该链接时，才自然说明暂时没看清。]"
                )
        return "\n\n".join(parts)
