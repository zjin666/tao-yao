from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _decode_ddg_url(href: str) -> str:
    href = unescape(href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    try:
        query = parse_qs(urlparse(href).query)
        if query.get("uddg"):
            return unquote(query["uddg"][0])
    except Exception:
        pass
    return href


class _DuckParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        self._pending_title = ""
        self._pending_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self._in_title = True
            self._title_parts = []
            self._pending_url = _decode_ddg_url(values.get("href") or "")
        elif tag in ("a", "div") and "result__snippet" in classes:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            self._in_title = False
            self._pending_title = _clean("".join(self._title_parts))
        elif self._in_snippet and tag in ("a", "div"):
            self._in_snippet = False
            if self._pending_title and self._pending_url:
                self.results.append(SearchResult(
                    title=self._pending_title,
                    url=self._pending_url,
                    snippet=_clean("".join(self._snippet_parts)),
                ))
                self._pending_title = ""
                self._pending_url = ""

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        if self._in_snippet:
            self._snippet_parts.append(data)


class WebSearchClient:
    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max(1, min(8, int(max_results)))
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        }

    async def search(self, query: str) -> list[SearchResult]:
        query = (query or "").strip()[:200]
        if not query:
            return []
        async with httpx.AsyncClient(
            headers=self.headers, timeout=15, follow_redirects=True
        ) as client:
            results = await self._duckduckgo(client, query)
            if not results:
                results = await self._bing(client, query)
        return results[: self.max_results]

    async def _duckduckgo(self, client: httpx.AsyncClient, query: str) -> list[SearchResult]:
        try:
            resp = await client.get(
                "https://html.duckduckgo.com/html/", params={"q": query}
            )
            if resp.status_code != 200:
                return []
            parser = _DuckParser()
            parser.feed(resp.text)
            return [item for item in parser.results if item.title and item.url]
        except Exception:
            return []

    async def _bing(self, client: httpx.AsyncClient, query: str) -> list[SearchResult]:
        try:
            resp = await client.get(
                "https://www.bing.com/search",
                params={"q": query, "setlang": "zh-Hans"},
            )
            if resp.status_code != 200:
                return []
            results: list[SearchResult] = []
            for block in re.findall(r'<li class="b_algo".*?</li>', resp.text, re.S):
                title_match = re.search(
                    r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                    block,
                    re.S,
                )
                if not title_match:
                    continue
                snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
                results.append(SearchResult(
                    title=_clean(title_match.group(2)),
                    url=unescape(title_match.group(1)),
                    snippet=_clean(snippet_match.group(1) if snippet_match else ""),
                ))
            return results
        except Exception:
            return []

    @staticmethod
    def format_results(query: str, results: list[SearchResult]) -> str:
        if not results:
            return ""
        lines = [f"查询：{query}"]
        for index, item in enumerate(results, 1):
            lines.append(f"{index}. {item.title}")
            if item.url:
                lines.append(f"   来源：{item.url}")
            if item.snippet:
                lines.append(f"   摘要：{item.snippet}")
        return "\n".join(lines)
