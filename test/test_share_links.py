import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from taoyao.app import TaoYaoBot
from taoyao.linkreader import LinkReader, extract_links


class ShareLinkTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = TaoYaoBot.__new__(TaoYaoBot)
        self.share = {'type': 'share', 'data': {
            'url': 'https://example.com/article', 'title': '测试文章',
            'content': '卡片简介'}}

    async def test_share_preserves_metadata_and_reads_actual_page_body(self):
        text = self.bot._extract_text({'message': [
            {'type': 'text', 'data': {'text': '看看'}}, self.share,
            {'type': 'text', 'data': {'text': '说说你的看法'}}]})
        self.assertIn('测试文章', text)
        self.assertIn('卡片简介', text)
        self.assertEqual(extract_links(text), ['https://example.com/article'])
        reader = LinkReader({})
        page = httpx.Response(200, headers={'content-type': 'text/html; charset=utf-8'},
            text='<html><title>网页标题</title><body>网页实际正文，不只是卡片简介。</body></html>')
        with patch.object(reader, '_safe_host', return_value=(True, '')), \
                patch('taoyao.linkreader.httpx.AsyncClient.get', new=AsyncMock(return_value=page)) as get:
            context = await reader.fetch_text(text)
        get.assert_awaited_once()
        self.assertEqual(get.await_args.args[0], 'https://example.com/article')
        self.assertIn('网页实际正文', context)
        self.assertIn('网页标题', context)

    async def test_share_inside_forward_retains_fetchable_url(self):
        self.bot.server = SimpleNamespace(call=AsyncMock(return_value={
            'data': {'messages': [{'sender': {'nickname': '测试用户'},
                'content': [self.share]}]}}))
        text = await self.bot._forward_text({'message': [
            {'type': 'forward', 'data': {'id': 'synthetic'}}]})
        self.assertIn('测试文章', text)
        self.assertIn('卡片简介', text)
        self.assertEqual(extract_links(text), ['https://example.com/article'])

    async def test_title_only_share_remains_visible_without_fetch(self):
        text = self.bot._extract_text({'message': [
            {'type': 'share', 'data': {'title': '没有网址的分享'}}]})
        self.assertIn('没有网址的分享', text)
        reader = LinkReader({})
        with patch.object(reader, 'fetch', new=AsyncMock()) as fetch:
            self.assertEqual(await reader.fetch_text(text), '')
        fetch.assert_not_awaited()

    async def test_failed_page_does_not_invent_content(self):
        text = self.bot._extract_text({'message': [self.share]})
        reader = LinkReader({})
        with patch.object(reader, '_safe_host', return_value=(True, '')), \
                patch('taoyao.linkreader.httpx.AsyncClient.get', new=AsyncMock(
                    return_value=httpx.Response(403))):
            context = await reader.fetch_text(text)
        self.assertIn('暂未读到正文', context)
        self.assertNotIn('网页实际正文', context)
