"""Separate trailing model metadata without waiting before yielding prose."""
import re

from . import markers


class MemeTail:
    """回复末尾的 [[meme:ID]]：只认 markers 里定义的写法（逐字匹配，区分大小写）；写错的不当作标签。"""
    marker = markers.MEME_OPEN

    def __init__(self):
        self.category = ''

    async def stream(self, source):
        pending = ''
        metadata = None
        async for chunk in source:
            if metadata is not None:
                metadata += chunk
                continue
            pending += chunk
            start = pending.find(self.marker)
            if start >= 0:
                if start:
                    yield pending[:start]
                metadata = pending[start:]
                pending = ''
                continue
            hold = 0
            for size in range(1, min(len(pending), len(self.marker)-1)+1):
                if self.marker.startswith(pending[-size:]):
                    hold = size
            ready = pending[:-hold] if hold else pending
            pending = pending[-hold:] if hold else ''
            if ready:
                yield ready
        # A truncated metadata prefix must never reach the recipient.
        if pending and not self.marker.startswith(pending):
            yield pending
        if metadata is not None:
            match = re.fullmatch(markers.MEME_RE.pattern + r'\s*', metadata)
            if match:
                self.category = match[1]
