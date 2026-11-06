"""Separate trailing model metadata without waiting before yielding prose."""
import re


class MemeTail:
    marker = '[[meme:'

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
            start = pending.lower().find(self.marker)
            if start >= 0:
                if start:
                    yield pending[:start]
                metadata = pending[start:]
                pending = ''
                continue
            hold = 0
            for size in range(1, min(len(pending), len(self.marker)-1)+1):
                if self.marker.startswith(pending[-size:].lower()):
                    hold = size
            ready = pending[:-hold] if hold else pending
            pending = pending[-hold:] if hold else ''
            if ready:
                yield ready
        # A truncated metadata prefix must never reach the recipient.
        if pending and not self.marker.startswith(pending.lower()):
            yield pending
        if metadata is not None:
            match = re.fullmatch(r'\[\[meme:([^\[\]\r\n]{1,40})\]\]\s*', metadata, re.I)
            if match:
                self.category = match[1].strip().lower()
