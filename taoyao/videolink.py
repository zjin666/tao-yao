"""链接里的视频：用 yt-dlp 解析出视频流地址，ffmpeg 直接从流里按时间点截几帧，交给现有的视觉模型看。

不下载整个视频，也不处理声音（没有语音识别）。只对 yt-dlp 认识的视频站生效，通用网页不会走这里。
没有装 yt-dlp、没有 ffmpeg、解析失败、直播、太长等情况一律返回空，由调用方按“没看到画面”处理。
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

FRAME_PROMPT = (
    "这是同一个视频按时间顺序抽取的几帧画面。请用中文概括视频大概在讲什么、画面里出现了什么，"
    "并提取画面中出现的文字；只描述画面，看不出来的不要编造，也不要说你听到了声音。"
    "控制在 150 字以内，不要用列表和 Markdown 格式。"
)


@dataclass
class VideoInfo:
    url: str
    title: str = ""
    uploader: str = ""
    description: str = ""
    duration: float = 0.0
    stream_url: str = ""
    headers: dict | None = None


def is_video_site(url: str) -> bool:
    """yt-dlp 是否有这个网址专用的解析器（排除“什么网页都试一遍”的通用解析器）。"""
    try:
        from yt_dlp.extractor import gen_extractor_classes
    except Exception:
        return False
    for extractor in gen_extractor_classes():
        name = getattr(extractor, "IE_NAME", "")
        if name in ("generic", "default") or getattr(extractor, "__name__", "") == "GenericIE":
            continue
        try:
            if extractor.suitable(url):
                return True
        except Exception:
            continue
    return False


def pick_stream(formats: list[dict]) -> dict | None:
    """选画质最低、有画面、能直接拉流的那一路——只要几帧，不需要清晰。"""
    candidates = []
    for fmt in formats or []:
        if not fmt.get("url") or (fmt.get("vcodec") or "none") == "none":
            continue
        if (fmt.get("protocol") or "") not in ("https", "http", "m3u8", "m3u8_native"):
            continue
        height = fmt.get("height") or 0
        if height and height < 144:
            continue
        candidates.append((height or 9999, fmt))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def resolve(url: str, timeout: float = 20.0) -> VideoInfo | None:
    """同步（在线程里调用）：解析视频信息和流地址。"""
    try:
        import yt_dlp
    except Exception:
        return None
    options = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": timeout}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        log.info("视频解析失败：%s %s", url, str(exc).splitlines()[0][:120] if str(exc) else "")
        return None
    if not info or info.get("is_live") or info.get("_type") == "playlist":
        return None
    fmt = pick_stream(info.get("formats") or [])
    if fmt is None and info.get("url"):
        fmt = {"url": info["url"], "http_headers": info.get("http_headers")}
    if fmt is None:
        return None
    return VideoInfo(
        url=url, title=str(info.get("title") or ""), uploader=str(info.get("uploader") or info.get("channel") or ""),
        description=str(info.get("description") or ""), duration=float(info.get("duration") or 0),
        stream_url=str(fmt["url"]), headers=dict(fmt.get("http_headers") or info.get("http_headers") or {}),
    )


def frame_times(duration: float, count: int) -> list[float]:
    """均匀取点，避开最开头和最末尾（常是黑屏/片头/片尾）。"""
    if duration <= 0:
        return [1.0]
    count = max(1, count)
    return [round(duration * (i + 0.5) / count, 2) for i in range(count)]


def grab_frame(ffmpeg: Path, info: VideoInfo, at: float, out: Path, timeout: float = 30.0) -> bool:
    header_text = "".join(f"{k}: {v}\r\n" for k, v in (info.headers or {}).items())
    cmd = [str(ffmpeg), "-y", "-loglevel", "error"]
    if header_text:
        cmd += ["-headers", header_text]
    cmd += ["-ss", f"{at}", "-i", info.stream_url, "-frames:v", "1", "-vf", "scale=-2:480", "-q:v", "3", str(out)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:
        log.info("视频截帧失败：%s", exc)
        return False
    return out.exists() and out.stat().st_size > 0


class VideoWatcher:
    """看链接里的视频：返回一段文字（标题、作者、简介、画面概括）；看不了就返回空字符串。"""

    def __init__(self, config: dict[str, Any] | None, ffmpeg: Path | None,
                 describe_frames: Callable[[list[Path], str], Awaitable[str]], work_dir: Path):
        cfg = config or {}
        self.enabled = bool(cfg.get("watch_video", True))
        self.frames = max(1, int(cfg.get("video_frames", 4)))
        self.timeout = float(cfg.get("video_timeout", 45))
        self.max_minutes = float(cfg.get("video_max_minutes", 180))
        self.ffmpeg = ffmpeg
        self._describe = describe_frames
        self.work_dir = work_dir
        self._cache: OrderedDict[str, str] = OrderedDict()

    async def watch(self, url: str, with_meta: bool = True) -> str:
        if not self.enabled or self.ffmpeg is None:
            return ""
        key = f"{int(with_meta)}|{url}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            text = await asyncio.wait_for(self._watch(url, with_meta), self.timeout)
        except asyncio.TimeoutError:
            log.info("看视频超时：%s", url)
            return ""
        except Exception:
            log.exception("看视频失败：%s", url)
            return ""
        if text:
            self._cache[key] = text
            while len(self._cache) > 32:
                self._cache.popitem(last=False)
        return text

    async def _watch(self, url: str, with_meta: bool) -> str:
        if not await asyncio.to_thread(is_video_site, url):
            return ""
        info = await asyncio.to_thread(resolve, url)
        if info is None or (info.duration and info.duration > self.max_minutes * 60):
            return ""
        frame_dir = self.work_dir / f"video_{uuid.uuid4().hex}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        paths = [frame_dir / f"f{i}.jpg" for i in range(self.frames)]
        try:
            times = frame_times(info.duration, self.frames)
            if len(times) < len(paths):
                paths = paths[: len(times)]
            done = await asyncio.gather(*(asyncio.to_thread(grab_frame, self.ffmpeg, info, at, path)
                                          for at, path in zip(times, paths)))
            frames = [path for path, ok in zip(paths, done) if ok]
            summary = await self._describe(frames, FRAME_PROMPT) if frames else ""
        finally:
            for path in frame_dir.glob("*"):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                frame_dir.rmdir()
            except OSError:
                pass
        parts = []
        if with_meta and info.uploader:
            parts.append(f"作者：{info.uploader}")
        if with_meta and info.duration:
            parts.append(f"时长约 {int(info.duration)} 秒")
        if with_meta and info.description.strip():
            parts.append("简介：" + " ".join(info.description.split())[:400])
        if summary:
            parts.append("画面概括（按时间截取的几帧，没有声音内容）：" + " ".join(summary.split())[:400])
        else:
            parts.append("（没能看到画面）")
        return "\n".join(parts)
