from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import platform
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

import httpx

from .llm import collect_stream

log = logging.getLogger("taoyao.vision")


IMAGE_PROMPT = (
    "用中文描述这张图片的内容，并提取图片里出现的文字。要求：先数清画面里一共有几个独立的角色或生物；"
    "角色身上长出来的耳朵、尾巴、鱼鳍、翅膀、角等属于角色本身，不要当成另一个生物或物品；"
    "如果某个物体既可能是单独的东西、也可能是角色自己的身体部分（比如从身后绕到身前或被抱着的尾巴），"
    "两种可能都写出来，不要只选一种；写明角色的动作、表情和姿势；不确定的写“可能”，不要编造；"
    "控制在 200 字以内，不要用列表和 Markdown 格式。"
)


class MediaDescriber:
    """把图片/视频转成文字：优先视觉 API，没配置时用本地 OCR。"""

    def __init__(self, config: dict[str, Any] | None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.provider = str(self.config.get("provider") or "auto").lower()
        self.api_base = str(self.config.get("api_base") or "").rstrip("/")
        self.model = str(self.config.get("model") or "")
        self.max_frames = max(1, int(self.config.get("max_frames", 4)))
        self.max_tokens = max(500, int(self.config.get("max_tokens", 1500)))
        self.detail = str(self.config.get("detail") or "low")
        self.max_media_mb = float(self.config.get("max_media_mb", 20))
        self.max_upload_mb = float(self.config.get("max_upload_mb", 6))
        self.timeout = float(self.config.get("timeout", 60))
        self.work_dir = Path(str(self.config.get("work_dir") or "data/media"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = ""
        key_file = str(self.config.get("api_key_file") or "")
        if key_file:
            path = Path(key_file)
            if path.exists():
                self._api_key = path.read_text(encoding="utf-8").strip()
        self._ocr = None
        self._ocr_failed = False
        self._ocr_script = Path(__file__).with_name("windows_ocr.ps1")
        self._ffmpeg = self._find_ffmpeg()

    def _find_ffmpeg(self) -> Path | None:
        configured = str(self.config.get("ffmpeg_path") or "")
        if configured and Path(configured).exists():
            return Path(configured)
        roots = [Path(__file__).resolve().parent.parent / "bin", self.work_dir.parent / "bin"]
        for root in roots:
            for path in root.glob("**/ffmpeg.exe"):
                return path
        return None

    def _api_ready(self) -> bool:
        if self.provider in ("ocr", "local"):
            return False
        return bool(self.api_base and self.model and self._api_key)

    async def fetch_segment(self, segment: dict) -> tuple[Path | None, bool]:
        data = segment.get("data") or {}
        raw_url = str(data.get("url") or "")
        raw_file = str(data.get("file") or "")
        raw_path = str(data.get("path") or "")
        for candidate in (raw_path, raw_file):
            if candidate.startswith("file://"):
                candidate = candidate[7:]
            if candidate and not candidate.startswith(("http://", "https://", "base64://")):
                path = Path(candidate)
                if path.exists() and path.is_file():
                    return path, False
        if raw_url.startswith(("http://", "https://")):
            return await self._download(raw_url), True
        if raw_file.startswith("base64://"):
            try:
                blob = base64.b64decode(raw_file[9:])
                out = self.work_dir / f"{uuid.uuid4().hex}.png"
                out.write_bytes(blob)
                return out, True
            except Exception:
                return None, False
        return None, False

    async def _download(self, url: str) -> Path | None:
        limit = int(self.max_media_mb * 1024 * 1024)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    suffix = Path(url.split("?")[0]).suffix or ".bin"
                    out = self.work_dir / f"{uuid.uuid4().hex}{suffix[:8]}"
                    size = 0
                    with out.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > limit:
                                raise ValueError("媒体文件太大")
                            handle.write(chunk)
                    return out
        except Exception as exc:
            log.warning("下载媒体失败：%s", exc)
            return None

    def _load_ocr(self):
        if self._ocr is not None or self._ocr_failed:
            return self._ocr
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        except Exception as exc:
            self._ocr_failed = True
            log.warning("本地 OCR 不可用：%s", exc)
        return self._ocr

    def _windows_ocr(self, path: Path) -> str:
        if platform.system() != "Windows" or not self._ocr_script.exists():
            return ""
        try:
            proc = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(self._ocr_script), "-Path", str(path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=45,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if proc.returncode != 0:
                return ""
            text = (proc.stdout or "").strip()
            # 中文 OCR 常把每个字用空格隔开，压回去；英文单词间的空格保留。
            text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
            text = re.sub(r"[ \t]{2,}", " ", text)
            return text.strip()
        except Exception as exc:
            log.warning("Windows OCR 失败：%s", exc)
            return ""

    def _ocr_image(self, path: Path) -> str:
        text = self._windows_ocr(path)
        if text:
            return text
        ocr = self._load_ocr()
        if ocr is None:
            return ""
        try:
            result, _ = ocr(str(path))
        except Exception as exc:
            log.warning("OCR 失败：%s", exc)
            return ""
        lines = []
        for item in result or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text = str(item[1]).strip()
                if text and text not in lines:
                    lines.append(text)
        return "\n".join(lines)

    def _extract_frames(self, path: Path) -> list[Path]:
        if self._ffmpeg is not None:
            try:
                frame_dir = self.work_dir / f"frames_{uuid.uuid4().hex}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                ffprobe = self._ffmpeg.with_name("ffprobe.exe")
                duration = 0.0
                if ffprobe.exists():
                    proc = subprocess.run(
                        [str(ffprobe), "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=nw=1:nk=1", str(path)],
                        capture_output=True, text=True, timeout=20,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    try:
                        duration = float((proc.stdout or "0").strip())
                    except ValueError:
                        duration = 0.0
                fps = max(0.1, min(2.0, self.max_frames / max(1.0, duration)))
                out_pattern = frame_dir / "frame_%02d.jpg"
                proc = subprocess.run(
                    [str(self._ffmpeg), "-y", "-i", str(path), "-vf", f"fps={fps:.3f}",
                     "-frames:v", str(self.max_frames), "-q:v", "2", str(out_pattern)],
                    capture_output=True, text=True, timeout=90,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                frames = sorted(frame_dir.glob("frame_*.jpg"))
                if frames:
                    return frames
            except Exception as exc:
                log.warning("ffmpeg 抽帧失败：%s", exc)
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                cap.release()
                return []
            count = min(self.max_frames, max(1, total))
            indices = [int(i * (total - 1) / max(1, count - 1)) for i in range(count)]
            frames = []
            for idx in sorted(set(indices)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                out = self.work_dir / f"{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(out), frame)
                frames.append(out)
            cap.release()
            return frames
        except Exception as exc:
            log.warning("视频抽帧失败：%s", exc)
            return []

    def _normalize_image(self, path: Path) -> tuple[Path, bool]:
        """把大图缩到可上传尺寸并压成 JPEG，避免 10MB 图片转 PNG 后超限。"""
        try:
            from PIL import Image

            limit = int(self.max_upload_mb * 1024 * 1024)
            with Image.open(path) as image:
                image.seek(0)
                if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    canvas = Image.new("RGB", rgba.size, (255, 255, 255))
                    canvas.paste(rgba, mask=rgba.split()[-1])
                    image = canvas
                else:
                    image = image.convert("RGB")
                image.thumbnail((1600, 1600))

                # 先降质量，再降尺寸；目标是让 base64 请求体留在安全范围内。
                for quality in (88, 82, 74, 66, 58):
                    out = self.work_dir / f"{uuid.uuid4().hex}.jpg"
                    image.save(out, "JPEG", quality=quality, optimize=True)
                    if out.stat().st_size <= limit:
                        return out, True
                    try:
                        out.unlink()
                    except OSError:
                        pass
                for scale in (0.85, 0.7, 0.55, 0.45):
                    resized = image.resize(
                        (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                    )
                    out = self.work_dir / f"{uuid.uuid4().hex}.jpg"
                    resized.save(out, "JPEG", quality=76, optimize=True)
                    if out.stat().st_size <= limit:
                        return out, True
                    try:
                        out.unlink()
                    except OSError:
                        pass
            return path, False
        except Exception as exc:
            log.warning("图片格式转换失败：%s", exc)
            return path, False

    async def _api_describe(self, paths: list[Path], prompt: str) -> str:
        if not self._api_ready():
            return ""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in paths:
            raw = path.read_bytes()
            if len(raw) > int(self.max_upload_mb * 1024 * 1024):
                log.warning("图片压缩后仍超过上传上限：%s bytes", len(raw))
                continue
            data = base64.b64encode(raw).decode("ascii")
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{data}",
                    "detail": self.detail,
                },
            })
        if len(content) == 1:
            return ""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            data = await collect_stream(client,
                f"{self.api_base}/chat/completions", headers, payload)
            content = str(data["choices"][0]["message"].get("content") or "").strip()
            if not content:
                log.warning("视觉 API 返回空正文，finish_reason=%s", data["choices"][0].get("finish_reason"))
            return content

    async def describe(self, path: Path, kind: str) -> str:
        if not self.enabled or not path.exists():
            return ""
        if kind == "video":
            frames = await asyncio.to_thread(self._extract_frames, path)
            if not frames:
                return ""
            try:
                if self._api_ready():
                    try:
                        return await self._api_describe(
                            frames,
                            "这是同一个视频按时间顺序抽取的帧。请用中文概括视频内容，"
                            "并提取画面中出现的文字；不确定的不要编造。",
                        )
                    except Exception as exc:
                        log.warning("视觉 API 失败，退回 OCR：%s", exc)
                texts = await asyncio.gather(
                    *(asyncio.to_thread(self._ocr_image, frame) for frame in frames)
                )
                merged = []
                for text in texts:
                    for line in str(text or "").splitlines():
                        line = line.strip()
                        if line and line not in merged:
                            merged.append(line)
                return "\n".join(merged)
            finally:
                for frame in frames:
                    try:
                        frame.unlink()
                    except OSError:
                        pass

        if self._api_ready():
            normalized, temporary = await asyncio.to_thread(self._normalize_image, path)
            try:
                return await self._api_describe(
                    [normalized],
                    IMAGE_PROMPT,
                )
            except Exception as exc:
                log.warning("视觉 API 失败，退回 OCR：%s", exc)
            finally:
                if temporary:
                    try:
                        normalized.unlink()
                    except OSError:
                        pass
        return await asyncio.to_thread(self._ocr_image, path)

    async def describe_frames(self, frames: list[Path], prompt: str) -> str:
        """把已经截好的几帧交给视觉模型；没配置视觉 API 时返回空。"""
        if not self.enabled or not self._api_ready() or not frames:
            return ""
        try:
            return await self._api_describe(frames, prompt)
        except Exception as exc:
            log.warning("视觉 API 看视频帧失败：%s", exc)
            return ""

    async def close(self) -> None:
        return None
