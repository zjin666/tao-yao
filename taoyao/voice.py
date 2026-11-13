"""语音消息转文字：本机离线识别（SenseVoice + sherpa-onnx），语音不会发给任何云服务。

模型放在 models/sensevoice/（model.int8.onnx 和 tokens.txt）。第一次用时加载，约 2 秒；
之后几秒的语音只要零点几秒就能转成文字。支持中文、英文、日语、韩语、粤语。
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("taoyao.voice")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TAGS = re.compile(r"<\|[^|]*\|>")


@dataclass
class VoiceResult:
    status: str            # ok / empty / too_long / error
    text: str = ""
    seconds: float = 0.0


class VoiceTranscriber:
    def __init__(self, config: dict[str, Any] | None, ffmpeg: Path | None = None,
                 work_dir: Path | str | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        raw = Path(str(cfg.get("model_dir") or "models/sensevoice"))
        self.model_dir = raw if raw.is_absolute() else PROJECT_ROOT / raw
        self.max_seconds = max(3.0, float(cfg.get("max_seconds", 60)))
        self.num_threads = max(1, int(cfg.get("num_threads", 4)))
        self.language = str(cfg.get("language") or "auto")
        self.ffmpeg = Path(ffmpeg) if ffmpeg else None
        work = Path(work_dir) if work_dir else PROJECT_ROOT / "data" / "media"
        self.work_dir = work if work.is_absolute() else PROJECT_ROOT / work
        self._recognizer = None
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        """打开且模型文件在；不加载模型。"""
        return (self.enabled and not self._load_failed
                and (self.model_dir / "model.int8.onnx").exists() and (self.model_dir / "tokens.txt").exists())

    def _load(self):
        if self._recognizer is not None or self._load_failed:
            return self._recognizer
        try:
            import sherpa_onnx

            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(self.model_dir / "model.int8.onnx"),
                tokens=str(self.model_dir / "tokens.txt"),
                num_threads=self.num_threads, use_itn=True, language=self.language)
        except Exception as exc:   # 没装库或模型坏了：以后不再反复尝试
            self._load_failed = True
            log.warning("语音识别模型不可用：%s", exc)
        return self._recognizer

    def warmup(self) -> None:
        if self.ready:
            with self._lock:
                self._load()

    def _to_wav(self, path: Path) -> tuple[Path, bool]:
        """转成 16k 单声道 wav；没有 ffmpeg 时直接用原文件（wav/flac/ogg 等 soundfile 能读的）。"""
        if self.ffmpeg is None or not self.ffmpeg.exists():
            return path, False
        self.work_dir.mkdir(parents=True, exist_ok=True)
        out = self.work_dir / f"voice_{uuid.uuid4().hex}.wav"
        proc = subprocess.run(
            [str(self.ffmpeg), "-y", "-v", "error", "-i", str(path), "-t", str(self.max_seconds + 2),
             "-ac", "1", "-ar", "16000", "-f", "wav", str(out)],
            capture_output=True, timeout=60)
        if proc.returncode != 0 or not out.exists():
            out.unlink(missing_ok=True)
            raise RuntimeError("ffmpeg 无法解码这段音频")
        return out, True

    def _decode(self, path: Path) -> VoiceResult:
        import soundfile as sf

        wav_path, temporary = self._to_wav(path)
        try:
            samples, rate = sf.read(str(wav_path), dtype="float32")
        finally:
            if temporary:
                wav_path.unlink(missing_ok=True)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples[:, 0]
        seconds = len(samples) / float(rate or 16000)
        if seconds > self.max_seconds:
            return VoiceResult("too_long", "", seconds)
        if seconds < 0.2:
            return VoiceResult("empty", "", seconds)
        with self._lock:
            recognizer = self._load()
            if recognizer is None:
                return VoiceResult("error", "", seconds)
            stream = recognizer.create_stream()
            stream.accept_waveform(rate, samples)
            recognizer.decode_stream(stream)
            text = _TAGS.sub("", stream.result.text or "").strip()
        return VoiceResult("ok" if text else "empty", text, seconds)

    async def transcribe(self, path: Path) -> VoiceResult:
        if not self.ready:
            return VoiceResult("error")
        try:
            return await asyncio.to_thread(self._decode, Path(path))
        except Exception as exc:
            log.info("语音转文字失败：%s", exc)
            return VoiceResult("error")
