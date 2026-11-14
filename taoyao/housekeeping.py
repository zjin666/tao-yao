"""备份、日志压缩和空间控制。

- 每天对数据库做一次在线快照（sqlite3 的 backup 接口，机器人运行中也安全），连同聊天记录等一起压缩成 zip，
  只留最近几份，总大小超限时先删最旧的。
- 旧日志（.bak）压成 .gz。
- 数据目录太大时，先压缩旧的群聊图片（缩小再存），还是超限就删最旧的并在目录里留一份文字总结。
- 全部是纯函数，不依赖机器人本体，方便单独测试。
"""
from __future__ import annotations

import gzip
import logging
import re
import shutil
import sqlite3
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

log = logging.getLogger("taoyao.housekeeping")
BACKUP_RE = re.compile(r"^taoyao-(\d{8})-(\d{6})\.zip$")


def folder_size(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def mb(size: int) -> float:
    return round(size / 1048576, 1)


# ---------- 备份 ----------
def _snapshot(db: Path, out: Path) -> None:
    source = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(out)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def source_size(databases: Iterable[Path], folders: Iterable[tuple[str, Path]] = ()) -> int:
    """要备份的内容现在有多大（数据库 + 各个文件夹，不含 -wal 之类的临时文件）。"""
    total = sum(Path(db).stat().st_size for db in databases if Path(db).exists())
    return total + sum(folder_size(Path(folder)) for _, folder in folders)


def read_last_check(dest: Path) -> str:
    try:
        return (Path(dest) / "last_check.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_last_check(dest: Path, day: str) -> None:
    Path(dest).mkdir(parents=True, exist_ok=True)
    (Path(dest) / "last_check.txt").write_text(day, encoding="utf-8")


def list_backups(dest: Path) -> list[Path]:
    dest = Path(dest)
    if not dest.exists():
        return []
    return sorted((p for p in dest.iterdir() if BACKUP_RE.match(p.name)), key=lambda p: p.name)


def last_backup_date(dest: Path) -> str:
    """最近一份备份的日期（YYYYMMDD），没有则空字符串。"""
    backups = list_backups(dest)
    return BACKUP_RE.match(backups[-1].name).group(1) if backups else ""


def make_backup(dest: Path, databases: Iterable[Path], folders: Iterable[tuple[str, Path]] = (),
                keep: int = 7, max_total_mb: float = 300, now: datetime | None = None) -> dict:
    """做一份备份并清理旧的。databases 是 sqlite 文件；folders 是 (压缩包里的名字, 目录)。"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    final = dest / f"taoyao-{stamp}.zip"
    partial = dest / f"taoyao-{stamp}.zip.part"
    saved = []
    with tempfile.TemporaryDirectory(dir=dest) as workdir:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for db in databases:
                db = Path(db)
                if not db.exists():
                    continue
                copy = Path(workdir) / db.name
                try:
                    _snapshot(db, copy)
                    archive.write(copy, f"db/{db.name}")
                    saved.append(db.name)
                except Exception as exc:
                    log.warning("备份数据库失败 %s：%s", db.name, exc)
            for name, folder in folders:
                folder = Path(folder)
                if not folder.exists():
                    continue
                for path in folder.rglob("*"):
                    if path.is_file():
                        archive.write(path, f"{name}/{path.relative_to(folder).as_posix()}")
    partial.replace(final)
    removed = prune_backups(dest, keep, max_total_mb)
    return {"file": final, "size": final.stat().st_size, "databases": saved, "removed": removed}


def prune_backups(dest: Path, keep: int = 7, max_total_mb: float = 300) -> list[str]:
    removed = []
    backups = list_backups(dest)
    while len(backups) > max(1, int(keep)):
        victim = backups.pop(0)
        victim.unlink(missing_ok=True)
        removed.append(victim.name)
    while len(backups) > 1 and sum(p.stat().st_size for p in backups) > max_total_mb * 1048576:
        victim = backups.pop(0)
        victim.unlink(missing_ok=True)
        removed.append(victim.name)
    return removed


# ---------- 日志 ----------
def compress_old_logs(logs_dir: Path, min_age_hours: float = 1.0) -> dict:
    """把 .bak 和轮转出来的 bot.log.N 压成 .gz，原文件删掉。"""
    logs_dir = Path(logs_dir)
    saved = count = 0
    if not logs_dir.exists():
        return {"files": 0, "saved": 0}
    cutoff = time.time() - min_age_hours * 3600
    for path in logs_dir.iterdir():
        if not path.is_file() or path.suffix == ".gz":
            continue
        if not (path.suffix == ".bak" or re.search(r"\.log\.\d+$", path.name)):
            continue
        if path.stat().st_mtime > cutoff:
            continue
        target = path.with_name(path.name + ".gz")
        try:
            before = path.stat().st_size
            with path.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
            path.unlink()
            saved += max(0, before - target.stat().st_size)
            count += 1
        except Exception as exc:
            log.warning("压缩日志失败 %s：%s", path.name, exc)
            target.unlink(missing_ok=True)
    return {"files": count, "saved": saved}


# ---------- 群聊图片 ----------
def compress_images(folder: Path, older_than_days: float = 7, max_side: int = 1280,
                    min_bytes: int = 300 * 1024) -> dict:
    """把旧的、较大的图片缩小重存（JPEG）；小图和已经处理过的不动。动图只在很大时才转成静态图。"""
    from PIL import Image

    folder = Path(folder)
    cutoff = time.time() - older_than_days * 86400
    done = 0
    saved = 0
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix == ".desc" or path.name.startswith("_"):
            continue
        try:
            stat = path.stat()
            if stat.st_mtime > cutoff:
                continue
            with Image.open(path) as image:
                animated = bool(getattr(image, "is_animated", False))
                big_side = max(image.size) > max_side
                if stat.st_size < min_bytes and not big_side:
                    continue
                if animated and stat.st_size < 1024 * 1024:
                    continue
                image.seek(0)
                rgb = image.convert("RGBA") if image.mode in ("RGBA", "LA", "P") else image.convert("RGB")
                if rgb.mode == "RGBA":
                    canvas = Image.new("RGB", rgb.size, (255, 255, 255))
                    canvas.paste(rgb, mask=rgb.split()[-1])
                    rgb = canvas
                rgb.thumbnail((max_side, max_side))
            temp = path.with_name(path.name + ".tmp")
            rgb.save(temp, "JPEG", quality=75, optimize=True)
            if temp.stat().st_size < stat.st_size:
                saved += stat.st_size - temp.stat().st_size
                temp.replace(path)
                done += 1
            else:
                temp.unlink()
        except Exception as exc:
            log.debug("压缩图片失败 %s：%s", path.name, exc)
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)
    return {"files": done, "saved": saved}


def trim_to_cap(folder: Path, cap_mb: float, data_root: Path | None = None) -> dict:
    """数据目录还是超限：从最旧的群聊图片开始删，并在那个群目录里留一份文字总结。"""
    folder = Path(folder)
    root = Path(data_root) if data_root else folder
    removed = freed = 0
    if mb(folder_size(root)) <= cap_mb:
        return {"files": 0, "freed": 0}
    files = sorted((p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("_") and p.suffix != ".desc"),
                   key=lambda p: p.stat().st_mtime)
    summary: dict[Path, list] = {}
    for path in files:
        if mb(folder_size(root)) <= cap_mb:
            break
        stat = path.stat()
        summary.setdefault(path.parent, []).append((stat.st_mtime, stat.st_size))
        path.unlink(missing_ok=True)
        removed += 1
        freed += stat.st_size
    for parent, items in summary.items():
        first = datetime.fromtimestamp(min(t for t, _ in items)).strftime("%Y-%m-%d")
        last = datetime.fromtimestamp(max(t for t, _ in items)).strftime("%Y-%m-%d")
        line = (f"{datetime.now():%Y-%m-%d %H:%M} 空间超限，清理了 {len(items)} 张旧图片"
                f"（{first} 至 {last}，共 {mb(sum(s for _, s in items))} MB）。\n")
        with (parent / "_cleanup_summary.txt").open("a", encoding="utf-8") as handle:
            handle.write(line)
    return {"files": removed, "freed": freed}


# ---------- 汇总 ----------
def run_housekeeping(*, backup_dir: Path, databases: Iterable[Path], folders: Iterable[tuple[str, Path]],
                     logs_dir: Path, media_dir: Path, data_root: Path, keep: int = 7, max_backup_mb: float = 300,
                     media_compress_days: float = 7, media_max_side: int = 1280, max_data_mb: float = 400,
                     do_backup: bool = True, backup_note: str = "") -> str:
    """做完所有整理，返回一行总结（写日志用）。"""
    parts = []
    if do_backup:
        result = make_backup(backup_dir, databases, folders, keep, max_backup_mb)
        parts.append(f"备份 {mb(result['size'])} MB（{len(result['databases'])} 个库，清掉旧备份 {len(result['removed'])} 份）")
    elif backup_note:
        parts.append(backup_note)
    logs = compress_old_logs(logs_dir)
    if logs["files"]:
        parts.append(f"压缩旧日志 {logs['files']} 个，省 {mb(logs['saved'])} MB")
    images = compress_images(media_dir, media_compress_days, media_max_side) if Path(media_dir).exists() else {"files": 0, "saved": 0}
    if images["files"]:
        parts.append(f"缩小旧图片 {images['files']} 张，省 {mb(images['saved'])} MB")
    trimmed = trim_to_cap(media_dir, max_data_mb, data_root) if Path(media_dir).exists() else {"files": 0, "freed": 0}
    if trimmed["files"]:
        parts.append(f"数据目录超过 {max_data_mb:g} MB，清理最旧图片 {trimmed['files']} 张，腾出 {mb(trimmed['freed'])} MB")
    parts.append(f"数据目录现在 {mb(folder_size(data_root))} MB")
    return "；".join(parts)
