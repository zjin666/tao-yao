"""统一的 SQLite 写入设置：WAL + synchronous=NORMAL。

默认的完整同步让每次提交都要等磁盘 fsync（实测 26 到 79 毫秒，偶尔数百毫秒）。
WAL + NORMAL 下数据库不会损坏，应用崩溃也不丢已提交的数据；只有断电或系统崩溃时，
最后几次提交可能丢失。读取不受影响。
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)


def tune(conn: sqlite3.Connection, *, wal: bool = True) -> sqlite3.Connection:
    """synchronous 是连接级设置，每个新连接都要设；journal_mode=WAL 写进数据库文件，设一次即可。"""
    if wal:
        conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    return conn



class WalCheckpointer:
    """定时把 WAL 并回数据库文件并同步磁盘，收紧断电时的最坏丢失范围。

    用单独的短连接做 PASSIVE 检查点，不阻塞各存储自己的写入；WAL 没有新内容时几乎没有开销。
    """

    def __init__(self, paths, interval: float = 60.0):
        self.paths = [Path(p) for p in paths]
        self.interval = float(interval)

    def checkpoint_once(self) -> dict[str, tuple[int, int]]:
        """返回 {库文件名: (WAL 帧数, 已并回帧数)}；失败的库只记日志，不影响其他库。"""
        result: dict[str, tuple[int, int]] = {}
        for path in self.paths:
            wal = Path(str(path) + '-wal')
            try:
                if not path.exists() or not wal.exists() or wal.stat().st_size == 0:
                    continue
                started = time.monotonic()
                conn = sqlite3.connect(str(path), timeout=5)
                try:
                    conn.execute('PRAGMA synchronous=NORMAL')
                    _busy, frames, done = conn.execute('PRAGMA wal_checkpoint(PASSIVE)').fetchone()
                finally:
                    conn.close()
                result[path.name] = (frames, done)
                elapsed = time.monotonic() - started
                if elapsed > 1.0:
                    log.warning('WAL 检查点较慢：%s %.2fs', path.name, elapsed)
            except Exception:
                log.exception('WAL 检查点失败：%s', path)
        return result

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            work = asyncio.ensure_future(asyncio.to_thread(self.checkpoint_once))
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                # 取消时让正在跑的检查点先结束，不留线程占着数据库文件。
                await asyncio.gather(work, return_exceptions=True)
                raise
            except Exception:
                log.exception('WAL 定时检查点失败')
