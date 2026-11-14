import gzip
import os
import sqlite3
import tempfile
import time
import unittest
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from taoyao import housekeeping as hk


def make_db(path: Path, rows: int = 5) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t(v) VALUES (?)", [(f"行{i}",) for i in range(rows)])
    conn.commit()
    conn.close()
    return path


def age(path: Path, days: float) -> None:
    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))


class BackupTests(unittest.TestCase):
    def test_backup_contains_consistent_snapshots_and_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = make_db(root / "a.db")
            (root / "hist").mkdir()
            (root / "hist" / "group.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            result = hk.make_backup(root / "bk", [db, root / "missing.db"], [("chat_history", root / "hist")])
            self.assertTrue(result["file"].exists())
            self.assertEqual(result["databases"], ["a.db"])
            with zipfile.ZipFile(result["file"]) as archive:
                names = archive.namelist()
                self.assertIn("db/a.db", names)
                self.assertIn("chat_history/group.jsonl", names)
                archive.extract("db/a.db", root / "out")
            restored = sqlite3.connect(root / "out" / "db" / "a.db")
            self.assertEqual(restored.execute("SELECT COUNT(*) FROM t").fetchone()[0], 5)
            restored.close()

    def test_backup_while_database_is_being_written_in_wal_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = root / "w.db"
            conn = sqlite3.connect(db)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE t(v)")
            conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(50)])
            conn.commit()                                   # 数据还在 -wal 里，主文件可能没有
            result = hk.make_backup(root / "bk", [db], [])
            with zipfile.ZipFile(result["file"]) as archive:
                archive.extract("db/w.db", root / "out")
            check = sqlite3.connect(root / "out" / "db" / "w.db")
            self.assertEqual(check.execute("SELECT COUNT(*) FROM t").fetchone()[0], 50)
            check.close()
            conn.close()

    def test_only_the_newest_copies_are_kept_and_the_size_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = make_db(root / "a.db")
            for day in range(1, 6):
                hk.make_backup(root / "bk", [db], [], keep=3, now=datetime(2026, 11, day, 4, 0, 0))
            names = [p.name for p in hk.list_backups(root / "bk")]
            self.assertEqual(names, ["taoyao-20261103-040000.zip", "taoyao-20261104-040000.zip", "taoyao-20261105-040000.zip"])
            self.assertEqual(hk.last_backup_date(root / "bk"), "20261105")
            hk.prune_backups(root / "bk", keep=10, max_total_mb=0.000001)
            self.assertEqual(len(hk.list_backups(root / "bk")), 1)        # 超限只留最新一份


class LogAndImageTests(unittest.TestCase):
    def test_old_bak_logs_are_gzipped_and_bot_log_is_left_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            (logs / "bot.log").write_text("live\n", encoding="utf-8")
            old = logs / "restart.log.20261105.bak"
            old.write_text("旧日志" * 1000, encoding="utf-8")
            rotated = logs / "bot.log.1"
            rotated.write_text("轮转" * 1000, encoding="utf-8")
            for path in (old, rotated):
                age(path, 2)
            fresh = logs / "fresh.bak"
            fresh.write_text("刚写的", encoding="utf-8")
            result = hk.compress_old_logs(logs)
            self.assertEqual(result["files"], 2)
            self.assertTrue((logs / "bot.log").exists())
            self.assertTrue(fresh.exists())                              # 太新的不动
            self.assertFalse(old.exists())
            with gzip.open(logs / "restart.log.20261105.bak.gz", "rt", encoding="utf-8") as handle:
                self.assertTrue(handle.read().startswith("旧日志"))

    def test_big_old_images_are_shrunk_small_and_young_ones_are_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            media = Path(folder)
            big, small, young = media / "big.bin", media / "small.bin", media / "young.bin"
            import random
            noise = Image.frombytes("RGB", (2400, 1800), bytes(random.getrandbits(8) for _ in range(2400 * 1800 * 3)))
            noise.save(big, "PNG")
            Image.new("RGB", (64, 64), (255, 0, 0)).save(small, "PNG")
            noise.save(young, "PNG")
            for path in (big, small):
                age(path, 10)
            before = big.stat().st_size
            result = hk.compress_images(media, older_than_days=7, max_side=1280)
            self.assertEqual(result["files"], 1)
            self.assertLess(big.stat().st_size, before / 3)
            with Image.open(big) as shrunk:
                self.assertLessEqual(max(shrunk.size), 1280)
            self.assertEqual(Image.open(small).size, (64, 64))
            self.assertEqual(Image.open(young).size, (2400, 1800))

    def test_over_cap_removes_oldest_images_and_leaves_a_written_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            group = data / "media" / "group" / "1"
            group.mkdir(parents=True)
            for index in range(4):
                path = group / f"img{index}.bin"
                path.write_bytes(b"x" * 400_000)
                age(path, 10 - index)
            result = hk.trim_to_cap(data / "media", cap_mb=1.0, data_root=data)
            self.assertGreaterEqual(result["files"], 1)
            self.assertFalse((group / "img0.bin").exists())              # 最旧的先删
            self.assertTrue((group / "img3.bin").exists())
            summary = (group / "_cleanup_summary.txt").read_text(encoding="utf-8")
            self.assertIn("清理了", summary)
            self.assertLessEqual(hk.mb(hk.folder_size(data)), 1.2)

    def test_nothing_is_removed_when_under_the_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            (data / "media").mkdir()
            (data / "media" / "a.bin").write_bytes(b"x" * 1000)
            self.assertEqual(hk.trim_to_cap(data / "media", cap_mb=10, data_root=data)["files"], 0)


class OrchestrationTests(unittest.TestCase):
    def test_run_housekeeping_returns_a_one_line_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data" / "media").mkdir(parents=True)
            db = make_db(root / "data" / "a.db")
            (root / "logs").mkdir()
            summary = hk.run_housekeeping(
                backup_dir=root / "bk", databases=[db], folders=[], logs_dir=root / "logs",
                media_dir=root / "data" / "media", data_root=root / "data")
            self.assertIn("备份", summary)
            self.assertIn("数据目录现在", summary)
            self.assertEqual(len(hk.list_backups(root / "bk")), 1)


class DailyGateTests(unittest.IsolatedAsyncioTestCase):
    def make_bot(self, root: Path, **cfg):
        from types import SimpleNamespace
        from taoyao.app import TaoYaoBot
        (root / "data" / "media").mkdir(parents=True)
        (root / "logs").mkdir()
        db = make_db(root / "data" / "a.db", rows=200)
        bot = TaoYaoBot.__new__(TaoYaoBot)
        bot.config = {"_root": str(root), "memory": {"db": str(db)},
                      "housekeeping": {"hour": 0, "include_annotation": False, **cfg}}
        bot._sqlite_paths = lambda: [db]
        bot.sticker_memory = SimpleNamespace(path=root / "data" / "none.db")
        bot.vision = SimpleNamespace(work_dir=root / "data" / "media")
        return bot

    async def test_backup_is_skipped_below_the_size_threshold_and_checked_once_a_day(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bot = self.make_bot(root, backup_min_mb=50)
            summary = await bot.run_housekeeping()
            self.assertIn("今天不备份", summary)
            self.assertEqual(hk.list_backups(root / "backups" / "auto"), [])
            self.assertEqual(await bot.run_housekeeping(), "")             # 今天已经检查过，不再重复
            forced = await bot.run_housekeeping(force=True)               # 手动强制才备份
            self.assertIn("备份", forced)
            self.assertEqual(len(hk.list_backups(root / "backups" / "auto")), 1)

    async def test_backup_is_made_once_the_content_reaches_the_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bot = self.make_bot(root, backup_min_mb=0.0001)
            summary = await bot.run_housekeeping()
            self.assertNotIn("今天不备份", summary)
            self.assertEqual(len(hk.list_backups(root / "backups" / "auto")), 1)
            self.assertEqual(hk.read_last_check(root / "backups" / "auto"), datetime.now().strftime("%Y%m%d"))

    async def test_source_size_counts_databases_and_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = make_db(root / "a.db", rows=100)
            (root / "hist").mkdir()
            (root / "hist" / "x.jsonl").write_bytes(b"x" * 1000)
            self.assertEqual(hk.source_size([db, root / "missing.db"], [("h", root / "hist")]), db.stat().st_size + 1000)


if __name__ == "__main__":
    unittest.main()
