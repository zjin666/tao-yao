import asyncio
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from taoyao.memory import MemoryStore
from taoyao.sqlite_tuning import WalCheckpointer, tune


def count_rows(path, table='messages'):
    conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        return conn.execute(f'select count(*) from {table}').fetchone()[0]
    finally:
        conn.close()


class TuneTests(unittest.TestCase):
    def test_tune_sets_wal_and_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = sqlite3.connect(str(Path(tmp) / 'a.db'))
            try:
                tune(conn)
                self.assertEqual(conn.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
                self.assertEqual(conn.execute('PRAGMA synchronous').fetchone()[0], 1)
            finally:
                conn.close()

    def test_stores_open_in_wal_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / 'm.db')
            try:
                self.assertEqual(store._conn.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
                self.assertEqual(store._conn.execute('PRAGMA synchronous').fetchone()[0], 1)
            finally:
                store._conn.close()


class CheckpointerTests(unittest.IsolatedAsyncioTestCase):
    async def test_checkpoint_moves_committed_rows_into_the_main_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.db'
            store = MemoryStore(path)
            try:
                store._conn.execute('PRAGMA wal_autocheckpoint=0')  # only the timed checkpoint may flush
                for i in range(20):
                    await store.add_message('s', 'user', f'消息{i}')
                # A copy of just the main file (no -wal) shows what survives losing the WAL.
                before = Path(tmp) / 'before.db'
                shutil.copy(path, before)
                try:
                    rows_before = count_rows(before)
                except sqlite3.OperationalError:
                    rows_before = 0  # even the schema is still only in the WAL
                self.assertLess(rows_before, 20)
                result = WalCheckpointer([path]).checkpoint_once()
                after = Path(tmp) / 'after.db'
                shutil.copy(path, after)
                self.assertEqual(count_rows(after), 20)
                frames, done = result['m.db']
                self.assertEqual(frames, done)
            finally:
                store._conn.close()

    async def test_store_keeps_writing_while_checkpoints_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.db'
            store = MemoryStore(path)
            try:
                checkpointer = WalCheckpointer([path], interval=0.01)
                task = asyncio.create_task(checkpointer.run())
                for i in range(50):
                    await store.add_message('s', 'user', f'消息{i}')
                    await asyncio.sleep(0.002)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self.assertEqual(count_rows(path), 50)
            finally:
                store._conn.close()

    def test_missing_or_idle_databases_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            idle = Path(tmp) / 'idle.db'
            conn = sqlite3.connect(str(idle))
            conn.execute('create table t(x)')
            conn.commit()
            conn.close()  # rollback-journal db: no -wal file
            result = WalCheckpointer([Path(tmp) / 'missing.db', idle]).checkpoint_once()
            self.assertEqual(result, {})

    def test_one_broken_database_does_not_stop_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / 'broken.db'
            broken.write_bytes(b'not a database')
            Path(str(broken) + '-wal').write_bytes(b'junk')
            good = Path(tmp) / 'good.db'
            store = MemoryStore(good)
            try:
                store._conn.execute('PRAGMA wal_autocheckpoint=0')
                store._conn.execute("insert into messages(session,role,content,created_at) values('s','user','x','t')")
                store._conn.commit()
                with self.assertLogs('taoyao.sqlite_tuning', level='ERROR'):
                    result = WalCheckpointer([broken, good]).checkpoint_once()
                self.assertIn('good.db', result)
            finally:
                store._conn.close()


if __name__ == '__main__':
    unittest.main()
