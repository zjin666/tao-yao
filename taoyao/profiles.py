from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .personality import merge_traits, is_tao_nickname_fragment
from .name_preferences import remove_disallowed_names
from .sqlite_tuning import tune

DEFAULT_SCORES = {"AGG": 50.0, "INT": 50.0, "HUM": 50.0, "DEP": 50.0, "RAT": 50.0, "ACT": 50.0}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProfileStore:
    """每个 QQ 用户一份画像 + 私聊/群聊记录。"""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        tune(self._conn)
        self._lock = asyncio.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT PRIMARY KEY,
                    nickname TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT 'user',
                    basic_info TEXT NOT NULL DEFAULT '{}',
                    basic_info_group TEXT NOT NULL DEFAULT '{}',
                    basic_info_private TEXT NOT NULL DEFAULT '{}',
                    personality TEXT NOT NULL DEFAULT '',
                    scores TEXT NOT NULL DEFAULT '{}',
                    summary TEXT NOT NULL DEFAULT '',
                    summary_upto INTEGER NOT NULL DEFAULT 0,
                    relationship_private REAL NOT NULL DEFAULT 0,
                    relationship_group REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    group_id INTEGER NOT NULL DEFAULT 0,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_user_messages_user
                    ON user_messages(user_id, id);
                CREATE INDEX IF NOT EXISTS idx_user_messages_group
                    ON user_messages(user_id, scene, group_id, id);
                CREATE TABLE IF NOT EXISTS user_messages_archive (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    group_id INTEGER NOT NULL DEFAULT 0,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    archived_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_user_messages_archive_user
                    ON user_messages_archive(user_id, id);
                CREATE TABLE IF NOT EXISTS profile_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_profile_updates_user
                    ON profile_updates(user_id, id);
                CREATE TABLE IF NOT EXISTS profile_group_summaries (
                    user_id TEXT NOT NULL,
                    group_id INTEGER NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_upto INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, group_id)
                );
                CREATE TABLE IF NOT EXISTS group_summaries (
                    group_id INTEGER PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_upto INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS greetings (
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    local_date TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, kind, local_date)
                );
                CREATE TABLE IF NOT EXISTS user_memories (
                    user_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS group_topic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS group_requirements (
                    group_id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS group_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    instruction TEXT NOT NULL,
                    target_group INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_group_topic_memory_group
                    ON group_topic_memory(group_id, id);
                CREATE TABLE IF NOT EXISTS group_requirement_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'standing',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    confirmed_at TEXT NOT NULL DEFAULT '',
                    asked INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_group_requirement_items_group
                    ON group_requirement_items(group_id, status, id);
                """
            )
            item_cols = [row[1] for row in self._conn.execute("PRAGMA table_info(group_requirement_items)").fetchall()]
            if "asked" not in item_cols:
                self._conn.execute("ALTER TABLE group_requirement_items ADD COLUMN asked INTEGER NOT NULL DEFAULT 0")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS wait_habits (key TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL)")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS reminders ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, scene TEXT NOT NULL DEFAULT 'private', "
                "group_id INTEGER NOT NULL DEFAULT 0, kind TEXT NOT NULL DEFAULT 'reminder', content TEXT NOT NULL, "
                "subject TEXT NOT NULL DEFAULT '', repeat TEXT NOT NULL DEFAULT 'none', spec TEXT NOT NULL DEFAULT '{}', "
                "tz TEXT NOT NULL DEFAULT 'Asia/Shanghai', due_at REAL NOT NULL, lead_days INTEGER NOT NULL DEFAULT 0, "
                "lead_sent REAL NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0, "
                "status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, due_at)")
            # 旧版每个群只有一条要求：一次性搬进新表（长期、已生效），搬完清空旧表，不会重复搬。
            self._conn.execute(
                "INSERT INTO group_requirement_items(group_id, content, kind, status, created_at, confirmed_at) "
                "SELECT group_id, content, 'standing', 'active', updated_at, updated_at "
                "FROM group_requirements WHERE TRIM(content) != ''"
            )
            self._conn.execute("DELETE FROM group_requirements")
            # 兼容旧库：已有 profiles 表没有 role 字段时补上。
            cols = [row[1] for row in self._conn.execute("PRAGMA table_info(profiles)").fetchall()]
            if "role" not in cols:
                self._conn.execute("ALTER TABLE profiles ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
            if "summary" not in cols:
                self._conn.execute("ALTER TABLE profiles ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
            if "summary_upto" not in cols:
                self._conn.execute("ALTER TABLE profiles ADD COLUMN summary_upto INTEGER NOT NULL DEFAULT 0")
            if "basic_info_group" not in cols:
                self._conn.execute("ALTER TABLE profiles ADD COLUMN basic_info_group TEXT NOT NULL DEFAULT '{}'")
            if "basic_info_private" not in cols:
                self._conn.execute("ALTER TABLE profiles ADD COLUMN basic_info_private TEXT NOT NULL DEFAULT '{}'")
                # 旧库的基础信息混了私聊和群聊，只归到私聊，避免进入群 prompt。
                self._conn.execute("UPDATE profiles SET basic_info_private = basic_info")
            if "relationship_private" not in cols:
                self._conn.execute("ALTER TABLE profiles ADD COLUMN relationship_private REAL NOT NULL DEFAULT 0")
            if "relationship_group" not in cols:
                self._conn.execute("ALTER TABLE profiles ADD COLUMN relationship_group REAL NOT NULL DEFAULT 0")
            msg_cols = [row[1] for row in self._conn.execute("PRAGMA table_info(user_messages)").fetchall()]
            if "message_id" not in msg_cols:
                self._conn.execute("ALTER TABLE user_messages ADD COLUMN message_id TEXT NOT NULL DEFAULT ''")
            for table in ("user_messages", "user_messages_archive"):
                columns = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
                if "routing" not in columns:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN routing TEXT NOT NULL DEFAULT '{{}}'")
            self._clean_bad_basic_info()

    def _clean_bad_basic_info(self) -> None:
        """清掉旧版本误提取的关系词残句，例如「的妹妹」。"""
        def valid(item: str) -> bool:
            item = str(item).strip()
            if not item or item.startswith(("的", "地", "得")):
                return False
            return not any(word in item for word in ("你", "哥哥", "桃夭", "妹妹", "姐姐", "弟弟", "妈妈", "爸爸", "家人"))

        rows = self._conn.execute(
            "SELECT user_id, basic_info, basic_info_group, basic_info_private FROM profiles"
        ).fetchall()
        for row in rows:
            values = {}
            changed = False
            for col in ("basic_info", "basic_info_group", "basic_info_private"):
                try:
                    data = json.loads(row[col] or "{}")
                except json.JSONDecodeError:
                    data = {}
                before = json.dumps(data, ensure_ascii=False, sort_keys=True)
                targets = [data]
                if col == "basic_info_group" and data and all(
                    isinstance(v, dict) for v in data.values()
                ):
                    targets = list(data.values())
                for target in targets:
                    for key in ("hobbies", "dislikes"):
                        if isinstance(target.get(key), list):
                            target[key] = [x for x in target[key] if valid(x)]
                values[col] = json.dumps(data, ensure_ascii=False)
                if values[col] != before:
                    changed = True
            if changed:
                with self._conn:
                    self._conn.execute(
                        "UPDATE profiles SET basic_info = ?, basic_info_group = ?, basic_info_private = ? WHERE user_id = ?",
                        (values["basic_info"], values["basic_info_group"], values["basic_info_private"], row["user_id"]),
                    )

    async def _run(self, fn, *args, **kwargs):
        async with self._lock:
            return await asyncio.to_thread(fn, *args, **kwargs)

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        data = dict(row)
        for key in ("basic_info", "basic_info_group", "basic_info_private"):
            try:
                data[key] = json.loads(data.get(key) or "{}")
            except json.JSONDecodeError:
                data[key] = {}
        try:
            data["scores"] = json.loads(data.get("scores") or "{}")
        except json.JSONDecodeError:
            data["scores"] = dict(DEFAULT_SCORES)
        data["scores"] = {**DEFAULT_SCORES, **data["scores"]}
        return data

    def _ensure(self, user_id: str, nickname: str, role: str) -> dict:
        role = "owner" if role == "owner" else "user"
        row = self._conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        created = row is None
        if row is None:
            now = _now()
            with self._conn:
                self._conn.execute(
                    "INSERT INTO profiles(user_id, nickname, role, basic_info, personality, scores, created_at, updated_at) "
                    "VALUES (?, ?, ?, '{}', '', ?, ?, ?)",
                    (user_id, nickname or "", role, json.dumps(DEFAULT_SCORES), now, now),
                )
            row = self._conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        elif (nickname and nickname != row["nickname"]) or role != row["role"]:
            with self._conn:
                self._conn.execute(
                    "UPDATE profiles SET nickname = ?, role = ?, updated_at = ? WHERE user_id = ?",
                    (nickname or row["nickname"], role, _now(), user_id),
                )
            row = self._conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        if role == "owner":
            # owner 的初始关系：私聊 35 + 群聊 50 = 85。
            with self._conn:
                self._conn.execute(
                    "UPDATE profiles SET "
                    "relationship_private = MAX(relationship_private, 35), "
                    "relationship_group = MAX(relationship_group, 50), "
                    "updated_at = ? WHERE user_id = ?",
                    (_now(), user_id),
                )
            row = self._conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        data = self._row_to_dict(row)
        if created:
            data["is_new"] = True
        return data

    async def ensure(self, user_id: int | str, nickname: str = "", role: str = "user") -> dict:
        return await self._run(self._ensure, str(user_id), nickname, role)

    def _sync_roles(self, owner_qq: int) -> None:
        # 全库先降为 user，再把唯一 owner 标回去，保证不会出现两个 owner。
        with self._conn:
            self._conn.execute("UPDATE profiles SET role = 'user'")
            if owner_qq:
                self._conn.execute(
                    "UPDATE profiles SET role = 'owner', "
                    "relationship_private = MAX(relationship_private, 35), "
                    "relationship_group = MAX(relationship_group, 50), "
                    "updated_at = ? WHERE user_id = ?",
                    (_now(), str(owner_qq)),
                )

    async def sync_roles(self, owner_qq: int) -> None:
        await self._run(self._sync_roles, int(owner_qq or 0))

    def _get(self, user_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    async def get(self, user_id: int | str) -> dict | None:
        return await self._run(self._get, str(user_id))

    def _add_relationship(self, user_id: str, scene: str, amount: float) -> dict:
        field = "relationship_group" if scene == "group" else "relationship_private"
        cap = 60.0 if field == "relationship_group" else 40.0
        step = max(0.0, float(amount))
        with self._conn:
            self._conn.execute(
                f"UPDATE profiles SET {field} = MIN(?, {field} + ?), updated_at = ? WHERE user_id = ?",
                (cap, step, _now(), user_id),
            )
        row = self._conn.execute(
            "SELECT relationship_private, relationship_group FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return {
            "private": round(float(row["relationship_private"]), 2) if row else 0.0,
            "group": round(float(row["relationship_group"]), 2) if row else 0.0,
        }

    async def add_relationship(self, user_id: int | str, scene: str, amount: float) -> dict:
        return await self._run(self._add_relationship, str(user_id), str(scene), float(amount))

    def _update_scores(self, user_id: str, scores: dict, personality: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE profiles SET scores = ?, personality = ?, updated_at = ? WHERE user_id = ?",
                (json.dumps(scores, ensure_ascii=False), personality, _now(), user_id),
            )

    async def update_scores(self, user_id: int | str, scores: dict, personality: str) -> None:
        await self._run(self._update_scores, str(user_id), scores, personality)

    _ITEM_PREFIX_RE = re.compile(
        r"^(?:我|平时|喜欢|爱好|玩|开|打|看|听|做|去|吃|喝|买|学|练|写|聊|养|观察)"
    )
    _ITEM_STOPWORDS = {"而已", "什么", "这个", "那个", "这些", "那些", "之类", "等等", "东西", "事情"}

    @classmethod
    def _canonical_item(cls, value: str) -> str:
        text = re.sub(r"[\s，。！？,.!?、；;：:\"'“”‘’（）()\[\]【】]", "", str(value).lower())
        return re.sub(r"(?:啦|了|啊|呀|吧|呢|嘛|哦|喔|哈)+$", "", text)

    @classmethod
    def _same_item(cls, left: str, right: str) -> bool:
        a = cls._canonical_item(left)
        b = cls._canonical_item(right)
        if not a or not b:
            return False
        if a == b:
            return True
        a = cls._ITEM_PREFIX_RE.sub("", a)
        b = cls._ITEM_PREFIX_RE.sub("", b)
        if not a or not b:
            return False
        if a == b:
            return True
        short, long = sorted((a, b), key=len)
        return len(short) >= 2 and short in long and len(long) - len(short) <= 3

    @classmethod
    def _clean_items(cls, items) -> list[str]:
        result: list[str] = []
        for raw in items or []:
            item = str(raw).strip()
            if not item or cls._canonical_item(item) in cls._ITEM_STOPWORDS:
                continue
            result = [old for old in result if not cls._same_item(old, item)]
            result.append(item)
        return result

    @classmethod
    def _drop_conflicts(cls, items, others) -> list[str]:
        return [
            item for item in items
            if not any(cls._same_item(item, other) for other in others)
        ]

    @classmethod
    def _normalize_info(cls, info: dict) -> dict:
        data = dict(info or {})
        # Keep malformed nickname writes from any extraction/call site out of storage.
        if 'tao_nickname' in data and is_tao_nickname_fragment(data['tao_nickname']):
            data.pop('tao_nickname')
        hobbies: list[str] = []
        dislikes: list[str] = []
        for key, value in data.items():
            if key not in ("hobbies", "dislikes") or not isinstance(value, list):
                continue
            fresh = cls._clean_items(value)
            if key == "hobbies":
                hobbies = cls._clean_items([*hobbies, *fresh])
                dislikes = cls._drop_conflicts(dislikes, fresh)
            else:
                dislikes = cls._clean_items([*dislikes, *fresh])
                hobbies = cls._drop_conflicts(hobbies, fresh)
        if "hobbies" in data:
            data["hobbies"] = cls._drop_conflicts(hobbies, dislikes)
        if "dislikes" in data:
            data["dislikes"] = cls._drop_conflicts(dislikes, hobbies)
        if isinstance(data.get("traits"), list):
            data["traits"] = [
                item for item in data["traits"]
                if not any(cls._same_item(item, other) for other in [*hobbies, *dislikes])
            ]
        return data

    @classmethod
    def _merge_basic_info(cls, current: dict, info: dict) -> dict:
        current = dict(current or {})
        info = cls._normalize_info(info or {})
        hobbies = cls._clean_items(current.get("hobbies") or [])
        dislikes = cls._clean_items(current.get("dislikes") or [])
        # 旧数据里已经有矛盾时，以 dislikes 为准。
        hobbies = cls._drop_conflicts(hobbies, dislikes)
        current["hobbies"] = hobbies[:8]
        current["dislikes"] = dislikes[:8]
        for key, value in (info or {}).items():
            if key == "hobbies" and isinstance(value, list):
                fresh = cls._clean_items(value)
                # 同列表去重：后说的保留。
                hobbies = cls._clean_items([*hobbies, *fresh])
                # 新喜欢覆盖旧的不喜欢。
                dislikes = cls._drop_conflicts(dislikes, fresh)
                current["hobbies"] = cls._drop_conflicts(hobbies, dislikes)[:8]
                current["dislikes"] = dislikes[:8]
            elif key == "dislikes" and isinstance(value, list):
                fresh = cls._clean_items(value)
                # 同列表去重：后说的保留。
                dislikes = cls._clean_items([*dislikes, *fresh])
                # 新不喜欢覆盖旧的喜欢。
                hobbies = cls._drop_conflicts(hobbies, fresh)
                current["hobbies"] = hobbies[:8]
                current["dislikes"] = cls._drop_conflicts(dislikes, hobbies)[:8]
            elif key == "traits" and isinstance(value, list):
                current["traits"] = merge_traits(current.get("traits") or [], value)
            elif key in ("relationships", "affection") and isinstance(value, list):
                old = current.get(key) or []
                current[key] = list(dict.fromkeys(
                    [str(x).strip() for x in old if str(x).strip()]
                    + [str(x).strip() for x in value if str(x).strip()]
                ))[:8]
            elif key == 'disallowed_names' and isinstance(value,list):
                current[key]=list(dict.fromkeys([*(current.get(key) or []),*value]))
            else:
                # 名字/年龄/性别/所在地等标量：新值直接覆盖旧值。
                current[key] = value
        return remove_disallowed_names(current)

    def _update_basic_info(self, user_id: str, info: dict, scene: str, group_id: int = 0) -> None:
        row = self._conn.execute(
            "SELECT basic_info, basic_info_group, basic_info_private FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return
        def load(value):
            try:
                return json.loads(value or "{}")
            except json.JSONDecodeError:
                return {}
        info=dict(info)
        blocked=load(row['basic_info']).get('disallowed_names') or []
        if blocked:info['disallowed_names']=list(dict.fromkeys([*blocked,*(info.get('disallowed_names') or [])]))
        combined = self._merge_basic_info(load(row["basic_info"]), info)
        group_map = load(row["basic_info_group"])
        # 兼容旧的扁平 group 信息，包到 group_id=0 下，不再跨群混用。
        if any(k in group_map for k in ("name", "age", "gender", "hobbies", "dislikes", "relationships", "affection")):
            group_map = {"0": group_map}
        private_info = load(row["basic_info_private"])
        if scene == "group":
            key = str(int(group_id or 0))
            group_map[key] = self._merge_basic_info(group_map.get(key) or {}, info)
        else:
            private_info = self._merge_basic_info(private_info, info)
        with self._conn:
            self._conn.execute(
                "UPDATE profiles SET basic_info = ?, basic_info_group = ?, basic_info_private = ?, updated_at = ? WHERE user_id = ?",
                (
                    json.dumps(combined, ensure_ascii=False),
                    json.dumps(group_map, ensure_ascii=False),
                    json.dumps(private_info, ensure_ascii=False),
                    _now(),
                    user_id,
                ),
            )

    async def update_basic_info(
        self, user_id: int | str, info: dict, scene: str = "private", group_id: int = 0
    ) -> None:
        if info:
            await self._run(self._update_basic_info, str(user_id), info, scene, int(group_id or 0))

    def _reject_user_names(self, user_id: str, names: list[str]) -> None:
        row=self._conn.execute('SELECT basic_info,basic_info_private,basic_info_group FROM profiles WHERE user_id=?',(user_id,)).fetchone()
        if row is None:return
        combined,private,groups=(json.loads(value or '{}') for value in row)
        combined=remove_disallowed_names(combined,names)
        private=remove_disallowed_names(private,names)
        # Only this user's alias fields, including stale copies in other scenes.
        if any(k in groups for k in ('name','aliases','nicknames')):
            groups=remove_disallowed_names(groups,names)
        else:
            groups={gid:remove_disallowed_names(value,names) for gid,value in groups.items()}
        with self._conn:
            self._conn.execute('UPDATE profiles SET basic_info=?,basic_info_private=?,basic_info_group=?,updated_at=? WHERE user_id=?',
                (json.dumps(combined,ensure_ascii=False),json.dumps(private,ensure_ascii=False),json.dumps(groups,ensure_ascii=False),_now(),user_id))

    async def reject_user_names(self, user_id: int | str, names: list[str]) -> None:
        if names:await self._run(self._reject_user_names,str(user_id),names)

    def _set_summary(self, user_id: str, summary: str, upto: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE profiles SET summary = ?, summary_upto = ?, updated_at = ? WHERE user_id = ?",
                (summary, int(upto), _now(), user_id),
            )

    def _count_after(self, user_id: str, upto: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM user_messages WHERE user_id = ? AND id > ?",
            (user_id, int(upto)),
        ).fetchone()
        return int(row["n"]) if row else 0

    def _messages_after(self, user_id: str, upto: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, role, content FROM user_messages "
            "WHERE user_id = ? AND id > ? ORDER BY id LIMIT ?",
            (user_id, int(upto), int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    def _prune_messages(self, user_id: str, upto: int, keep: int = 35) -> int:
        """摘要后把旧原文移到归档表，热表只留最近 keep 条。"""
        keep = max(1, int(keep))
        row = self._conn.execute(
            "SELECT id FROM user_messages WHERE user_id = ? ORDER BY id DESC LIMIT 1 OFFSET ?",
            (user_id, keep - 1),
        ).fetchone()
        if not row:
            return 0
        cutoff = int(row["id"])
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO user_messages_archive"
                "(id, user_id, scene, group_id, role, content, created_at, routing, archived_at) "
                "SELECT id, user_id, scene, group_id, role, content, created_at, routing, ? "
                "FROM user_messages WHERE user_id = ? AND id < ? AND id <= ?",
                (_now(), user_id, cutoff, int(upto)),
            )
            cur = self._conn.execute(
                "DELETE FROM user_messages WHERE user_id = ? AND id < ? AND id <= ?",
                (user_id, cutoff, int(upto)),
            )
        return int(cur.rowcount or 0)

    async def maybe_summarize(
        self, user_id: int | str, llm, after: int = 50, max_chars: int = 500, keep_raw: int = 35
    ) -> str:
        """每 50 条新消息把用户历史压成一段画像摘要；热表保留最近 35 条原文。"""
        if after <= 0:
            return ""
        uid = str(user_id)
        profile = await self.get(uid)
        if not profile:
            return ""
        upto = int(profile.get("summary_upto") or 0)
        count = await self._run(self._count_after, uid, upto)
        if count < after:
            return ""
        rows = await self._run(self._messages_after, uid, upto, 80)
        if not rows:
            return ""
        transcript = "\n".join(
            f"{'用户' if row['role'] == 'user' else '桃夭'}：{row['content']}" for row in rows
        )
        old = str(profile.get("summary") or "")
        prompt = (
            "你是用户画像整理器。把下面这段聊天整理成一段简短的中文画像摘要，"
            f"不超过 {max_chars} 字。只保留关于这个用户的事实：基础信息、喜好/厌恶、"
            "重要经历、近期状态、沟通偏好和约定；不要编造，不要写分析过程。\n"
            "如果新信息与旧摘要冲突，以新信息为准，不要同时保留矛盾项。\n\n"
            f"旧摘要：\n{old or '（无）'}\n\n新聊天：\n{transcript}"
        )
        summary = await llm.chat(
            [
                {"role": "system", "content": "你只负责压缩用户画像，输出一段中文事实摘要。"},
                {"role": "user", "content": prompt},
            ]
        )
        summary = str(summary or "").strip()
        if not summary:
            return ""
        upto = int(rows[-1]["id"])
        await self._run(self._set_summary, uid, summary[: max(50, max_chars)], upto)
        await self._run(self._prune_messages, uid, upto, keep_raw)
        return summary

    def _add_message(
        self, user_id: str, scene: str, group_id: int, role: str, content: str, message_id: str = "", routing: dict | None = None
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO user_messages(user_id, scene, group_id, role, content, message_id, created_at, routing) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, scene, int(group_id or 0), role, content, message_id, _now(), json.dumps(routing or {}, ensure_ascii=False)),
            )

    async def add_message(
        self, user_id: int | str, scene: str, group_id: int, role: str, content: str,
        message_id: str = "",
        routing: dict | None = None,
    ) -> None:
        await self._run(
            self._add_message, str(user_id), scene, int(group_id or 0), role, content, message_id, routing
        )

    def _impression_seed(self):
        # Deliberately do not select personality, scores, summaries or private fields.
        rows=self._conn.execute("SELECT m.id,m.user_id,m.group_id,m.role,m.routing,m.content,m.message_id,m.created_at,p.nickname,p.basic_info_group FROM user_messages m LEFT JOIN profiles p ON p.user_id=m.user_id WHERE m.scene='group' AND m.role IN ('user','assistant') AND m.group_id>0 ORDER BY m.id DESC LIMIT 6000").fetchall()
        previous={};interacted=[]
        for row in reversed(rows):
            r=dict(row);prior=previous.get(r['group_id'])
            previous[r['group_id']]=r
            if r['role']!='assistant' or not prior or prior['role']!='user' or prior['user_id']!=r['user_id']:continue
            try:
                source=json.loads(prior.get('routing') or '{}');reply=json.loads(r.get('routing') or '{}')
                elapsed=(datetime.fromisoformat(r['created_at'])-datetime.fromisoformat(prior['created_at'])).total_seconds()
                if source.get('state')=='bot' and reply.get('evidence')=='bot_reply' and str(reply.get('target_user_id'))==str(r['user_id']) and 0<=elapsed<=180:
                    interacted.append(prior)
            except (ValueError,TypeError,AttributeError):continue
        counts={};result=[]
        for row in reversed(interacted):
            r=dict(row);key=(r['user_id'],r['group_id'])
            if counts.get(key,0)>=40:continue
            counts[key]=counts.get(key,0)+1
            names=[r.get('nickname') or '']
            try:
                info=json.loads(r.get('basic_info_group') or '{}').get(str(r['group_id']),{})
                names.append(info.get('name') or '')
                for field in ('aliases','nicknames'):
                    if isinstance(info.get(field),list):names.extend(info[field])
            except (ValueError,AttributeError):pass
            result.append({'user_id':r['user_id'],'group_id':r['group_id'],'names':names,
                'content':r['content'],'message_key':r['message_id'] or 'history:'+str(r['id']),
                'sent_at':r['created_at']})
        return list(reversed(result))

    async def impression_seed(self):return await self._run(self._impression_seed)

    def _add_update(self, user_id: str, content: str, details: dict) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO profile_updates(user_id, content, details, created_at) VALUES (?, ?, ?, ?)",
                (user_id, content[:500], json.dumps(details, ensure_ascii=False), _now()),
            )

    async def add_update(self, user_id: int | str, content: str, details: dict) -> None:
        await self._run(self._add_update, str(user_id), content, details)

    def _latest_update(self, user_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT content, details, created_at FROM profile_updates "
            "WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["details"] = json.loads(data.get("details") or "{}")
        except json.JSONDecodeError:
            data["details"] = {}
        return data

    async def latest_update(self, user_id: int | str) -> dict | None:
        return await self._run(self._latest_update, str(user_id))

    def _recent_context(
        self, user_id: str, scene: str, group_id: int, limit: int, include_private: bool = False
    ) -> list[dict]:
        if scene == "group":
            rows = self._conn.execute(
                "SELECT role, content, scene, group_id FROM user_messages "
                "WHERE user_id = ? AND scene = 'group' AND group_id = ? ORDER BY id DESC LIMIT ?",
                (user_id,int(group_id or 0),limit),
            ).fetchall()
        else:
            # 私聊可以看到自己的群聊记录；群聊不会看到私聊记录。
            rows = self._conn.execute(
                "SELECT role, content, scene, group_id FROM user_messages "
                "WHERE user_id = ? AND (scene = 'private' OR scene = 'group') "
                "ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    async def recent_context(
        self,
        user_id: int | str,
        scene: str,
        group_id: int,
        limit: int = 12,
        include_private: bool = False,
    ) -> list[dict]:
        return await self._run(
            self._recent_context, str(user_id), scene, int(group_id or 0), limit, include_private
        )

    def _annotation_context(self, user_id: str, scene: str, group_id: int, limit: int, message_id: str) -> list[dict]:
        # Strict scene isolation; do not use recent_context's private+group union.
        where = "scene = ? AND group_id = ?" if scene == "group" else "scene = ? AND user_id = ?"
        args = [scene, int(group_id) if scene == "group" else user_id]
        if message_id:
            # For a pre-stored group message exclude it AND anything stored later.
            cutoff = self._conn.execute(
                f"SELECT MIN(id) FROM user_messages WHERE {where} AND message_id = ?", [*args,message_id]
            ).fetchone()[0]
            if cutoff is not None:
                where += " AND id < ?"
                args.append(cutoff)
        rows = self._conn.execute(
            f"SELECT user_id,role,content,created_at,message_id FROM user_messages WHERE {where} ORDER BY id DESC LIMIT ?",
            [*args,max(0,min(20,int(limit)))],
        ).fetchall()
        speakers = {}
        out = []
        for row in reversed(rows):
            speaker = 'taoyao' if row['role']=='assistant' else ('self' if row['user_id']==user_id else speakers.setdefault(row['user_id'],f'other_{len(speakers)+1}'))
            text = re.sub(r'^\s*\[(?:群友|哥哥|用户|群聊|私聊)\]\s*','',row['content'])
            out.append(dict(speaker=speaker,text=text[-1500:],recorded_at=row['created_at']))
        return out

    async def annotation_context(self, user_id: int | str, scene: str, group_id: int, limit: int = 6, message_id: str = '') -> list[dict]:
        return await self._run(self._annotation_context,str(user_id),scene,int(group_id or 0),limit,message_id)

    def _group_context(self, group_id: int, limit: int, max_age_seconds: int = 0,
                       recall_seconds: int = 0) -> list[dict]:
        """近 max_age_seconds 内的群消息；更早的只保留桃夭实际回复过的那段
        （桃夭自己的回复和它所回应的那条消息），最远回溯 recall_seconds。"""
        recent_age = int(max_age_seconds or 0)
        recall_age = max(int(recall_seconds or 0), recent_age)
        params: list = [int(group_id)]
        age_clause, fetch_limit = "", int(limit)
        recent_cutoff = ""
        now = datetime.now(timezone.utc).timestamp()
        if recent_age > 0:
            recent_cutoff = datetime.fromtimestamp(now - recent_age, timezone.utc).isoformat(timespec="seconds")
            cutoff = datetime.fromtimestamp(now - recall_age, timezone.utc).isoformat(timespec="seconds")
            age_clause, params = " AND m.created_at >= ?", [int(group_id), cutoff]
            fetch_limit = int(limit) * 6 if recall_age > recent_age else int(limit)
        rows = self._conn.execute(
            "SELECT m.id, m.user_id, m.role, m.content, m.scene, m.group_id, m.routing, "
            "m.created_at, COALESCE(p.nickname, '') AS nickname "
            "FROM user_messages m LEFT JOIN profiles p ON p.user_id = m.user_id "
            "WHERE m.scene = 'group' AND m.group_id = ?" + age_clause +
            " ORDER BY m.id DESC LIMIT ?",
            (*params, fetch_limit),
        ).fetchall()
        rows = [dict(row) for row in reversed(rows)]
        if recent_cutoff:
            kept = []
            for i, row in enumerate(rows):
                replied_to = i + 1 < len(rows) and rows[i + 1].get("role") == "assistant"
                if (str(row.get("created_at") or "") >= recent_cutoff
                        or row.get("role") == "assistant" or replied_to):
                    kept.append(row)
            rows = kept[-int(limit):]
        return rows

    async def group_context(self, group_id: int, limit: int = 16, max_age_seconds: int = 0) -> list[dict]:
        return await self._run(self._group_context, int(group_id or 0), int(limit), int(max_age_seconds))

    def _reply_inputs(self, user_id: str, scene: str, group_id: int, limit: int,
                      include_private: bool, include_user_memory: bool,
                      include_analysis: bool, analysis_limit: int, message_id: str,
                      group_max_age_seconds: int = 0, group_recall_seconds: int = 0) -> dict:
        if scene == "group":
            history = self._group_context(group_id, limit, group_max_age_seconds, group_recall_seconds)
        else:
            history = self._recent_context(user_id, scene, group_id, limit, include_private)
        return {
            "history": history,
            "group_summary": self._get_group_summary(group_id) if scene == "group" else "",
            "group_requirements": self._get_group_requirements(group_id) if scene == "group" else [],
            "user_memory": self._get_user_memory(user_id) if include_user_memory and scene == "private" else "",
            # 评分上下文仍按场景隔离，不能直接拿私聊可见的混合历史来评分。
            "analysis_context": self._annotation_context(user_id, scene, group_id, analysis_limit, message_id)
                                if include_analysis else None,
        }

    async def reply_inputs(self, user_id: int | str, scene: str, group_id: int,
                           limit: int = 12, include_private: bool = False,
                           include_user_memory: bool = False, include_analysis: bool = False,
                           analysis_limit: int = 6, message_id: str = "",
                           group_max_age_seconds: int = 0, group_recall_seconds: int = 0) -> dict:
        """每轮一次调度，复用上下文；不跨轮缓存，不影响后续画像更新。"""
        return await self._run(self._reply_inputs, str(user_id), scene, int(group_id or 0),
                               int(limit), include_private, include_user_memory,
                               include_analysis, int(analysis_limit), str(message_id),
                               int(group_max_age_seconds), int(group_recall_seconds))

    def _private_context(self, user_id: str, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, role, content, scene, group_id FROM user_messages "
            "WHERE user_id = ? AND scene = 'private' ORDER BY id DESC LIMIT ?",
            (user_id, int(limit)),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    async def private_context(self, user_id: int | str, limit: int = 16) -> list[dict]:
        return await self._run(self._private_context, str(user_id), int(limit))

    def _get_group_summary(self, group_id: int) -> str:
        row = self._conn.execute(
            "SELECT summary FROM group_summaries WHERE group_id = ?", (int(group_id),)
        ).fetchone()
        return str(row["summary"]) if row else ""

    async def get_group_summary(self, group_id: int) -> str:
        return await self._run(self._get_group_summary, int(group_id or 0))

    def _set_group_summary(self, group_id: int, summary: str, upto: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO group_summaries(group_id, summary, summary_upto, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(group_id) DO UPDATE SET "
                "summary = excluded.summary, summary_upto = excluded.summary_upto, updated_at = excluded.updated_at",
                (int(group_id), summary, int(upto), _now()),
            )

    def _group_upto(self, group_id: int) -> int:
        row = self._conn.execute(
            "SELECT summary_upto FROM group_summaries WHERE group_id = ?", (int(group_id),)
        ).fetchone()
        return int(row["summary_upto"]) if row else 0

    def _count_group_after(self, group_id: int, upto: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM ("
            "SELECT id FROM user_messages WHERE scene='group' AND group_id=? AND id>? "
            "UNION ALL "
            "SELECT id FROM user_messages_archive WHERE scene='group' AND group_id=? AND id>?"
            ")",
            (int(group_id), int(upto), int(group_id), int(upto)),
        ).fetchone()
        return int(row["n"]) if row else 0

    def _group_messages_after(self, group_id: int, upto: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT m.id, m.user_id, m.role, m.content, COALESCE(p.nickname, '') AS nickname "
            "FROM ("
            "SELECT id, user_id, role, content FROM user_messages "
            "WHERE scene='group' AND group_id=? AND id>? "
            "UNION ALL "
            "SELECT id, user_id, role, content FROM user_messages_archive "
            "WHERE scene='group' AND group_id=? AND id>?"
            ") m LEFT JOIN profiles p ON p.user_id = m.user_id "
            "ORDER BY m.id LIMIT ?",
            (int(group_id), int(upto), int(group_id), int(upto), int(limit)),
        ).fetchall()
        return [dict(row) for row in rows]

    async def maybe_summarize_group(
        self, group_id: int, llm, after: int = 50, max_chars: int = 500
    ) -> str:
        """只总结群内公共消息，绝不混入任何私聊内容。"""
        if after <= 0:
            return ""
        gid = int(group_id)
        upto = await self._run(self._group_upto, gid)
        count = await self._run(self._count_group_after, gid, upto)
        if count < after:
            return ""
        rows = await self._run(self._group_messages_after, gid, upto, 100)
        if not rows:
            return ""
        lines = []
        for row in rows:
            if row["role"] == "assistant":
                lines.append(f"桃夭：{row['content']}")
            else:
                name = row.get("nickname") or f"群友{row.get('user_id')}"
                lines.append(f"{name}：{row['content']}")
        transcript = "\n".join(lines)
        old = await self.get_group_summary(gid)
        prompt = (
            "你是群聊记忆整理器。只根据下面的群聊记录，整理成一段简短中文摘要，"
            f"不超过 {max_chars} 字。保留群内公共事件、话题、约定和群友信息；"
            "不要写私聊内容，不要编造，不要分析过程。\n"
            "如果新信息与旧摘要冲突，以新信息为准。\n\n"
            f"旧群摘要：\n{old or '（无）'}\n\n新群聊：\n{transcript}"
        )
        summary = await llm.chat(
            [
                {"role": "system", "content": "你只负责压缩群聊公共记忆，输出中文事实摘要。"},
                {"role": "user", "content": prompt},
            ]
        )
        summary = str(summary or "").strip()
        if not summary:
            return ""
        await self._run(self._set_group_summary, gid, summary[: max(50, max_chars)], int(rows[-1]["id"]))
        return summary

    def _delete_user(self, user_id: str) -> dict:
        groups = [
            int(row["group_id"])
            for row in self._conn.execute(
                "SELECT DISTINCT group_id FROM user_messages WHERE user_id = ? "
                "UNION SELECT DISTINCT group_id FROM user_messages_archive WHERE user_id = ?",
                (user_id, user_id),
            ).fetchall()
        ]
        with self._conn:
            self._conn.execute("DELETE FROM profile_updates WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM user_messages WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM user_messages_archive WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM profile_group_summaries WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM greetings WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
            # group_summaries 是群内所有人共用的，不能因为一个用户退出就删掉。
        return {"groups": groups}

    async def delete_user(self, user_id: int | str) -> dict:
        return await self._run(self._delete_user, str(user_id))

    def _find_message(self, message_id: str, scene: str, group_id: int, user_id: str) -> dict | None:
        if scene == "group":
            row = self._conn.execute(
                "SELECT m.role, m.content, m.user_id, COALESCE(p.nickname, '') AS nickname "
                "FROM user_messages m LEFT JOIN profiles p ON p.user_id = m.user_id "
                "WHERE m.message_id = ? AND m.scene = 'group' AND m.group_id = ? "
                "ORDER BY m.id DESC LIMIT 1",
                (str(message_id), int(group_id or 0)),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT m.role, m.content, m.user_id, COALESCE(p.nickname, '') AS nickname "
                "FROM user_messages m LEFT JOIN profiles p ON p.user_id = m.user_id "
                "WHERE m.message_id = ? AND m.user_id = ? AND m.scene = 'private' "
                "ORDER BY m.id DESC LIMIT 1",
                (str(message_id), user_id),
            ).fetchone()
        return dict(row) if row else None

    async def find_message(
        self, message_id: str, scene: str, group_id: int, user_id: int | str
    ) -> dict | None:
        return await self._run(
            self._find_message, str(message_id), scene, int(group_id or 0), str(user_id)
        )

    def _all_profiles(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM profiles").fetchall()
        return [self._row_to_dict(row) for row in rows]

    async def all_profiles(self) -> list[dict]:
        return await self._run(self._all_profiles)

    def _recent_group_topics(self, group_id: int, limit: int = 12) -> list[str]:
        rows = self._conn.execute(
            "SELECT topic FROM group_topic_memory WHERE group_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (int(group_id), int(limit)),
        ).fetchall()
        return [str(row["topic"]) for row in rows if row["topic"]]

    def _add_group_topic(self, group_id: int, topic: str) -> None:
        topic = str(topic or "").strip()
        if not topic:
            return
        with self._conn:
            self._conn.execute(
                "INSERT INTO group_topic_memory(group_id, topic, created_at) VALUES (?, ?, ?)",
                (int(group_id), topic, _now()),
            )
            self._conn.execute(
                "DELETE FROM group_topic_memory WHERE group_id = ? AND id NOT IN ("
                "SELECT id FROM group_topic_memory WHERE group_id = ? "
                "ORDER BY id DESC LIMIT 50)",
                (int(group_id), int(group_id)),
            )

    async def recent_group_topics(self, group_id: int, limit: int = 12) -> list[str]:
        return await self._run(self._recent_group_topics, int(group_id), int(limit))

    async def add_group_topic(self, group_id: int, topic: str) -> None:
        await self._run(self._add_group_topic, int(group_id), topic)

    def _get_group_requirements(self, group_id: int) -> list[str]:
        """写进提示词的要求（已生效的），“所有群”(0) 在前，一次性要求带说明。"""
        from .group_requirements import format_for_prompt
        rows = self._conn.execute(
            "SELECT content, kind FROM group_requirement_items "
            "WHERE group_id IN (0, ?) AND status = 'active' ORDER BY group_id ASC, id ASC",
            (int(group_id),),
        ).fetchall()
        return [text for text in (format_for_prompt(dict(row)) for row in rows) if text.strip()]

    def _set_group_requirement(self, group_id: int, content: str) -> None:
        """兼容旧接口：用一条长期要求替换该群全部已生效的长期要求；空内容就是清空。"""
        content = str(content or "").strip()
        with self._conn:
            self._conn.execute(
                "DELETE FROM group_requirement_items WHERE group_id = ? AND status = 'active' AND kind = 'standing'",
                (int(group_id),),
            )
            if content:
                now = _now()
                self._conn.execute(
                    "INSERT INTO group_requirement_items(group_id, content, kind, status, created_at, confirmed_at) "
                    "VALUES (?, ?, 'standing', 'active', ?, ?)",
                    (int(group_id), content, now, now),
                )

    def _add_group_requirement(self, group_id: int, content: str, kind: str, status: str, created_by: str) -> int:
        now = _now()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO group_requirement_items(group_id, content, kind, status, created_by, created_at, confirmed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (int(group_id), str(content).strip(), kind, status, str(created_by or ""), now,
                 now if status == "active" else ""),
            )
            return int(cur.lastrowid)

    def _list_group_requirements(self, group_id: int, status: str = "active") -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, group_id, content, kind, status, created_by, created_at FROM group_requirement_items "
            "WHERE group_id = ? AND status = ? ORDER BY id ASC",
            (int(group_id), status),
        ).fetchall()
        return [dict(row) for row in rows]

    def _count_group_requirements(self, group_id: int) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM group_requirement_items WHERE group_id = ? AND status = 'active'",
            (int(group_id),),
        ).fetchone()[0])

    def _confirm_group_requirement(self, item_id: int, limit: int) -> str:
        """待确认 -> 已生效。返回 ok / full / missing，条数在写入这一刻再核一次。"""
        with self._conn:
            row = self._conn.execute(
                "SELECT group_id FROM group_requirement_items WHERE id = ? AND status = 'pending'",
                (int(item_id),),
            ).fetchone()
            if row is None:
                return "missing"
            active = self._conn.execute(
                "SELECT COUNT(*) FROM group_requirement_items WHERE group_id = ? AND status = 'active'",
                (int(row["group_id"]),),
            ).fetchone()[0]
            if active >= int(limit):
                return "full"
            self._conn.execute(
                "UPDATE group_requirement_items SET status = 'active', confirmed_at = ? WHERE id = ?",
                (_now(), int(item_id)),
            )
            return "ok"

    def _delete_group_requirement(self, item_id: int, group_id: int | None = None) -> bool:
        sql, args = "DELETE FROM group_requirement_items WHERE id = ?", [int(item_id)]
        if group_id is not None:
            sql, args = sql + " AND group_id = ?", args + [int(group_id)]
        with self._conn:
            return self._conn.execute(sql, args).rowcount > 0

    def _pending_group_requirement(self, created_by: str, max_age_seconds: int) -> dict | None:
        """该 owner 最新的待确认要求；超时的顺手作废。"""
        cutoff = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() - int(max_age_seconds), timezone.utc
        ).isoformat(timespec="seconds")
        with self._conn:
            self._conn.execute(
                "DELETE FROM group_requirement_items WHERE status = 'pending' AND created_at < ?", (cutoff,))
        row = self._conn.execute(
            "SELECT id, group_id, content, kind, status, created_by, created_at FROM group_requirement_items "
            "WHERE status = 'pending' AND asked = 1 AND created_by = ? ORDER BY id DESC LIMIT 1",
            (str(created_by),),
        ).fetchone()
        return dict(row) if row else None

    def _mark_group_requirement_asked(self, item_id: int) -> bool:
        """桃夭确实把确认问题发出去之后才算“已询问”；没发出去的待确认内容不能被确认。"""
        with self._conn:
            return self._conn.execute(
                "UPDATE group_requirement_items SET asked = 1 WHERE id = ? AND status = 'pending'",
                (int(item_id),)).rowcount > 0

    def _clear_pending_group_requirements(self, created_by: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM group_requirement_items WHERE status = 'pending' AND created_by = ?",
                (str(created_by),))

    def _load_wait_habits(self) -> dict[str, dict]:
        rows = self._conn.execute("SELECT key, data FROM wait_habits").fetchall()
        result = {}
        for row in rows:
            try:
                value = json.loads(row["data"])
            except ValueError:
                continue
            if isinstance(value, dict):
                result[str(row["key"])] = value
        return result

    def _save_wait_habit(self, key: str, data: dict) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO wait_habits(key, data, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
                (str(key), json.dumps(data, ensure_ascii=False), _now()))

    async def load_wait_habits(self) -> dict[str, dict]:
        return await self._run(self._load_wait_habits)

    async def save_wait_habit(self, key: str, data: dict) -> None:
        await self._run(self._save_wait_habit, str(key), data)

    # ---------- 提醒、生日和纪念日 ----------

    def _reminder_row(self, row) -> dict:
        data = dict(row)
        try:
            data["spec"] = json.loads(data.get("spec") or "{}")
        except ValueError:
            data["spec"] = {}
        return data

    def _add_reminder(self, user_id: str, scene: str, group_id: int, item: dict, lead_sent: float) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO reminders(user_id, scene, group_id, kind, content, subject, repeat, spec, tz, due_at, "
                "lead_days, lead_sent, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (str(user_id), scene, int(group_id or 0), item["kind"], item["content"], item.get("subject", ""),
                 item["repeat"], json.dumps(item["spec"], ensure_ascii=False), item["tz"], float(item["due_at"]),
                 int(item.get("lead_days") or 0), float(lead_sent), _now()))
            return int(cursor.lastrowid)

    def _active_reminders(self, user_id: str, scene: str = "", group_id: int = 0) -> list[dict]:
        sql = "SELECT * FROM reminders WHERE user_id = ? AND status = 'active'"
        args: list = [str(user_id)]
        if scene == "group":                      # 群里只看这个群里设的，别人私聊设的（尤其生日）不在群里露出
            sql += " AND scene = 'group' AND group_id = ?"
            args.append(int(group_id or 0))
        sql += " ORDER BY due_at"
        return [self._reminder_row(r) for r in self._conn.execute(sql, args).fetchall()]

    def _due_reminders(self, now: float) -> list[dict]:
        """到点的，以及带提前提醒、提前提醒也到点了的。"""
        rows = self._conn.execute(
            "SELECT * FROM reminders WHERE status = 'active' AND (due_at <= ? OR "
            "(lead_days > 0 AND lead_sent != due_at AND due_at - lead_days * 86400 <= ?)) ORDER BY due_at LIMIT 50",
            (float(now), float(now))).fetchall()
        return [self._reminder_row(r) for r in rows]

    def _update_reminder(self, reminder_id: int, **fields) -> None:
        allowed = {"due_at", "lead_sent", "attempts", "status", "spec"}
        sets, args = [], []
        for key, value in fields.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                args.append(json.dumps(value, ensure_ascii=False) if key == "spec" else value)
        if sets:
            with self._conn:
                self._conn.execute(f"UPDATE reminders SET {', '.join(sets)} WHERE id = ?", [*args, int(reminder_id)])

    def _cancel_reminder(self, reminder_id: int, user_id: str) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND user_id = ? AND status = 'active'",
                (int(reminder_id), str(user_id)))
            return cursor.rowcount > 0

    def _count_reminders(self, user_id: str) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM reminders WHERE user_id = ? AND status = 'active'", (str(user_id),)).fetchone()[0])

    async def add_reminder(self, user_id, scene: str, group_id: int, item: dict, lead_sent: float = 0.0) -> int:
        return await self._run(self._add_reminder, str(user_id), scene, int(group_id or 0), item, lead_sent)

    async def active_reminders(self, user_id, scene: str = "", group_id: int = 0) -> list[dict]:
        return await self._run(self._active_reminders, str(user_id), scene, int(group_id or 0))

    async def due_reminders(self, now: float) -> list[dict]:
        return await self._run(self._due_reminders, float(now))

    async def update_reminder(self, reminder_id: int, **fields) -> None:
        await self._run(self._update_reminder, int(reminder_id), **fields)

    async def cancel_reminder(self, reminder_id: int, user_id) -> bool:
        return await self._run(self._cancel_reminder, int(reminder_id), str(user_id))

    async def count_reminders(self, user_id) -> int:
        return await self._run(self._count_reminders, str(user_id))

    # ---------- 撤回：把这条消息当作没发过 ----------

    def _remove_message_by_id(self, message_id: str, scene: str, group_id: int, user_id: str) -> list[str]:
        """按 QQ 消息 id 删掉这条消息（只限定在对应的私聊/群里），返回被删的内容，方便同步清掉别处的副本。"""
        if scene == "group":
            where, args = "message_id = ? AND scene = 'group' AND group_id = ?", [str(message_id), int(group_id or 0)]
        else:
            where, args = "message_id = ? AND scene = 'private' AND user_id = ?", [str(message_id), str(user_id)]
        removed = []
        with self._conn:
            for table in ("user_messages", "user_messages_archive"):
                if table == "user_messages_archive" and not self._has_column(table, "message_id"):
                    continue
                rows = self._conn.execute(f"SELECT id, content FROM {table} WHERE {where}", args).fetchall()
                removed += [row["content"] for row in rows]
                if rows:
                    self._conn.execute(f"DELETE FROM {table} WHERE {where}", args)
        return removed

    def _has_column(self, table: str, column: str) -> bool:
        return any(row[1] == column for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall())

    async def remove_message_by_id(self, message_id: str, scene: str, group_id: int, user_id) -> list[str]:
        return await self._run(self._remove_message_by_id, str(message_id), scene, int(group_id or 0), str(user_id))

    async def get_group_requirements(self, group_id: int) -> list[str]:
        return await self._run(self._get_group_requirements, int(group_id))

    async def set_group_requirement(self, group_id: int, content: str) -> None:
        await self._run(self._set_group_requirement, int(group_id), content)

    async def add_group_requirement(self, group_id: int, content: str, kind: str = "standing",
                                    status: str = "active", created_by: str = "") -> int:
        return await self._run(self._add_group_requirement, int(group_id), content, kind, status, created_by)

    async def list_group_requirements(self, group_id: int, status: str = "active") -> list[dict]:
        return await self._run(self._list_group_requirements, int(group_id), status)

    async def count_group_requirements(self, group_id: int) -> int:
        return await self._run(self._count_group_requirements, int(group_id))

    async def confirm_group_requirement(self, item_id: int, limit: int) -> str:
        return await self._run(self._confirm_group_requirement, int(item_id), int(limit))

    async def delete_group_requirement(self, item_id: int, group_id: int | None = None) -> bool:
        return await self._run(self._delete_group_requirement, int(item_id), group_id)

    async def pending_group_requirement(self, created_by: str, max_age_seconds: int = 600) -> dict | None:
        return await self._run(self._pending_group_requirement, str(created_by), int(max_age_seconds))

    async def mark_group_requirement_asked(self, item_id: int) -> bool:
        return await self._run(self._mark_group_requirement_asked, int(item_id))

    async def clear_pending_group_requirements(self, created_by: str) -> None:
        await self._run(self._clear_pending_group_requirements, str(created_by))

    def _add_group_task(self, user_id: str, instruction: str, target_group: int) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO group_tasks(user_id, instruction, target_group, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, instruction, int(target_group), _now()),
            )
        return int(cur.lastrowid)

    def _delete_group_task(self, task_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM group_tasks WHERE id = ?", (int(task_id),))

    async def add_group_task(self, user_id: int | str, instruction: str, target_group: int) -> int:
        return await self._run(self._add_group_task, str(user_id), instruction, int(target_group))

    async def delete_group_task(self, task_id: int) -> None:
        await self._run(self._delete_group_task, int(task_id))

    def _get_user_memory(self, user_id: str) -> str:
        row = self._conn.execute(
            "SELECT summary FROM user_memories WHERE user_id = ?", (user_id,)
        ).fetchone()
        return str(row["summary"]) if row else ""

    def _set_user_memory(self, user_id: str, summary: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO user_memories(user_id, summary, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET summary = excluded.summary, "
                "updated_at = excluded.updated_at",
                (user_id, summary, _now()),
            )

    async def get_user_memory(self, user_id: int | str) -> str:
        return await self._run(self._get_user_memory, str(user_id))

    def _recent_user_messages(self, user_id: str, limit: int = 12) -> list[str]:
        rows = self._conn.execute(
            "SELECT content FROM user_messages WHERE user_id = ? AND role = 'user' "
            "ORDER BY id DESC LIMIT ?",
            (user_id, int(limit)),
        ).fetchall()
        return [str(row["content"]) for row in reversed(rows)]

    async def recent_user_messages(self, user_id: int | str, limit: int = 12) -> list[str]:
        return await self._run(self._recent_user_messages, str(user_id), int(limit))

    async def set_user_memory(self, user_id: int | str, summary: str) -> None:
        await self._run(self._set_user_memory, str(user_id), summary)

    def _was_greeted(self, user_id: str, kind: str, local_date: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM greetings WHERE user_id = ? AND kind = ? AND local_date = ?",
            (user_id, kind, local_date),
        ).fetchone()
        return row is not None

    def _mark_greeted(self, user_id: str, kind: str, local_date: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO greetings(user_id, kind, local_date, sent_at) VALUES (?, ?, ?, ?)",
                (user_id, kind, local_date, _now()),
            )

    async def was_greeted(self, user_id: int | str, kind: str, local_date: str) -> bool:
        return await self._run(self._was_greeted, str(user_id), kind, local_date)

    async def mark_greeted(self, user_id: int | str, kind: str, local_date: str) -> None:
        await self._run(self._mark_greeted, str(user_id), kind, local_date)

    def _stats(self, user_id: str) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT scene, COUNT(*) AS n FROM user_messages WHERE user_id = ? GROUP BY scene",
            (user_id,),
        ).fetchall()
        data = {"private": 0, "group": 0}
        for row in rows:
            data[str(row["scene"])] = int(row["n"])
        return data

    async def stats(self, user_id: int | str) -> dict[str, int]:
        return await self._run(self._stats, str(user_id))

    def close(self) -> None:
        self._conn.close()
