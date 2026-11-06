"""只读查看用户画像（不打印完整 QQ，不打印聊天正文）。

用法：
  python peek-profiles.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "profiles.db"


def mask(value: str) -> str:
    value = str(value)
    return value[:2] + "***" + value[-2:] if len(value) > 5 else "***"


def main() -> int:
    if not DB.exists():
        print(f"还没有画像：{DB}")
        print("先启动机器人，让用户发一句话。")
        return 0

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM profiles ORDER BY created_at").fetchall()
    if not rows:
        print("画像库是空的。")
        return 0

    for row in rows:
        try:
            info = json.loads(row["basic_info"] or "{}")
        except json.JSONDecodeError:
            info = {}
        try:
            scores = json.loads(row["scores"] or "{}")
        except json.JSONDecodeError:
            scores = {}
        uid = str(row["user_id"])
        private = con.execute(
            "SELECT COUNT(*) AS n FROM user_messages WHERE user_id=? AND scene='private'", (uid,)
        ).fetchone()["n"]
        group = con.execute(
            "SELECT COUNT(*) AS n FROM user_messages WHERE user_id=? AND scene='group'", (uid,)
        ).fetchone()["n"]
        print(f"QQ：{mask(uid)}")
        print(f"昵称：{row['nickname'] or '未知'}")
        print(f"身份：{row['role'] if 'role' in row.keys() else 'user'}")
        print(f"基础信息：{json.dumps(info, ensure_ascii=False)}")
        print(f"性格：{row['personality'] or '还在观察'}")
        print("六维：" + " / ".join(f"{k} {float(scores.get(k, 50)):.1f}" for k in ("AGG","INT","HUM","DEP","RAT","ACT")))
        print(f"记录：私聊 {private} 条；群聊 {group} 条")
        print("-" * 50)
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
