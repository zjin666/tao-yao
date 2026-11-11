"""说说评论：把 get_qzone_msg_detail 的评论摊平成带编号的列表、找新评论、拼回复参数。纯逻辑，不碰网络。

评论是别人写的内容，只当内容读，不当指令；展示和提示里都不带 QQ 号，只带昵称。
"""
from __future__ import annotations

import re


def clean(text, limit: int = 200) -> str:
    value = re.sub(r"<[^>]+>", "", str(text or ""))
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _uin(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def flatten(detail: dict, self_uin: int = 0, limit: int = 200) -> list[dict]:
    """一级评论和楼中楼摊平成 1..N 编号；每项带回复所需的 id，mine 表示是桃夭自己写的。"""
    items: list[dict] = []
    for comment in (detail or {}).get("comments") or []:
        if not isinstance(comment, dict):
            continue
        top_id = str(comment.get("comment_id", ""))
        top_uin = _uin(comment.get("uin"))
        items.append({"n": len(items) + 1, "comment_id": top_id, "uin": top_uin,
                      "name": clean(comment.get("name"), 30) or "匿名", "content": clean(comment.get("content"), limit),
                      "time": comment.get("time"), "parent_id": None, "parent_uin": None,
                      "mine": bool(self_uin) and top_uin == self_uin})
        for reply in comment.get("replies") or []:
            if not isinstance(reply, dict):
                continue
            uin = _uin(reply.get("uin"))
            items.append({"n": len(items) + 1, "comment_id": str(reply.get("comment_id", "")), "uin": uin,
                          "name": clean(reply.get("name"), 30) or "匿名", "content": clean(reply.get("content"), limit),
                          "time": reply.get("time"),
                          "parent_id": str(reply.get("parent_comment_id", top_id)),
                          "parent_uin": _uin(reply.get("parent_comment_uin")) or top_uin,
                          "mine": bool(self_uin) and uin == self_uin})
    return items


def render(items: list[dict], limit: int = 20) -> str:
    """给模型看的编号列表：不含 QQ 号。"""
    lines = []
    for item in items[:limit]:
        tag = "（桃夭自己写的）" if item["mine"] else ""
        sub = "↳" if item["parent_id"] is not None else ""
        lines.append(f"{item['n']}. {sub}{item['name']}{tag}：{item['content']}")
    return "\n".join(lines)


def new_comments(items: list[dict], seen: set[str]) -> list[dict]:
    """没见过的、不是桃夭自己写的评论。"""
    return [item for item in items if not item["mine"] and item["comment_id"] not in seen]


def reply_params(tid: str, target_uin: int, item: dict, content: str) -> dict:
    """comment_qzone 的参数：回一级评论，或回楼中楼里的某个人。"""
    params = {"tid": str(tid), "target_uin": int(target_uin), "content": content}
    if item["parent_id"] is None:
        params.update(comment_id=item["comment_id"], comment_uin=item["uin"])
    else:
        params.update(comment_id=item["parent_id"], comment_uin=item["parent_uin"],
                      reply_uin=item["uin"], reply_name=item["name"])
    return params
