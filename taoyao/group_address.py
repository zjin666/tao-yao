"""群消息对象来自传输元数据，不把正文里的角色自称当作事实。"""
import json


def message_route(data: dict, quoted_sender: str = "", quoted_is_bot: bool = False) -> dict:
    at_ids, reply_id = [], ""
    for segment in data.get("message") or []:
        if not isinstance(segment, dict):
            continue
        payload = segment.get("data") or {}
        if segment.get("type") == "at" and payload.get("qq"):
            at_ids.append(str(payload["qq"]))
        if segment.get("type") == "reply":
            reply_id = str(payload.get("id") or payload.get("message_id") or "")
    self_id = str(data.get("self_id") or "")
    if self_id and self_id in at_ids:
        state, evidence = "bot", "explicit_at"
    elif any(uid != "all" for uid in at_ids):
        state, evidence = "other", "explicit_at"
    elif reply_id:
        if quoted_is_bot or (self_id and quoted_sender == self_id):
            state, evidence = "bot", "reply"
        elif quoted_sender:
            state, evidence = "other", "reply"
        else:
            state, evidence = "unknown", "unresolved_reply"
    elif "all" in at_ids:
        state, evidence = "broad", "at_all"
    else:
        state, evidence = "none", "unspecified"
    return dict(state=state, evidence=evidence, at_ids=at_ids,
                reply_id=reply_id, reply_user_id=quoted_sender)


def route_note(route: dict | str | None) -> str:
    if isinstance(route, str):
        try:
            route = json.loads(route)
        except (ValueError, TypeError):
            route = {}
    if not isinstance(route, dict):
        route = {}
    label = {"bot": "桃夭", "other": "其他群友", "broad": "全群", "none": "未明确", "unknown": "引用对象未确认"}.get(route.get("state"), "旧记录未标注")
    return json.dumps({"指向": label, "依据": route.get("evidence", "未标注"),
                       "at": route.get("at_ids", []), "引用消息": route.get("reply_id", ""),
                       "引用发言者": route.get("reply_user_id", ""),
                       "回复给": route.get("target_user_id", "")}, ensure_ascii=False)
