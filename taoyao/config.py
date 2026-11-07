from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "bot_name": "桃夭",
    "text_provider": "deepseek",
    "privacy_index": {
        "enabled": True,
        "db": "private/privacy-index.sqlite3",
        "interval_seconds": 60,
        "batch_size": 8,
    },
    "qwen": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-plus",
        "key_file": "private/qwen-key.txt",
        "thinking": False,
        "temperature": 1.0,
        "max_tokens": 600,
        "timeout": 60,
    },
    "onebot": {
        "host": "127.0.0.1",
        "port": 6199,
        "access_token": "",
    },
    "deepseek": {
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_file": "private/deepseek-key.txt",
        "temperature": 0.8,
        "max_tokens": 600,
        "timeout": 60,
    },
    "owner_qq": 0,
    "impressions": {
        "enabled": True,
        "db": "",
        "minimum_messages": 8,
        "update_every": 5,
    },
    "knowledge": {
        "enabled": True,
        "db": "",
        "max_references": 3,
        "semantic_threshold": 0.55,
        "summarize": True,
        "summary_batch_size": 20,
        "summary_max_chars": 1000,
    },
    "language_analysis": {
        "enabled": False,
        "directory": "data/annotation_samples",
        "model": "deepseek-flash",
        "thinking": False,
        "timeout": 15,
        "max_tokens": 1800,
        "minimum_confidence": 0.65,
        "concurrency": 2,
        "context_messages": 6,
        "timezone": "Asia/Shanghai",
        "background_group_probability": 0.10,
    },
    "allowed_groups": [],
    "group_require_at": True,
    "group_followup_seconds": 40,
    "group_followup_similarity": 0.50,
    "group_followup_use_llm": True,
    "roles": {
        "allow_user_private": True,
        "allow_all_groups": False,
        "auto_whitelist_joined_groups": True,
        "owner_group_require_at": True,
        "user_context_messages": 6,
        "group_context_messages": 10,
        "user_max_chars": 500,
        "user_rate_seconds": 2,
        "user_rate_per_minute": 12,
        "user_meme_probability": 0.2,
    },
    "batch": {
        "enabled": True,
        "window_seconds": 0.8,
        "max_messages": 6,
    },
    "links": {
        "enabled": True,
        "timeout": 15,
        "max_chars": 2000,
        "max_bytes": 1048576,
        "max_urls": 2,
    },
    "media_memory": {
        "enabled": True,
        "dir": "data/media_memory",
        "ttl_seconds": 86400,
        "max_items": 8,
    },
    "response_speed": {
        "two_part_reply": False,
        "speculative_reply": True,
        "defer_six_axis": True,
        "fast_helpers": True,
        "utility_model": "deepseek-flash",
        "utility_max_tokens": 220,
        "utility_timeout": 12,
    },
    "search": {
        "fast_daily_gate": True,
        "enabled": True,
        "max_results": 5,
        "decision_llm": True,
    },
    "poke": {
        "enabled": True,
        "cooldown_seconds": 5,
    },
    "poke_reply": {
        "enabled": True,
        "emotion_probability": 0.20,
        "notify_poke_probability": 0.55,
        "recent_poke_seconds": 900,
        "flow_seconds": 300,
        "cooldown_seconds": 60,
        "max_per_hour": 4,
    },
    "group_tasks": {
        "enabled": True,
        "max_targets": 10,
    },
    "memory_requests": {
        "enabled": True,
        "max_chars": 500,
        "max_items": 12,
    },
    "qzone": {
        "enabled": True,
        "post_cooldown_seconds": 120,
        "max_chars": 500,
        "read_limit": 3,
    },
    "intro": {
        "enabled": True,
        "include_owner": False,
        "text": "……你好。我是桃夭。\n平时话不多，但不代表我没在听。\n以后请多关照。……先说好，我可不是对谁都这么客气。",
    },
    "group_join_greeting": {
        "enabled": True,
        "text": "大家好，我是桃夭。以后请多关照。",
    },
    "group_ambient": {
        "enabled": True,
        "threshold": 8,
        "probability": 0.35,
        "dice_probability": 0.08,
        "no_target_probability": 0.40,
        "broad_target_probability": 0.60,
        "bot_target_probability": 0.80,
        "cooldown_seconds": 300,
        "max_per_hour": 3,
        "topic_continuation_seconds": 900,
        "topic_similarity_threshold": 0.58,
        "topic_max_chain": 3,
        "topic_min_interval_seconds": 20,
        "max_consecutive_replies": 6,
        "reply_window_seconds": 120,
        "pause_seconds": 120,
    },
    "group_topics": {
        "enabled": True,
        "per_day": 2,
        "quiet_minutes": 45,
        "active_start_hour": 9,
        "active_end_hour": 23,
        "min_gap_minutes": 180,
        "timezone": "Asia/Shanghai",
    },
    "greetings": {
        "enabled": True,
        "morning_hour": 8,
        "night_hour": 23,
        "window_minutes": 10,
        "include_owner": True,
        "check_seconds": 30,
    },
    "vision": {
        "enabled": True,
        "provider": "api",
        "api_base": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",
        "api_key_file": "private/deepseek-key.txt",
        "max_frames": 4,
        "max_tokens": 2000,
        "max_media_mb": 20,
        "max_upload_mb": 6,
        "timeout": 90,
        "work_dir": "data/media",
        "detail": "low",
    },
    "reply": {
        "segment": True,
        "max_chars": 70,
        "max_segments": 4,
        "segment_gap_min_seconds": 1.0,
        "segment_gap_max_seconds": 1.6,
        "context_messages": 16,
        "polish_fixed": True,
    },
    "memes": {
        "enabled": True,
        "dir": "memes",
        "probability": 0.45,
    },
    "user_memes": {
        "enabled": True,
        "dir": "user_memes",
        "store_probability": 0.08,
        "use_probability": 0.15,
        "min_description_chars": 4,
        "min_use_score": 0.42,
        "max_files": 200,
    },
    "profile": {
        "enabled": True,
        "db": "data/profiles.db",
        "context_messages": 12,
        "summarize_after": 50,
        "summary_max_chars": 500,
        "summary_keep_raw": 35,
        "llm_extract": True,
        "default_timezone": "Asia/Shanghai",
        "scoring": {
            "ema_alpha": 0.45,
            "max_single_delta": 7,
            "delta_scale": 1.2,
            "confidence_power": 0.7,
            "confidence_floor": 0.35,
        },
        "classifier": "local",
        "local_high_confidence": 0.55,
        "local_low_confidence": 0.35,
        "local_tie_break_margin": 0.0,
        "embedding_model": "models/bge-small-zh-v1.5",
        "embedding_device": "auto",
        "embedding_threshold": 0.48,
        "embedding_temperature": 0.06,
        "adapt_strength": 0.40,
        "band_current_weight": 0.60,
    },
    "memory": {
        "enabled": True,
        "db": "data/tao-yao.db",
        "summarize_every": 20,
        "summary_max_chars": 700,
    },
    "mock_llm": False,
    "log_level": "INFO",
}


def _deep_update(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else ROOT / "config.json"
    data = copy.deepcopy(DEFAULTS)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8-sig") as fh:
            _deep_update(data, json.load(fh))
    data["_root"] = str(ROOT)
    data["_config_path"] = str(config_path)
    # Resolve paths relative to the project root.
    for section, key in (
        ("deepseek", "key_file"),
        ("qwen", "key_file"),
        ("privacy_index", "db"),
        ("memory", "db"),
        ("knowledge", "db"),
        ("impressions", "db"),
        ("memes", "dir"),
        ("user_memes", "dir"),
        ("profile", "db"),
        ("profile", "embedding_model"),
        ("language_analysis", "directory"),
        ("vision", "api_key_file"),
        ("vision", "work_dir"),
        ("vision", "ffmpeg_path"),
    ):
        value = data.get(section, {}).get(key)
        if value:
            p = Path(value)
            if not p.is_absolute():
                p = ROOT / p
            data[section][key] = str(p.resolve())
    return data


def save_config(config: dict, path: str | Path | None = None) -> None:
    """Persist runtime changes such as auto-whitelisted group IDs."""
    target = Path(path or config.get("_config_path") or (ROOT / "config.json"))
    data = {k: v for k, v in config.items() if not k.startswith("_")}
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
