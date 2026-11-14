from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from taoyao.app import TaoYaoBot
from taoyao.config import load_config
from taoyao.llm import LLMError
from taoyao.persona import load_persona

ROOT = Path(__file__).resolve().parent


def setup_logging(level: str = "INFO") -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            # 日志按大小轮转：bot.log 超过 10 MB 就换一个新的，最多留 5 份旧的（旧的会被每日整理压成 .gz）。
            RotatingFileHandler(ROOT / "logs" / "bot.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"),
        ],
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="桃夭 QQ 机器人")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--mock", action="store_true", help="使用本地假模型，不调用 DeepSeek")
    parser.add_argument("--check", action="store_true", help="只测模型/key，不启动 OneBot")
    parser.add_argument("--show-persona", action="store_true", help="打印实际加载的人设全文")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.show_persona:
        text = load_persona()
        print(f"# persona chars={len(text)}")
        print(text)
        return 0

    config = load_config(args.config)
    if args.host:
        config["onebot"]["host"] = args.host
    if args.port:
        config["onebot"]["port"] = args.port

    setup_logging(str(config.get("log_level", "INFO")))

    try:
        bot = TaoYaoBot(config, mock=args.mock)
    except LLMError as exc:
        print(f"启动失败：{exc}")
        print("把 key 粘进 private/deepseek-key.txt，或者先用 --mock 跑通消息链路。")
        return 2

    if args.check:
        print("正在测试模型...")
        reply = await bot.llm.chat([
            {"role": "system", "content": "你是桃夭，用一句很短的中文打个招呼。"},
            {"role": "user", "content": "在吗"},
        ])
        print("模型回复：", reply)
        await bot.llm.close()
        bot.memory.close()
        bot.profiles.close()
        return 0

    try:
        await bot.start()
    except OSError as exc:
        code = getattr(exc, "winerror", None) or getattr(exc, "errno", None)
        if code == 10048:
            print("6199 端口被占用了：先关掉另一个机器人窗口（mock 或真实），再启动这个。")
            await bot.stop()
            return 3
        raise
    print(f"桃夭已启动：ws://{config['onebot']['host']}:{config['onebot']['port']}")
    print("按 Ctrl+C 停止。")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await bot.stop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(0)
