#!/usr/bin/env python3
"""Famou 配置与项目文件管理工具。"""
import argparse
import json
import os
import sys

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".famou-ctl", "settings.json")
DEFAULT_API_URL = os.environ.get("FAMOU_API_URL", "https://api.example.com")
DEFAULT_USER_ID = "default"


def mask(s: str) -> str:
    return s[:3] + "***" + s[-3:] if len(s) > 6 else "***"


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: 配置文件解析失败: {e}", file=sys.stderr)
        return {}


def cmd_read():
    """读取配置文件，检查 api_url 和 api_key 是否完整"""
    settings = load_settings()
    api_url = settings.get("api_url", "").strip()
    api_key = settings.get("api_key", "").strip()

    missing = []
    if not api_url:
        missing.append("api_url")
    if not api_key:
        missing.append("api_key")

    result = {
        "status": "ok" if not missing else "missing",
        "api_url": api_url,
        "masked_key": mask(api_key) if api_key else "",
        "missing": missing
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_write(api_key: str):
    """写入 API key，使用默认的 api_url 和 user_id"""
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)

    settings = {
        "api_url": DEFAULT_API_URL,
        "api_key": api_key.strip(),
        "user_id": DEFAULT_USER_ID
    }

    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
        result = {
            "success": True,
            "message": "配置已保存",
            "config": {
                "api_url": settings["api_url"],
                "masked_key": mask(settings["api_key"])
            }
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Famou 配置与项目文件管理工具")
    parser.add_argument("command", choices=["read", "write"], help="命令：read 或 write")
    parser.add_argument("api_key", nargs="?", help="API 密钥（仅 write 命令需要）")
    
    args = parser.parse_args()

    if args.command == "read":
        cmd_read()
    elif args.command == "write":
        if not args.api_key:
            parser.error("write 命令需要 api_key 参数")
        cmd_write(args.api_key)


if __name__ == "__main__":
    main()
