import json
import os
from datetime import datetime, timedelta
import threading
import pytest

from core.compat import get_output_directory, get_temp_directory, ProgressBar, VIDEO_TYPE
from core.config import (
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    clean_user_path,
    format_yesapi_error,
)
from core.task_store import JsonTaskStore


def test_resolve_api_key_priority(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "default_api_key": "sk-default-cfg",
                "image_api_key": "sk-image-cfg",
                "llm_api_key": "",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("YESAPI_API_KEY", "sk-default-env")
    monkeypatch.setenv("YESAPI_LLM_KEY", "sk-llm-env")
    monkeypatch.setenv("YESAPI_IMAGE_KEY", "sk-image-env")

    # 1. Node input wins over everything
    assert resolve_api_key("  sk-node  ", category="image", config_path=str(cfg_file)) == "sk-node"
    # 2. Category key in config.json wins over category env & default
    assert resolve_api_key("", category="image", config_path=str(cfg_file)) == "sk-image-cfg"
    # 3. Category env wins when config category key is empty
    assert resolve_api_key("", category="llm", config_path=str(cfg_file)) == "sk-llm-env"
    # 4. Default key in config.json wins when video key not set in config or env
    monkeypatch.delenv("YESAPI_VIDEO_KEY", raising=False)
    assert resolve_api_key("", category="video", config_path=str(cfg_file)) == "sk-default-cfg"
    # 5. Env fallback when config file does not exist
    assert resolve_api_key("", category="video", config_path=str(tmp_path / "missing.json")) == "sk-default-env"


def test_resolve_base_url_normalizes_trailing_slash_and_v1(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"base_url": "https://yesapi.online/v1/"}), encoding="utf-8")
    assert resolve_base_url("", config_path=str(cfg_file)) == "https://yesapi.online"
    assert resolve_base_url("https://custom.example.com/v1") == "https://custom.example.com"
    assert resolve_base_url("https://yesapi.online///") == "https://yesapi.online"


def test_resolve_model_and_clean_user_path():
    assert resolve_model("Nano-banana-Pro", "") == "Nano-banana-Pro"
    assert resolve_model("Nano-banana-Pro", "  gpt-image-3-preview  ") == "gpt-image-3-preview"
    assert clean_user_path('  "E:\\Desktop\\prompts.csv"  ') == r"E:\Desktop\prompts.csv"
    assert clean_user_path("  'D:/imgs/a.png' ") == "D:/imgs/a.png"


def test_format_yesapi_error_classifies_fatal_and_adds_hints():
    msg, fatal = format_yesapi_error('Model "Nano-banana-Pro" is not supported（分组不匹配）', status_code=400, category="image")
    assert fatal is True
    assert "分组" in msg and "image_api_key" in msg

    msg2, fatal2 = format_yesapi_error("API_KEY_QUOTA_EXHAUSTED", status_code=402, category="image")
    assert fatal2 is True
    assert "额度" in msg2 or "耗尽" in msg2

    msg3, fatal3 = format_yesapi_error("upstream timeout", status_code=504, category="image")
    assert fatal3 is False


def test_json_task_store_thread_safe_numbering_and_trimming(tmp_path):
    store_path = tmp_path / "tasks.json"
    store = JsonTaskStore(str(store_path), max_completed_history=3)

    assert store.next_numbered_id("任务") == "任务1"
    store.upsert_task("任务1", {"status": "pending", "submitted_at": "2026-11-25 10:00:00"})
    store.upsert_task("批量任务1", {"status": "pending", "submitted_at": "2026-11-25 10:01:00"})
    assert store.next_numbered_id("任务") == "任务2"
    assert store.next_numbered_id("批量任务", exclude_prefixes=()) == "批量任务2"

    # Trim downloaded/completed beyond max_completed_history while preserving active
    for i in range(1, 6):
        store.upsert_task(
            f"done_{i}",
            {"status": "downloaded", "submitted_at": f"2026-11-25 11:0{i}:00"},
        )
    tasks = store.get_tasks()
    assert "任务1" in tasks
    assert "批量任务1" in tasks
    downloaded_keys = [k for k, v in tasks.items() if v.get("status") == "downloaded"]
    assert len(downloaded_keys) == 3
    assert set(downloaded_keys) == {"done_3", "done_4", "done_5"}


def test_task_store_detects_two_hour_expiration(tmp_path):
    store = JsonTaskStore(str(tmp_path / "t.json"))
    now = datetime(2026, 11, 25, 12, 0, 0)
    old_ts = (now - timedelta(hours=2, minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    fresh_ts = (now - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    assert store.is_expired(old_ts, ttl_seconds=7200, now=now) is True
    assert store.is_expired(fresh_ts, ttl_seconds=7200, now=now) is False


def test_compat_shims_work_outside_comfyui(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("COMFYUI_TEMP_DIR", str(tmp_path / "tmp"))
    assert os.path.isdir(get_output_directory())
    assert os.path.isdir(get_temp_directory())
    pbar = ProgressBar(100)
    pbar.update_absolute(45)
    assert pbar.current == 45
    assert isinstance(VIDEO_TYPE, str)
