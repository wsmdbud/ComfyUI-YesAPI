import json
import os
from datetime import datetime, timedelta
import threading
import pytest

from core.compat import get_output_directory, get_temp_directory, ProgressBar, VIDEO_TYPE
from core.config import (
    IMAGE_ASPECT_RATIOS,
    IMAGE_RESOLUTIONS,
    LLM_MODELS,
    model_family,
    strip_model_label,
    resolve_api_key,
    resolve_base_url,
    resolve_image_size,
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


def test_llm_catalog_has_group_labels_and_unique_ids():
    assert "[Gemini] gemini-4-carbon" in LLM_MODELS
    assert "[Gemini] gemini-4-neon" in LLM_MODELS
    assert "[Gemini-VT] gemini-3.1-pro-preview" in LLM_MODELS
    assert "[ChatGPT] codex-auto-review" in LLM_MODELS
    assert "[Claude] claude-fable-5-2" in LLM_MODELS
    assert "[国模CN] deepseek-v4.1-flash" in LLM_MODELS
    ids = [strip_model_label(m) for m in LLM_MODELS]
    assert len(ids) == len(set(ids))
    assert all(not i.startswith("[") for i in ids)
    assert "claude-fable-5.2" not in ids  # does not exist on model plaza
    assert len(ids) >= 40
    assert resolve_model("[Claude] claude-opus-5.5", "") == "claude-opus-5.5"
    assert resolve_model("[Claude] claude-opus-5.5", " my-new-model ") == "my-new-model"


def test_model_family_detection():
    assert model_family("gemini-4-carbon") == "gemini"
    assert model_family("gemini-3.1-pro-preview") == "gemini_vt"
    assert model_family("gemini-3.7-flash") == "gemini_vt"
    assert model_family("gemini-3.6-flash") == "gemini_vt"
    assert model_family("gemini-3.1-flash-lite-preview") == "gemini_vt"
    assert model_family("[Gemini-VT] gemini-3.8-flash") == "gemini_vt"
    assert model_family("[Gemini] gemini-3.8-flash") == "gemini"
    assert model_family("gpt-6.2-sol") == "openai"
    assert model_family("codex-auto-review") == "openai"
    assert model_family("claude-sonnet-9") == "claude"  # prefix fallback for future models
    assert model_family("kimi-k9") == "cn"
    assert model_family("glm-6") == "cn"
    assert model_family("mystery-x") is None


def test_llm_key_routes_by_model_group(tmp_path, monkeypatch):
    for env in (
        "YESAPI_API_KEY", "YESAPI_LLM_KEY", "YESAPI_GEMINI_KEY", "YESAPI_GEMINI_VT_KEY",
        "YESAPI_OPENAI_KEY", "YESAPI_CLAUDE_KEY", "YESAPI_CN_KEY",
    ):
        monkeypatch.delenv(env, raising=False)
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "llm_api_key": "sk-llm-generic",
                "gemini_api_key": "sk-gem",
                "gemini_vt_api_key": "",
                "openai_api_key": "sk-gpt",
                "claude_api_key": "sk-cla",
                "cn_api_key": "sk-cn",
            }
        ),
        encoding="utf-8",
    )
    cp = str(cfg_file)
    assert resolve_api_key("", category="llm", model="gemini-4-carbon", config_path=cp) == "sk-gem"
    assert resolve_api_key("", category="llm", model="gpt-6", config_path=cp) == "sk-gpt"
    assert resolve_api_key("", category="llm", model="claude-opus-5.5", config_path=cp) == "sk-cla"
    assert resolve_api_key("", category="llm", model="deepseek-v4-pro", config_path=cp) == "sk-cn"
    # Unknown model falls back to generic llm_api_key
    assert resolve_api_key("", category="llm", model="mystery-x", config_path=cp) == "sk-llm-generic"
    # Gemini-VT model falls back to gemini key when VT key is empty
    assert resolve_api_key("", category="llm", model="gemini-3.1-pro-preview", config_path=cp) == "sk-gem"
    monkeypatch.setenv("YESAPI_GEMINI_VT_KEY", "sk-vt-env")
    assert resolve_api_key("", category="llm", model="gemini-3.1-pro-preview", config_path=cp) == "sk-vt-env"
    # Node input still wins
    assert resolve_api_key("sk-node", category="llm", model="gpt-6", config_path=cp) == "sk-node"
    # Group name usable directly as category (for model list node)
    assert resolve_api_key("", category="claude", config_path=cp) == "sk-cla"


def test_aspect_ratio_and_resolution_are_strictly_separated():
    # Aspect ratios must NEVER contain resolution tiers or WxH pixels
    for ar in IMAGE_ASPECT_RATIOS:
        assert ar in {"auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"}
    # Resolutions must NEVER contain aspect ratios
    for res in IMAGE_RESOLUTIONS:
        assert ":" not in res

    assert resolve_image_size("1:1", "1K") == "1024x1024"
    assert resolve_image_size("16:9", "1K") == "1920x1080"
    assert resolve_image_size("16:9", "4K") == "3840x2160"
    assert resolve_image_size("9:16", "2K") == "1440x2560"
    assert resolve_image_size("1:1", "512 (小)") == "512x512"
    assert resolve_image_size("auto", "默认") is None
    assert resolve_image_size("auto", "1K") is None
    assert resolve_image_size("auto", "2K") == "2K"
    assert resolve_image_size("auto", "4K") == "4K"
    assert resolve_image_size("16:9", "2K", "1792×1024") == "1792x1024"


def test_format_yesapi_error_classifies_fatal_and_adds_hints():
    msg, fatal = format_yesapi_error('Model "Nano-banana-Pro" is not supported（分组不匹配）', status_code=400, category="image")
    assert fatal is True
    assert "分组" in msg and "image_api_key" in msg

    msg2, fatal2 = format_yesapi_error("API_KEY_QUOTA_EXHAUSTED", status_code=402, category="image")
    assert fatal2 is True
    assert "额度" in msg2 or "耗尽" in msg2

    msg3, fatal3 = format_yesapi_error("upstream timeout", status_code=504, category="image")
    assert fatal3 is False

    msg4, fatal4 = format_yesapi_error("Unauthorized", status_code=401, category="claude")
    assert fatal4 is True
    assert "claude_api_key" in msg4


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
