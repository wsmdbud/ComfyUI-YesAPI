"""Configuration, API key group resolution, model presets, and Chinese error diagnostics."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config.json")
DEFAULT_BASE_URL = "https://yesapi.online"
DEFAULT_TIMEOUT_SECONDS = 180

# Synced from https://yesapi.online/model-plaza (Image group: [生图Image])
IMAGE_MODELS = [
    "Nano-banana-Pro",
    "Nano-banana-2.1",
    "Nano-banana-2",
    "gpt-image-2.6-sunburst",
    "gpt-image-2.6-flare",
    "gpt-image-2.5-sunburst",
    "gpt-image-2.5-flare",
    "gpt-image-2",
]

IMAGE_SIZES = [
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "2K",
    "4K",
    "3840x2160",
]

IMAGE_QUALITIES = ["auto", "high", "medium", "low"]

# Synced from https://yesapi.online/model-plaza across Gemini / ChatGPT / Claude / 国模聚合CN groups
LLM_MODELS = [
    "gemini-4-carbon",
    "gemini-4-argon",
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.1-pro-preview",
    "gpt-6",
    "gpt-6-astra",
    "gpt-6.2-astra",
    "gpt-5.6-sol",
    "claude-sonnet-5.5",
    "claude-opus-5.5",
    "claude-fable-5.2",
    "claude-haiku-4-5-20251001",
    "deepseek-v4.1-flash",
    "deepseek-v4-pro",
    "kimi-k3.1-pro",
    "kimi-k3",
    "glm-5.3",
    "glm-5.3-flash",
]

# Synced from https://yesapi.online/guide (视频生成)
VIDEO_MODELS = [
    "Veo-3.1-fast-1080P",
    "Veo-3.1-lite-1080P",
    "无限制-MiniMax-H3-特惠",
    "官方sd2.0-fast",
    "稳seadanco2.5-720",
]

VIDEO_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load local config.json safely; return empty dict if absent or malformed."""
    target = config_path or os.environ.get("YESAPI_CONFIG_PATH", "").strip() or DEFAULT_CONFIG_PATH
    if not target or not os.path.exists(target):
        return {}
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_api_key(
    node_api_key: str = "",
    category: str = "default",
    config_path: Optional[str] = None,
) -> str:
    """
    Resolve API key in strict priority order:
    1. Non-empty node_api_key widget input
    2. Category-specific key in config.json (image_api_key / llm_api_key / video_api_key)
    3. Category-specific environment variable (YESAPI_IMAGE_KEY / YESAPI_LLM_KEY / YESAPI_VIDEO_KEY)
    4. default_api_key in config.json
    5. YESAPI_API_KEY environment variable
    """
    if node_api_key and node_api_key.strip():
        return node_api_key.strip()

    cfg = load_config(config_path)
    cat = (category or "default").strip().lower()

    if cat in ("image", "llm", "video"):
        cfg_cat_val = str(cfg.get(f"{cat}_api_key", "") or "").strip()
        if cfg_cat_val:
            return cfg_cat_val
        env_cat_val = os.environ.get(f"YESAPI_{cat.upper()}_KEY", "").strip()
        if env_cat_val:
            return env_cat_val

    cfg_default = str(cfg.get("default_api_key", "") or "").strip()
    if cfg_default:
        return cfg_default

    env_default = os.environ.get("YESAPI_API_KEY", "").strip()
    if env_default:
        return env_default

    return ""


def resolve_base_url(node_base_url: str = "", config_path: Optional[str] = None) -> str:
    """Normalize base URL by stripping trailing slashes and redundant trailing /v1."""
    raw = ""
    if node_base_url and node_base_url.strip():
        raw = node_base_url.strip()
    else:
        cfg = load_config(config_path)
        raw = (
            str(cfg.get("base_url", "") or "").strip()
            or os.environ.get("YESAPI_BASE_URL", "").strip()
            or DEFAULT_BASE_URL
        )
    normalized = raw.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized or DEFAULT_BASE_URL


def resolve_timeout(config_path: Optional[str] = None, default: int = DEFAULT_TIMEOUT_SECONDS) -> int:
    cfg = load_config(config_path)
    try:
        val = int(cfg.get("timeout_seconds", default))
        return max(10, val)
    except Exception:
        return default


def resolve_model(selected_model: str, custom_model: str = "") -> str:
    """Prefer custom_model when non-empty; otherwise use dropdown selection."""
    if custom_model and custom_model.strip():
        return custom_model.strip()
    return (selected_model or "").strip()


def clean_user_path(raw_path: str) -> str:
    """Strip whitespace and surrounding single/double quotes from copied paths."""
    if not raw_path:
        return ""
    cleaned = raw_path.strip()
    while len(cleaned) >= 2 and (
        (cleaned[0] == '"' and cleaned[-1] == '"')
        or (cleaned[0] == "'" and cleaned[-1] == "'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def format_yesapi_error(
    error_text: str,
    status_code: Optional[int] = None,
    category: str = "default",
) -> Tuple[str, bool]:
    """
    Format raw API error into an actionable Chinese message and return (message, is_fatal).
    is_fatal=True signals batch nodes to trip the circuit breaker immediately.
    """
    text = (error_text or "未知错误").strip()
    lower = text.lower()
    cat_key_hint = f"{category}_api_key" if category in ("image", "llm", "video") else "default_api_key"

    if status_code == 402 or "api_key_quota_exhausted" in lower or "insufficient_quota" in lower or "quota" in lower:
        return (
            f"API Key 额度不足或已耗尽 (HTTP {status_code or 402}): {text}。请前往 yesapi.online 检查令牌额度。",
            True,
        )

    if status_code in (401, 403) or "invalid_api_key" in lower or "unauthorized" in lower:
        return (
            f"API Key 无效或无权限 (HTTP {status_code or 401}): {text}。请检查节点 api_key 或 config.json 中的 {cat_key_hint}。",
            True,
        )

    if "分组不匹配" in text:
        group_tip = (
            "生图模型需使用绑定 [生图Image] 分组的 Key"
            if category == "image"
            else "请确认该 Key 绑定的分组包含所选模型"
        )
        return (
            f"模型与当前 Key 分组不匹配: {text}。提示：{group_tip}（可在 config.json 中分别配置 {cat_key_hint}，或在节点面板单独填写 api_key）。",
            True,
        )

    if "在售清单之外" in text or "is not supported" in lower:
        extra = (
            "（注意：若当前模型广场视频分类暂无在售模型，请等待平台上架或通过『🔍 YesAPI 可用模型查询』节点核对可用模型名）"
            if category == "video"
            else "（可使用『🔍 YesAPI 可用模型查询』节点核对当前 Key 可用的完整模型名）"
        )
        # Model unsupported is fatal for a batch using that exact model
        return (f"模型不受支持: {text} {extra}", True)

    if "payload is too large" in lower or status_code == 413:
        return (
            f"请求体过大: {text}。请减少单次传入的参考图数量或缩小图像尺寸（单图上限 20MB）。",
            False,
        )

    prefix = f"HTTP {status_code}: " if status_code else ""
    return (f"{prefix}{text}", False)
