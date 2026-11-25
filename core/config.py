"""Configuration, API key group resolution, model presets, and Chinese error diagnostics."""
from __future__ import annotations

import json
import os
import re
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

IMAGE_ASPECT_RATIOS = [
    "auto",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "21:9",
]

IMAGE_RESOLUTIONS = [
    "默认",
    "512 (小)",
    "1K",
    "2K",
    "4K",
]

# Pixel lookup table aligned with Yesapi ImageStudioView (ye) + standard extended ratios
ASPECT_RESOLUTION_MAP: Dict[str, Dict[str, str]] = {
    "1:1": {
        "512 (小)": "512x512",
        "默认": "1024x1024",
        "1K": "1024x1024",
        "2K": "2048x2048",
        "4K": "4096x4096",
    },
    "16:9": {
        "512 (小)": "1280x720",
        "默认": "1920x1080",
        "1K": "1920x1080",
        "2K": "2560x1440",
        "4K": "3840x2160",
    },
    "9:16": {
        "512 (小)": "720x1280",
        "默认": "1080x1920",
        "1K": "1080x1920",
        "2K": "1440x2560",
        "4K": "2160x3840",
    },
    "4:3": {
        "512 (小)": "1024x768",
        "默认": "1600x1200",
        "1K": "1600x1200",
        "2K": "2048x1536",
        "4K": "3840x2880",
    },
    "3:4": {
        "512 (小)": "768x1024",
        "默认": "1200x1600",
        "1K": "1200x1600",
        "2K": "1536x2048",
        "4K": "2880x3840",
    },
    "3:2": {
        "512 (小)": "1080x720",
        "默认": "1536x1024",
        "1K": "1536x1024",
        "2K": "2304x1536",
        "4K": "3840x2560",
    },
    "2:3": {
        "512 (小)": "720x1080",
        "默认": "1024x1536",
        "1K": "1024x1536",
        "2K": "1536x2304",
        "4K": "2560x3840",
    },
    "21:9": {
        "512 (小)": "1280x548",
        "默认": "2560x1080",
        "1K": "2560x1080",
        "2K": "2560x1080",
        "4K": "3840x1646",
    },
}

IMAGE_QUALITIES = ["auto", "high", "medium", "low"]

# Synced from the public Yesapi model-plaza catalog. The Gemini-VT preview model
# is retained from the documented model-group example when absent from the catalog.
LLM_MODELS = [
    "[Gemini] gemini-3.8-flash",
    "[Gemini] gemini-4-argon",
    "[Gemini] gemini-4-carbon",
    "[Gemini] gemini-4-hydrogen",
    "[Gemini] gemini-4-neon",
    "[Gemini-VT] gemini-3.1-pro-preview",
    "[Gemini-VT] gemini-3.1-flash-lite-preview",
    "[Gemini-VT] gemini-3.7-flash",
    "[Gemini-VT] gemini-3.6-flash",
    "[ChatGPT] codex-auto-review",
    "[ChatGPT] gpt-5.6-luna",
    "[ChatGPT] gpt-5.6-sol",
    "[ChatGPT] gpt-5.6-terra",
    "[ChatGPT] gpt-6",
    "[ChatGPT] gpt-6-astra",
    "[ChatGPT] gpt-6-luna",
    "[ChatGPT] gpt-6-sol",
    "[ChatGPT] gpt-6.1-sol",
    "[ChatGPT] gpt-6.2-astra",
    "[ChatGPT] gpt-6.2-sol",
    "[Claude] claude-fable-5",
    "[Claude] claude-fable-5-1",
    "[Claude] claude-fable-5-2",
    "[Claude] claude-fable-5.1",
    "[Claude] claude-haiku-4-5-20251001",
    "[Claude] claude-opus-4-6",
    "[Claude] claude-opus-4-7",
    "[Claude] claude-opus-4-8",
    "[Claude] claude-opus-4.6",
    "[Claude] claude-opus-5",
    "[Claude] claude-opus-5-5",
    "[Claude] claude-opus-5.5",
    "[Claude] claude-sonnet-4-6",
    "[Claude] claude-sonnet-5",
    "[Claude] claude-sonnet-5-5",
    "[Claude] claude-sonnet-5.5",
    "[国模CN] deepseek-v4-pro",
    "[国模CN] deepseek-v4-pro-0813",
    "[国模CN] deepseek-v4.1-flash",
    "[国模CN] deepseek-v4.1-flash-1017",
    "[国模CN] glm-5.3",
    "[国模CN] glm-5.3-flash",
    "[国模CN] kimi-k3",
    "[国模CN] kimi-k3.1-pro",
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
VIDEO_RESOLUTIONS = ["默认", "480P", "720P", "1080P"]


def resolve_image_size(
    aspect_ratio: str = "1:1",
    resolution: str = "1K",
    custom_size: str = "",
) -> Optional[str]:
    """
    Combine separated aspect_ratio and resolution into Yesapi's OpenAI Images 'size' parameter.
    - custom_size non-empty: normalize '1792×1024' -> '1792x1024' and return directly.
    - aspect_ratio == 'auto': return '2K' / '4K' when explicitly requested, else None (omit size so model decides).
    - concrete ratio: map (aspect_ratio, resolution) -> exact 'WxH' pixel string.
    """
    if custom_size and custom_size.strip():
        cleaned = re.sub(r"(\d+)\s*[×xX]\s*(\d+)", r"\1x\2", custom_size.strip())
        return cleaned

    ar = (aspect_ratio or "auto").strip()
    res = (resolution or "默认").strip()

    if ar == "auto":
        if res in ("2K", "4K"):
            return res
        return None

    ratio_table = ASPECT_RESOLUTION_MAP.get(ar)
    if ratio_table:
        return ratio_table.get(res, ratio_table["1K"])
    return "1024x1024"


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


def strip_model_label(model: str) -> str:
    """Remove the display-only group prefix from a model dropdown value."""
    return re.sub(r"^\[[^\]]+\]\s*", "", (model or "").strip()).strip()


def model_family(model: str) -> Optional[str]:
    """Map known model IDs (or group-prefixed labels) to the Yesapi key group."""
    raw = (model or "").strip()
    if raw.startswith("[Gemini-VT]"):
        return "gemini_vt"
    if raw.startswith("[Gemini]"):
        return "gemini"
    if raw.startswith("[ChatGPT]"):
        return "openai"
    if raw.startswith("[Claude]"):
        return "claude"
    if raw.startswith("[国模CN]"):
        return "cn"

    model_id = strip_model_label(raw).lower()
    if model_id.startswith(("gemini-3.1-", "gemini-3.6-", "gemini-3.7-")):
        return "gemini_vt"
    if model_id.startswith("gemini-"):
        return "gemini"
    if model_id.startswith(("gpt-", "codex-")):
        return "openai"
    if model_id.startswith("claude-"):
        return "claude"
    if model_id.startswith(("deepseek-", "kimi-", "glm-")):
        return "cn"
    return None


def resolve_api_key(
    node_api_key: str = "",
    category: str = "default",
    config_path: Optional[str] = None,
    model: str = "",
) -> str:
    """
    Resolve API key in strict priority order:
    1. Non-empty node_api_key widget input
    LLM family-specific config/environment keys precede the generic LLM key.
    Gemini-VT can override its Gemini fallback with a dedicated key.
    """
    if node_api_key and node_api_key.strip():
        return node_api_key.strip()

    cfg = load_config(config_path)
    cat = (category or "default").strip().lower()

    family = model_family(model) if cat == "llm" else (cat if cat in {"gemini", "gemini_vt", "openai", "claude", "cn"} else None)
    family_chain = [family] if family else []
    if family == "gemini_vt":
        family_chain.append("gemini")
    elif family == "gemini":
        family_chain.append("gemini_vt")

    for item in family_chain:
        cfg_group_val = str(cfg.get(f"{item}_api_key", "") or "").strip()
        if cfg_group_val:
            return cfg_group_val
        env_group_val = os.environ.get(f"YESAPI_{item.upper()}_KEY", "").strip()
        if env_group_val:
            return env_group_val

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
        return strip_model_label(custom_model)
    return strip_model_label(selected_model)


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
    VALID_KEY_CATEGORIES = {"image", "llm", "video", "gemini", "gemini_vt", "openai", "claude", "cn"}
    text = (error_text or "未知错误").strip()
    lower = text.lower()
    cat_clean = (category or "default").strip().lower()
    cat_key_hint = f"{cat_clean}_api_key" if cat_clean in VALID_KEY_CATEGORIES else "default_api_key"

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
        return (f"模型不受支持: {text} {extra}", True)

    if "payload is too large" in lower or status_code == 413:
        return (
            f"请求体过大: {text}。请减少单次传入的参考图数量或缩小图像尺寸（单图上限 20MB）。",
            False,
        )

    prefix = f"HTTP {status_code}: " if status_code else ""
    return (f"{prefix}{text}", False)
