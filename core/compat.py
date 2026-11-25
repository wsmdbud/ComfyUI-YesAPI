"""Compatibility layer for running inside ComfyUI and standalone pytest environments."""
from __future__ import annotations

import os
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import folder_paths as _folder_paths  # type: ignore
except ImportError:
    _folder_paths = None

try:
    from comfy.comfy_types import IO as _COMFY_IO  # type: ignore

    VIDEO_TYPE: Any = _COMFY_IO.VIDEO
except Exception:
    VIDEO_TYPE = "VIDEO"

try:
    import comfy.utils as _comfy_utils  # type: ignore
except Exception:
    _comfy_utils = None


def get_output_directory() -> str:
    """Return ComfyUI's output directory or fallback local output directory."""
    env_override = os.environ.get("COMFYUI_OUTPUT_DIR", "").strip()
    if env_override:
        os.makedirs(env_override, exist_ok=True)
        return env_override
    if _folder_paths is not None and hasattr(_folder_paths, "get_output_directory"):
        out_dir = _folder_paths.get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    fallback = os.path.join(REPO_ROOT, "output")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def get_temp_directory() -> str:
    """Return ComfyUI's temp directory or fallback local temp directory."""
    env_override = os.environ.get("COMFYUI_TEMP_DIR", "").strip()
    if env_override:
        os.makedirs(env_override, exist_ok=True)
        return env_override
    if _folder_paths is not None and hasattr(_folder_paths, "get_temp_directory"):
        tmp_dir = _folder_paths.get_temp_directory()
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir
    fallback = os.path.join(REPO_ROOT, "temp")
    os.makedirs(fallback, exist_ok=True)
    return fallback


class ProgressBar:
    """Thin wrapper over comfy.utils.ProgressBar with standalone fallback."""

    def __init__(self, total: int = 100) -> None:
        self.total = max(1, int(total))
        self.current = 0
        self._inner = (
            _comfy_utils.ProgressBar(self.total)
            if _comfy_utils is not None and hasattr(_comfy_utils, "ProgressBar")
            else None
        )

    def update_absolute(self, value: int, total: int | None = None) -> None:
        if total is not None:
            self.total = max(1, int(total))
        self.current = max(0, min(int(value), self.total))
        if self._inner is not None:
            try:
                self._inner.update_absolute(self.current, self.total)
            except TypeError:
                self._inner.update_absolute(self.current)

    def update(self, delta: int = 1) -> None:
        self.update_absolute(self.current + int(delta))
