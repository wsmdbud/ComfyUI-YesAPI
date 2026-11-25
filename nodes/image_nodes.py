"""Synchronous and batch image generation & editing nodes for ComfyUI-YesAPI (6 nodes)."""
from __future__ import annotations

import concurrent.futures
from datetime import datetime
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch

try:
    from ..core.client import YesAPIClient, YesAPIError
    from ..core.compat import get_output_directory, get_temp_directory
    from ..core.config import (
        IMAGE_ASPECT_RATIOS,
        IMAGE_MODELS,
        IMAGE_QUALITIES,
        IMAGE_RESOLUTIONS,
        clean_user_path,
        resolve_api_key,
        resolve_base_url,
        resolve_image_size,
        resolve_model,
        resolve_timeout,
    )
    from ..core.media import (
        is_valid_image_file,
        make_error_image_tensor,
        mask_to_png_bytes,
        read_table_file,
        safe_pil_batch_to_tensor,
        safe_pil_to_rgb,
        sanitize_filename,
        tensor_to_pil,
    )
except ImportError:
    from core.client import YesAPIClient, YesAPIError
    from core.compat import get_output_directory, get_temp_directory
    from core.config import (
        IMAGE_ASPECT_RATIOS,
        IMAGE_MODELS,
        IMAGE_QUALITIES,
        IMAGE_RESOLUTIONS,
        clean_user_path,
        resolve_api_key,
        resolve_base_url,
        resolve_image_size,
        resolve_model,
        resolve_timeout,
    )
    from core.media import (
        is_valid_image_file,
        make_error_image_tensor,
        mask_to_png_bytes,
        read_table_file,
        safe_pil_batch_to_tensor,
        safe_pil_to_rgb,
        sanitize_filename,
        tensor_to_pil,
    )


def _collect_ref_pils(kwargs: Dict[str, Any], max_slots: int) -> List[Image.Image]:
    refs: List[Image.Image] = []
    for idx in range(1, max_slots + 1):
        tensor = kwargs.get(f"image_{idx}")
        if tensor is not None:
            pils = tensor_to_pil(tensor)
            if pils:
                refs.extend(pils)
    return refs


class _YesAPIImageNodeBase:
    CATEGORY = "YesAPI/图像"
    FUNCTION = "execute"

    @staticmethod
    def _error_2tuple(msg: str) -> Dict[str, Any]:
        full = f"失败: {msg}"
        return {"ui": {"string": [full]}, "result": (make_error_image_tensor(), full)}

    @staticmethod
    def _error_3tuple(msg: str) -> Dict[str, Any]:
        full = f"失败: {msg}"
        return {"ui": {"string": [full]}, "result": (make_error_image_tensor(), full, "")}


# ==================== Node 1: YesAPIImageGenerate ====================


class YesAPIImageGenerate(_YesAPIImageNodeBase):
    """Unified multi-reference (up to 14 images) + mask editing & text-to-image node."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        optional: Dict[str, Any] = {
            "custom_model": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：填写平台新模型名覆盖上方下拉框"}),
            "custom_size": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：自定义像素尺寸如 1792x1024"}),
            "mask": ("MASK",),
        }
        for i in range(1, 15):
            optional[f"image_{i}"] = ("IMAGE",)
        optional["api_key"] = ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 image_api_key"})
        optional["base_url"] = ("STRING", {"default": "https://yesapi.online", "multiline": False})

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat wearing a tiny astronaut helmet"}),
                "model": (IMAGE_MODELS, {"default": "Nano-banana-Pro"}),
                "aspect_ratio": (IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (IMAGE_RESOLUTIONS, {"default": "1K"}),
                "quality": (IMAGE_QUALITIES, {"default": "auto"}),
                "concurrency": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "random_mode": (["固定", "随机"], {"default": "随机"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> Any:
        if kwargs.get("random_mode") == "随机":
            return time.time_ns()
        return False

    def execute(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        concurrency: int,
        random_mode: str = "随机",
        custom_model: str = "",
        custom_size: str = "",
        mask: Optional[torch.Tensor] = None,
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            return self._error_2tuple("提示词 (prompt) 不能为空。")

        final_key = resolve_api_key(api_key, category="image")
        if not final_key:
            return self._error_2tuple("API Key 不能为空。请在节点填入 api_key 或在 config.json 中配置 image_api_key。")

        final_model = resolve_model(model, custom_model)
        final_size = resolve_image_size(aspect_ratio, resolution, custom_size)
        refs = _collect_ref_pils(kwargs, max_slots=14)

        if mask is not None and not refs:
            return self._error_2tuple("使用局部重绘遮罩 (mask) 时，必须同时连接至少一张基础图像 (image_1)。")

        mask_bytes = mask_to_png_bytes(mask, target_size=refs[0].size) if mask is not None and refs else None

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="image",
                timeout=resolve_timeout(),
            )
        except YesAPIError as e:
            return self._error_2tuple(str(e))

        all_images: List[Image.Image] = []
        all_errors: List[str] = []
        abort_event = threading.Event()

        def _worker(_idx: int) -> Tuple[List[Image.Image], List[str]]:
            if abort_event.is_set():
                return [], []
            try:
                imgs = client.generate_images(
                    prompt=clean_prompt,
                    model=final_model,
                    size=final_size,
                    quality=quality,
                    ref_images=refs,
                    mask_png_bytes=mask_bytes,
                )
                return imgs, []
            except YesAPIError as err:
                if err.is_fatal:
                    abort_event.set()
                return [], [str(err)]
            except Exception as err:
                return [], [f"{type(err).__name__}: {err}"]

        workers = max(1, min(int(concurrency), 10))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for pils, errs in pool.map(_worker, range(workers)):
                all_images.extend(pils)
                all_errors.extend(errs)

        if not all_images:
            return self._error_2tuple("; ".join(all_errors) if all_errors else "未生成任何图像")

        mode_label = "图生图/遮罩编辑" if (refs or mask_bytes) else "文生图"
        size_label = final_size or "auto"
        status = f"成功: {len(all_images)} 张 | 失败: {len(all_errors)} | 模式: {mode_label} | 尺寸: {size_label} | 模型: {final_model}"
        if all_errors:
            status += f" | 首条错误: {all_errors[0]}"
        return {"ui": {"string": [status]}, "result": (safe_pil_batch_to_tensor(all_images), status)}


class YesAPIImageText2Image(YesAPIImageGenerate):
    """Text-to-image node with no reference-image or mask sockets."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs = YesAPIImageGenerate.INPUT_TYPES()
        optional = inputs["optional"]
        return {
            "required": inputs["required"],
            "optional": {
                key: optional[key]
                for key in ("custom_model", "custom_size", "api_key", "base_url")
            },
        }


# ==================== Node 2: YesAPIImageBatchCSV ====================


class YesAPIImageBatchCSV(_YesAPIImageNodeBase):
    """CSV / Excel batch image generation with retry and fatal-error circuit breaker."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽或粘贴 CSV/Excel 文件路径"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "model": (IMAGE_MODELS, {"default": "Nano-banana-Pro"}),
                "aspect_ratio": (IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (IMAGE_RESOLUTIONS, {"default": "1K"}),
                "quality": (IMAGE_QUALITIES, {"default": "auto"}),
                "concurrency": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "max_count": ("INT", {"default": 50, "min": 1, "max": 500, "step": 1}),
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "custom_size": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：自定义像素尺寸如 1792x1024"}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 image_api_key"}),
                "base_url": ("STRING", {"default": "https://yesapi.online", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_batch", "status")

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def execute(
        self,
        file_path: str,
        column_name: str,
        prompt_prefix: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        concurrency: int,
        max_count: int,
        executions_per_prompt: int,
        custom_model: str = "",
        custom_size: str = "",
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        final_key = resolve_api_key(api_key, category="image")
        if not final_key:
            return self._error_2tuple("API Key 不能为空。请配置 image_api_key 或在节点填入。")

        try:
            df = read_table_file(file_path)
        except Exception as e:
            return self._error_2tuple(str(e))

        col = (column_name or "prompt").strip()
        if col not in df.columns:
            return self._error_2tuple(f"列名 '{col}' 不存在。当前表头: {list(df.columns)}")

        raw_prompts = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()][: max(1, int(max_count))]
        if not raw_prompts:
            return self._error_2tuple(f"列 '{col}' 中没有找到有效提示词。")

        base_prompts = [f"{prompt_prefix}{p}" for p in raw_prompts]
        prompts = [p for p in base_prompts for _ in range(max(1, int(executions_per_prompt)))]

        final_model = resolve_model(model, custom_model)
        final_size = resolve_image_size(aspect_ratio, resolution, custom_size)
        refs = _collect_ref_pils(kwargs, max_slots=4)

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="image",
                timeout=resolve_timeout(),
            )
        except YesAPIError as e:
            return self._error_2tuple(str(e))

        abort_event = threading.Event()
        lock = threading.Lock()
        consecutive_failures = 0
        max_consecutive_failures = 10
        abort_reason = ""

        def _run_prompt(p_text: str) -> Tuple[List[Image.Image], List[str]]:
            nonlocal consecutive_failures, abort_reason
            if abort_event.is_set():
                return [], []

            last_err = ""
            for attempt in range(2):
                if abort_event.is_set():
                    return [], []
                try:
                    imgs = client.generate_images(
                        prompt=p_text,
                        model=final_model,
                        size=final_size,
                        quality=quality,
                        ref_images=refs,
                    )
                    with lock:
                        consecutive_failures = 0
                    return imgs, []
                except YesAPIError as err:
                    last_err = str(err)
                    if err.is_fatal:
                        with lock:
                            abort_reason = f"触发致命错误熔断: {err}"
                            abort_event.set()
                        return [], [last_err]
                    if attempt == 0:
                        time.sleep(0.5)
                except Exception as err:
                    last_err = f"{type(err).__name__}: {err}"
                    if attempt == 0:
                        time.sleep(0.5)

            with lock:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures and not abort_event.is_set():
                    abort_reason = f"连续失败达 {max_consecutive_failures} 次，触发保护熔断"
                    abort_event.set()
            return [], [last_err]

        all_images: List[Image.Image] = []
        errors: List[str] = []
        workers = max(1, min(int(concurrency), 20))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for imgs, errs in pool.map(_run_prompt, prompts):
                all_images.extend(imgs)
                errors.extend(errs)

        if not all_images:
            msg = abort_reason or ("; ".join(errors[:3]) if errors else "全部任务失败")
            return self._error_2tuple(msg)

        status = f"批量完成 | 总任务: {len(prompts)} | 成功图像: {len(all_images)} | 失败: {len(errors)} | 尺寸: {final_size or 'auto'} | 模型: {final_model}"
        if abort_reason:
            status += f" | ⚠️ {abort_reason}"
        return {"ui": {"string": [status]}, "result": (safe_pil_batch_to_tensor(all_images), status)}


# ==================== Node 3: YesAPIImageSaveWithPrompt ====================


class YesAPIImageSaveWithPrompt(_YesAPIImageNodeBase):
    """Batch CSV/Excel generation saving unpadded PNGs named after cleaned prompts."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽或粘贴 CSV/Excel 文件路径"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "filename_prefix": ("STRING", {"default": "YesAPI"}),
                "rm_prompt_prefix": ("STRING", {"multiline": False, "default": ""}),
                "rm_prompt_suffix": ("STRING", {"multiline": False, "default": ""}),
                "model": (IMAGE_MODELS, {"default": "Nano-banana-Pro"}),
                "aspect_ratio": (IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (IMAGE_RESOLUTIONS, {"default": "1K"}),
                "quality": (IMAGE_QUALITIES, {"default": "auto"}),
                "concurrency": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "max_count": ("INT", {"default": 50, "min": 1, "max": 500, "step": 1}),
                "ui_display_mode": (["保存 (Saved)", "预览 (Preview)"], {"default": "保存 (Saved)"}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "custom_size": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：自定义像素尺寸如 1792x1024"}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 image_api_key"}),
                "base_url": ("STRING", {"default": "https://yesapi.online", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "status", "filenames")

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def execute(
        self,
        file_path: str,
        column_name: str,
        prompt_prefix: str,
        filename_prefix: str,
        rm_prompt_prefix: str,
        rm_prompt_suffix: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        concurrency: int,
        max_count: int,
        ui_display_mode: str,
        custom_model: str = "",
        custom_size: str = "",
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        final_key = resolve_api_key(api_key, category="image")
        if not final_key:
            return self._error_3tuple("API Key 不能为空。请配置 image_api_key 或在节点填入。")

        try:
            df = read_table_file(file_path)
        except Exception as e:
            return self._error_3tuple(str(e))

        col = (column_name or "prompt").strip()
        if col not in df.columns:
            return self._error_3tuple(f"列名 '{col}' 不存在。")

        prompts = [f"{prompt_prefix}{str(x).strip()}" for x in df[col].dropna().tolist() if str(x).strip()][: max(1, int(max_count))]
        if not prompts:
            return self._error_3tuple(f"列 '{col}' 中没有找到有效提示词。")

        final_model = resolve_model(model, custom_model)
        final_size = resolve_image_size(aspect_ratio, resolution, custom_size)
        refs = _collect_ref_pils(kwargs, max_slots=4)

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="image",
                timeout=resolve_timeout(),
            )
        except YesAPIError as e:
            return self._error_3tuple(str(e))

        abort_event = threading.Event()
        abort_reason = ""

        def _worker(p_text: str) -> Tuple[List[Image.Image], List[str]]:
            nonlocal abort_reason
            if abort_event.is_set():
                return [], []
            try:
                imgs = client.generate_images(
                    prompt=p_text,
                    model=final_model,
                    size=final_size,
                    quality=quality,
                    ref_images=refs,
                )
                return imgs, []
            except YesAPIError as err:
                if err.is_fatal:
                    abort_reason = f"触发熔断: {err}"
                    abort_event.set()
                return [], [str(err)]
            except Exception as err:
                return [], [f"{type(err).__name__}: {err}"]

        main_output_dir = get_output_directory()
        temp_dir = get_temp_directory()
        subfolder, actual_prefix = os.path.split((filename_prefix or "YesAPI").strip())

        all_pil_images: List[Image.Image] = []
        ui_image_info: List[Dict[str, str]] = []
        saved_filenames: List[str] = []
        errors: List[str] = []
        saved_count = 0

        workers = max(1, min(int(concurrency), 20))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_worker, prompts)
            for original_prompt, (pils, errs) in zip(prompts, results):
                errors.extend(errs)
                if not pils:
                    continue
                cleaned_p = original_prompt
                if rm_prompt_prefix and cleaned_p.startswith(rm_prompt_prefix):
                    cleaned_p = cleaned_p[len(rm_prompt_prefix) :]
                if rm_prompt_suffix and cleaned_p.endswith(rm_prompt_suffix):
                    cleaned_p = cleaned_p[: -len(rm_prompt_suffix)]
                stem = sanitize_filename(cleaned_p)
                final_dir = os.path.join(main_output_dir, subfolder) if subfolder else main_output_dir
                os.makedirs(final_dir, exist_ok=True)

                for pil_img in pils:
                    rgb = safe_pil_to_rgb(pil_img)
                    all_pil_images.append(rgb)

                    base_name = f"{actual_prefix}_{stem}" if actual_prefix else stem
                    final_path = os.path.join(final_dir, f"{base_name}.png")
                    counter = 1
                    while os.path.exists(final_path):
                        final_path = os.path.join(final_dir, f"{base_name} ({counter}).png")
                        counter += 1

                    saved_filenames.append(final_path)

                    if ui_display_mode == "保存 (Saved)":
                        rgb.save(final_path, "PNG", compress_level=4)
                        saved_count += 1
                        ui_image_info.append(
                            {
                                "filename": os.path.basename(final_path),
                                "subfolder": subfolder,
                                "type": "output",
                            }
                        )
                    else:
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, os.path.basename(final_path))
                        t_counter = 1
                        while os.path.exists(temp_path):
                            t_stem = re.sub(r" \(\d+\)$", "", os.path.splitext(os.path.basename(final_path))[0])
                            temp_path = os.path.join(temp_dir, f"{t_stem} ({t_counter}).png")
                            t_counter += 1
                        rgb.save(temp_path, "PNG", compress_level=4)
                        ui_image_info.append(
                            {
                                "filename": os.path.basename(temp_path),
                                "subfolder": "",
                                "type": "temp",
                            }
                        )

        if not all_pil_images:
            return self._error_3tuple(abort_reason or ("; ".join(errors[:3]) if errors else "全部图像生成失败"))

        mode_summary = f"成功保存: {saved_count}" if ui_display_mode == "保存 (Saved)" else f"成功预览: {len(all_pil_images)}"
        status = f"批量完成 | 总Prompt: {len(prompts)} | {mode_summary} | 失败: {len(errors)} | 尺寸: {final_size or 'auto'} | 模型: {final_model}"
        if abort_reason:
            status += f" | ⚠️ {abort_reason}"

        return {
            "ui": {"string": [status], "images": ui_image_info},
            "result": (safe_pil_batch_to_tensor(all_pil_images), status, "\n".join(saved_filenames)),
        }


# ==================== Node 4: YesAPIImageBatchDir ====================


class YesAPIImageBatchDir(_YesAPIImageNodeBase):
    """Folder + CSV/fixed-prompt batch image editing with bad-file precheck, retries, and circuit breaker."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "placeholder": "输入本地图片文件夹路径"}),
                "csv_file_path": ("STRING", {"default": "", "placeholder": "可选：CSV/Excel 提示词文件路径"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "model": (IMAGE_MODELS, {"default": "Nano-banana-Pro"}),
                "aspect_ratio": (IMAGE_ASPECT_RATIOS, {"default": "auto"}),
                "resolution": (IMAGE_RESOLUTIONS, {"default": "1K"}),
                "quality": (IMAGE_QUALITIES, {"default": "auto"}),
                "max_concurrent_files": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "fixed_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "不用 CSV 时在此填固定提示词"}),
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "custom_size": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：自定义像素尺寸如 1792x1024"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 image_api_key"}),
                "base_url": ("STRING", {"default": "https://yesapi.online", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images_preview_15", "status_report")

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def execute(
        self,
        directory_path: str,
        csv_file_path: str,
        column_name: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        max_concurrent_files: int,
        executions_per_prompt: int,
        fixed_prompt: str = "",
        custom_model: str = "",
        custom_size: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        final_key = resolve_api_key(api_key, category="image")
        if not final_key:
            return self._error_2tuple("API Key 不能为空。请配置 image_api_key 或在节点填入。")

        clean_dir = clean_user_path(directory_path)
        if not clean_dir or not os.path.isdir(clean_dir):
            return self._error_2tuple(f"图片文件夹路径不存在: {clean_dir or '(空)'}")

        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_files = sorted(
            os.path.join(clean_dir, f)
            for f in os.listdir(clean_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        )
        if not image_files:
            return self._error_2tuple("文件夹内未找到支持的图片文件 (.png/.jpg/.jpeg/.webp/.bmp)。")

        base_prompts: List[str] = []
        source_mode = "Fixed Prompt"
        clean_csv = clean_user_path(csv_file_path)
        if clean_csv and os.path.exists(clean_csv):
            try:
                df = read_table_file(clean_csv)
                col = (column_name or "prompt").strip()
                if col in df.columns:
                    base_prompts = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
                    if base_prompts:
                        source_mode = "CSV/Excel"
            except Exception as e:
                return self._error_2tuple(f"读取提示词表格失败: {e}")

        if not base_prompts:
            if not fixed_prompt or not fixed_prompt.strip():
                return self._error_2tuple("必须提供有效的 CSV/Excel 文件，或填写 fixed_prompt 固定提示词。")
            base_prompts = [fixed_prompt.strip()]

        final_prompts = [p for p in base_prompts for _ in range(max(1, int(executions_per_prompt)))]
        final_model = resolve_model(model, custom_model)
        final_size = resolve_image_size(aspect_ratio, resolution, custom_size)

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="image",
                timeout=resolve_timeout(),
            )
        except YesAPIError as e:
            return self._error_2tuple(str(e))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = os.path.join(get_output_directory(), f"YesAPI_Batch_{timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)

        abort_event = threading.Event()
        lock = threading.Lock()
        consecutive_failures = 0
        MAX_FAILURES = 10
        abort_reason = ""

        def _process_single_file(img_idx: int, img_path: str) -> Tuple[str, int, List[Image.Image], List[str]]:
            nonlocal abort_reason
            fname = os.path.basename(img_path)
            if abort_event.is_set():
                return fname, 0, [], ["任务已熔断跳过"]

            if not is_valid_image_file(img_path):
                return fname, 0, [], [f"文件损坏或非有效图像: {fname}"]

            try:
                with Image.open(img_path) as raw_img:
                    ref_pil = safe_pil_to_rgb(raw_img)
            except Exception as e:
                return fname, 0, [], [f"打开图像失败: {e}"]

            base_stem = sanitize_filename(os.path.splitext(fname)[0])
            saved_count = 0
            preview_pils: List[Image.Image] = []
            file_errs: List[str] = []

            for p_idx, p_text in enumerate(final_prompts):
                if abort_event.is_set():
                    break
                last_err = ""
                for attempt in range(3):
                    if abort_event.is_set():
                        break
                    try:
                        out_pils = client.generate_images(
                            prompt=p_text,
                            model=final_model,
                            size=final_size,
                            quality=quality,
                            ref_images=[ref_pil],
                        )
                        for j, gen_img in enumerate(out_pils):
                            rgb_gen = safe_pil_to_rgb(gen_img)
                            prefix = f"Img{img_idx + 1:03d}_{base_stem}"
                            suffix = f"P{p_idx + 1:03d}_{j + 1}" if len(out_pils) > 1 else f"P{p_idx + 1:03d}"
                            save_path = os.path.join(batch_output_dir, f"{prefix}_{suffix}.png")
                            c = 1
                            while os.path.exists(save_path):
                                save_path = os.path.join(batch_output_dir, f"{prefix}_{suffix}_{c}.png")
                                c += 1
                            rgb_gen.save(save_path, "PNG", compress_level=4)
                            saved_count += 1
                            if len(preview_pils) < 15:
                                preview_pils.append(rgb_gen)
                        last_err = ""
                        break
                    except YesAPIError as err:
                        last_err = str(err)
                        if err.is_fatal:
                            with lock:
                                abort_reason = f"触发致命错误熔断: {err}"
                                abort_event.set()
                            break
                        time.sleep(0.5)
                    except Exception as err:
                        last_err = f"{type(err).__name__}: {err}"
                        time.sleep(0.5)

                if last_err:
                    file_errs.append(last_err)

            return fname, saved_count, preview_pils, file_errs

        total_success_imgs = 0
        failed_records: List[str] = []
        preview_images: List[Image.Image] = []
        logs: List[str] = [
            f"=== YesAPI Folder Batch Log ({timestamp}) ===",
            f"Source: {source_mode}, Model: {final_model}, Size: {final_size or 'auto'}, Prompts: {len(base_prompts)}, Images: {len(image_files)}",
        ]

        workers = max(1, min(int(max_concurrent_files), 20))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_single_file, idx, p) for idx, p in enumerate(image_files)]
            for fut in concurrent.futures.as_completed(futures):
                fname, cnt, pils, errs = fut.result()
                if cnt > 0:
                    total_success_imgs += cnt
                    with lock:
                        consecutive_failures = 0
                    logs.append(f"[OK] {fname} -> 生成 {cnt} 张")
                    if len(preview_images) < 15 and pils:
                        preview_images.extend(pils[: 15 - len(preview_images)])
                else:
                    err_detail = errs[0] if errs else "0 张输出"
                    logs.append(f"[FAIL] {fname} -> {err_detail}")
                    failed_records.append(f"{fname} -> {err_detail}")
                    with lock:
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_FAILURES and not abort_event.is_set():
                            abort_reason = f"连续 {MAX_FAILURES} 个文件失败，触发熔断保护"
                            abort_event.set()

        if failed_records:
            log_file = os.path.join(batch_output_dir, "batch_error_log.txt")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("\n".join(logs) + "\n\n=== 失败详情 ===\n" + "\n".join(failed_records))

        summary = [
            "📢 提示：节点预览为兼容不同比例会自动补齐黑边，输出文件夹内保存的原图均无黑边！",
            "✅ 文件夹批量完成" if not abort_event.is_set() else f"❌ 任务中断 ({abort_reason})",
            f"📁 输入目录: {os.path.basename(clean_dir)} ({len(image_files)} 个文件)",
            f"📄 提示词源: {source_mode} ({len(base_prompts)} 条 x {executions_per_prompt} 次)",
            f"📐 尺寸设置: 比例={aspect_ratio}, 分辨率={resolution} ({final_size or 'auto'})",
            f"🖼️ 成功: {total_success_imgs}",
            f"❌ 失败: {len(failed_records)}",
            f"💾 输出目录: {batch_output_dir}",
            "⚠️ 预览仅最多展示前 15 张图像。",
        ]
        if failed_records:
            summary.append("\n--- 失败记录 (前 5 条) ---")
            for item in failed_records[:5]:
                summary.append(f"• {item}")

        report_text = "\n".join(summary)
        out_tensor = safe_pil_batch_to_tensor(preview_images) if preview_images else make_error_image_tensor()
        return {"ui": {"string": [report_text]}, "result": (out_tensor, report_text)}


# ==================== Node 5: YesAPISaveImageByFilename ====================


class YesAPISaveImageByFilename:
    """Smart output node saving an IMAGE batch aligned with newline-delimited filenames."""

    CATEGORY = "YesAPI/图像"
    FUNCTION = "execute"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "filenames": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "placeholder": "可选：相对于 ComfyUI output 的子文件夹"}),
                "suffix": ("STRING", {"default": "", "placeholder": "可选：文件名追加后缀如 _HD"}),
            },
        }

    def execute(
        self,
        images: torch.Tensor,
        filenames: str,
        output_path: str = "",
        suffix: str = "",
    ) -> Dict[str, Any]:
        pil_images = tensor_to_pil(images)
        raw_paths = [clean_user_path(line) for line in (filenames or "").splitlines() if line.strip()]
        if not raw_paths:
            return {"ui": {"string": ["错误: filenames 文件名列表为空"], "images": []}}

        main_output_dir = get_output_directory()
        clean_sub = clean_user_path(output_path)
        saved_count = 0
        ui_images: List[Dict[str, str]] = []

        for idx, pil_img in enumerate(pil_images):
            if idx >= len(raw_paths):
                break
            orig = raw_paths[idx]
            orig_name = os.path.basename(orig)
            stem, _ = os.path.splitext(orig_name)
            stem = sanitize_filename(stem)

            if clean_sub:
                save_dir = os.path.join(main_output_dir, clean_sub)
            else:
                orig_dir = os.path.dirname(orig)
                save_dir = orig_dir if orig_dir else main_output_dir

            os.makedirs(save_dir, exist_ok=True)
            new_stem = f"{stem}{suffix}"
            target_path = os.path.join(save_dir, f"{new_stem}.png")
            counter = 1
            while os.path.exists(target_path):
                target_path = os.path.join(save_dir, f"{new_stem} ({counter}).png")
                counter += 1

            safe_pil_to_rgb(pil_img).save(target_path, "PNG", compress_level=4)
            saved_count += 1

            try:
                rel = os.path.relpath(target_path, main_output_dir)
                ui_sub = "" if rel.startswith("..") else os.path.dirname(rel)
            except ValueError:
                ui_sub = ""

            ui_images.append(
                {
                    "filename": os.path.basename(target_path),
                    "subfolder": ui_sub,
                    "type": "output",
                }
            )

        msg = f"成功保存 {saved_count} 张图像"
        return {"ui": {"images": ui_images, "string": [msg]}}


NODE_CLASS_MAPPINGS = {
    "YesAPIImageGenerate": YesAPIImageGenerate,
    "YesAPIImageText2Image": YesAPIImageText2Image,
    "YesAPIImageBatchCSV": YesAPIImageBatchCSV,
    "YesAPIImageSaveWithPrompt": YesAPIImageSaveWithPrompt,
    "YesAPIImageBatchDir": YesAPIImageBatchDir,
    "YesAPISaveImageByFilename": YesAPISaveImageByFilename,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YesAPIImageGenerate": "🎨 YesAPI 图像生成与编辑",
    "YesAPIImageText2Image": "🎨 YesAPI 纯文生图",
    "YesAPIImageBatchCSV": "📊 YesAPI 图像批量生成 (CSV/Excel)",
    "YesAPIImageSaveWithPrompt": "🏷️ YesAPI 图像批量生成 (命名细化版)",
    "YesAPIImageBatchDir": "📂 YesAPI 文件夹批量处理 (CSV)",
    "YesAPISaveImageByFilename": "💾 YesAPI 按文件名智能保存",
}
