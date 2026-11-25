"""Asynchronous image submission, batch submission, and polling/download nodes (3 nodes)."""
from __future__ import annotations

import concurrent.futures
from datetime import datetime
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch

try:
    from ..core.client import YesAPIClient, YesAPIError
    from ..core.compat import get_temp_directory
    from ..core.config import (
        IMAGE_ASPECT_RATIOS,
        IMAGE_MODELS,
        IMAGE_QUALITIES,
        IMAGE_RESOLUTIONS,
        resolve_api_key,
        resolve_base_url,
        resolve_image_size,
        resolve_model,
        resolve_timeout,
    )
    from ..core.media import (
        make_error_image_tensor,
        mask_to_png_bytes,
        read_table_file,
        safe_pil_batch_to_tensor,
        tensor_to_pil,
    )
    from ..core.task_store import JsonTaskStore
except ImportError:
    from core.client import YesAPIClient, YesAPIError
    from core.compat import get_temp_directory
    from core.config import (
        IMAGE_ASPECT_RATIOS,
        IMAGE_MODELS,
        IMAGE_QUALITIES,
        IMAGE_RESOLUTIONS,
        resolve_api_key,
        resolve_base_url,
        resolve_image_size,
        resolve_model,
        resolve_timeout,
    )
    from core.media import (
        make_error_image_tensor,
        mask_to_png_bytes,
        read_table_file,
        safe_pil_batch_to_tensor,
        tensor_to_pil,
    )
    from core.task_store import JsonTaskStore


def _get_image_task_store() -> JsonTaskStore:
    path = os.path.join(get_temp_directory(), "yesapi_image_tasks.json")
    return JsonTaskStore(path, max_completed_history=10)


def _collect_refs(kwargs: Dict[str, Any], max_slots: int) -> List[Image.Image]:
    refs: List[Image.Image] = []
    for i in range(1, max_slots + 1):
        t = kwargs.get(f"image_{i}")
        if t is not None:
            refs.extend(tensor_to_pil(t))
    return refs


# ==================== Node 6: YesAPIImageAsyncSubmit ====================


class YesAPIImageAsyncSubmit:
    """Submit single or multi-subtask async image generation/editing job."""

    CATEGORY = "YesAPI/图像异步"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("task_id", "status")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        optional: Dict[str, Any] = {
            "custom_model": ("STRING", {"default": "", "multiline": False}),
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
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def execute(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        resolution: str,
        quality: str,
        concurrency: int,
        custom_model: str = "",
        custom_size: str = "",
        mask: Optional[torch.Tensor] = None,
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            msg = "提交失败: 提示词不能为空。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        final_key = resolve_api_key(api_key, category="image")
        if not final_key:
            msg = "提交失败: API Key 不能为空。请配置 image_api_key 或在节点填入。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        refs = _collect_refs(kwargs, max_slots=14)
        if mask is not None and not refs:
            msg = "提交失败: 使用局部重绘遮罩 (mask) 时，必须同时连接至少一张基础图像 (image_1)。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        mask_bytes = mask_to_png_bytes(mask, target_size=refs[0].size) if mask is not None and refs else None
        final_model = resolve_model(model, custom_model)
        final_size = resolve_image_size(aspect_ratio, resolution, custom_size)

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="image",
                timeout=resolve_timeout(),
            )
            subtasks: List[Dict[str, Any]] = []
            for _ in range(max(1, min(int(concurrency), 10))):
                api_id = client.submit_async_image(
                    prompt=clean_prompt,
                    model=final_model,
                    size=final_size,
                    quality=quality,
                    ref_images=refs,
                    mask_png_bytes=mask_bytes,
                )
                subtasks.append(
                    {
                        "api_task_id": api_id,
                        "status": "pending",
                        "content_url": None,
                        "error": "",
                    }
                )

            store = _get_image_task_store()
            task_id = store.next_numbered_id("任务", exclude_prefixes=("批量任务",))
            store.upsert_task(
                task_id,
                {
                    "type": "normal",
                    "prompt": clean_prompt,
                    "model": final_model,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "size": final_size or "auto",
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "pending",
                    "concurrency": len(subtasks),
                    "subtasks": subtasks,
                },
            )
            status = f"任务提交成功 | {task_id} | 尺寸: {final_size or 'auto'} | 模型: {final_model} | 子任务数: {len(subtasks)}（请在 2 小时内使用异步查询下载节点提取结果）"
            return {"ui": {"string": [status]}, "result": (task_id, status)}
        except Exception as e:
            msg = f"提交失败: {e}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}


# ==================== Node 7: YesAPIImageAsyncBatchSubmit ====================


class YesAPIImageAsyncBatchSubmit:
    """Batch submit CSV/Excel prompts to Yesapi async image queue."""

    CATEGORY = "YesAPI/图像异步"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("batch_id", "status")

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
                "executions_per_prompt": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "concurrency": ("INT", {"default": 3, "min": 1, "max": 15, "step": 1}),
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
        executions_per_prompt: int,
        concurrency: int = 3,
        custom_model: str = "",
        custom_size: str = "",
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        final_key = resolve_api_key(api_key, category="image")
        if not final_key:
            msg = "批量提交失败: API Key 不能为空。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        try:
            df = read_table_file(file_path)
        except Exception as e:
            msg = f"批量提交失败: {e}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        col = (column_name or "prompt").strip()
        if col not in df.columns:
            msg = f"批量提交失败: 列 '{col}' 不存在。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        base_prompts = [f"{prompt_prefix}{str(x).strip()}" for x in df[col].dropna().tolist() if str(x).strip()]
        if not base_prompts:
            msg = f"批量提交失败: 列 '{col}' 中没有有效提示词。"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        prompts = [p for p in base_prompts for _ in range(max(1, int(executions_per_prompt)))]
        refs = _collect_refs(kwargs, max_slots=4)
        final_model = resolve_model(model, custom_model)
        final_size = resolve_image_size(aspect_ratio, resolution, custom_size)

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="image",
                timeout=resolve_timeout(),
            )
        except Exception as e:
            msg = f"批量提交失败: {e}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        abort_event = threading.Event()
        subtasks: List[Dict[str, Any]] = []
        errors: List[str] = []

        def _submit_one(p_text: str) -> Dict[str, Any]:
            if abort_event.is_set():
                return {"error": "已熔断跳过"}
            try:
                api_id = client.submit_async_image(
                    prompt=p_text,
                    model=final_model,
                    size=final_size,
                    quality=quality,
                    ref_images=refs,
                )
                return {
                    "api_task_id": api_id,
                    "status": "pending",
                    "content_url": None,
                    "error": "",
                    "original_prompt": p_text,
                }
            except YesAPIError as err:
                if err.is_fatal:
                    abort_event.set()
                return {"error": str(err)}
            except Exception as err:
                return {"error": str(err)}

        workers = max(1, min(int(concurrency), 15))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(_submit_one, prompts):
                if "error" in res and "api_task_id" not in res:
                    errors.append(res["error"])
                else:
                    subtasks.append(res)

        if not subtasks:
            msg = f"全部任务提交失败。错误: {errors[0] if errors else '未知'}"
            return {"ui": {"string": [msg]}, "result": ("", msg)}

        store = _get_image_task_store()
        batch_id = store.next_numbered_id("批量任务", exclude_prefixes=())
        store.upsert_task(
            batch_id,
            {
                "type": "batch",
                "prompt": f"批量文件: {os.path.basename(file_path)}",
                "model": final_model,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "size": final_size or "auto",
                "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "pending",
                "concurrency": len(subtasks),
                "subtasks": subtasks,
            },
        )
        status = f"批量提交成功 | {batch_id} | 成功提交: {len(subtasks)}/{len(prompts)} | 尺寸: {final_size or 'auto'} | 模型: {final_model}"
        if errors:
            status += f" | 提交失败: {len(errors)}"
        return {"ui": {"string": [status]}, "result": (batch_id, status)}


# ==================== Node 8: YesAPIImageAsyncQuery ====================


class YesAPIImageAsyncQuery:
    """Poll active async image tasks and download completed results."""

    CATEGORY = "YesAPI/图像异步"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "optional": {
                "target_task_id": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则自动下载最早完成的任务（如 任务1）"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 image_api_key"}),
                "base_url": ("STRING", {"default": "https://yesapi.online", "multiline": False}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    @staticmethod
    def _format_board(tasks: Dict[str, Dict[str, Any]], store: JsonTaskStore) -> str:
        status_cn = {
            "pending": "生成中",
            "running": "生成中",
            "completed": "待下载",
            "failed": "失败",
            "downloaded": "已下载",
        }
        lines: List[str] = ["--- 图像异步队列总览 ---"]
        for tid, info in sorted(tasks.items(), key=lambda x: str(x[1].get("submitted_at", "")), reverse=True):
            st = str(info.get("status", "pending")).lower()
            label = status_cn.get(st, st)
            subs = info.get("subtasks") or []
            done_cnt = sum(1 for s in subs if s.get("status") in ("completed", "failed"))
            ok_cnt = sum(1 for s in subs if s.get("status") == "completed")
            fail_cnt = sum(1 for s in subs if s.get("status") == "failed")
            expired_note = (
                " [⚠️超2小时可能已过期]"
                if st not in ("downloaded", "failed") and store.is_expired(str(info.get("submitted_at", "")))
                else ""
            )
            if info.get("type") == "batch":
                lines.append(f"[{label}] {tid} - 进度: {done_cnt}/{len(subs)} (成功:{ok_cnt} 失败:{fail_cnt}){expired_note}")
            else:
                snippet = str(info.get("prompt", ""))[:18]
                lines.append(f"[{label}] {tid} ({snippet}) - {len(subs)}张{expired_note}")
        return "\n".join(lines) if len(lines) > 1 else "当前无图像异步任务记录。"

    def execute(
        self,
        target_task_id: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        final_key = resolve_api_key(api_key, category="image")
        if not final_key:
            msg = "查询失败: API Key 不能为空。"
            return {"ui": {"string": [msg]}, "result": (make_error_image_tensor(), msg)}

        store = _get_image_task_store()
        tasks = store.get_tasks()
        if not tasks:
            msg = "当前没有图像异步任务记录。"
            return {"ui": {"string": [msg]}, "result": (make_error_image_tensor(), msg)}

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="image",
                timeout=resolve_timeout(),
            )
        except Exception as e:
            msg = f"查询失败: {e}"
            return {"ui": {"string": [msg]}, "result": (make_error_image_tensor(), msg)}

        updated_tasks: Dict[str, Dict[str, Any]] = {}
        for tid, info in tasks.items():
            if info.get("status") in ("completed", "failed", "downloaded"):
                continue
            subs = info.get("subtasks") or []
            any_pending = False
            any_completed = False
            for sub in subs:
                if sub.get("status") == "completed":
                    any_completed = True
                    continue
                if sub.get("status") == "failed":
                    continue
                api_id = sub.get("api_task_id")
                if not api_id:
                    sub["status"] = "failed"
                    continue
                try:
                    p = client.poll_image_task(str(api_id))
                    sub["status"] = p["status"]
                    sub["content_url"] = p.get("content_url")
                    sub["error"] = p.get("error", "")
                except Exception as err:
                    sub["error"] = str(err)

                if sub["status"] == "pending":
                    any_pending = True
                elif sub["status"] == "completed":
                    any_completed = True

            if any_pending:
                info["status"] = "pending"
            elif any_completed:
                info["status"] = "completed"
            else:
                info["status"] = "failed"
            info["subtasks"] = subs
            updated_tasks[tid] = info

        if updated_tasks:
            def _apply_updates(curr: Dict[str, Dict[str, Any]]) -> None:
                for k, v in updated_tasks.items():
                    curr[k] = v
            store.mutate(_apply_updates)
            tasks = store.get_tasks()

        clean_target = (target_task_id or "").strip()
        chosen: Optional[Tuple[str, Dict[str, Any]]] = None
        if clean_target:
            if clean_target in tasks and tasks[clean_target].get("status") == "completed":
                chosen = (clean_target, tasks[clean_target])
        else:
            candidates = [(k, v) for k, v in tasks.items() if v.get("status") == "completed"]
            if candidates:
                candidates.sort(key=lambda item: str(item[1].get("submitted_at", "9999-99-99")))
                chosen = candidates[0]

        board = self._format_board(tasks, store)
        if chosen is None:
            return {"ui": {"string": [board]}, "result": (make_error_image_tensor(), board)}

        dl_tid, dl_info = chosen
        subs = dl_info.get("subtasks") or []
        completed_subs = [s for s in subs if s.get("status") == "completed" and s.get("api_task_id")]

        downloaded_pils: List[Image.Image] = []
        dl_errors: List[str] = []

        def _dl_one(sub_item: Dict[str, Any]) -> Optional[Image.Image]:
            return client.download_image_content(
                str(sub_item["api_task_id"]),
                sub_item.get("content_url"),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(completed_subs)))) as pool:
            futs = [pool.submit(_dl_one, s) for s in completed_subs]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    img = fut.result()
                    if img is not None:
                        downloaded_pils.append(img)
                except Exception as err:
                    dl_errors.append(str(err))

        if not downloaded_pils:
            err_msg = f"{dl_tid} 下载失败: {dl_errors[0] if dl_errors else '无有效图像'}\n{board}"
            return {"ui": {"string": [err_msg]}, "result": (make_error_image_tensor(), err_msg)}

        def _mark_downloaded(curr: Dict[str, Dict[str, Any]]) -> None:
            if dl_tid in curr:
                curr[dl_tid]["status"] = "downloaded"

        store.mutate(_mark_downloaded)
        fresh_board = self._format_board(store.get_tasks(), store)
        header = f"✅ 下载成功: {dl_tid} | 成功获取: {len(downloaded_pils)}/{len(subs)} 张"
        if dl_errors:
            header += f" | 下载异常: {len(dl_errors)}"
        full_status = f"{header}\n{fresh_board}"
        return {
            "ui": {"string": [full_status]},
            "result": (safe_pil_batch_to_tensor(downloaded_pils), full_status),
        }


NODE_CLASS_MAPPINGS = {
    "YesAPIImageAsyncSubmit": YesAPIImageAsyncSubmit,
    "YesAPIImageAsyncBatchSubmit": YesAPIImageAsyncBatchSubmit,
    "YesAPIImageAsyncQuery": YesAPIImageAsyncQuery,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YesAPIImageAsyncSubmit": "🚀 YesAPI 图像异步提交",
    "YesAPIImageAsyncBatchSubmit": "🚀 YesAPI 图像异步批量提交 (CSV/Excel)",
    "YesAPIImageAsyncQuery": "📥 YesAPI 图像异步查询下载",
}
