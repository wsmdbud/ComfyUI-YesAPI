"""Video generation and async video task queue nodes for ComfyUI-YesAPI (/v1/videos, 5 nodes)."""
from __future__ import annotations

import concurrent.futures
from datetime import datetime
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import torch

try:
    from ..core.client import YesAPIClient, YesAPIError
    from ..core.compat import VIDEO_TYPE, ProgressBar, get_output_directory, get_temp_directory
    from ..core.config import (
        VIDEO_ASPECT_RATIOS,
        VIDEO_MODELS,
        VIDEO_RESOLUTIONS,
        resolve_api_key,
        resolve_base_url,
        resolve_model,
        resolve_timeout,
    )
    from ..core.media import YesAPIVideoAdapter, read_table_file, sanitize_filename, tensor_to_pil
    from ..core.task_store import JsonTaskStore
except ImportError:
    from core.client import YesAPIClient, YesAPIError
    from core.compat import VIDEO_TYPE, ProgressBar, get_output_directory, get_temp_directory
    from core.config import (
        VIDEO_ASPECT_RATIOS,
        VIDEO_MODELS,
        VIDEO_RESOLUTIONS,
        resolve_api_key,
        resolve_base_url,
        resolve_model,
        resolve_timeout,
    )
    from core.media import YesAPIVideoAdapter, read_table_file, sanitize_filename, tensor_to_pil
    from core.task_store import JsonTaskStore


def _get_video_task_store() -> JsonTaskStore:
    path = os.path.join(get_temp_directory(), "yesapi_video_tasks.json")
    return JsonTaskStore(path, max_completed_history=10)


def _parse_extra_json(extra_body_json: str) -> Dict[str, Any]:
    if not extra_body_json or not extra_body_json.strip():
        return {}
    data = json.loads(extra_body_json.strip())
    if not isinstance(data, dict):
        raise ValueError("extra_body_json 必须是合法的 JSON 对象，例如 {\"seed\": 42}")
    return data


def _first_pil(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    pils = tensor_to_pil(tensor)
    return pils[0] if pils else None


# ==================== Node 14: YesAPIVideoGenerate ====================


class YesAPIVideoGenerate:
    """Submit /v1/videos task, poll until completion with ComfyUI ProgressBar, and download stream."""

    CATEGORY = "YesAPI/视频"
    FUNCTION = "execute"
    RETURN_TYPES = (VIDEO_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "status")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"default": "A cinematic drone shot over misty mountains at sunrise", "multiline": True}),
                "model": (VIDEO_MODELS, {"default": "Veo-3.1-fast-1080P"}),
                "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (VIDEO_RESOLUTIONS, {"default": "默认"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                "poll_timeout_sec": ("INT", {"default": 600, "min": 30, "max": 3600, "step": 30}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：填写平台新上架视频模型名"}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "audio_url": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：驱动音频直链 (.mp3/.wav)"}),
                "extra_body_json": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：透传额外 JSON 参数"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 video_api_key"}),
                "base_url": ("STRING", {"default": "https://yesapi.online", "multiline": False}),
            },
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
        duration: int,
        poll_timeout_sec: int = 600,
        custom_model: str = "",
        first_frame: Optional[torch.Tensor] = None,
        last_frame: Optional[torch.Tensor] = None,
        audio_url: str = "",
        extra_body_json: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            err = "失败: 视频提示词不能为空。"
            return {"ui": {"string": [err]}, "result": (YesAPIVideoAdapter(""), "", err)}

        final_key = resolve_api_key(api_key, category="video")
        if not final_key:
            err = "失败: API Key 不能为空。请在节点填入 api_key 或在 config.json 中配置 video_api_key。"
            return {"ui": {"string": [err]}, "result": (YesAPIVideoAdapter(""), "", err)}

        try:
            extra_body = _parse_extra_json(extra_body_json)
        except Exception as e:
            err = f"失败: extra_body_json 格式错误: {e}"
            return {"ui": {"string": [err]}, "result": (YesAPIVideoAdapter(""), "", err)}

        final_model = resolve_model(model, custom_model)
        pbar = ProgressBar(100)
        pbar.update_absolute(10)

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="video",
                timeout=resolve_timeout(),
            )
            vid_id = client.create_video_task(
                model=final_model,
                prompt=clean_prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                first_frame=_first_pil(first_frame),
                last_frame=_first_pil(last_frame),
                audio_url=audio_url,
                extra_body=extra_body,
            )
            pbar.update_absolute(20)

            start_ts = time.time()
            deadline = start_ts + max(30, int(poll_timeout_sec))
            while time.time() < deadline:
                p_info = client.poll_video_task(vid_id)
                st = p_info["status"]
                prog = int(p_info.get("progress") or 0)
                if prog > 0:
                    pbar.update_absolute(20 + int(prog * 0.7))

                if st == "completed":
                    pbar.update_absolute(92)
                    safe_id = sanitize_filename(vid_id, max_length=24)
                    filename = f"yesapi_video_{safe_id}_{uuid.uuid4().hex[:6]}.mp4"
                    out_path = os.path.join(get_output_directory(), filename)
                    client.download_video_content(vid_id, out_path)
                    pbar.update_absolute(100)
                    status = f"成功 | 任务ID: {vid_id} | 模型: {final_model} | 已保存至: {out_path}"
                    return {"ui": {"string": [status]}, "result": (YesAPIVideoAdapter(out_path), out_path, status)}

                if st == "failed":
                    reason = p_info.get("error") or "上游视频生成失败"
                    raise YesAPIError(f"视频任务失败 ({vid_id}): {reason}", is_fatal=False)

                wait_sec = max(1.0, float(p_info.get("poll_after_ms", 3000)) / 1000.0)
                time.sleep(wait_sec)

            raise YesAPIError(f"轮询超时 ({poll_timeout_sec}s)，任务 {vid_id} 仍在处理中。", is_fatal=False)
        except Exception as e:
            err = f"失败: {e}"
            return {"ui": {"string": [err]}, "result": (YesAPIVideoAdapter(""), "", err)}


# ==================== Node 15: YesAPIVideoAsyncSubmit ====================


class YesAPIVideoAsyncSubmit:
    """Submit a video task asynchronously and record it in yesapi_video_tasks.json."""

    CATEGORY = "YesAPI/视频"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("task_id", "status")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"default": "A cinematic drone shot over misty mountains at sunrise", "multiline": True}),
                "model": (VIDEO_MODELS, {"default": "Veo-3.1-fast-1080P"}),
                "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (VIDEO_RESOLUTIONS, {"default": "默认"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "audio_url": ("STRING", {"default": "", "multiline": False}),
                "extra_body_json": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 video_api_key"}),
                "base_url": ("STRING", {"default": "https://yesapi.online", "multiline": False}),
            },
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
        duration: int,
        custom_model: str = "",
        first_frame: Optional[torch.Tensor] = None,
        last_frame: Optional[torch.Tensor] = None,
        audio_url: str = "",
        extra_body_json: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            err = "提交失败: 提示词不能为空。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        final_key = resolve_api_key(api_key, category="video")
        if not final_key:
            err = "提交失败: API Key 不能为空。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        try:
            extra_body = _parse_extra_json(extra_body_json)
            final_model = resolve_model(model, custom_model)
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="video",
                timeout=resolve_timeout(),
            )
            vid_id = client.create_video_task(
                model=final_model,
                prompt=clean_prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                first_frame=_first_pil(first_frame),
                last_frame=_first_pil(last_frame),
                audio_url=audio_url,
                extra_body=extra_body,
            )
            store = _get_video_task_store()
            store.upsert_task(
                vid_id,
                {
                    "prompt": clean_prompt,
                    "model": final_model,
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "in_progress",
                    "progress": 0,
                    "error": "",
                    "video_path": "",
                },
            )
            msg = f"视频任务提交成功!\nID: {vid_id}\n模型: {final_model}\n请使用『📋 YesAPI 视频任务状态查询』与『📥 YesAPI 获取下一个完成视频』节点。"
            return {"ui": {"string": [msg]}, "result": (vid_id, msg)}
        except Exception as e:
            err = f"提交失败: {e}"
            return {"ui": {"string": [err]}, "result": ("", err)}


# ==================== Node 16: YesAPIVideoAsyncBatchSubmit ====================


class YesAPIVideoAsyncBatchSubmit:
    """Batch-submit video tasks from CSV/Excel into yesapi_video_tasks.json."""

    CATEGORY = "YesAPI/视频"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("task_ids", "report")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "placeholder": "拖拽或粘贴 CSV/Excel 文件路径"}),
                "column_name": ("STRING", {"default": "prompt"}),
                "prompt_prefix": ("STRING", {"multiline": True, "default": ""}),
                "model": (VIDEO_MODELS, {"default": "Veo-3.1-fast-1080P"}),
                "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (VIDEO_RESOLUTIONS, {"default": "默认"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                "concurrency": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "max_count": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "first_frame": ("IMAGE",),
                "audio_url": ("STRING", {"default": "", "multiline": False}),
                "extra_body_json": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 video_api_key"}),
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
        duration: int,
        concurrency: int = 2,
        max_count: int = 50,
        custom_model: str = "",
        first_frame: Optional[torch.Tensor] = None,
        audio_url: str = "",
        extra_body_json: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        final_key = resolve_api_key(api_key, category="video")
        if not final_key:
            err = "批量提交失败: API Key 不能为空。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        try:
            df = read_table_file(file_path)
            extra_body = _parse_extra_json(extra_body_json)
        except Exception as e:
            err = f"批量提交失败: {e}"
            return {"ui": {"string": [err]}, "result": ("", err)}

        col = (column_name or "prompt").strip()
        if col not in df.columns:
            err = f"批量提交失败: 列 '{col}' 不存在。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        prompts = [f"{prompt_prefix}{str(x).strip()}" for x in df[col].dropna().tolist() if str(x).strip()][: max(1, int(max_count))]
        if not prompts:
            err = f"批量提交失败: 列 '{col}' 中没有有效提示词。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        final_model = resolve_model(model, custom_model)
        first_pil = _first_pil(first_frame)
        store = _get_video_task_store()

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="video",
                timeout=resolve_timeout(),
            )
        except Exception as e:
            err = f"批量提交失败: {e}"
            return {"ui": {"string": [err]}, "result": ("", err)}

        abort_event = threading.Event()
        submitted_ids: List[str] = []
        errors: List[str] = []

        def _submit_one(p_text: str) -> Tuple[Optional[str], Optional[str]]:
            if abort_event.is_set():
                return None, "已熔断跳过"
            try:
                vid_id = client.create_video_task(
                    model=final_model,
                    prompt=p_text,
                    duration=duration,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    first_frame=first_pil,
                    audio_url=audio_url,
                    extra_body=extra_body,
                )
                store.upsert_task(
                    vid_id,
                    {
                        "prompt": p_text,
                        "model": final_model,
                        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "in_progress",
                        "progress": 0,
                        "error": "",
                        "video_path": "",
                    },
                )
                return vid_id, None
            except YesAPIError as err:
                if err.is_fatal:
                    abort_event.set()
                return None, str(err)
            except Exception as err:
                return None, str(err)

        workers = max(1, min(int(concurrency), 10))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for vid_id, err_str in pool.map(_submit_one, prompts):
                if vid_id:
                    submitted_ids.append(vid_id)
                elif err_str:
                    errors.append(err_str)

        report = f"视频批量提交完成 | 总数: {len(prompts)} | 成功: {len(submitted_ids)} | 失败: {len(errors)}"
        if errors:
            report += f"\n首条错误: {errors[0]}"
        return {"ui": {"string": [report]}, "result": ("\n".join(submitted_ids), report)}


# ==================== Node 17: YesAPIVideoQueryTasks ====================


class YesAPIVideoQueryTasks:
    """Poll active tasks in yesapi_video_tasks.json and display queue status."""

    CATEGORY = "YesAPI/视频"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 video_api_key"}),
                "base_url": ("STRING", {"default": "https://yesapi.online", "multiline": False}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def execute(self, api_key: str = "", base_url: str = "") -> Dict[str, Any]:
        final_key = resolve_api_key(api_key, category="video")
        if not final_key:
            err = "查询失败: API Key 不能为空。"
            return {"ui": {"string": [err]}, "result": (err,)}

        store = _get_video_task_store()
        tasks = store.get_tasks()
        if not tasks:
            msg = "当前无视频任务记录。"
            return {"ui": {"string": [msg]}, "result": (msg,)}

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="video",
                timeout=30,
            )
        except Exception as e:
            err = f"查询失败: {e}"
            return {"ui": {"string": [err]}, "result": (err,)}

        updates: Dict[str, Dict[str, Any]] = {}
        for tid, info in tasks.items():
            if info.get("status") in ("completed", "downloaded", "failed", "download_failed"):
                continue
            try:
                p = client.poll_video_task(tid)
                info["status"] = p["status"]
                info["progress"] = p.get("progress", 0)
                info["error"] = p.get("error", "")
                updates[tid] = info
            except Exception as err:
                info["error"] = str(err)
                updates[tid] = info

        if updates:
            def _merge(curr: Dict[str, Dict[str, Any]]) -> None:
                for k, v in updates.items():
                    curr[k] = v
            store.mutate(_merge)
            tasks = store.get_tasks()

        lines = ["--- 视频任务队列总览 ---"]
        for tid, info in sorted(tasks.items(), key=lambda x: str(x[1].get("submitted_at", "")), reverse=True):
            st = info.get("status", "in_progress")
            prog = info.get("progress", 0)
            st_str = f"处理中 {prog}%" if st == "in_progress" else st
            snippet = str(info.get("prompt", ""))[:24]
            lines.append(f"[{st_str}] {tid} - {snippet}")

        report = "\n".join(lines)
        return {"ui": {"string": [report]}, "result": (report,)}


# ==================== Node 18: YesAPIVideoGetNext ====================


class YesAPIVideoGetNext:
    """Download the earliest completed video task from yesapi_video_tasks.json."""

    CATEGORY = "YesAPI/视频"
    FUNCTION = "execute"
    RETURN_TYPES = (VIDEO_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "report")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "optional": {
                "target_task_id": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则自动下载最早已完成的视频任务"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取 config.json 的 video_api_key"}),
                "base_url": ("STRING", {"default": "https://yesapi.online", "multiline": False}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def execute(
        self,
        target_task_id: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        final_key = resolve_api_key(api_key, category="video")
        if not final_key:
            err = "下载失败: API Key 不能为空。"
            return {"ui": {"string": [err]}, "result": (YesAPIVideoAdapter(""), "", err)}

        store = _get_video_task_store()
        tasks = store.get_tasks()
        clean_target = (target_task_id or "").strip()

        chosen_id: Optional[str] = None
        if clean_target:
            if clean_target in tasks and tasks[clean_target].get("status") == "completed":
                chosen_id = clean_target
        else:
            ready = [(k, v) for k, v in tasks.items() if v.get("status") == "completed"]
            if ready:
                ready.sort(key=lambda item: str(item[1].get("submitted_at", "9999-99-99")))
                chosen_id = ready[0][0]

        if not chosen_id:
            msg = "当前无待下载的已完成视频任务（请先运行『📋 YesAPI 视频任务状态查询』刷新队列）。"
            return {"ui": {"string": [msg]}, "result": (YesAPIVideoAdapter(""), "", msg)}

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category="video",
                timeout=300,
            )
            safe_id = sanitize_filename(chosen_id, max_length=24)
            out_path = os.path.join(get_output_directory(), f"yesapi_video_{safe_id}_{uuid.uuid4().hex[:6]}.mp4")
            client.download_video_content(chosen_id, out_path)

            def _mark_ok(curr: Dict[str, Dict[str, Any]]) -> None:
                if chosen_id in curr:
                    curr[chosen_id]["status"] = "downloaded"
                    curr[chosen_id]["video_path"] = out_path

            store.mutate(_mark_ok)
            report = f"✅ 下载成功: {chosen_id}\n保存路径: {out_path}"
            return {"ui": {"string": [report]}, "result": (YesAPIVideoAdapter(out_path), out_path, report)}
        except Exception as e:
            def _mark_fail(curr: Dict[str, Dict[str, Any]]) -> None:
                if chosen_id in curr:
                    curr[chosen_id]["status"] = "download_failed"
                    curr[chosen_id]["error"] = str(e)

            store.mutate(_mark_fail)
            err = f"❌ 视频下载失败 ({chosen_id}): {e}"
            return {"ui": {"string": [err]}, "result": (YesAPIVideoAdapter(""), "", err)}


NODE_CLASS_MAPPINGS = {
    "YesAPIVideoGenerate": YesAPIVideoGenerate,
    "YesAPIVideoAsyncSubmit": YesAPIVideoAsyncSubmit,
    "YesAPIVideoAsyncBatchSubmit": YesAPIVideoAsyncBatchSubmit,
    "YesAPIVideoQueryTasks": YesAPIVideoQueryTasks,
    "YesAPIVideoGetNext": YesAPIVideoGetNext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YesAPIVideoGenerate": "🎬 YesAPI 视频生成 (同步等待)",
    "YesAPIVideoAsyncSubmit": "🚀 YesAPI 视频异步提交",
    "YesAPIVideoAsyncBatchSubmit": "🚀 YesAPI 视频异步批量提交 (CSV/Excel)",
    "YesAPIVideoQueryTasks": "📋 YesAPI 视频任务状态查询",
    "YesAPIVideoGetNext": "📥 YesAPI 获取下一个完成视频",
}
