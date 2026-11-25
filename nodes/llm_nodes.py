"""LLM chat, VLM multimodal chat, prompt-table writer, CSV batch processor, and model discovery nodes (5 nodes)."""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import pandas as pd

try:
    from ..core.client import YesAPIClient, YesAPIError
    from ..core.compat import get_output_directory
    from ..core.config import (
        LLM_MODELS,
        clean_user_path,
        model_family,
        resolve_api_key,
        resolve_base_url,
        resolve_model,
        resolve_timeout,
    )
    from ..core.media import read_table_file, sanitize_filename, tensor_to_pil
except ImportError:
    from core.client import YesAPIClient, YesAPIError
    from core.compat import get_output_directory
    from core.config import (
        LLM_MODELS,
        clean_user_path,
        model_family,
        resolve_api_key,
        resolve_base_url,
        resolve_model,
        resolve_timeout,
    )
    from core.media import read_table_file, sanitize_filename, tensor_to_pil

PROTOCOLS = ["auto", "openai", "anthropic"]


def _parse_prompt_list(raw_text: str) -> List[str]:
    """Extract clean list of prompts from JSON array, fenced code block, or numbered lines."""
    if not raw_text or not raw_text.strip():
        return []
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json|text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Try JSON array extraction first
    bracket_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if bracket_match:
        try:
            arr = json.loads(bracket_match.group(0))
            if isinstance(arr, list):
                items: List[str] = []
                for elem in arr:
                    if isinstance(elem, str) and elem.strip():
                        items.append(elem.strip())
                    elif isinstance(elem, dict):
                        val = elem.get("prompt") or elem.get("text") or elem.get("content")
                        if val and str(val).strip():
                            items.append(str(val).strip())
                if items:
                    return items
        except json.JSONDecodeError:
            pass

    # Fallback: line-by-line stripping numbering/bullets
    lines: List[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:\d+[\.\)、]\s*|[-*•]\s*)", "", line).strip()
        line = line.strip('"').strip("'").strip()
        if line:
            lines.append(line)
    return lines


def _save_dataframe(df: pd.DataFrame, target_path: str) -> str:
    parent = os.path.dirname(os.path.abspath(target_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    lower = target_path.lower()
    if lower.endswith(".csv"):
        df.to_csv(target_path, index=False, encoding="utf-8-sig")
    elif lower.endswith((".xls", ".xlsx")):
        df.to_excel(target_path, index=False)
    else:
        target_path = f"{target_path}.csv"
        df.to_csv(target_path, index=False, encoding="utf-8-sig")
    return target_path


# ==================== Node 9: YesAPILLMChat ====================


class YesAPILLMChat:
    """Pure-text LLM chat supporting both OpenAI /chat/completions and Anthropic /messages."""

    CATEGORY = "YesAPI/语言模型"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "status")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": (LLM_MODELS, {"default": "[Gemini] gemini-4-carbon"}),
                "protocol": (PROTOCOLS, {"default": "auto"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "prompt": ("STRING", {"default": "你好，请介绍一下你自己。", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 32768, "step": 64}),
                "random_mode": (["固定", "随机"], {"default": "固定"}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：填写平台其他模型全名"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则自动按模型读取对应分组 Key"}),
                "base_url": ("STRING", {"default": "", "multiline": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> Any:
        if kwargs.get("random_mode") == "随机":
            return time.time_ns()
        return False

    def execute(
        self,
        model: str,
        protocol: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        random_mode: str = "固定",
        custom_model: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            err = "失败: prompt 不能为空。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        raw_selector = custom_model.strip() if (custom_model and custom_model.strip()) else model
        final_model = resolve_model(model, custom_model)
        fam = model_family(raw_selector) or "llm"
        final_key = resolve_api_key(api_key, category="llm", model=raw_selector)
        if not final_key:
            err = f"失败: API Key 不能为空。请在节点填入 api_key 或在 config.json 中配置 {fam}_api_key / llm_api_key。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category=fam,
                timeout=resolve_timeout(),
            )
            reply = client.chat_completion(
                model=final_model,
                prompt=clean_prompt,
                system_prompt=system_prompt,
                images=None,
                protocol=protocol,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            status = f"成功 | 模型: {final_model} | 字数: {len(reply)}"
            return {"ui": {"string": [status]}, "result": (reply, status)}
        except Exception as e:
            err = f"失败: {e}"
            return {"ui": {"string": [err]}, "result": ("", err)}


# ==================== Node 10: YesAPIVLMChat ====================


class YesAPIVLMChat:
    """Multimodal VLM chat accepting up to 6 optional images (gracefully degrades to pure text)."""

    CATEGORY = "YesAPI/语言模型"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "status")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        optional: Dict[str, Any] = {
            "custom_model": ("STRING", {"default": "", "multiline": False}),
        }
        for i in range(1, 7):
            optional[f"image_{i}"] = ("IMAGE",)
        optional["api_key"] = ("STRING", {"default": "", "multiline": False, "placeholder": "留空则自动按模型读取对应分组 Key"})
        optional["base_url"] = ("STRING", {"default": "", "multiline": False})

        return {
            "required": {
                "model": (LLM_MODELS, {"default": "[Gemini] gemini-4-carbon"}),
                "protocol": (PROTOCOLS, {"default": "auto"}),
                "system_prompt": ("STRING", {"default": "You are an expert visual analyst and prompt engineer.", "multiline": True}),
                "prompt": ("STRING", {"default": "请详细描述画面内容、构图与光影风格。", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 32768, "step": 64}),
                "random_mode": (["固定", "随机"], {"default": "固定"}),
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> Any:
        if kwargs.get("random_mode") == "随机":
            return time.time_ns()
        return False

    def execute(
        self,
        model: str,
        protocol: str,
        system_prompt: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        random_mode: str = "固定",
        custom_model: str = "",
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            err = "失败: prompt 不能为空。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        raw_selector = custom_model.strip() if (custom_model and custom_model.strip()) else model
        final_model = resolve_model(model, custom_model)
        fam = model_family(raw_selector) or "llm"
        final_key = resolve_api_key(api_key, category="llm", model=raw_selector)
        if not final_key:
            err = f"失败: API Key 不能为空。请配置 {fam}_api_key / llm_api_key 或在节点填入。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        pils: List[Image.Image] = []
        for i in range(1, 7):
            t = kwargs.get(f"image_{i}")
            if t is not None:
                pils.extend(tensor_to_pil(t))

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category=fam,
                timeout=resolve_timeout(),
            )
            reply = client.chat_completion(
                model=final_model,
                prompt=clean_prompt,
                system_prompt=system_prompt,
                images=pils,
                protocol=protocol,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            img_info = f"{len(pils)} 张图" if pils else "纯文本模式"
            status = f"成功 | 模型: {final_model} | 输入: {img_info} | 字数: {len(reply)}"
            return {"ui": {"string": [status]}, "result": (reply, status)}
        except Exception as e:
            err = f"失败: {e}"
            return {"ui": {"string": [err]}, "result": ("", err)}


# ==================== Node 11: YesAPILLMWriter ====================


class YesAPILLMWriter:
    """Generate a list of image/video prompts via LLM/VLM and export directly to CSV or Excel."""

    CATEGORY = "YesAPI/语言模型"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("file_path", "prompts_preview", "status")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": (LLM_MODELS, {"default": "[Gemini] gemini-3.8-flash"}),
                "protocol": (PROTOCOLS, {"default": "auto"}),
                "main_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "请为我生成 5 条关于“夏日海边咖啡馆”的高细节绘画提示词，仅输出 JSON 字符串数组。",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "你是一名专业的 AI 绘画与视频分镜提示词专家。请严格只输出一个 JSON 字符串数组，例如 [\"prompt 1\", \"prompt 2\"]，不要输出任何多余解释。",
                    },
                ),
                "output_filename": ("STRING", {"default": "generated_prompts.csv"}),
                "column_name": ("STRING", {"default": "prompt"}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则自动按模型读取对应分组 Key"}),
                "base_url": ("STRING", {"default": "", "multiline": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def execute(
        self,
        model: str,
        protocol: str,
        main_prompt: str,
        system_prompt: str,
        output_filename: str,
        column_name: str,
        custom_model: str = "",
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raw_selector = custom_model.strip() if (custom_model and custom_model.strip()) else model
        final_model = resolve_model(model, custom_model)
        fam = model_family(raw_selector) or "llm"
        final_key = resolve_api_key(api_key, category="llm", model=raw_selector)
        if not final_key:
            err = f"失败: API Key 不能为空。请配置 {fam}_api_key / llm_api_key 或在节点填入。"
            return {"ui": {"string": [err]}, "result": ("", "", err)}

        fname = (output_filename or "generated_prompts.csv").strip()
        if not fname.lower().endswith((".csv", ".xls", ".xlsx")):
            err = "失败: output_filename 必须以 .csv, .xls 或 .xlsx 结尾。"
            return {"ui": {"string": [err]}, "result": ("", "", err)}

        pils: List[Image.Image] = []
        for i in range(1, 4):
            t = kwargs.get(f"image_{i}")
            if t is not None:
                pils.extend(tensor_to_pil(t))

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category=fam,
                timeout=resolve_timeout(),
            )
            raw_reply = client.chat_completion(
                model=final_model,
                prompt=(main_prompt or "").strip(),
                system_prompt=system_prompt,
                images=pils,
                protocol=protocol,
            )
            items = _parse_prompt_list(raw_reply)
            if not items:
                err = f"失败: 未能从模型输出解析出有效提示词列表。原始回复: {raw_reply[:200]}"
                return {"ui": {"string": [err]}, "result": ("", raw_reply, err)}

            col = (column_name or "prompt").strip()
            df = pd.DataFrame(items, columns=[col])
            out_dir = get_output_directory()
            full_path = _save_dataframe(df, os.path.join(out_dir, os.path.basename(fname)))
            preview = "\n".join(items[:10])
            status = f"成功生成 {len(items)} 条提示词并写入: {full_path}"
            return {"ui": {"string": [status]}, "result": (full_path, preview, status)}
        except Exception as e:
            err = f"失败: {e}"
            return {"ui": {"string": [err]}, "result": ("", "", err)}


# ==================== Node 12: YesAPILLMBatchCSV ====================


class YesAPILLMBatchCSV:
    """Batch-process rows of a CSV/Excel table with an LLM (translate, expand storyboards, rewrite)."""

    CATEGORY = "YesAPI/语言模型"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_file_path", "status")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "placeholder": "输入待处理的 CSV/Excel 文件路径"}),
                "input_column": ("STRING", {"default": "prompt"}),
                "output_column": ("STRING", {"default": "llm_output"}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "prompt_template": ("STRING", {"default": "请将以下内容扩写为英文绘画提示词：\n{text}", "multiline": True}),
                "model": (LLM_MODELS, {"default": "[Gemini] gemini-3.8-flash"}),
                "protocol": (PROTOCOLS, {"default": "auto"}),
                "concurrency": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}),
                "max_count": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
                "output_filename": ("STRING", {"default": "", "placeholder": "留空则自动保存为 原文件名_processed.csv"}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则自动按模型读取对应分组 Key"}),
                "base_url": ("STRING", {"default": "", "multiline": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def execute(
        self,
        file_path: str,
        input_column: str,
        output_column: str,
        system_prompt: str,
        prompt_template: str,
        model: str,
        protocol: str,
        concurrency: int = 3,
        max_count: int = 100,
        output_filename: str = "",
        custom_model: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        raw_selector = custom_model.strip() if (custom_model and custom_model.strip()) else model
        final_model = resolve_model(model, custom_model)
        fam = model_family(raw_selector) or "llm"
        final_key = resolve_api_key(api_key, category="llm", model=raw_selector)
        if not final_key:
            err = f"失败: API Key 不能为空。请配置 {fam}_api_key / llm_api_key 或在节点填入。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        try:
            df = read_table_file(file_path)
        except Exception as e:
            err = f"失败: {e}"
            return {"ui": {"string": [err]}, "result": ("", err)}

        in_col = (input_column or "prompt").strip()
        if in_col not in df.columns:
            err = f"失败: 输入列 '{in_col}' 不存在。当前表头: {list(df.columns)}"
            return {"ui": {"string": [err]}, "result": ("", err)}

        work_df = df.head(max(1, int(max_count))).copy()
        texts = [str(v) if pd.notna(v) else "" for v in work_df[in_col].tolist()]
        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category=fam,
                timeout=resolve_timeout(),
            )
        except Exception as e:
            err = f"失败: {e}"
            return {"ui": {"string": [err]}, "result": ("", err)}

        abort_event = threading.Event()
        abort_reason = ""

        def _process_row(cell_text: str) -> Tuple[str, Optional[str]]:
            nonlocal abort_reason
            if not cell_text.strip() or abort_event.is_set():
                return "", None
            user_msg = prompt_template.replace("{text}", cell_text) if "{text}" in prompt_template else f"{prompt_template}\n{cell_text}"
            for attempt in range(2):
                if abort_event.is_set():
                    return "", None
                try:
                    ans = client.chat_completion(
                        model=final_model,
                        prompt=user_msg,
                        system_prompt=system_prompt,
                        protocol=protocol,
                    )
                    return ans, None
                except YesAPIError as err:
                    if err.is_fatal:
                        abort_reason = f"触发熔断: {err}"
                        abort_event.set()
                        return "", str(err)
                    if attempt == 0:
                        time.sleep(0.5)
                    else:
                        return "", str(err)
                except Exception as err:
                    if attempt == 0:
                        time.sleep(0.5)
                    else:
                        return "", str(err)
            return "", "重试失败"

        outputs: List[str] = []
        errors: List[str] = []
        workers = max(1, min(int(concurrency), 20))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for out_val, err_val in pool.map(_process_row, texts):
                outputs.append(out_val)
                if err_val:
                    errors.append(err_val)

        out_col = (output_column or "llm_output").strip()
        work_df[out_col] = outputs
        ok_count = sum(1 for v in outputs if v.strip())

        out_dir = get_output_directory()
        if output_filename and output_filename.strip():
            target_name = os.path.basename(clean_user_path(output_filename))
        else:
            src_stem = sanitize_filename(os.path.splitext(os.path.basename(clean_user_path(file_path)))[0])
            target_name = f"{src_stem}_processed.csv"

        saved_path = _save_dataframe(work_df, os.path.join(out_dir, target_name))
        status = f"表格处理完成 | 成功: {ok_count}/{len(texts)} | 失败: {len(errors)} | 输出: {saved_path}"
        if abort_reason:
            status += f" | ⚠️ {abort_reason}"
        return {"ui": {"string": [status]}, "result": (saved_path, status)}


# ==================== Node 13: YesAPIModelList ====================


class YesAPIModelList:
    """Query GET /v1/models to inspect exact model IDs bound to the active API key group."""

    CATEGORY = "YesAPI/语言模型"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("models_text", "status")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "key_category": (["llm", "gemini", "gemini_vt", "openai", "claude", "cn", "image", "video", "default"], {"default": "llm"}),
                "keyword_filter": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：按关键字过滤（如 banana / gemini / veo）"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空则读取对应分类的 config.json Key"}),
                "base_url": ("STRING", {"default": "", "multiline": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("NaN")

    def execute(
        self,
        key_category: str = "llm",
        keyword_filter: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> Dict[str, Any]:
        final_key = resolve_api_key(api_key, category=key_category)
        if not final_key:
            err = f"失败: 未找到 {key_category} 类别的 API Key。请在节点或 config.json 中配置。"
            return {"ui": {"string": [err]}, "result": ("", err)}

        try:
            client = YesAPIClient(
                api_key=final_key,
                base_url=resolve_base_url(base_url),
                category=key_category,
                timeout=30,
            )
            models = client.list_models(keyword=keyword_filter)
            text = "\n".join(models)
            status = f"查询成功 | 分类: {key_category} | 匹配到 {len(models)} 个可用模型"
            return {"ui": {"string": [status]}, "result": (text, status)}
        except Exception as e:
            err = f"查询失败: {e}"
            return {"ui": {"string": [err]}, "result": ("", err)}


NODE_CLASS_MAPPINGS = {
    "YesAPILLMChat": YesAPILLMChat,
    "YesAPIVLMChat": YesAPIVLMChat,
    "YesAPILLMWriter": YesAPILLMWriter,
    "YesAPILLMBatchCSV": YesAPILLMBatchCSV,
    "YesAPIModelList": YesAPIModelList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YesAPILLMChat": "💬 YesAPI LLM 文本对话",
    "YesAPIVLMChat": "👁️ YesAPI VLM 多图视觉对话",
    "YesAPILLMWriter": "✍️ YesAPI AI 批量写词导出表格",
    "YesAPILLMBatchCSV": "📊 YesAPI LLM 批量表格处理 (CSV/Excel)",
    "YesAPIModelList": "🔍 YesAPI 可用模型查询",
}
