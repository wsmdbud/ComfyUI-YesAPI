"""Unified HTTP client for Yesapi (yesapi.online): Images, Dual-Protocol LLM/VLM, Videos, and Models."""
from __future__ import annotations

import base64
from io import BytesIO
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import requests

from .config import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SECONDS, format_yesapi_error, resolve_base_url
from .media import encode_image_under_limit, pil_to_base64_jpeg, pil_to_data_uri

MultipartSpec = Tuple[str, Tuple[str, bytes, str]]
_NANO_OR_GEMINI_IMAGE_RE = re.compile(r"nano-?banana|gemini|imagen", re.IGNORECASE)


def _supports_quality_param(model: str) -> bool:
    """Matches Yesapi ImageStudioView: Nano-banana/Gemini/Imagen models omit 'quality'."""
    return not bool(_NANO_OR_GEMINI_IMAGE_RE.search(model or ""))


class YesAPIError(Exception):
    """Structured exception carrying HTTP status and circuit-breaker fatality flag."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        is_fatal: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.is_fatal = is_fatal


class YesAPIClient:
    """Stateless-friendly HTTP client for all Yesapi endpoints with /v1 <-> root 404 route fallback."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        category: str = "default",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        clean_key = (api_key or "").strip()
        if not clean_key:
            cat_hint = f"{category}_api_key" if category in ("image", "llm", "video") else "default_api_key"
            raise YesAPIError(
                f"API Key 不能为空。请在节点面板填入 api_key，或在 config.json 中配置 {cat_hint}。",
                is_fatal=True,
            )
        self.api_key = clean_key
        self.base_url = resolve_base_url(base_url)
        self.category = category
        self.timeout = max(10, int(timeout))
        self.session = session or requests.Session()

    def _bearer_headers(self, json_content: bool = True) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "ComfyUI-YesAPI/1.0",
        }
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    def _anthropic_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-YesAPI/1.0",
        }

    def _raise_from_response(self, resp: requests.Response) -> None:
        raw_msg = ""
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                err_obj = payload.get("error")
                if isinstance(err_obj, dict):
                    raw_msg = str(err_obj.get("message") or err_obj.get("code") or payload)
                elif isinstance(err_obj, str):
                    raw_msg = err_obj
                else:
                    raw_msg = str(payload.get("message") or payload.get("msg") or payload)
            else:
                raw_msg = str(payload)
        except Exception:
            raw_msg = (resp.text or f"HTTP {resp.status_code}").strip()[:400]

        formatted, is_fatal = format_yesapi_error(raw_msg, status_code=resp.status_code, category=self.category)
        raise YesAPIError(formatted, status_code=resp.status_code, is_fatal=is_fatal)

    def _request(
        self,
        method: str,
        primary_path: str,
        alt_path: Optional[str] = None,
        *,
        extra_alt_paths: Optional[Sequence[str]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        form_data: Optional[Dict[str, Any]] = None,
        files_spec: Optional[Sequence[MultipartSpec]] = None,
        stream: bool = False,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        paths = [primary_path]
        if alt_path and alt_path not in paths:
            paths.append(alt_path)
        for p in extra_alt_paths or ():
            if p and p not in paths:
                paths.append(p)

        req_timeout = timeout or self.timeout
        last_resp: Optional[requests.Response] = None

        for idx, path in enumerate(paths):
            url = path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"
            req_kwargs: Dict[str, Any] = {
                "headers": headers,
                "timeout": req_timeout,
            }
            if json_body is not None:
                req_kwargs["json"] = json_body
            if form_data is not None:
                req_kwargs["data"] = form_data
            if files_spec is not None:
                req_kwargs["files"] = list(files_spec)
            if stream:
                req_kwargs["stream"] = True

            try:
                resp = self.session.request(method, url, **req_kwargs)
            except requests.exceptions.Timeout as e:
                raise YesAPIError(f"请求超时 ({req_timeout}s): {e}", is_fatal=False) from e
            except requests.exceptions.RequestException as e:
                raise YesAPIError(f"网络连接失败: {e}", is_fatal=False) from e

            last_resp = resp
            if resp.status_code == 404 and idx < len(paths) - 1:
                continue
            if 200 <= resp.status_code < 300:
                return resp
            self._raise_from_response(resp)

        assert last_resp is not None
        self._raise_from_response(last_resp)
        raise AssertionError("unreachable")

    def _decode_pil_from_url(self, url: str) -> Image.Image:
        full_url = url if url.startswith(("http://", "https://")) else f"{self.base_url}{url}"
        try:
            resp = self.session.get(full_url, timeout=120)
            if resp.status_code >= 400:
                self._raise_from_response(resp)
            img = Image.open(BytesIO(resp.content))
            img.load()
            return img
        except YesAPIError:
            raise
        except Exception as e:
            raise YesAPIError(f"下载生成图像失败: {e}", is_fatal=False) from e

    def _extract_images_from_payload(self, payload: Dict[str, Any]) -> List[Image.Image]:
        items: List[Dict[str, Any]] = []
        if isinstance(payload.get("data"), list):
            items = [x for x in payload["data"] if isinstance(x, dict)]
        elif isinstance(payload.get("result"), dict) and isinstance(payload["result"].get("data"), list):
            items = [x for x in payload["result"]["data"] if isinstance(x, dict)]
        elif isinstance(payload.get("results"), list):
            items = [x for x in payload["results"] if isinstance(x, dict)]
        elif "b64_json" in payload or "url" in payload:
            items = [payload]

        images: List[Image.Image] = []
        for item in items:
            if item.get("b64_json"):
                raw = base64.b64decode(item["b64_json"])
                img = Image.open(BytesIO(raw))
                img.load()
                images.append(img)
            elif item.get("url"):
                images.append(self._decode_pil_from_url(str(item["url"])))

        if not images:
            raise YesAPIError(f"接口响应中未找到有效图像数据: {str(payload)[:240]}", is_fatal=False)
        return images

    @staticmethod
    def _build_edit_files_spec(
        refs: Sequence[Image.Image],
        mask_png_bytes: Optional[bytes] = None,
    ) -> List[MultipartSpec]:
        # Matches Yesapi ImageStudioView: singular 'image' when 1 ref, repeated 'image[]' when >1 refs
        field_name = "image" if len(refs) == 1 else "image[]"
        files_spec: List[MultipartSpec] = []
        for idx, pil_img in enumerate(refs):
            img_bytes, fname, mime = encode_image_under_limit(pil_img, index=idx)
            files_spec.append((field_name, (fname, img_bytes, mime)))
        if mask_png_bytes is not None:
            files_spec.append(("mask", ("mask.png", mask_png_bytes, "image/png")))
        return files_spec

    # ==================== 1. Image Generation & Editing ====================

    def generate_images(
        self,
        prompt: str,
        model: str,
        size: Optional[str] = "1024x1024",
        quality: str = "auto",
        n: int = 1,
        ref_images: Optional[Sequence[Image.Image]] = None,
        mask_png_bytes: Optional[bytes] = None,
    ) -> List[Image.Image]:
        refs = list(ref_images or [])
        if mask_png_bytes is not None and not refs:
            raise YesAPIError(
                "使用局部重绘遮罩 (mask) 时，必须同时连接至少一张基础图像 (image_1)。",
                is_fatal=True,
            )

        allow_quality = quality and quality != "auto" and _supports_quality_param(model)

        if not refs and mask_png_bytes is None:
            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "response_format": "b64_json",
            }
            if size and size.strip():
                payload["size"] = size.strip()
            if allow_quality:
                payload["quality"] = quality
            if n > 1:
                payload["n"] = int(n)

            resp = self._request(
                "POST",
                "/v1/images/generations",
                "/images/generations",
                headers=self._bearer_headers(json_content=True),
                json_body=payload,
            )
            return self._extract_images_from_payload(resp.json())

        files_spec = self._build_edit_files_spec(refs, mask_png_bytes)
        form_data: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json",
        }
        if size and size.strip():
            form_data["size"] = size.strip()
        if allow_quality:
            form_data["quality"] = quality
        if n > 1:
            form_data["n"] = str(int(n))

        resp = self._request(
            "POST",
            "/v1/images/edits",
            "/images/edits",
            headers=self._bearer_headers(json_content=False),
            form_data=form_data,
            files_spec=files_spec,
        )
        return self._extract_images_from_payload(resp.json())

    def submit_async_image(
        self,
        prompt: str,
        model: str,
        size: Optional[str] = "1024x1024",
        quality: str = "auto",
        ref_images: Optional[Sequence[Image.Image]] = None,
        mask_png_bytes: Optional[bytes] = None,
    ) -> str:
        refs = list(ref_images or [])
        if mask_png_bytes is not None and not refs:
            raise YesAPIError(
                "使用局部重绘遮罩 (mask) 时，必须同时连接至少一张基础图像 (image_1)。",
                is_fatal=True,
            )

        allow_quality = quality and quality != "auto" and _supports_quality_param(model)

        if not refs and mask_png_bytes is None:
            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "response_format": "url",
                "n": 1,
                "async": True,
            }
            if size and size.strip():
                payload["size"] = size.strip()
            if allow_quality:
                payload["quality"] = quality
            resp = self._request(
                "POST",
                "/v1/images/generations",
                "/images/generations",
                extra_alt_paths=("/v1/images/generations/async",),
                headers=self._bearer_headers(json_content=True),
                json_body=payload,
            )
        else:
            files_spec = self._build_edit_files_spec(refs, mask_png_bytes)
            form_data: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "response_format": "url",
                "n": "1",
                "async": "true",
            }
            if size and size.strip():
                form_data["size"] = size.strip()
            if allow_quality:
                form_data["quality"] = quality
            resp = self._request(
                "POST",
                "/v1/images/edits",
                "/images/edits",
                extra_alt_paths=("/v1/images/edits/async",),
                headers=self._bearer_headers(json_content=False),
                form_data=form_data,
                files_spec=files_spec,
            )

        data = resp.json()
        img_id = str(
            data.get("id")
            or data.get("task_id")
            or (data.get("data") or {}).get("id")
            or (data.get("data") or {}).get("task_id")
            or ""
        ).strip()
        if not img_id:
            raise YesAPIError(f"异步图像提交未返回有效任务 ID: {data}", is_fatal=False)
        return img_id

    def poll_image_task(self, image_id: str) -> Dict[str, Any]:
        resp = self._request(
            "GET",
            f"/v1/images/{image_id}",
            f"/images/{image_id}",
            extra_alt_paths=(f"/v1/images/tasks/{image_id}",),
            headers=self._bearer_headers(json_content=False),
            timeout=30,
        )
        raw = resp.json()
        payload = raw.get("data") if isinstance(raw.get("data"), dict) and "status" in raw["data"] else raw
        raw_status = str(payload.get("status", "pending")).lower()
        if raw_status in ("completed", "succeeded", "success"):
            norm_status = "completed"
        elif raw_status in ("failed", "error", "cancelled", "canceled"):
            norm_status = "failed"
        else:
            norm_status = "pending"

        err_val = payload.get("error")
        if isinstance(err_val, dict):
            err_str = str(err_val.get("message") or err_val.get("code") or err_val)
        else:
            err_str = str(err_val) if err_val else ""

        direct_url = None
        if isinstance(payload.get("result"), dict) and isinstance(payload["result"].get("data"), list):
            first_item = payload["result"]["data"][0] if payload["result"]["data"] else {}
            if isinstance(first_item, dict) and first_item.get("url"):
                direct_url = str(first_item["url"])

        return {
            "id": str(payload.get("id") or payload.get("task_id") or image_id),
            "status": norm_status,
            "content_url": direct_url or payload.get("content_url") or f"/v1/images/{image_id}/content",
            "error": err_str,
            "raw": raw,
        }

    def download_image_content(self, image_id: str, content_url: Optional[str] = None) -> Image.Image:
        primary = content_url or f"/v1/images/{image_id}/content"
        alt = f"/images/{image_id}/content" if primary.startswith("/v1/") else f"/v1/images/{image_id}/content"
        resp = self._request(
            "GET",
            primary,
            alt if not primary.startswith(("http://", "https://")) else None,
            headers=self._bearer_headers(json_content=False) if not primary.startswith(("http://", "https://")) else None,
            timeout=120,
        )
        ctype = str(resp.headers.get("Content-Type", "")).lower()
        if "application/json" in ctype:
            imgs = self._extract_images_from_payload(resp.json())
            return imgs[0]
        try:
            img = Image.open(BytesIO(resp.content))
            img.load()
            return img
        except Exception as e:
            raise YesAPIError(f"解析下载图像内容失败: {e}", is_fatal=False) from e

    # ==================== 2. LLM / VLM & Model Discovery ====================

    def chat_completion(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
        images: Optional[Sequence[Image.Image]] = None,
        protocol: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        proto = (protocol or "auto").strip().lower()
        if proto == "auto":
            proto = "anthropic" if model.strip().lower().startswith("claude-") else "openai"

        pil_list = list(images or [])

        if proto == "anthropic":
            content_blocks: List[Dict[str, Any]] = []
            for img in pil_list:
                b64 = pil_to_base64_jpeg(img)
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    }
                )
            content_blocks.append({"type": "text", "text": prompt})
            body: Dict[str, Any] = {
                "model": model,
                "max_tokens": max(64, int(max_tokens)),
                "temperature": float(temperature),
                "messages": [{"role": "user", "content": content_blocks}],
            }
            if system_prompt and system_prompt.strip():
                body["system"] = system_prompt.strip()

            resp = self._request(
                "POST",
                "/messages",
                "/v1/messages",
                headers=self._anthropic_headers(),
                json_body=body,
            )
            data = resp.json()
            blocks = data.get("content") or []
            texts = [str(b.get("text", "")) for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
            if texts:
                return "".join(texts).strip()
            raise YesAPIError(f"Anthropic 响应格式异常: {str(data)[:240]}", is_fatal=False)

        messages: List[Dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        if not pil_list:
            messages.append({"role": "user", "content": prompt})
        else:
            user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in pil_list:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_data_uri(img)},
                    }
                )
            messages.append({"role": "user", "content": user_content})

        body = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": max(64, int(max_tokens)),
            "stream": False,
        }
        resp = self._request(
            "POST",
            "/chat/completions",
            "/v1/chat/completions",
            headers=self._bearer_headers(json_content=True),
            json_body=body,
        )
        data = resp.json()
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [str(p.get("text", "")) for p in content if isinstance(p, dict)]
                return "".join(parts).strip()
        raise YesAPIError(f"Chat Completions 响应格式异常: {str(data)[:240]}", is_fatal=False)

    def list_models(self, keyword: str = "") -> List[str]:
        resp = self._request(
            "GET",
            "/v1/models",
            "/models",
            headers=self._bearer_headers(json_content=False),
            timeout=30,
        )
        data = resp.json()
        raw_list = data.get("data") if isinstance(data, dict) else data
        model_ids: List[str] = []
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, dict) and item.get("id"):
                    model_ids.append(str(item["id"]))
                elif isinstance(item, str):
                    model_ids.append(item)
        kw = (keyword or "").strip().lower()
        if kw:
            model_ids = [m for m in model_ids if kw in m.lower()]
        return sorted(set(model_ids))

    # ==================== 3. Video Generation (/v1/videos) ====================

    def create_video_task(
        self,
        model: str,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        resolution: str = "默认",
        first_frame: Optional[Image.Image] = None,
        last_frame: Optional[Image.Image] = None,
        audio_url: str = "",
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        if last_frame is not None and first_frame is None:
            raise YesAPIError("使用尾帧控制 (last_frame) 时，必须同时提供首帧图像 (first_frame)。", is_fatal=True)

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "duration": int(duration),
            "aspect_ratio": aspect_ratio,
        }
        if resolution and resolution.strip() and resolution.strip() != "默认":
            payload["resolution"] = resolution.strip().lower()
        if audio_url and audio_url.strip():
            payload["audio_url"] = audio_url.strip()
        if first_frame is not None:
            payload["image"] = pil_to_data_uri(first_frame)
        if last_frame is not None:
            payload["last_frame_image"] = pil_to_data_uri(last_frame)
        if extra_body:
            payload.update(extra_body)

        resp = self._request(
            "POST",
            "/v1/videos",
            "/videos",
            headers=self._bearer_headers(json_content=True),
            json_body=payload,
            timeout=90,
        )
        data = resp.json()
        vid_id = str(
            data.get("id")
            or data.get("task_id")
            or (data.get("data") or {}).get("id")
            or (data.get("data") or {}).get("task_id")
            or ""
        ).strip()
        if not vid_id:
            raise YesAPIError(f"视频任务提交成功但未返回有效任务 ID: {data}", is_fatal=False)
        return vid_id

    def poll_video_task(self, video_id: str) -> Dict[str, Any]:
        resp = self._request(
            "GET",
            f"/v1/videos/{video_id}",
            f"/videos/{video_id}",
            headers=self._bearer_headers(json_content=False),
            timeout=30,
        )
        raw = resp.json()
        payload = raw.get("data") if isinstance(raw.get("data"), dict) and "status" in raw["data"] else raw
        raw_status = str(payload.get("status", "queued")).lower()

        if raw_status in ("completed", "succeeded", "success"):
            norm_status = "completed"
        elif raw_status in ("failed", "error", "cancelled", "canceled"):
            norm_status = "failed"
        else:
            norm_status = "in_progress"

        err_val = payload.get("error") or payload.get("failure_reason")
        if isinstance(err_val, dict):
            err_str = str(err_val.get("message") or err_val.get("code") or err_val)
        else:
            err_str = str(err_val) if err_val else ""

        try:
            poll_ms = int(payload.get("poll_after_ms") or 3000)
        except Exception:
            poll_ms = 3000

        try:
            progress = int(payload.get("progress") or 0)
        except Exception:
            progress = 0

        return {
            "id": str(payload.get("id") or video_id),
            "status": norm_status,
            "raw_status": raw_status,
            "progress": max(0, min(progress, 100)),
            "poll_after_ms": max(1000, min(poll_ms, 15000)),
            "error": err_str,
            "raw": raw,
        }

    def download_video_content(self, video_id: str, output_path: str) -> str:
        resp = self._request(
            "GET",
            f"/v1/videos/{video_id}/content",
            f"/videos/{video_id}/content",
            headers=self._bearer_headers(json_content=False),
            stream=True,
            timeout=300,
        )
        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        part_path = f"{output_path}.part"
        total_bytes = 0
        try:
            with open(part_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
            if total_bytes == 0:
                raise YesAPIError("下载的视频文件为空 (0 bytes)", is_fatal=False)
            os.replace(part_path, output_path)
            return output_path
        except Exception:
            if os.path.exists(part_path):
                try:
                    os.unlink(part_path)
                except OSError:
                    pass
            raise
