"""Media conversion, mask encoding, >19MB smart image compression, CSV/Excel loading, and VideoAdapter."""
from __future__ import annotations

from io import BytesIO
import base64
import os
import re
import shutil
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from .config import clean_user_path

# Yesapi enforces 20MB per reference image; keep 19MB ceiling to prevent truncation.
MAX_UPLOAD_IMAGE_BYTES = 19 * 1024 * 1024


def safe_pil_to_rgb(image: Image.Image) -> Image.Image:
    """Convert any PIL image safely to RGB, compositing RGBA onto white."""
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def tensor_to_pil(tensor: Optional[torch.Tensor]) -> List[Image.Image]:
    """Convert ComfyUI (B, H, W, C) float32 tensor in [0,1] into a list of PIL Images."""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return []
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    images: List[Image.Image] = []
    for i in range(tensor.shape[0]):
        arr = (torch.clamp(tensor[i], 0.0, 1.0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        if arr.ndim == 2:
            images.append(Image.fromarray(arr, mode="L").convert("RGB"))
        elif arr.shape[-1] == 4:
            images.append(Image.fromarray(arr, mode="RGBA"))
        else:
            images.append(Image.fromarray(arr[:, :, :3], mode="RGB"))
    return images


def pil_to_tensor(pil_images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    """Convert one or more same-size PIL images into a ComfyUI (B, H, W, 3) float32 tensor."""
    if not isinstance(pil_images, list):
        pil_images = [pil_images]
    if not pil_images:
        return make_error_image_tensor()
    tensors: List[torch.Tensor] = []
    for img in pil_images:
        rgb = safe_pil_to_rgb(img)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).unsqueeze(0))
    return torch.cat(tensors, dim=0)


def safe_pil_batch_to_tensor(pil_images: List[Optional[Image.Image]]) -> torch.Tensor:
    """Safely batch PIL images of potentially different dimensions/modes by zero-padding to max (H, W)."""
    valid_tensors: List[torch.Tensor] = []
    max_h = 0
    max_w = 0

    for img in pil_images:
        if img is None:
            continue
        try:
            rgb = safe_pil_to_rgb(img)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[-1] > 3:
                arr = arr[:, :, :3]
            t = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)
            _, h, w, c = t.shape
            if h <= 0 or w <= 0 or c != 3:
                continue
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            valid_tensors.append(t)
        except Exception:
            continue

    if not valid_tensors:
        return make_error_image_tensor()

    padded: List[torch.Tensor] = []
    for t in valid_tensors:
        _, h, w, _ = t.shape
        if h == max_h and w == max_w:
            padded.append(t)
            continue
        chw = t.permute(0, 3, 1, 2)
        chw_padded = F.pad(chw, (0, max_w - w, 0, max_h - h), mode="constant", value=0.0)
        padded.append(chw_padded.permute(0, 2, 3, 1))

    return torch.cat(padded, dim=0)


def make_error_image_tensor(height: int = 64, width: int = 64) -> torch.Tensor:
    """Create a valid fallback black image tensor so downstream ComfyUI nodes do not crash."""
    return torch.zeros((1, height, width, 3), dtype=torch.float32)


def mask_to_png_bytes(
    mask_tensor: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
) -> bytes:
    """
    Convert ComfyUI MASK (where 1.0 = area to edit/inpaint, 0.0 = area to keep)
    into an OpenAI Images Edits RGBA PNG (where alpha=0 = area to edit, alpha=255 = area to keep).
    target_size is (width, height) in PIL convention.
    """
    m = mask_tensor.detach().cpu()
    if m.ndim == 3:
        m = m[0]
    elif m.ndim == 4:
        m = m[0, 0]
    m_clamped = torch.clamp(m.float(), 0.0, 1.0).numpy()
    # Invert for OpenAI alpha: 1.0 (edit) -> 0 (transparent), 0.0 (keep) -> 255 (opaque)
    alpha_arr = ((1.0 - m_clamped) * 255.0).round().astype(np.uint8)
    h, w = alpha_arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha_arr
    pil_mask = Image.fromarray(rgba, mode="RGBA")
    if target_size is not None and pil_mask.size != target_size:
        pil_mask = pil_mask.resize(target_size, resample=Image.Resampling.NEAREST)
    buf = BytesIO()
    pil_mask.save(buf, format="PNG")
    return buf.getvalue()


def encode_image_under_limit(
    pil_image: Image.Image,
    max_bytes: int = MAX_UPLOAD_IMAGE_BYTES,
    index: int = 0,
) -> Tuple[bytes, str, str]:
    """
    Encode a PIL image for multipart upload under max_bytes.
    Tries lossless PNG first; if larger than max_bytes, falls back to JPEG quality/downscaling.
    Returns (image_bytes, filename, mime_type).
    """
    rgb_or_rgba = pil_image if pil_image.mode in ("RGB", "RGBA") else safe_pil_to_rgb(pil_image)
    buf = BytesIO()
    rgb_or_rgba.save(buf, format="PNG", compress_level=6)
    png_bytes = buf.getvalue()
    if len(png_bytes) <= max_bytes:
        return png_bytes, f"ref_{index}.png", "image/png"

    working = safe_pil_to_rgb(pil_image)
    for scale_step in range(8):
        if scale_step > 0:
            new_w = max(64, int(working.width * 0.8))
            new_h = max(64, int(working.height * 0.8))
            working = working.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        for quality in (92, 85, 75, 65, 50):
            jbuf = BytesIO()
            working.save(jbuf, format="JPEG", quality=quality, optimize=True)
            data = jbuf.getvalue()
            if len(data) <= max_bytes:
                return data, f"ref_{index}.jpg", "image/jpeg"

    # Last-resort aggressive resize
    tiny = working.resize((256, 256), resample=Image.Resampling.LANCZOS)
    jbuf = BytesIO()
    tiny.save(jbuf, format="JPEG", quality=50)
    return jbuf.getvalue(), f"ref_{index}.jpg", "image/jpeg"


def pil_to_base64_jpeg(
    pil_image: Image.Image,
    quality: int = 92,
    max_bytes: int = MAX_UPLOAD_IMAGE_BYTES,
) -> str:
    """Encode PIL image into base64 JPEG string safely under max_bytes."""
    rgb = safe_pil_to_rgb(pil_image)
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    raw = buf.getvalue()
    if len(raw) > max_bytes:
        raw, _, _ = encode_image_under_limit(rgb, max_bytes=max_bytes)
    return base64.b64encode(raw).decode("utf-8")


def pil_to_data_uri(pil_image: Image.Image, quality: int = 92) -> str:
    """Return data:image/jpeg;base64,... URI for multimodal chat and video reference frames."""
    b64 = pil_to_base64_jpeg(pil_image, quality=quality)
    return f"data:image/jpeg;base64,{b64}"


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Clean arbitrary prompt/string into a safe cross-platform filename stem."""
    if not text:
        return "untitled"
    cleaned = re.sub(r'[\\/*?:"<>|\r\n\t]+', "_", text)
    cleaned = re.sub(r"[\s_]+", "_", cleaned).strip("_. ")
    if not cleaned:
        return "untitled"
    return cleaned[:max_length]


def is_valid_image_file(file_path: str) -> bool:
    """Verify a local image file is non-empty and readable by Pillow."""
    try:
        if not file_path or not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def read_table_file(file_path: str) -> pd.DataFrame:
    """Load .csv (trying UTF-8-SIG, UTF-8, GB18030, GBK) or .xls/.xlsx into a DataFrame."""
    clean_path = clean_user_path(file_path)
    if not clean_path or not os.path.exists(clean_path):
        raise FileNotFoundError(f"表格文件不存在: {clean_path or '(空路径)'}")

    lower = clean_path.lower()
    if lower.endswith(".csv"):
        last_err: Optional[Exception] = None
        for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return pd.read_csv(clean_path, encoding=enc)
            except Exception as e:
                last_err = e
        raise ValueError(f"CSV 文件读取失败: {last_err}")

    if lower.endswith((".xls", ".xlsx")):
        return pd.read_excel(clean_path)

    raise ValueError("仅支持 .csv, .xls, .xlsx 格式的表格文件")


class YesAPIVideoAdapter:
    """ComfyUI IO.VIDEO compatible wrapper around a downloaded local video file."""

    def __init__(self, video_path: Optional[str]) -> None:
        self.video_path = video_path or ""
        self.width, self.height, self.fps = self._probe(self.video_path)

    @staticmethod
    def _probe(path: str) -> Tuple[int, int, float]:
        try:
            if not path or not os.path.exists(path) or os.path.getsize(path) < 512:
                return 1280, 720, 30.0
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return 1280, 720, 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            cap.release()
            if w <= 0 or h <= 0:
                return 1280, 720, 30.0
            return w, h, (fps if fps > 0 else 30.0)
        except Exception:
            return 1280, 720, 30.0

    def get_dimensions(self) -> Tuple[int, int]:
        return self.width, self.height

    def save_to(self, output_path: str, format: str = "auto", codec: str = "auto", metadata: object = None, **kwargs: object) -> bool:
        try:
            if self.video_path and os.path.exists(self.video_path) and os.path.getsize(self.video_path) > 0:
                parent = os.path.dirname(os.path.abspath(output_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)
                shutil.copyfile(self.video_path, output_path)
                return True
            return False
        except Exception:
            return False
