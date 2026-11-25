from io import BytesIO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from core.media import (
    MAX_UPLOAD_IMAGE_BYTES,
    YesAPIVideoAdapter,
    encode_image_under_limit,
    make_error_image_tensor,
    mask_to_png_bytes,
    pil_to_base64_jpeg,
    pil_to_tensor,
    read_table_file,
    safe_pil_batch_to_tensor,
    sanitize_filename,
    tensor_to_pil,
)


def test_tensor_pil_roundtrip():
    arr = np.zeros((2, 32, 48, 3), dtype=np.float32)
    arr[0, :, :, 0] = 1.0  # pure red
    arr[1, :, :, 2] = 1.0  # pure blue
    tensor = torch.from_numpy(arr)

    pils = tensor_to_pil(tensor)
    assert len(pils) == 2
    assert pils[0].size == (48, 32)

    back = pil_to_tensor(pils)
    assert back.shape == (2, 32, 48, 3)
    assert back.dtype == torch.float32
    assert torch.allclose(tensor, back, atol=1.0 / 255.0)


def test_safe_pil_batch_to_tensor_pads_heterogeneous_sizes_and_modes():
    img1 = Image.new("RGB", (200, 100), (255, 0, 0))
    img2 = Image.new("RGBA", (120, 150), (0, 255, 0, 128))
    img3 = Image.new("L", (90, 80), 200)

    batch = safe_pil_batch_to_tensor([img1, img2, img3])
    assert batch.shape == (3, 150, 200, 3)
    assert batch.dtype == torch.float32


def test_mask_to_png_bytes_converts_comfy_mask_to_openai_transparent_edit_area():
    # ComfyUI MASK: (1, H, W), 1.0 = edit region (should become alpha=0), 0.0 = keep region (alpha=255)
    mask = torch.zeros((1, 40, 60), dtype=torch.float32)
    mask[0, 10:30, 15:45] = 1.0

    png_bytes = mask_to_png_bytes(mask, target_size=(120, 80))
    out_img = Image.open(BytesIO(png_bytes))
    assert out_img.mode == "RGBA"
    assert out_img.size == (120, 80)
    alpha = np.array(out_img)[:, :, 3]
    # Center (inside scaled [30:90, 20:60]) must be transparent (0), corner must be opaque (255)
    assert alpha[40, 60] == 0
    assert alpha[0, 0] == 255


def test_encode_image_under_limit_compresses_oversized_image():
    assert MAX_UPLOAD_IMAGE_BYTES == 19 * 1024 * 1024
    rng = np.random.default_rng(42)
    noisy = Image.fromarray(rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8), mode="RGB")
    # Force a strict 40KB limit so PNG (~750KB) must compress/downscale
    data, fname, mime = encode_image_under_limit(noisy, max_bytes=40_000, index=0)
    assert len(data) <= 40_000
    assert mime in ("image/png", "image/jpeg")
    assert fname.endswith((".png", ".jpg"))


def test_read_table_file_supports_gbk_csv_and_xlsx(tmp_path):
    gbk_csv = tmp_path / "cn_gbk.csv"
    gbk_csv.write_bytes("prompt,tag\n一只橘猫在晒太阳,cat\n赛博朋克城市夜景,city\n".encode("gbk"))

    df_csv = read_table_file(f'  "{gbk_csv}"  ')
    assert list(df_csv["prompt"]) == ["一只橘猫在晒太阳", "赛博朋克城市夜景"]

    xlsx_path = tmp_path / "prompts.xlsx"
    pd.DataFrame({"prompt": ["夕阳海滩", "森林木屋"]}).to_excel(xlsx_path, index=False)
    df_xlsx = read_table_file(str(xlsx_path))
    assert list(df_xlsx["prompt"]) == ["夕阳海滩", "森林木屋"]


def test_sanitize_filename_and_error_tensor():
    assert sanitize_filename('a/b\\c:d*e?"f<g>h|i   test') == "a_b_c_d_e_f_g_h_i_test"
    err_t = make_error_image_tensor()
    assert err_t.shape == (1, 64, 64, 3)


def test_video_adapter_handles_missing_and_valid_video(tmp_path):
    empty_adapter = YesAPIVideoAdapter("")
    assert empty_adapter.get_dimensions() == (1280, 720)
    assert empty_adapter.save_to(str(tmp_path / "out.mp4")) is False

    # Create a tiny real video file via OpenCV
    vid_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    for _ in range(15):
        frame = np.full((240, 320, 3), 120, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    adapter = YesAPIVideoAdapter(str(vid_path))
    assert adapter.get_dimensions() == (320, 240)
    copied = tmp_path / "copied.avi"
    assert adapter.save_to(str(copied)) is True
    assert copied.exists() and copied.stat().st_size > 1024
