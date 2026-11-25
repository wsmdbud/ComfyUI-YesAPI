import os
from unittest.mock import MagicMock, patch
import pandas as pd
from PIL import Image
import pytest
import torch

from core.client import YesAPIError
from nodes.image_nodes import (
    YesAPIImageBatchCSV,
    YesAPIImageBatchDir,
    YesAPIImageGenerate,
    YesAPIImageText2Image,
    YesAPIImageSaveWithPrompt,
    YesAPISaveImageByFilename,
)
from nodes.image_async_nodes import (
    YesAPIImageAsyncBatchSubmit,
    YesAPIImageAsyncQuery,
    YesAPIImageAsyncSubmit,
)


def _dummy_tensor(h=32, w=32, batch=1):
    return torch.ones((batch, h, w, 3), dtype=torch.float32) * 0.5


def test_all_image_nodes_separate_aspect_ratio_and_resolution():
    for cls in (
        YesAPIImageGenerate,
        YesAPIImageBatchCSV,
        YesAPIImageSaveWithPrompt,
        YesAPIImageBatchDir,
        YesAPIImageAsyncSubmit,
        YesAPIImageAsyncBatchSubmit,
    ):
        req = cls.INPUT_TYPES()["required"]
        assert "aspect_ratio" in req, f"{cls.__name__} missing aspect_ratio"
        assert "resolution" in req, f"{cls.__name__} missing resolution"
        assert "size" not in req, f"{cls.__name__} still has mixed size dropdown"
        for opt in req["aspect_ratio"][0]:
            assert opt not in ("1K", "2K", "4K", "1024x1024", "3840x2160")
        for opt in req["resolution"][0]:
            assert ":" not in opt


def test_text2image_node_has_no_image_slots_and_generates():
    it = YesAPIImageText2Image.INPUT_TYPES()
    all_keys = set(it["required"]) | set(it.get("optional", {}))
    assert not any(k.startswith("image_") for k in all_keys)
    assert "mask" not in all_keys
    assert "aspect_ratio" in it["required"] and "resolution" in it["required"]

    fake_client = MagicMock()
    fake_client.generate_images.return_value = [Image.new("RGB", (64, 36), (10, 10, 10))]
    with patch("nodes.image_nodes.YesAPIClient", return_value=fake_client):
        res = YesAPIImageText2Image().execute(
            prompt="sunset",
            model="gpt-image-2",
            aspect_ratio="16:9",
            resolution="1K",
            quality="auto",
            concurrency=1,
            random_mode="固定",
            api_key="sk-img",
        )
    out, status = res["result"]
    assert out.shape == (1, 36, 64, 3)
    assert "文生图" in status
    kw = fake_client.generate_images.call_args.kwargs
    assert kw["ref_images"] == []
    assert kw["mask_png_bytes"] is None
    assert kw["size"] == "1920x1080"


def test_image_generate_sync_and_mask_validation(monkeypatch):
    node = YesAPIImageGenerate()

    # 1. Mask without image_1 returns safe error result without crashing
    res_err = node.execute(
        prompt="fix hat",
        model="Nano-banana-Pro",
        aspect_ratio="1:1",
        resolution="1K",
        quality="high",
        concurrency=1,
        random_mode="固定",
        api_key="sk-img",
        mask=torch.ones((1, 32, 32), dtype=torch.float32),
    )
    img_out, status_out = res_err["result"]
    assert img_out.shape == (1, 64, 64, 3)
    assert "image_1" in status_out

    # 2. Normal concurrent generation with 16:9 + 4K -> sends size="3840x2160"
    fake_client = MagicMock()
    fake_client.generate_images.return_value = [Image.new("RGB", (64, 48), (120, 200, 80))]
    with patch("nodes.image_nodes.YesAPIClient", return_value=fake_client):
        res_ok = node.execute(
            prompt="cat astronaut",
            model="Nano-banana-Pro",
            aspect_ratio="16:9",
            resolution="4K",
            quality="high",
            concurrency=2,
            random_mode="固定",
            custom_model="Nano-banana-2.1",
            api_key="sk-img",
            image_1=_dummy_tensor(48, 64),
            mask=torch.zeros((1, 48, 64), dtype=torch.float32),
        )
    batch_tensor, status = res_ok["result"]
    assert batch_tensor.shape == (2, 48, 64, 3)
    assert "成功: 2" in status
    assert fake_client.generate_images.call_args.kwargs["model"] == "Nano-banana-2.1"
    assert fake_client.generate_images.call_args.kwargs["size"] == "3840x2160"
    assert fake_client.generate_images.call_args.kwargs["mask_png_bytes"] is not None


def test_image_batch_csv_trips_circuit_breaker_on_fatal_402(tmp_path):
    csv_path = tmp_path / "prompts.csv"
    pd.DataFrame({"prompt": [f"p_{i}" for i in range(10)]}).to_csv(csv_path, index=False)

    call_counter = {"count": 0}

    def side_effect(**kwargs):
        call_counter["count"] += 1
        raise YesAPIError("API Key 额度不足或已耗尽 (HTTP 402)", status_code=402, is_fatal=True)

    fake_client = MagicMock()
    fake_client.generate_images.side_effect = side_effect

    node = YesAPIImageBatchCSV()
    with patch("nodes.image_nodes.YesAPIClient", return_value=fake_client):
        res = node.execute(
            file_path=f'"{csv_path}"',
            column_name="prompt",
            prompt_prefix="masterpiece, ",
            model="gpt-image-2",
            aspect_ratio="1:1",
            resolution="1K",
            quality="auto",
            concurrency=1,
            max_count=10,
            executions_per_prompt=1,
            api_key="sk-img",
        )
    _, status = res["result"]
    assert call_counter["count"] == 1
    assert "熔断" in status or "额度" in status


def test_save_with_prompt_and_save_by_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("COMFYUI_TEMP_DIR", str(tmp_path / "tmp"))

    csv_path = tmp_path / "named.csv"
    pd.DataFrame({"prompt": ["prefix_可爱小狗_suffix", "prefix_星空城堡_suffix"]}).to_csv(csv_path, index=False)

    fake_client = MagicMock()
    fake_client.generate_images.side_effect = [
        [Image.new("RGB", (80, 60), (255, 0, 0))],
        [Image.new("RGB", (60, 80), (0, 255, 0))],
    ]

    save_node = YesAPIImageSaveWithPrompt()
    with patch("nodes.image_nodes.YesAPIClient", return_value=fake_client):
        res = save_node.execute(
            file_path=str(csv_path),
            column_name="prompt",
            prompt_prefix="",
            filename_prefix="sub/YesBatch",
            rm_prompt_prefix="prefix_",
            rm_prompt_suffix="_suffix",
            model="gpt-image-2",
            aspect_ratio="16:9",
            resolution="2K",
            quality="auto",
            concurrency=2,
            max_count=10,
            ui_display_mode="保存 (Saved)",
            api_key="sk-img",
        )
    tensor_out, status, filenames_str = res["result"]
    assert tensor_out.shape == (2, 80, 80, 3)
    saved_paths = [p for p in filenames_str.splitlines() if p.strip()]
    assert len(saved_paths) == 2
    assert any("可爱小狗" in os.path.basename(p) for p in saved_paths)

    by_name_node = YesAPISaveImageByFilename()
    ui_res = by_name_node.execute(
        images=tensor_out,
        filenames=filenames_str,
        output_path="upscaled",
        suffix="_HD",
    )
    assert len(ui_res["ui"]["images"]) == 2
    assert os.path.exists(os.path.join(str(tmp_path / "out"), "upscaled", "YesBatch_可爱小狗_HD.png"))


def test_image_batch_dir_skips_bad_file_and_saves_originals(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_OUTPUT_DIR", str(tmp_path / "out"))
    in_dir = tmp_path / "input_imgs"
    in_dir.mkdir()
    Image.new("RGB", (50, 50), (10, 20, 30)).save(in_dir / "good.png")
    (in_dir / "corrupt.png").write_bytes(b"")

    fake_client = MagicMock()
    fake_client.generate_images.return_value = [Image.new("RGB", (100, 60), (200, 100, 50))]

    dir_node = YesAPIImageBatchDir()
    with patch("nodes.image_nodes.YesAPIClient", return_value=fake_client):
        res = dir_node.execute(
            directory_path=str(in_dir),
            csv_file_path="",
            column_name="prompt",
            model="Nano-banana-2",
            aspect_ratio="auto",
            resolution="2K",
            quality="auto",
            max_concurrent_files=2,
            executions_per_prompt=1,
            fixed_prompt="restyle as watercolor",
            api_key="sk-img",
        )
    preview_tensor, report = res["result"]
    assert preview_tensor.shape == (1, 60, 100, 3)
    assert "成功: 1" in report
    assert "失败: 1" in report
    assert fake_client.generate_images.call_args.kwargs["size"] == "2K"


def test_async_image_submit_batch_and_query_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_TEMP_DIR", str(tmp_path / "tmp"))
    store_file = str(tmp_path / "tmp" / "yesapi_image_tasks.json")

    fake_client = MagicMock()
    fake_client.submit_async_image.side_effect = ["img_001", "img_002", "img_003"]
    fake_client.poll_image_task.side_effect = lambda tid: {
        "id": tid,
        "status": "completed",
        "content_url": f"/v1/images/{tid}/content",
        "error": "",
    }
    fake_client.download_image_content.side_effect = lambda tid, url=None: Image.new("RGB", (40, 40), (90, 90, 90))

    submit_node = YesAPIImageAsyncSubmit()
    batch_node = YesAPIImageAsyncBatchSubmit()
    query_node = YesAPIImageAsyncQuery()

    csv_path = tmp_path / "async.csv"
    pd.DataFrame({"prompt": ["p1", "p2"]}).to_csv(csv_path, index=False)

    with (
        patch("nodes.image_async_nodes.YesAPIClient", return_value=fake_client),
        patch("nodes.image_async_nodes._get_image_task_store", side_effect=lambda: __import__("core.task_store", fromlist=["JsonTaskStore"]).JsonTaskStore(store_file)),
    ):
        r1 = submit_node.execute(
            prompt="single async",
            model="gpt-image-2",
            aspect_ratio="9:16",
            resolution="2K",
            quality="auto",
            concurrency=1,
            api_key="sk-img",
        )
        tid1, status1 = r1["result"]
        assert tid1 == "任务1"
        assert "提交成功" in status1
        assert fake_client.submit_async_image.call_args.kwargs["size"] == "1440x2560"

        r2 = batch_node.execute(
            file_path=str(csv_path),
            column_name="prompt",
            prompt_prefix="",
            model="gpt-image-2",
            aspect_ratio="1:1",
            resolution="1K",
            quality="auto",
            executions_per_prompt=1,
            concurrency=2,
            api_key="sk-img",
        )
        tid2, _ = r2["result"]
        assert tid2 == "批量任务1"

        q1 = query_node.execute(target_task_id="", api_key="sk-img")
        img_batch1, q_status1 = q1["result"]
        assert img_batch1.shape == (1, 40, 40, 3)
        assert "任务1" in q_status1

        q2 = query_node.execute(target_task_id="", api_key="sk-img")
        img_batch2, q_status2 = q2["result"]
        assert img_batch2.shape == (2, 40, 40, 3)
        assert "批量任务1" in q_status2
