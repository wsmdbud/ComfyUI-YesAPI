import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import torch

from nodes.llm_nodes import (
    YesAPILLMBatchCSV,
    YesAPILLMChat,
    YesAPILLMWriter,
    YesAPIModelList,
    YesAPIVLMChat,
)
from nodes.video_nodes import (
    YesAPIVideoAsyncBatchSubmit,
    YesAPIVideoAsyncSubmit,
    YesAPIVideoGenerate,
    YesAPIVideoGetNext,
    YesAPIVideoQueryTasks,
)


def _dummy_img(h=32, w=32):
    return torch.ones((1, h, w, 3), dtype=torch.float32) * 0.4


def test_llm_chat_vlm_chat_and_model_list():
    fake_client = MagicMock()
    fake_client.chat_completion.return_value = "Hello from Gemini 4"
    fake_client.list_models.return_value = ["Nano-banana-2.1", "Nano-banana-Pro"]

    llm_node = YesAPILLMChat()
    vlm_node = YesAPIVLMChat()
    models_node = YesAPIModelList()

    with patch("nodes.llm_nodes.YesAPIClient", return_value=fake_client):
        r1 = llm_node.execute(
            model="gemini-4-carbon",
            protocol="auto",
            system_prompt="You are concise.",
            prompt="Hi",
            temperature=0.5,
            max_tokens=512,
            random_mode="固定",
            api_key="sk-llm",
        )
        text1, status1 = r1["result"]
        assert text1 == "Hello from Gemini 4"
        assert "成功" in status1

        r2 = vlm_node.execute(
            model="claude-sonnet-5.5",
            protocol="auto",
            system_prompt="",
            prompt="Describe both",
            temperature=0.7,
            max_tokens=1024,
            random_mode="固定",
            image_1=_dummy_img(),
            image_2=_dummy_img(),
            api_key="sk-llm",
        )
        text2, status2 = r2["result"]
        assert text2 == "Hello from Gemini 4"
        assert len(fake_client.chat_completion.call_args.kwargs["images"]) == 2
        assert "2 张图" in status2

        r3 = models_node.execute(key_category="image", keyword_filter="banana", api_key="sk-img")
        models_txt, m_status = r3["result"]
        assert "Nano-banana-Pro" in models_txt
        assert "2 个" in m_status


def test_llm_writer_and_llm_batch_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_OUTPUT_DIR", str(tmp_path / "out"))
    fake_client = MagicMock()
    # Markdown fenced JSON array output from LLM
    fake_client.chat_completion.side_effect = [
        '```json\n["sunset beach anime style", "cyberpunk neon alley"]\n```',
        "Translated: A",
        "Translated: B",
    ]

    writer_node = YesAPILLMWriter()
    batch_csv_node = YesAPILLMBatchCSV()

    with patch("nodes.llm_nodes.YesAPIClient", return_value=fake_client):
        w_res = writer_node.execute(
            model="gemini-3.8-flash",
            protocol="auto",
            main_prompt="Give me 2 prompts",
            system_prompt="Return JSON array",
            output_filename="my_prompts.csv",
            column_name="prompt",
            api_key="sk-llm",
        )
        csv_out_path, preview_txt, w_status = w_res["result"]
        assert os.path.exists(csv_out_path)
        df_written = pd.read_csv(csv_out_path)
        assert list(df_written["prompt"]) == ["sunset beach anime style", "cyberpunk neon alley"]
        assert "2 条" in w_status

        b_res = batch_csv_node.execute(
            file_path=csv_out_path,
            input_column="prompt",
            output_column="zh_prompt",
            system_prompt="Translate",
            prompt_template="请翻译：{text}",
            model="deepseek-v4.1-flash",
            protocol="openai",
            concurrency=2,
            max_count=10,
            output_filename="translated.xlsx",
            api_key="sk-llm",
        )
        xlsx_out, b_status = b_res["result"]
        assert os.path.exists(xlsx_out)
        df_trans = pd.read_excel(xlsx_out)
        assert set(df_trans["zh_prompt"]) == {"Translated: A", "Translated: B"}
        assert "成功: 2" in b_status


def test_video_sync_and_async_queue_nodes(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFYUI_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("COMFYUI_TEMP_DIR", str(tmp_path / "tmp"))
    vid_store_path = str(tmp_path / "tmp" / "yesapi_video_tasks.json")

    fake_client = MagicMock()
    fake_client.create_video_task.side_effect = ["vid_sync_1", "vid_async_1", "vid_batch_1", "vid_batch_2"]
    fake_client.poll_video_task.side_effect = [
        {"id": "vid_sync_1", "status": "in_progress", "progress": 50, "poll_after_ms": 10, "error": ""},
        {"id": "vid_sync_1", "status": "completed", "progress": 100, "poll_after_ms": 10, "error": ""},
        {"id": "vid_async_1", "status": "completed", "progress": 100, "poll_after_ms": 10, "error": ""},
        {"id": "vid_batch_1", "status": "in_progress", "progress": 30, "poll_after_ms": 10, "error": ""},
        {"id": "vid_batch_2", "status": "completed", "progress": 100, "poll_after_ms": 10, "error": ""},
    ]

    def _fake_dl(vid_id: str, out_path: str) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42" * 64)
        return out_path

    fake_client.download_video_content.side_effect = _fake_dl

    sync_node = YesAPIVideoGenerate()
    async_sub = YesAPIVideoAsyncSubmit()
    async_batch = YesAPIVideoAsyncBatchSubmit()
    query_node = YesAPIVideoQueryTasks()
    next_node = YesAPIVideoGetNext()

    csv_path = tmp_path / "vids.csv"
    pd.DataFrame({"prompt": ["wave 1", "wave 2"]}).to_csv(csv_path, index=False)

    with (
        patch("nodes.video_nodes.YesAPIClient", return_value=fake_client),
        patch("nodes.video_nodes.time.sleep", return_value=None),
        patch("nodes.video_nodes._get_video_task_store", side_effect=lambda: __import__("core.task_store", fromlist=["JsonTaskStore"]).JsonTaskStore(vid_store_path)),
    ):
        # 1. Sync video generate
        s_res = sync_node.execute(
            prompt="drone shot",
            model="Veo-3.1-fast-1080P",
            aspect_ratio="16:9",
            duration=5,
            poll_timeout_sec=60,
            first_frame=_dummy_img(),
            extra_body_json='{"fps": 24}',
            api_key="sk-vid",
        )
        adapter, v_path, v_status = s_res["result"]
        assert os.path.exists(v_path)
        assert "成功" in v_status

        # 2. Async single + batch submit
        sub_res = async_sub.execute(
            prompt="cat jump",
            model="Veo-3.1-fast-1080P",
            aspect_ratio="16:9",
            duration=5,
            api_key="sk-vid",
        )
        assert sub_res["result"][0] == "vid_async_1"

        b_res = async_batch.execute(
            file_path=str(csv_path),
            column_name="prompt",
            prompt_prefix="",
            model="Veo-3.1-fast-1080P",
            aspect_ratio="16:9",
            duration=5,
            concurrency=2,
            max_count=10,
            api_key="sk-vid",
        )
        assert "成功: 2" in b_res["result"][1]

        # 3. Query tasks updates statuses
        q_res = query_node.execute(api_key="sk-vid")
        assert "vid_async_1" in q_res["result"][0]

        # 4. GetNext downloads earliest completed video
        n_res = next_node.execute(target_task_id="", api_key="sk-vid")
        dl_adapter, dl_path, dl_report = n_res["result"]
        assert os.path.exists(dl_path)
        assert "下载成功" in dl_report


def test_comfyui_hyphenated_package_loader_registers_all_18_nodes():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    init_py = os.path.join(repo_root, "__init__.py")
    pkg_name = "ComfyUI-YesAPI"

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        init_py,
        submodule_search_locations=[repo_root],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)

    mappings = module.NODE_CLASS_MAPPINGS
    display = module.NODE_DISPLAY_NAME_MAPPINGS
    assert len(mappings) == 18
    assert set(mappings.keys()) == set(display.keys())

    for key, cls in mappings.items():
        inp = cls.INPUT_TYPES()
        assert isinstance(inp, dict)
        assert hasattr(cls, cls.FUNCTION)
        assert str(cls.CATEGORY).startswith("YesAPI/")
