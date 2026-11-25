import base64
from io import BytesIO
import json
from unittest.mock import MagicMock
from PIL import Image
import pytest

from core.client import YesAPIClient, YesAPIError


def _make_png_bytes(color=(10, 200, 30), size=(32, 32)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mock_response(status_code=200, json_data=None, content=b"", headers=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {"Content-Type": "application/json"}
    resp.content = content
    if json_data is not None:
        resp.json.return_value = json_data
        resp.text = json.dumps(json_data, ensure_ascii=False)
    else:
        resp.json.side_effect = ValueError("No JSON")
        resp.text = text or content.decode("utf-8", errors="ignore")
    return resp


def test_route_404_fallback_rewinds_and_caches_working_prefix():
    session = MagicMock()
    png_b64 = base64.b64encode(_make_png_bytes()).decode("utf-8")
    # First call to /v1/images/generations returns 404 HTML route not found, fallback /images/generations returns 200
    session.request.side_effect = [
        _mock_response(404, text="404 page not found", headers={"Content-Type": "text/plain"}),
        _mock_response(200, json_data={"data": [{"b64_json": png_b64}]}),
    ]
    client = YesAPIClient(api_key="sk-test", base_url="https://yesapi.online", category="image", session=session)
    imgs = client.generate_images(prompt="cat", model="Nano-banana-Pro", size="1024x1024")
    assert len(imgs) == 1
    assert imgs[0].size == (32, 32)
    assert session.request.call_count == 2
    assert session.request.call_args_list[0].args[1] == "https://yesapi.online/v1/images/generations"
    assert session.request.call_args_list[1].args[1] == "https://yesapi.online/images/generations"


def test_image_edits_multipart_and_mask_without_image_validation():
    session = MagicMock()
    client = YesAPIClient(api_key="sk-test", base_url="https://yesapi.online", category="image", session=session)

    # Mask without reference image must raise fatal validation error before network call
    with pytest.raises(YesAPIError) as exc_info:
        client.generate_images(prompt="inpaint", model="gpt-image-2", ref_images=[], mask_png_bytes=b"fake_mask")
    assert exc_info.value.is_fatal is True
    assert "image_1" in str(exc_info.value)
    assert session.request.call_count == 0

    # Valid edit with 2 reference images + mask -> uses image[]
    png_b64 = base64.b64encode(_make_png_bytes((255, 128, 0))).decode("utf-8")
    session.request.return_value = _mock_response(200, json_data={"data": [{"b64_json": png_b64}]})
    ref1 = Image.new("RGB", (64, 64), (10, 20, 30))
    ref2 = Image.new("RGB", (64, 64), (40, 50, 60))
    imgs = client.generate_images(
        prompt="combine",
        model="Nano-banana-2.1",
        size=None,
        quality="high",
        ref_images=[ref1, ref2],
        mask_png_bytes=_make_png_bytes(),
    )
    assert len(imgs) == 1
    call_kwargs = session.request.call_args.kwargs
    files_sent = call_kwargs["files"]
    field_names = [f[0] for f in files_sent]
    assert field_names.count("image[]") == 2
    assert "mask" in field_names
    # Nano-banana models must strip unsupported 'quality' and omit empty 'size'
    assert "quality" not in call_kwargs["data"]
    assert "size" not in call_kwargs["data"]

    # Valid edit with 1 reference image -> uses singular 'image' per ImageStudioView
    client.generate_images(
        prompt="single edit",
        model="gpt-image-2",
        size="1024x1024",
        quality="high",
        ref_images=[ref1],
    )
    single_kwargs = session.request.call_args.kwargs
    single_fields = [f[0] for f in single_kwargs["files"]]
    assert single_fields == ["image"]
    assert single_kwargs["data"]["quality"] == "high"
    assert single_kwargs["data"]["size"] == "1024x1024"


def test_async_image_submit_poll_and_binary_content_download():
    session = MagicMock()
    raw_png = _make_png_bytes((0, 100, 255), size=(48, 48))
    session.request.side_effect = [
        _mock_response(200, json_data={"id": "img_abc123", "object": "image", "status": "pending"}),
        _mock_response(
            200,
            json_data={
                "id": "img_abc123",
                "status": "completed",
                "content_url": "/v1/images/img_abc123/content",
                "error": None,
            },
        ),
        _mock_response(200, content=raw_png, headers={"Content-Type": "image/png"}),
    ]
    client = YesAPIClient(api_key="sk-img", base_url="https://yesapi.online", category="image", session=session)

    tid = client.submit_async_image(prompt="otter", model="gpt-image-2")
    assert tid == "img_abc123"
    # Verify async: True & n: 1 were sent
    sent_json = session.request.call_args_list[0].kwargs["json"]
    assert sent_json["async"] is True and sent_json["n"] == 1

    poll_info = client.poll_image_task("img_abc123")
    assert poll_info["status"] == "completed"

    dl_img = client.download_image_content("img_abc123", poll_info["content_url"])
    assert dl_img.size == (48, 48)


def test_fatal_quota_and_group_mismatch_errors():
    session = MagicMock()
    session.request.return_value = _mock_response(
        402,
        json_data={"error": {"code": "API_KEY_QUOTA_EXHAUSTED", "message": "API_KEY_QUOTA_EXHAUSTED"}},
    )
    client = YesAPIClient(api_key="sk-exhausted", category="image", session=session)
    with pytest.raises(YesAPIError) as exc_info:
        client.generate_images(prompt="test", model="gpt-image-2")
    assert exc_info.value.is_fatal is True
    assert exc_info.value.status_code == 402


def test_dual_protocol_chat_openai_and_anthropic_with_vision():
    session = MagicMock()
    session.request.side_effect = [
        _mock_response(200, json_data={"choices": [{"message": {"content": "OpenAI reply"}}]}),
        _mock_response(200, json_data={"content": [{"type": "text", "text": "Claude reply"}]}),
    ]
    client = YesAPIClient(api_key="sk-llm", category="llm", session=session)
    img = Image.new("RGB", (32, 32), (100, 100, 100))

    # Auto selects OpenAI for gemini-4-carbon
    res1 = client.chat_completion(model="gemini-4-carbon", prompt="describe", system_prompt="sys", images=[img])
    assert res1 == "OpenAI reply"
    req1 = session.request.call_args_list[0]
    assert req1.kwargs["headers"]["Authorization"] == "Bearer sk-llm"
    assert req1.kwargs["json"]["messages"][1]["content"][1]["type"] == "image_url"

    # Auto selects Anthropic for claude-sonnet-5.5
    res2 = client.chat_completion(model="claude-sonnet-5.5", prompt="describe", system_prompt="sys", images=[img])
    assert res2 == "Claude reply"
    req2 = session.request.call_args_list[1]
    assert req2.kwargs["headers"]["x-api-key"] == "sk-llm"
    assert req2.kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert req2.kwargs["json"]["messages"][0]["content"][0]["type"] == "image"


def test_video_create_poll_and_stream_download(tmp_path):
    session = MagicMock()
    # Create returns nested data.id; Poll returns url: null + completed; Content streams mp4 bytes
    mp4_Resp = MagicMock()
    mp4_Resp.status_code = 200
    mp4_Resp.headers = {"Content-Type": "video/mp4"}
    mp4_Resp.iter_content.return_value = [b"\x00\x00\x00\x18ftypmp42" * 100]

    session.request.side_effect = [
        _mock_response(200, json_data={"data": {"id": "vid_999"}, "status": "queued"}),
        _mock_response(200, json_data={"id": "vid_999", "status": "completed", "url": None, "poll_after_ms": 4000}),
        mp4_Resp,
    ]
    client = YesAPIClient(api_key="sk-vid", category="video", session=session)
    vid_id = client.create_video_task(
        model="Veo-3.1-fast-1080P",
        prompt="ocean waves",
        duration=5,
        aspect_ratio="16:9",
        extra_body={"seed": 123},
    )
    assert vid_id == "vid_999"
    assert session.request.call_args_list[0].kwargs["json"]["seed"] == 123

    pinfo = client.poll_video_task("vid_999")
    assert pinfo["status"] == "completed"

    out_file = tmp_path / "video.mp4"
    saved = client.download_video_content("vid_999", str(out_file))
    assert saved == str(out_file)
    assert out_file.exists() and out_file.stat().st_size > 500
