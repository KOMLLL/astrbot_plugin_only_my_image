from __future__ import annotations

import httpx
import pytest

from services.errors import PluginError, ValidationError
from services.models import ReferenceImage
from services.utils import resolve_relative_path
from tests.support import b64_image_response, make_config, make_png


async def test_weighted_selection_controls_actual_submitted_images(app_factory):
    face_png = make_png(color=(210, 40, 40))
    outfit_png = make_png(color=(40, 80, 220))
    scene_png = make_png(color=(40, 190, 90))
    recorded: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return b64_image_response(make_png(color=(120, 120, 120)))

    app = await app_factory(
        config=make_config(generation={"max_reference_images": 2}),
        transport=httpx.MockTransport(handler),
    )
    await app.gallery.save_reference(
        "test:alice",
        name="面部参考",
        note="正脸特写",
        tags=["正脸"],
        data=face_png,
        weights={
            "face": 100,
            "hair": 0,
            "outfit": 0,
            "scene": 0,
            "pose": 0,
            "composition": 0,
        },
    )
    await app.gallery.save_reference(
        "test:alice",
        name="服装参考",
        note="白色连衣裙",
        tags=["连衣裙"],
        data=outfit_png,
        weights={
            "face": 0,
            "hair": 0,
            "outfit": 100,
            "scene": 0,
            "pose": 0,
            "composition": 0,
        },
    )
    await app.gallery.save_reference(
        "test:alice",
        name="场景参考",
        note="海边栈道",
        tags=["海边"],
        data=scene_png,
        weights={
            "face": 0,
            "hair": 0,
            "outfit": 0,
            "scene": 100,
            "pose": 0,
            "composition": 0,
        },
    )

    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice", "shared"],
        description="在海边穿白色连衣裙散步",
        outfit_hint="白色连衣裙",
        scene="海边",
    )
    selected_names = [ref.name for ref in prepared.reference_plan.references]
    assert selected_names == ["面部参考", "服装参考"]
    assert prepared.reference_plan.dimensions["face"].reference_name == "面部参考"
    assert prepared.reference_plan.dimensions["outfit"].reference_name == "服装参考"
    assert "scene" not in prepared.reference_plan.dimensions
    assert any("场景" in warning and "未能纳入" in warning for warning in prepared.reference_plan.warnings)

    await app.coordinator.generate(prepared)
    assert len(recorded) == 1
    body = await recorded[0].aread()
    assert b'filename="ref-1.png"' in body
    assert b'filename="ref-2.png"' in body
    assert face_png in body
    assert outfit_png in body
    assert scene_png not in body


async def test_duplicate_generation_reuses_result_without_second_api_call(app_factory):
    calls = 0
    result_png = make_png(color=(30, 150, 170))

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return b64_image_response(result_png)

    app = await app_factory(
        config=make_config(generation={"dedupe_window": 60}),
        transport=httpx.MockTransport(handler),
    )
    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice", "shared"],
        description="在窗边喝茶看书",
    )

    first = await app.coordinator.generate(prepared)
    second = await app.coordinator.generate(prepared)

    assert calls == 1
    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.task_id == first.task_id
    assert second.image_path == first.image_path
    tasks = await app.storage.recent_tasks("test:alice", 10)
    assert len(tasks) == 1
    assert tasks[0].status == "succeeded"


async def test_reference_path_traversal_is_rejected(app_factory):
    app = await app_factory()
    for path in ("../outside.png", "../../outside.png", "/etc/passwd", "C:/Windows/win.ini"):
        with pytest.raises(ValidationError):
            resolve_relative_path(app.data_dir, path)

    malicious = ReferenceImage(
        id="ref-traversal",
        owner_id="test:alice",
        name="越界参考",
        file_path="../../outside.png",
        enabled=True,
        created_at=1.0,
        updated_at=1.0,
    )
    with pytest.raises(ValidationError, match="越界"):
        app.gallery.resolve_file(malicious)
    await app.storage.add_reference(malicious)
    references, warnings = await app.gallery.list_available(["test:alice"])
    assert references == []
    assert any("路径异常" in warning for warning in warnings)


async def test_private_download_url_is_rejected(app_factory):
    recorded: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return httpx.Response(
            200,
            json={"data": [{"url": "http://127.0.0.1:8080/internal.png"}]},
            headers={"content-type": "application/json"},
        )

    app = await app_factory(transport=httpx.MockTransport(handler))
    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice"],
        description="普通室内照片",
    )
    with pytest.raises(PluginError) as excinfo:
        await app.coordinator.generate(prepared)
    assert "内网" in excinfo.value.user_message or "本机" in excinfo.value.user_message
    assert len(recorded) == 1
    assert recorded[0].url.path == "/v1/images/generations"


async def test_redirect_to_private_host_is_rejected(app_factory):
    recorded: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"data": [{"url": "https://cdn.test/result.png"}]},
                headers={"content-type": "application/json"},
            )
        if request.url.host == "cdn.test":
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1:8080/internal.png"},
            )
        return b64_image_response(make_png(color=(1, 2, 3)))

    app = await app_factory(transport=httpx.MockTransport(handler))
    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice"],
        description="普通室内照片",
    )
    with pytest.raises(PluginError) as excinfo:
        await app.coordinator.generate(prepared)
    assert "内网" in excinfo.value.user_message or "本机" in excinfo.value.user_message
    assert [request.url.host for request in recorded] == ["primary.test", "cdn.test"]
