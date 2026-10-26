from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from services.errors import ConfigError, PluginError
from tests.support import b64_image_response, make_config, make_png


async def test_generation_submits_actual_reference_images(app_factory):
    reference_png = make_png(width=4, height=3, color=(200, 60, 60))
    result_png = make_png(color=(60, 160, 90))
    recorded: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return b64_image_response(result_png)

    app = await app_factory(transport=httpx.MockTransport(handler))
    await app.gallery.save_reference(
        "test:alice",
        name="身份参考",
        note="正脸",
        tags=["正脸", "日常"],
        data=reference_png,
        weights={"face": 90, "hair": 80, "outfit": 60, "scene": 20},
    )

    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice", "shared"],
        description="穿着白衬衫在窗边看书",
    )
    assert len(prepared.reference_plan.references) == 1
    result = await app.coordinator.generate(prepared)

    assert len(recorded) == 1
    request = recorded[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/images/edits"
    assert request.headers["authorization"] == "Bearer sk-primary-test-secret"
    body = await request.aread()
    assert b'name="image[]"' in body
    assert b'filename="ref-1.png"' in body
    assert reference_png in body
    assert b"reference_weight" not in body
    assert b"response_format" not in body
    assert "参考图使用规则" in body.decode("utf-8", "ignore")
    assert Path(result.image_path).is_file()
    task = await app.storage.get_task(result.task_id)
    assert task is not None
    assert task.status == "succeeded"
    assert task.result_path


async def test_generation_without_references_uses_text_only_endpoint(app_factory):
    result_png = make_png(color=(40, 90, 200))
    recorded: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return b64_image_response(result_png)

    app = await app_factory(transport=httpx.MockTransport(handler))
    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice", "shared"],
        description="傍晚在街道散步",
    )
    assert prepared.reference_plan.references == []
    assert "本次没有提交参考图" in prepared.prompt_plan.prompt
    result = await app.coordinator.generate(prepared)

    request = recorded[0]
    assert request.url.path == "/v1/images/generations"
    payload = json.loads((await request.aread()).decode("utf-8"))
    assert payload["model"] == "gpt-image-2"
    assert "傍晚在街道散步" in payload["prompt"]
    assert "image" not in payload
    assert "reference_weight" not in payload
    assert Path(result.image_path).is_file()


async def test_require_reference_config_stops_without_reference(app_factory):
    app = await app_factory(config=make_config(api={"require_reference": True}))
    with pytest.raises(ConfigError, match="参考图"):
        await app.coordinator.prepare(
            actor_id="test:alice",
            state_owner="test:alice",
            scopes=["test:alice"],
            description="普通照片",
        )


async def test_api_timeout_returns_user_friendly_failure(app_factory):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    app = await app_factory(
        config=make_config(api={"max_retries": 0}),
        transport=httpx.MockTransport(handler),
    )
    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice"],
        description="在阳台晒太阳",
    )
    with pytest.raises(PluginError) as excinfo:
        await app.coordinator.generate(prepared)
    assert "超时" in excinfo.value.user_message
    assert "所有生图接口均失败" in excinfo.value.user_message

    tasks = await app.storage.recent_tasks("test:alice", 1)
    assert tasks
    assert tasks[0].status == "failed"
    assert "超时" in tasks[0].error


async def test_retryable_timeout_is_retried_then_succeeds(app_factory):
    attempts = 0
    result_png = make_png(color=(10, 180, 180))

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("first attempt timeout", request=request)
        return b64_image_response(result_png)

    app = await app_factory(
        config=make_config(api={"max_retries": 1}),
        transport=httpx.MockTransport(handler),
    )
    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice"],
        description="在厨房做早餐",
    )
    result = await app.coordinator.generate(prepared)
    assert attempts == 2
    assert Path(result.image_path).is_file()


async def test_primary_api_failure_falls_back_to_enabled_backup(app_factory):
    result_png = make_png(color=(180, 120, 20))
    hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "primary.test":
            return httpx.Response(
                500,
                json={"error": {"message": "上游服务异常"}},
                headers={"content-type": "application/json"},
            )
        return b64_image_response(result_png)

    app = await app_factory(transport=httpx.MockTransport(handler))
    await app.api.add_backup(
        name="备用接口",
        base_url="https://backup.test/v1",
        model="gpt-image-2",
        api_key="sk-backup-test-secret",
        priority=10,
    )
    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice"],
        description="周末在公园长椅上看书",
    )
    result = await app.coordinator.generate(prepared)

    assert hosts == ["primary.test", "backup.test"]
    assert result.endpoint_name == "备用接口"
    assert Path(result.image_path).is_file()
    endpoints = await app.api.list_endpoint_info()
    backup = next(item for item in endpoints if item.name == "备用接口")
    assert backup.last_ok_at is not None
    assert backup.last_error == ""


async def test_auth_failure_does_not_leak_api_key(app_factory):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": "invalid key sk-primary-test-secret",
                }
            },
            headers={"content-type": "application/json"},
        )

    app = await app_factory(transport=httpx.MockTransport(handler))
    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice"],
        description="在客厅拍一张照片",
    )
    with pytest.raises(PluginError) as excinfo:
        await app.coordinator.generate(prepared)
    message = excinfo.value.user_message
    assert "鉴权失败" in message
    assert "sk-primary-test-secret" not in message


async def test_endpoint_list_masks_secret(app_factory):
    app = await app_factory()
    await app.api.add_backup(
        name="备用接口",
        base_url="https://backup.test/v1",
        model="gpt-image-2",
        api_key="sk-backup-sensitive-value",
        priority=5,
    )
    infos = await app.api.list_endpoint_info()
    serialized = "\n".join(
        f"{item.name}|{item.masked_key}|{item.base_url}|{item.last_error}" for item in infos
    )
    assert "sk-backup-sensitive-value" not in serialized
    assert "sk-primary-test-secret" not in serialized
    assert "****" in serialized
