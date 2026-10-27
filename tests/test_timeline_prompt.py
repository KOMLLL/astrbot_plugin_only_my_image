from __future__ import annotations

import json

import httpx
import pytest

from services.errors import SafetyError
from services.models import ReferencePlan, TimelineContext, WardrobePlan
from services.reference_selector import SelectorIntent
from tests.support import b64_image_response, make_config, make_png


def _recording_transport(recorded: list[httpx.Request], png: bytes):
    async def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return b64_image_response(png)

    return httpx.MockTransport(handler)


async def test_timeline_adapter_context_reaches_prompt_and_request(app_factory):
    recorded: list[httpx.Request] = []
    png = make_png(color=(120, 160, 90))

    async def resolver(plugin_name: str, function_name: str):
        assert plugin_name == "timeline_plugin"
        assert function_name == "get_context"
        return {
            "activity": "在咖啡店写稿",
            "location": "街角咖啡店",
            "weather": "小雨",
            "temperature": 18,
            "indoor": True,
            "season": "秋",
        }

    app = await app_factory(
        config=make_config(
            timeline={
                "mode": "auto",
                "plugin_name": "timeline_plugin",
                "function_name": "get_context",
            }
        ),
        transport=_recording_transport(recorded, png),
        resolver=resolver,
    )

    context = await app.timeline.get_context("test:alice")
    assert context.source == "plugin"
    assert context.activity == "在咖啡店写稿"
    assert context.location == "街角咖啡店"
    assert context.weather == "小雨"
    assert context.temperature == 18
    assert context.indoor is True
    assert context.season == "秋"

    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice", "shared"],
        description="记录今天在咖啡店的工作状态",
    )
    assert "当前活动：在咖啡店写稿" in prepared.prompt_plan.prompt
    assert "地点：街角咖啡店" in prepared.prompt_plan.prompt
    assert "天气：小雨" in prepared.prompt_plan.prompt
    assert "气温：约 18℃" in prepared.prompt_plan.prompt
    assert prepared.timeline.source == "plugin"

    result = await app.coordinator.generate(prepared)
    assert len(recorded) == 1
    payload = json.loads((await recorded[0].aread()).decode("utf-8"))
    assert "在咖啡店写稿" in payload["prompt"]
    assert "街角咖啡店" in payload["prompt"]
    assert "小雨" in payload["prompt"]
    assert result.image_path
    from pathlib import Path

    assert Path(result.image_path).is_file()


async def test_timeline_unavailable_falls_back_and_still_generates(app_factory):
    recorded: list[httpx.Request] = []
    png = make_png(color=(90, 120, 180))

    def broken_resolver(_plugin_name: str, _function_name: str):
        raise RuntimeError("模拟时间轴插件未加载")

    app = await app_factory(
        config=make_config(
            timeline={
                "mode": "auto",
                "plugin_name": "missing_timeline",
                "function_name": "get_context",
            }
        ),
        transport=_recording_transport(recorded, png),
        resolver=broken_resolver,
    )

    context = await app.timeline.get_context("test:alice")
    assert context.source == "time"
    assert "回退" in context.note
    assert context.activity == "日常状态"

    prepared = await app.coordinator.prepare(
        actor_id="test:alice",
        state_owner="test:alice",
        scopes=["test:alice", "shared"],
        description="在窗边安静地看书",
    )
    assert "当前时段：" in prepared.prompt_plan.prompt
    assert "不要虚构具体地点、天气或行程" in prepared.prompt_plan.prompt
    result = await app.coordinator.generate(prepared)
    assert len(recorded) == 1
    assert result.image_path
    assert await app.storage.recent_tasks("test:alice", 1)


async def test_manual_timeline_overrides_fallback(app_factory):
    app = await app_factory(
        config=make_config(timeline={"mode": "off"}),
    )
    await app.timeline.set_manual(
        "test:alice",
        activity="在家整理书架",
        location="书房",
        weather="晴",
        temperature="22",
        indoor="是",
        season="秋",
    )
    context = await app.timeline.get_context("test:alice")
    assert context.source == "manual"
    assert context.activity == "在家整理书架"
    assert context.location == "书房"
    assert context.temperature == 22
    assert context.indoor is True

    await app.timeline.clear_manual("test:alice")
    fallback = await app.timeline.get_context("test:alice")
    assert fallback.source == "time"


async def test_reference_weights_appear_in_prompt_roles(app_factory):
    app = await app_factory()
    face = await app.gallery.save_reference(
        "test:alice",
        name="正脸参考",
        note="五官清晰",
        tags=["正脸"],
        data=make_png(color=(200, 100, 100)),
        weights={"face": 95, "hair": 80, "outfit": 10, "scene": 0},
    )
    outfit = await app.gallery.save_reference(
        "test:alice",
        name="服装参考",
        note="白色连衣裙",
        tags=["连衣裙"],
        data=make_png(color=(240, 240, 240)),
        weights={"face": 10, "hair": 10, "outfit": 95, "scene": 20},
    )
    plan = app.selector.select(
        [face, outfit],
        intent=SelectorIntent(description="穿白色连衣裙在公园散步", outfit="白色连衣裙"),
        primary_id=face.id,
    )
    prompt = app.prompts.build(
        description="穿白色连衣裙在公园散步",
        selfie=False,
        timeline=TimelineContext(activity="散步", season="春", source="manual"),
        wardrobe=WardrobePlan(items=[], source="auto"),
        outfit_description="白色连衣裙",
        reference_plan=plan,
        outfit_hint="白色连衣裙",
    )
    assert "面部特征、五官比例与妆容以图1为主要依据" in prompt.prompt
    assert "面部权重95/100" in prompt.prompt
    assert "服装的款式、颜色与材质参考图2" in prompt.prompt
    assert "参考图2（权重95/100" in prompt.prompt
    assert "图1（正脸参考）" in prompt.prompt
    assert "图2（服装参考）" in prompt.prompt
    assert prompt.reference_roles[face.id]
    assert prompt.reference_roles[outfit.id]

    assert "裸露" in prompt.negative_prompt
    assert "低清晰度" in prompt.negative_prompt


async def test_safety_guard_rejects_explicit_nudity_before_generation(app_factory):
    app = await app_factory()
    with pytest.raises(SafetyError):
        app.prompts.assert_safe("帮我拍一张裸照")
    with pytest.raises(SafetyError):
        app.prompts.assert_safe("不穿上衣在客厅自拍")
    with pytest.raises(SafetyError):
        await app.coordinator.prepare(
            actor_id="test:alice",
            state_owner="test:alice",
            scopes=["test:alice"],
            description="脱掉上衣拍一张",
        )


async def test_prompt_without_references_and_style_negative_filter(app_factory):
    app = await app_factory(
        config=make_config(generation={"default_negative": "二次元,模糊,低清晰度"})
    )
    prompt = app.prompts.build(
        description="二次元插画风格，在窗边看书",
        selfie=False,
        timeline=TimelineContext(activity="看书", season="春", source="manual"),
        wardrobe=WardrobePlan(items=[], source="auto"),
        outfit_description="基础白T恤 + 黑色短裤",
        reference_plan=ReferencePlan(),
    )
    assert "本次没有提交参考图" in prompt.prompt
    assert "用户已指定非摄影风格" in prompt.prompt
    assert "二次元" not in prompt.negative_prompt
    assert "模糊" in prompt.negative_prompt
