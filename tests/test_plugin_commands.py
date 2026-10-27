from __future__ import annotations

import inspect

import httpx
from astrbot.core.star.filter.command import GreedyStr

from tests.support import (
    ProviderRequest,
    b64_image_response,
    collect_results,
    make_config,
    make_png,
)


def _generation_transport(recorded: list[httpx.Request]) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        return b64_image_response(make_png(color=(40, 120, 200)))

    return httpx.MockTransport(handler)


async def test_llm_tool_registration_and_explicit_only_hint(plugin_factory, event_factory):
    plugin = await plugin_factory()
    assert plugin._ready is True
    assert plugin._tool_registered is True
    tool = plugin.generate_character_image
    assert tool._astrbot_llm_tool_name == "generate_character_image"
    docstring = tool.__doc__ or ""
    assert "明确要求" in docstring
    assert "在干嘛" in docstring
    assert "不要调用" in docstring

    event = event_factory(sender="alice", origin="test:private:alice")
    request = ProviderRequest("你是一个聊天助手。")
    await plugin.add_generation_hint(event, request)
    assert "[astrbot_plugin_only_my_image]" in request.system_prompt
    marker_count = request.system_prompt.count("[astrbot_plugin_only_my_image]")
    assert marker_count == 1
    await plugin.add_generation_hint(event, request)
    assert request.system_prompt.count("[astrbot_plugin_only_my_image]") == 1

    disabled = await plugin_factory(
        config=make_config(features={"on_llm_request_hint": False})
    )
    disabled_request = ProviderRequest("只做普通聊天。")
    await disabled.add_generation_hint(event, disabled_request)
    assert disabled_request.system_prompt == "只做普通聊天。"


async def test_explicit_generation_command_works_without_registered_tool(
    plugin_factory,
    event_factory,
):
    recorded: list[httpx.Request] = []
    plugin = await plugin_factory(
        tool_registered=False,
        transport=_generation_transport(recorded),
    )
    assert plugin._tool_registered is False
    event = event_factory(sender="alice", origin="test:private:alice")

    results = await collect_results(plugin.generate_command(event, "傍晚在阳台吹风"))
    assert len(results) == 1
    assert "[image:" in results[0].text
    assert "正在生成图片" in event.sent_text()
    assert len(recorded) == 1
    assert recorded[0].url.path == "/v1/images/generations"


async def test_llm_tool_handler_generates_and_stops_event(plugin_factory, event_factory):
    recorded: list[httpx.Request] = []
    plugin = await plugin_factory(transport=_generation_transport(recorded))
    event = event_factory(sender="alice", origin="test:private:alice")

    results = await collect_results(
        plugin.generate_character_image(
            event,
            description="在窗边低头画画",
            selfie=True,
        )
    )
    assert len(results) == 1
    assert "[image:" in results[0].text
    assert event.stopped is True
    assert len(recorded) == 1
    body = (await recorded[0].aread()).decode("utf-8")
    assert "自拍" in body


async def test_management_commands_work_without_buttons(plugin_factory, event_factory):
    plugin = await plugin_factory()
    user = event_factory(sender="alice", origin="test:private:alice")
    await plugin.gallery.save_reference(
        "test:alice",
        name="主参考",
        note="正脸",
        tags=[],
        data=make_png(color=(180, 40, 40)),
    )

    weighted = await plugin._handle_gallery(user, "权重 主参考 面部=高 服装=低")
    assert "已更新《主参考》的权重" in weighted
    assert "面部85" in weighted
    assert "服装35" in weighted

    tagged = await plugin._handle_gallery(user, "标签 添加 主参考 日常 人像")
    assert "已更新《主参考》的标签" in tagged
    assert "日常" in tagged

    found = await plugin._handle_gallery(user, "搜索 日常")
    assert "主参考" in found

    removed_tag = await plugin._handle_gallery(user, "标签 删除 日常")
    assert "1 张参考图" in removed_tag

    shirt = await plugin._handle_wardrobe(
        user,
        "添加 白色衬衫 上装 颜色=白色 材质=棉质 季节=春,秋 场景=工作,通勤 "
        "温度=12-26 正式度=4 优先级=70",
    )
    assert "已添加衣物《白色衬衫》" in shirt
    trousers = await plugin._handle_wardrobe(
        user,
        "添加 黑色西裤 下装 颜色=黑色 季节=春,秋,冬 温度=-5-22 正式度=4 优先级=70",
    )
    assert "已添加衣物《黑色西裤》" in trousers

    outfit = await plugin._handle_wardrobe(
        user,
        "套装 添加 通勤装 白色衬衫 黑色西裤",
    )
    assert "已创建套装《通勤装》" in outfit
    worn = await plugin._handle_wardrobe(user, "套装 穿上 通勤装")
    assert "已穿上套装《通勤装》" in worn
    assert "白色衬衫" in worn
    assert "黑色西裤" in worn

    changed = await plugin._handle_wardrobe(user, "脱下 上装")
    assert "基础白T恤" in changed
    assert "黑色西裤" in changed

    listed = await plugin._handle_wardrobe(user, "列表")
    assert "白色衬衫" in listed
    assert "黑色西裤" in listed

    admin = event_factory(sender="admin", origin="test:private:admin", admin=True)
    added = await plugin._handle_api(
        admin,
        "添加 备用接口 地址=https://backup.test/v1 模型=gpt-image-2 "
        "密钥=sk-backup-command-secret 优先级=20",
    )
    assert "已添加备用接口" in added
    assert "sk-backup-command-secret" not in added

    listed_api = await plugin._handle_api(admin, "列表")
    assert "备用接口" in listed_api
    assert "sk-backup-command-secret" not in listed_api
    assert "sk-primary-test-secret" not in listed_api
    assert "****" in listed_api

    deleted = await plugin._handle_api(admin, "删除 备用接口 --确认")
    assert "已删除备用接口《备用接口》" in deleted

    timeline = await plugin._handle_profile(
        user,
        "活动 设置 在家看书 地点=客厅 天气=晴 温度=22 季节=秋 室内=是",
    )
    assert "已更新手动时间轴" in timeline
    current = await plugin._handle_profile(user, "活动")
    assert "在家看书" in current
    assert "客厅" in current

    status = await plugin._handle_profile(admin, "状态")
    assert "插件状态：正常运行" in status


def _astrbot_parse_params(handler, message: str) -> dict[str, object]:
    """复刻 AstrBot CommandFilter 的参数转换，防止默认值把 GreedyStr 降级成普通字符串。"""

    signature = inspect.signature(handler, eval_str=True)
    specs: dict[str, object] = {}
    index = 0
    for name, param in signature.parameters.items():
        if index < 2:
            index += 1
            continue
        specs[name] = (
            param.annotation if param.default is inspect.Parameter.empty else param.default
        )
    tokens = [token for token in message.split(" ") if token]
    converted: dict[str, object] = {}
    items = list(specs.items())
    for position, (name, spec) in enumerate(items):
        if spec is GreedyStr:
            converted[name] = " ".join(tokens[position:])
            break
        if position >= len(tokens):
            converted[name] = spec
        else:
            converted[name] = tokens[position]
    return converted


async def test_greedy_command_parameters_receive_full_text(plugin_factory, event_factory):
    plugin = await plugin_factory()
    for name, handler in inspect.getmembers(type(plugin), inspect.isfunction):
        if not hasattr(handler, "_astrbot_command_name"):
            continue
        signature = inspect.signature(handler, eval_str=True)
        for param_name, param in list(signature.parameters.items())[2:]:
            if param.annotation is GreedyStr:
                assert param.default is inspect.Parameter.empty, (
                    f"{name}.{param_name} 带默认值会被 AstrBot 当成普通字符串，"
                    "只接收第一个词"
                )

    message = "添加 白色衬衫 上装 颜色=白色 季节=春,秋 场景=工作,约会 优先级=80"
    parsed = _astrbot_parse_params(type(plugin).wardrobe_command, message)
    assert parsed["text"] == message

    user = event_factory(sender="alice", origin="test:private:alice")
    result = await plugin._handle_wardrobe(user, str(parsed["text"]))
    assert "已添加衣物《白色衬衫》" in result


async def test_missing_timeline_adapter_does_not_block_plugin_or_generation(
    plugin_factory,
    event_factory,
):
    recorded: list[httpx.Request] = []
    plugin = await plugin_factory(
        config=make_config(
            timeline={
                "mode": "auto",
                "plugin_name": "missing_timeline_plugin",
                "function_name": "get_context",
            }
        ),
        transport=_generation_transport(recorded),
    )
    assert plugin._ready is True
    event = event_factory(sender="alice", origin="test:private:alice")

    preview = await plugin._generate_reply(
        event,
        "傍晚散步",
        selfie=False,
        preview_only=True,
    )
    assert "生图预览" in preview.text
    assert "时间轴来源：time" in preview.text
    assert recorded == []

    results = await collect_results(plugin.generate_command(event, "傍晚散步"))
    assert len(results) == 1
    assert "[image:" in results[0].text
    assert len(recorded) == 1
