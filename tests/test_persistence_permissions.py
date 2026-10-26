from __future__ import annotations

import pytest

from services.errors import NotFoundError, PermissionDeniedError
from tests.support import (
    AppHarness,
    Image,
    make_config,
    make_png,
)


async def test_gallery_wardrobe_outfit_api_and_timeline_survive_restart(tmp_path):
    data_dir = tmp_path / "persist"
    first = await AppHarness.create(data_dir, config=make_config())
    await first.gallery.save_reference(
        "test:alice",
        name="重启参考",
        note="用于验证持久化",
        tags=["持久化"],
        data=make_png(color=(80, 140, 200)),
    )
    shirt = await first.wardrobe.add_item(
        "test:alice",
        name="持久化衬衫",
        category_query="上装",
        color="蓝色",
    )
    trousers = await first.wardrobe.add_item(
        "test:alice",
        name="持久化长裤",
        category_query="下装",
        color="黑色",
    )
    await first.wardrobe.create_outfit(
        "test:alice",
        name="重启套装",
        queries=["持久化衬衫", "持久化长裤"],
    )
    await first.wardrobe.wear("test:alice", ["test:alice"], ["持久化衬衫", "持久化长裤"])
    backup = await first.api.add_backup(
        name="持久化备用",
        base_url="https://backup.test/v1",
        model="gpt-image-2",
        api_key="sk-persist-secret-value",
        priority=7,
    )
    await first.timeline.set_manual(
        "test:alice",
        activity="在家整理房间",
        location="卧室",
        season="秋",
    )
    await first.close()

    second = await AppHarness.create(data_dir, config=make_config())
    try:
        references = await second.gallery.list_references(["test:alice"])
        assert [item.name for item in references] == ["重启参考"]
        assert second.gallery.resolve_file(references[0]).is_file()
        assert references[0].tags == ["持久化"]

        items = await second.wardrobe.list_items(["test:alice"])
        assert {"持久化衬衫", "持久化长裤"} <= {item.name for item in items}
        outfits = await second.wardrobe.list_outfits(["test:alice"])
        assert [item.name for item in outfits] == ["重启套装"]
        assert outfits[0].item_ids == [shirt.id, trousers.id]

        _state, worn, _notes = await second.wardrobe.get_state("test:alice")
        worn_names = {item.name for item in worn}
        assert {"持久化衬衫", "持久化长裤"} <= worn_names

        persisted_backup = await second.storage.find_api_endpoint_by_name("持久化备用")
        assert persisted_backup is not None
        assert persisted_backup.id == backup.id
        assert persisted_backup.api_key == "sk-persist-secret-value"
        assert persisted_backup.priority == 7

        manual = await second.timeline.get_manual("test:alice")
        assert manual is not None
        assert manual["activity"] == "在家整理房间"
        assert manual["location"] == "卧室"
    finally:
        await second.close()


async def test_permissions_and_shared_scope(plugin_factory, event_factory):
    plugin = await plugin_factory(
        config=make_config(
            generation={"allow_user_generation": False},
            permissions={"admin_ids": ["admin"]},
        )
    )
    user = event_factory(sender="alice", origin="test:private:alice")
    other = event_factory(sender="bob", origin="test:private:bob")
    admin = event_factory(sender="admin", origin="test:private:admin", admin=True)

    await plugin.gallery.save_reference(
        "test:alice",
        name="Alice 的参考",
        note="",
        tags=[],
        data=make_png(color=(200, 20, 20)),
    )
    shared = await plugin.gallery.save_reference(
        "shared",
        name="共享参考",
        note="",
        tags=[],
        data=make_png(color=(20, 200, 20)),
    )

    user_list = await plugin._handle_gallery(user, "列表")
    assert "Alice 的参考" in user_list
    assert "共享参考" in user_list
    other_list = await plugin._handle_gallery(other, "列表")
    assert "Alice 的参考" not in other_list
    assert "共享参考" in other_list

    shared_read = await plugin._handle_gallery(user, "列表 --共享")
    assert "共享参考" in shared_read
    assert "Alice 的参考" not in shared_read

    with pytest.raises(PermissionDeniedError, match="只有管理员"):
        await plugin._handle_gallery(user, "主图 共享参考 --共享")
    with pytest.raises(PermissionDeniedError, match="只有管理员"):
        await plugin._handle_gallery(
            user,
            "删除 共享参考 --共享 --确认",
        )

    admin_delete = await plugin._handle_gallery(
        admin,
        "删除 共享参考 --共享 --确认",
    )
    assert "已删除" in admin_delete
    with pytest.raises(NotFoundError):
        await plugin.gallery.get_owned(shared.id, "shared")

    result = await plugin._generate_reply(
        user,
        "在窗边看书",
        selfie=False,
        preview_only=True,
    )
    assert "管理员已关闭普通用户的生图权限" in result.text
    admin_preview = await plugin._generate_reply(
        admin,
        "在窗边看书",
        selfie=False,
        preview_only=True,
    )
    assert "生图预览" in admin_preview.text

    with pytest.raises(PermissionDeniedError, match="仅限管理员"):
        await plugin._handle_api(user, "列表")
    api_list = await plugin._handle_api(admin, "列表")
    assert "sk-primary-test-secret" not in api_list
    assert "****" in api_list


async def test_upload_sessions_and_saved_data_are_isolated_per_user(
    plugin_factory,
    event_factory,
    tmp_path,
):
    plugin = await plugin_factory()
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(make_png(color=(210, 20, 20)))
    image_b.write_bytes(make_png(color=(20, 20, 210)))

    reply_a_image = event_factory(
        sender="alice",
        origin="test:private:alice",
        messages=[Image(str(image_a))],
    )
    reply_a_done = event_factory(
        sender="alice",
        origin="test:private:alice",
        message_str="完成",
    )
    reply_b_image = event_factory(
        sender="bob",
        origin="test:private:bob",
        messages=[Image(str(image_b))],
    )
    reply_b_done = event_factory(
        sender="bob",
        origin="test:private:bob",
        message_str="完成",
    )
    request_a = event_factory(
        sender="alice",
        origin="test:private:alice",
        upload_replies=[reply_a_image, reply_a_done],
    )
    request_b = event_factory(
        sender="bob",
        origin="test:private:bob",
        upload_replies=[reply_b_image, reply_b_done],
    )

    summary_a = await plugin._run_upload_session(request_a, "test:alice", "A图|A备注|A标签")
    summary_b = await plugin._run_upload_session(request_b, "test:bob", "B图|B备注|B标签")
    assert "保存 1 张" in summary_a
    assert "保存 1 张" in summary_b

    filter_a = request_a.last_session_filter
    filter_b = request_b.last_session_filter
    assert filter_a is not None and filter_b is not None
    assert filter_a.filter(reply_a_image) != ""
    assert filter_a.filter(reply_b_image) == ""
    assert filter_b.filter(reply_b_image) != ""
    assert filter_b.filter(reply_a_image) == ""
    assert filter_a.filter(reply_a_image) != filter_b.filter(reply_b_image)

    refs_a = await plugin.gallery.find_references(["test:alice"], "A图")
    refs_b = await plugin.gallery.find_references(["test:bob"], "B图")
    assert [item.name for item in refs_a] == ["A图"]
    assert [item.name for item in refs_b] == ["B图"]
    assert refs_a[0].tags == ["A标签"]
    assert refs_b[0].tags == ["B标签"]
    assert [item.name for item in await plugin.gallery.list_references(["test:alice"])] == [
        "A图"
    ]
    assert [item.name for item in await plugin.gallery.list_references(["test:bob"])] == [
        "B图"
    ]


async def test_wardrobe_state_is_isolated_between_users(plugin_factory, event_factory):
    plugin = await plugin_factory()
    await plugin.wardrobe.add_item(
        "test:alice",
        name="Alice 的夹克",
        category_query="外套",
    )
    await plugin.wardrobe.wear("test:alice", ["test:alice"], ["Alice 的夹克"])

    alice_state, alice_items, _notes = await plugin.wardrobe.get_state("test:alice")
    bob_state, bob_items, _notes = await plugin.wardrobe.get_state("test:bob")
    assert "Alice 的夹克" in {item.name for item in alice_items}
    assert "Alice 的夹克" not in {item.name for item in bob_items}
    assert alice_state.owner_id == "test:alice"
    assert bob_state.owner_id == "test:bob"
    assert {item.name for item in bob_items} == {"基础白T恤", "基础黑短裤"}
