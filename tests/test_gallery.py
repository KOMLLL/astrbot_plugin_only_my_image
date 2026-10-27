from __future__ import annotations

import pytest

from services.errors import ValidationError
from services.reference_selector import ReferenceSelector, SelectorIntent
from tests.support import make_png


async def test_add_multiple_reference_images(app_factory):
    app = await app_factory()
    first = await app.gallery.save_reference(
        "test:alice",
        name="主参考",
        note="正脸清晰，适合确定五官",
        tags=["日常", "人像"],
        data=make_png(color=(190, 40, 40)),
    )
    second = await app.gallery.save_reference(
        "test:alice",
        name="海边参考",
        note="海边场景与夏日穿搭",
        tags=["海边", "夏日"],
        data=make_png(width=3, color=(20, 80, 200)),
        weights={"face": 25, "outfit": 90, "scene": 95},
    )

    references = await app.gallery.list_references(["test:alice"])
    assert [item.name for item in references] == ["主参考", "海边参考"]
    assert first.is_primary is True
    assert second.is_primary is False
    assert await app.gallery.count_owned("test:alice") == 2
    for reference in references:
        path = app.gallery.resolve_file(reference)
        assert path.is_file()
        assert path.read_bytes()

    with pytest.raises(ValidationError, match="已存在"):
        await app.gallery.save_reference(
            "test:alice",
            name="重复图",
            note="",
            tags=[],
            data=make_png(color=(190, 40, 40)),
        )


async def test_weights_on_different_references_drive_dimension_choices(app_factory):
    app = await app_factory()
    face_ref = await app.gallery.save_reference(
        "test:alice",
        name="正脸特写",
        note="五官与妆容参考",
        tags=["正脸"],
        data=make_png(color=(210, 60, 60)),
    )
    style_ref = await app.gallery.save_reference(
        "test:alice",
        name="海边穿搭",
        note="海边连衣裙与场景",
        tags=["海边", "连衣裙"],
        data=make_png(color=(30, 120, 220)),
    )

    await app.gallery.update_metadata(
        face_ref.id,
        "test:alice",
        weights={"face": "高", "hair": 70, "outfit": "低", "scene": "关"},
    )
    await app.gallery.update_metadata(
        style_ref.id,
        "test:alice",
        weights={"face": 10, "hair": 15, "outfit": 95, "scene": 90},
    )

    refreshed_face = await app.gallery.get_owned(face_ref.id, "test:alice")
    refreshed_style = await app.gallery.get_owned(style_ref.id, "test:alice")
    assert refreshed_face.weights["face"] == 85
    assert refreshed_face.weights["outfit"] == 35
    assert refreshed_face.weights["scene"] == 0
    assert refreshed_style.weights["outfit"] == 95
    assert refreshed_style.weights["scene"] == 90

    plan = app.selector.select(
        [refreshed_face, refreshed_style],
        intent=SelectorIntent(
            description="在海边穿着连衣裙散步",
            scene="海边",
            outfit="连衣裙",
        ),
        primary_id=refreshed_face.id,
    )
    assert plan.dimensions["face"].reference_id == refreshed_face.id
    assert plan.dimensions["face"].weight == 85
    assert plan.dimensions["outfit"].reference_id == refreshed_style.id
    assert plan.dimensions["outfit"].weight == 95
    assert plan.dimensions["scene"].reference_id == refreshed_style.id
    assert plan.dimensions["scene"].weight == 90
    assert {ref.id for ref in plan.references} == {refreshed_face.id, refreshed_style.id}

    zero_weight = ReferenceSelector(app.settings).select(
        [refreshed_face],
        intent=SelectorIntent(description="海边"),
    )
    assert "scene" not in zero_weight.dimensions


async def test_add_search_and_delete_tags(app_factory):
    app = await app_factory()
    first = await app.gallery.save_reference(
        "test:alice",
        name="第一张",
        note="",
        tags=[],
        data=make_png(color=(200, 10, 10)),
    )
    second = await app.gallery.save_reference(
        "test:alice",
        name="第二张",
        note="",
        tags=[],
        data=make_png(color=(10, 10, 200)),
    )
    await app.gallery.update_metadata(first.id, "test:alice", tags=["日常", "夏日"])
    await app.gallery.update_metadata(second.id, "test:alice", tags=["海边", "夏日"])

    stats = dict(await app.gallery.all_tags(["test:alice"]))
    assert stats["夏日"] == 2
    assert stats["日常"] == 1

    found = await app.gallery.find_references(["test:alice"], "夏日")
    assert {item.name for item in found} == {"第一张", "第二张"}
    assert [item.name for item in await app.gallery.find_references(["test:alice"], "海边")] == [
        "第二张"
    ]

    changed = await app.gallery.delete_tag("test:alice", "夏日")
    assert changed == 2
    assert "夏日" not in dict(await app.gallery.all_tags(["test:alice"]))

    renamed = await app.gallery.rename_tag("test:alice", "日常", "日常照")
    assert renamed == 1
    assert [item.name for item in await app.gallery.find_references(["test:alice"], "日常照")] == [
        "第一张"
    ]

    await app.gallery.update_metadata(first.id, "test:alice", tags=["日常照", "夏日限定"])
    await app.gallery.update_metadata(second.id, "test:alice", tags=["海边", "夏日限定"])
    assert await app.gallery.delete_tag("test:alice", "夏日限定") == 2
