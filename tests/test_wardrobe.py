from __future__ import annotations


async def _add_basic_wardrobe(app) -> dict[str, object]:
    shirt = await app.wardrobe.add_item(
        "test:alice",
        name="白色衬衫",
        category_query="上装",
        color="白色",
        material="棉质",
        seasons="春,秋",
        scenes="工作,通勤",
        temperature="12-26",
        formality=4,
        priority=70,
    )
    trousers = await app.wardrobe.add_item(
        "test:alice",
        name="黑色西裤",
        category_query="下装",
        color="黑色",
        material="羊毛混纺",
        seasons="春,秋,冬",
        scenes="工作",
        temperature="-5-22",
        formality=4,
        priority=70,
    )
    jacket = await app.wardrobe.add_item(
        "test:alice",
        name="蓝色牛仔外套",
        category_query="外套",
        color="蓝色",
        material="牛仔",
        seasons="春,秋",
        scenes="通勤,约会",
        temperature="5-22",
        formality=3,
        priority=80,
    )
    return {"shirt": shirt, "trousers": trousers, "jacket": jacket}


async def test_add_single_item_and_save_outfit(app_factory):
    app = await app_factory()
    items = await _add_basic_wardrobe(app)
    shirt = items["shirt"]
    assert shirt.category_key == "top"
    assert shirt.color == "白色"
    assert shirt.seasons == ["春", "秋"]
    assert shirt.temp_min == 12
    assert shirt.temp_max == 26
    assert shirt.formality == 4
    assert shirt.priority == 70

    outfit = await app.wardrobe.create_outfit(
        "test:alice",
        name="通勤套装",
        queries=["白色衬衫", "黑色西裤"],
    )
    assert outfit.name == "通勤套装"
    assert outfit.item_ids == [items["shirt"].id, items["trousers"].id]
    listed = await app.wardrobe.list_outfits(["test:alice"])
    assert [item.name for item in listed] == ["通勤套装"]

    await app.wardrobe.add_item(
        "test:alice",
        name="夏季碎花连衣裙",
        category_query="连衣裙",
        color="碎花",
        seasons="夏",
        scenes="约会,逛街",
        temperature="20-35",
        priority=90,
    )
    dress = await app.wardrobe.resolve_item(["test:alice"], "夏季碎花连衣裙")
    assert dress.category_key == "onepiece"


async def test_wear_swap_remove_and_base_layer_fallback(app_factory):
    app = await app_factory()
    await _add_basic_wardrobe(app)
    scopes = ["test:alice"]

    _state, current, _notes = await app.wardrobe.wear(
        "test:alice",
        scopes,
        ["白色衬衫", "黑色西裤", "蓝色牛仔外套"],
    )
    assert {item.name for item in current} >= {"白色衬衫", "黑色西裤", "蓝色牛仔外套"}

    _state, current, _notes = await app.wardrobe.swap(
        "test:alice",
        scopes,
        "上装",
        "白色衬衫",
    )
    assert {item.name for item in current} == {
        "白色衬衫",
        "黑色西裤",
        "蓝色牛仔外套",
    }

    _state, current, _notes = await app.wardrobe.remove(
        "test:alice",
        scopes,
        "外套",
    )
    assert "蓝色牛仔外套" not in {item.name for item in current}

    _state, current, _notes = await app.wardrobe.remove(
        "test:alice",
        scopes,
        "上装",
    )
    assert "白色衬衫" not in {item.name for item in current}
    assert "基础白T恤" in {item.name for item in current}
    assert "黑色西裤" in {item.name for item in current}

    _state, current, _notes = await app.wardrobe.remove(
        "test:alice",
        scopes,
        "下装",
    )
    names = {item.name for item in current}
    assert "黑色西裤" not in names
    assert {"基础白T恤", "基础黑短裤"} <= names
    description = await app.wardrobe.outfit_description("test:alice", scopes, current)
    assert "基础白T恤" in description
    assert "基础黑短裤" in description

    _state, restored, _notes = await app.wardrobe.restore("test:alice", scopes)
    assert restored


async def test_auto_outfit_by_activity_and_season(app_factory):
    app = await app_factory()
    await _add_basic_wardrobe(app)
    await app.wardrobe.add_item(
        "test:alice",
        name="夏季碎花连衣裙",
        category_query="连衣裙",
        color="碎花",
        seasons="夏",
        scenes="约会,逛街",
        temperature="20-35",
        formality=3,
        priority=90,
    )
    await app.wardrobe.add_item(
        "test:alice",
        name="长款羽绒服",
        category_query="外套",
        color="米白",
        material="羽绒",
        seasons="冬",
        scenes="工作,通勤",
        temperature="-10-8",
        formality=3,
        priority=90,
    )

    summer = await app.wardrobe.auto_outfit(
        "test:alice",
        ["test:alice"],
        activity="约会",
        season="夏",
        temperature=30,
        indoor=False,
    )
    summer_names = {item.name for item in summer.items}
    assert "夏季碎花连衣裙" in summer_names
    assert "长款羽绒服" not in summer_names
    assert summer.source == "auto"

    winter = await app.wardrobe.auto_outfit(
        "test:alice",
        ["test:alice"],
        activity="工作",
        season="冬",
        temperature=2,
        indoor=True,
    )
    winter_names = {item.name for item in winter.items}
    assert "白色衬衫" in winter_names
    assert "黑色西裤" in winter_names
    assert "长款羽绒服" in winter_names
    assert "夏季碎花连衣裙" not in winter_names


async def test_explicit_outfit_fills_missing_base_layer(app_factory):
    app = await app_factory()
    await _add_basic_wardrobe(app)
    await app.wardrobe.add_item(
        "test:alice",
        name="夏季碎花连衣裙",
        category_query="连衣裙",
        seasons="夏",
        priority=80,
    )

    dress_plan = await app.wardrobe.plan_for_generation(
        "test:alice",
        ["test:alice"],
        outfit_hint="夏季碎花连衣裙",
    )
    assert dress_plan.source == "explicit"
    assert [item.name for item in dress_plan.items] == ["夏季碎花连衣裙"]

    shirt_plan = await app.wardrobe.plan_for_generation(
        "test:alice",
        ["test:alice"],
        outfit_hint="白色衬衫",
    )
    shirt_names = {item.name for item in shirt_plan.items}
    assert "白色衬衫" in shirt_names
    assert "基础黑短裤" in shirt_names
    assert "基础白T恤" not in shirt_names
