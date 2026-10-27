from __future__ import annotations

from collections.abc import Iterable, Sequence

from .errors import NotFoundError, ValidationError
from .models import (
    BASE_OUTFIT_LABEL,
    ClothingItem,
    Outfit,
    WardrobeCategory,
    WardrobePlan,
    WearState,
)
from .settings import Settings
from .storage import Storage
from .utils import (
    new_id,
    normalize_tags,
    now_ts,
    parse_bool,
    parse_temperature_range,
    sha256_bytes,
    split_list,
)

BUILTIN_CATEGORIES: tuple[WardrobeCategory, ...] = (
    WardrobeCategory("top", "上装", "top", sort_order=10),
    WardrobeCategory("bottom", "下装", "bottom", sort_order=20),
    WardrobeCategory("inner", "内搭/打底", "inner", sort_order=30),
    WardrobeCategory("outer", "外套", "outer", sort_order=40),
    WardrobeCategory("shoes", "鞋履", "shoes", sort_order=50),
    WardrobeCategory("accessory", "配饰", "accessory", sort_order=60),
    WardrobeCategory("headwear", "头饰", "headwear", sort_order=70),
    WardrobeCategory("onepiece", "连衣裙/连体装", "onepiece", sort_order=80),
    WardrobeCategory("set", "成套服装", "set", sort_order=90),
)

CATEGORY_ALIASES: dict[str, str] = {
    "上衣": "top",
    "上装": "top",
    "外套": "outer",
    "大衣": "outer",
    "裤子": "bottom",
    "下装": "bottom",
    "半身裙": "bottom",
    "裙子": "onepiece",
    "连衣裙": "onepiece",
    "连体装": "onepiece",
    "套装": "set",
    "成套": "set",
    "鞋": "shoes",
    "鞋子": "shoes",
    "配饰": "accessory",
    "首饰": "accessory",
    "头饰": "headwear",
    "帽子": "headwear",
    "内搭": "inner",
    "打底": "inner",
}

COMPLETE_KINDS = {"onepiece", "set"}
ALL_SEASONS = ("春", "夏", "秋", "冬")

ACTIVITY_SCENE_HINTS: dict[str, tuple[str, ...]] = {
    "居家": ("居家", "在家", "家里", "宅", "卧室", "客厅", "休息", "看书", "追剧", "做饭", "睡觉"),
    "工作": ("上班", "工作", "办公", "公司", "会议", "通勤", "加班"),
    "约会": ("约会", "逛街", "咖啡", "餐厅", "看电影", "散步"),
    "运动": ("运动", "健身", "跑步", "瑜伽", "球", "游泳"),
    "旅行": ("旅行", "旅游", "出游", "海边", "爬山", "露营"),
    "正式": ("正式", "宴会", "婚礼", "典礼", "面试"),
    "睡": ("睡觉", "睡衣", "睡前", "卧室"),
}

FORMALITY_HINTS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("睡觉", "睡衣", "居家", "在家", "运动", "健身"), 1),
    (("咖啡", "逛街", "约会", "散步", "旅行", "出游"), 3),
    (("上班", "工作", "办公", "通勤", "会议"), 4),
    (("正式", "宴会", "婚礼", "典礼", "面试"), 5),
)

BASE_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "name": "基础白T恤",
        "category_key": "top",
        "role": "base_top",
        "color": "白色",
    },
    {
        "name": "基础黑短裤",
        "category_key": "bottom",
        "role": "base_bottom",
        "color": "黑色",
    },
)


def _contains_any(haystack: Iterable[str], needles: Iterable[str]) -> bool:
    text = " ".join(str(value) for value in haystack).casefold()
    return any(str(needle).casefold() in text for needle in needles if str(needle).strip())


def _text_matches(text: str, values: Sequence[str]) -> bool:
    lowered = (text or "").casefold()
    for value in values:
        token = str(value).strip().casefold()
        if token and (token in lowered or token in {"四季", "all", "any"}):
            return True
    return False


class WardrobeService:
    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings

    # ---------- categories ----------

    async def categories(self, owner_ids: Iterable[str]) -> list[WardrobeCategory]:
        result = list(BUILTIN_CATEGORIES)
        for owner_id in dict.fromkeys(str(item) for item in owner_ids):
            result.extend(await self.storage.list_custom_categories(owner_id))
        result.sort(key=lambda item: (0 if not item.custom else 1, item.sort_order, item.label))
        return result

    async def resolve_category(
        self,
        owner_ids: Iterable[str],
        raw: str,
    ) -> WardrobeCategory:
        text = (raw or "").strip()
        if not text:
            raise ValidationError("请提供衣物分类。")
        lowered = text.casefold()
        categories = await self.categories(owner_ids)
        for category in categories:
            if lowered in {category.key.casefold(), category.label.casefold()}:
                return category
        alias_key = CATEGORY_ALIASES.get(text)
        if alias_key:
            for category in categories:
                if category.kind == alias_key and not category.custom:
                    return category
        for category in categories:
            if lowered in category.label.casefold() or category.label.casefold() in lowered:
                return category
        names = "、".join(category.label for category in categories)
        raise ValidationError(f"未找到分类“{text}”。可用分类：{names}")

    async def add_custom_category(
        self,
        owner_id: str,
        label: str,
        kind: str,
    ) -> WardrobeCategory:
        clean_label = " ".join((label or "").split())[:20]
        if not clean_label:
            raise ValidationError("分类名称不能为空。")
        valid_kinds = {category.kind for category in BUILTIN_CATEGORIES}
        clean_kind = (kind or "other").strip()
        if clean_kind not in valid_kinds and clean_kind != "other":
            raise ValidationError("分类类型必须是上装、下装、外套、鞋履等基础类型之一。")
        categories = await self.categories([owner_id])
        if any(item.label.casefold() == clean_label.casefold() for item in categories):
            raise ValidationError("已存在同名分类。")
        key = f"custom_{sha256_bytes(clean_label.encode('utf-8'))[:10]}"
        category = WardrobeCategory(
            key=key,
            label=clean_label,
            kind=clean_kind,
            owner_id=owner_id,
            custom=True,
            sort_order=200,
        )
        await self.storage.upsert_custom_category(category)
        return category

    async def delete_custom_category(self, owner_id: str, raw: str) -> WardrobeCategory:
        category = await self.resolve_category([owner_id], raw)
        if not category.custom or category.owner_id != owner_id:
            raise ValidationError("只能删除自己数据空间中的自定义分类。")
        if await self.storage.category_in_use(owner_id, category.key):
            raise ValidationError("该分类下仍有衣物，请先移动或删除衣物。")
        if not await self.storage.delete_custom_category(owner_id, category.key):
            raise NotFoundError("分类不存在。")
        return category

    # ---------- base layers ----------

    async def ensure_base_layers(self, owner_id: str) -> tuple[ClothingItem, ClothingItem]:
        existing: dict[str, ClothingItem] = {}
        for definition in BASE_DEFINITIONS:
            item = await self.storage.find_base_clothing(owner_id, definition["role"])
            if item is None:
                stamp = now_ts()
                item = ClothingItem(
                    id=new_id("base"),
                    owner_id=owner_id,
                    name=definition["name"],
                    category_key=definition["category_key"],
                    color=definition["color"],
                    material="棉质",
                    seasons=list(ALL_SEASONS),
                    scenes=["居家", "日常", "睡觉"],
                    temp_min=-10,
                    temp_max=40,
                    formality=1,
                    indoor_outdoor="both",
                    priority=1000,
                    enabled=True,
                    is_base=True,
                    base_role=definition["role"],
                    note="安全基础层，自动创建且不可删除。",
                    created_at=stamp,
                    updated_at=stamp,
                )
                await self.storage.add_clothing(item)
            existing[definition["role"]] = item
        return existing["base_top"], existing["base_bottom"]

    # ---------- item lookup ----------

    async def list_items(
        self,
        owner_ids: Iterable[str],
        *,
        enabled_only: bool = True,
    ) -> list[ClothingItem]:
        scopes = list(dict.fromkeys(str(item) for item in owner_ids))
        return await self.storage.list_clothing(scopes, enabled_only=enabled_only)

    async def _categories_by_key(
        self,
        owner_ids: Iterable[str],
    ) -> dict[str, WardrobeCategory]:
        return {category.key: category for category in await self.categories(owner_ids)}

    def _kind_of(self, item: ClothingItem, categories: dict[str, WardrobeCategory]) -> str:
        category = categories.get(item.category_key)
        if category:
            return category.kind
        return item.category_key

    async def match_items(
        self,
        owner_ids: Iterable[str],
        query: str,
        *,
        include_base: bool = True,
    ) -> list[ClothingItem]:
        text = (query or "").strip().casefold()
        if not text:
            return []
        items = await self.list_items(owner_ids, enabled_only=False)
        if not include_base:
            items = [item for item in items if not item.is_base]
        exact = [
            item
            for item in items
            if text in {item.id.casefold(), item.name.casefold()}
        ]
        if exact:
            return exact
        partial = [
            item
            for item in items
            if text in item.name.casefold()
            or text in item.color.casefold()
            or any(text in tag.casefold() or tag.casefold() in text for tag in item.scenes)
        ]
        return partial

    async def resolve_item(
        self,
        owner_ids: Iterable[str],
        query: str,
        *,
        include_base: bool = True,
    ) -> ClothingItem:
        matches = await self.match_items(owner_ids, query, include_base=include_base)
        if not matches:
            raise NotFoundError(f"衣柜里没有找到“{query}”。")
        enabled = [item for item in matches if item.enabled]
        matches = enabled or matches
        if len(matches) > 1:
            names = "、".join(f"{item.name}({item.id})" for item in matches[:8])
            raise ValidationError(f"“{query}”匹配到多件衣物：{names}。请使用更精确的名称或 ID。")
        return matches[0]

    # ---------- wear state ----------

    async def _resolve_worn(
        self,
        state: WearState,
        scopes: list[str],
        categories: dict[str, WardrobeCategory],
        notes: list[str],
    ) -> list[ClothingItem]:
        if not state.worn_ids:
            return []
        placeholders = list(dict.fromkeys(state.worn_ids))
        items = await self.list_items(scopes, enabled_only=False)
        lookup = {item.id: item for item in items}
        resolved: list[ClothingItem] = []
        for item_id in placeholders:
            item = lookup.get(item_id)
            if item is None:
                notes.append("部分衣物已不存在，已从当前穿搭中移除。")
                continue
            if not item.enabled:
                notes.append(f"《{item.name}》已停用，已从当前穿搭中移除。")
                continue
            resolved.append(item)
        return resolved

    def _dedupe_by_kind(
        self,
        items: list[ClothingItem],
        categories: dict[str, WardrobeCategory],
    ) -> list[ClothingItem]:
        ordered: list[ClothingItem] = []
        slots: dict[str, ClothingItem] = {}
        complete: ClothingItem | None = None
        for item in items:
            kind = self._kind_of(item, categories)
            if kind in COMPLETE_KINDS:
                if complete is None:
                    complete = item
                continue
            previous = slots.get(kind)
            if previous is None or item.updated_at >= previous.updated_at:
                slots[kind] = item
        if complete is not None:
            ordered.append(complete)
            for kind in ("outer", "shoes", "accessory", "headwear", "inner"):
                if kind in slots:
                    ordered.append(slots[kind])
        else:
            for kind in ("inner", "top", "bottom", "outer", "shoes", "accessory", "headwear"):
                if kind in slots:
                    ordered.append(slots[kind])
        return ordered

    async def _normalize_state(
        self,
        state: WearState,
        scopes: list[str],
        *,
        notes: list[str] | None = None,
    ) -> tuple[WearState, list[ClothingItem], list[str]]:
        note_list = list(notes or [])
        categories = await self._categories_by_key(scopes)
        items = await self._resolve_worn(state, scopes, categories, note_list)
        items = self._dedupe_by_kind(items, categories)
        kinds = {self._kind_of(item, categories) for item in items}
        has_complete = bool(kinds & COMPLETE_KINDS)
        base_top, base_bottom = await self.ensure_base_layers(state.owner_id)
        if not has_complete:
            if "top" not in kinds:
                items.append(base_top)
                note_list.append("上装为空，已自动补上基础白T恤。")
                kinds.add("top")
            if "bottom" not in kinds:
                items.append(base_bottom)
                note_list.append("下装为空，已自动补上基础黑色短裤。")
                kinds.add("bottom")
        items = self._dedupe_by_kind(items, categories)
        normalized_ids = [item.id for item in items]
        if normalized_ids != state.worn_ids:
            state.worn_ids = normalized_ids
            state.updated_at = now_ts()
            await self.storage.save_wear_state(state)
        return state, items, note_list

    async def get_state(
        self,
        state_owner: str,
        scopes: Sequence[str] | None = None,
    ) -> tuple[WearState, list[ClothingItem], list[str]]:
        scope_list = list(dict.fromkeys([state_owner, *(scopes or [state_owner])]))
        state = await self.storage.get_wear_state(state_owner)
        notes: list[str] = []
        if not state.worn_ids:
            base_top, base_bottom = await self.ensure_base_layers(state_owner)
            state.worn_ids = [base_top.id, base_bottom.id]
            state.mode = state.mode or "auto"
            state.updated_at = now_ts()
            await self.storage.save_wear_state(state)
        return await self._normalize_state(state, scope_list, notes=notes)

    async def _push_history(self, state: WearState) -> None:
        if not state.worn_ids:
            return
        if state.history and state.history[-1] == list(state.worn_ids):
            return
        state.history.append(list(state.worn_ids))
        state.history = state.history[-20:]

    async def _apply_wear(
        self,
        state_owner: str,
        scopes: Sequence[str],
        new_items: Sequence[ClothingItem],
        *,
        mode: str = "manual",
    ) -> tuple[WearState, list[ClothingItem], list[str]]:
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        categories = await self._categories_by_key(scope_list)
        state, current, notes = await self.get_state(state_owner, scope_list)
        await self._push_history(state)
        incoming_kinds = {self._kind_of(item, categories) for item in new_items}
        wants_complete = bool(incoming_kinds & COMPLETE_KINDS)
        merged = list(current)
        for item in new_items:
            kind = self._kind_of(item, categories)
            if (
                wants_complete
                and kind not in COMPLETE_KINDS
                and kind not in {"outer", "shoes", "accessory", "headwear"}
            ):
                # 穿一整件套装时，保留外层和配饰，顶部与底部由套装覆盖。
                continue
            if kind in COMPLETE_KINDS:
                merged = [
                    existing
                    for existing in merged
                    if self._kind_of(existing, categories) not in {"top", "bottom"} | COMPLETE_KINDS
                ]
            elif self._kind_of(item, categories) in {"top", "bottom"}:
                merged = [
                    existing
                    for existing in merged
                    if self._kind_of(existing, categories) not in COMPLETE_KINDS
                ]
            merged = [
                existing
                for existing in merged
                if not (
                    self._kind_of(existing, categories) == kind
                    and existing.id != item.id
                )
            ]
            if all(existing.id != item.id for existing in merged):
                merged.append(item)
        state.worn_ids = [item.id for item in merged]
        state.mode = mode
        state.updated_at = now_ts()
        await self.storage.save_wear_state(state)
        return await self._normalize_state(state, scope_list, notes=notes)

    async def wear(
        self,
        state_owner: str,
        scopes: Sequence[str],
        queries: Sequence[str],
    ) -> tuple[WearState, list[ClothingItem], list[str]]:
        if not queries:
            raise ValidationError("请告诉我要穿哪件衣物。")
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        items: list[ClothingItem] = []
        for query in queries:
            for part in split_list(query) or [query]:
                if part.strip():
                    items.append(await self.resolve_item(scope_list, part))
        return await self._apply_wear(state_owner, scope_list, items)

    async def remove(
        self,
        state_owner: str,
        scopes: Sequence[str],
        target: str,
    ) -> tuple[WearState, list[ClothingItem], list[str]]:
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        categories = await self._categories_by_key(scope_list)
        state, current, notes = await self.get_state(state_owner, scope_list)
        text = (target or "").strip().casefold()
        if not text:
            raise ValidationError("请告诉我要脱下哪件衣物或哪一类衣物。")
        to_remove_kinds: set[str] = set()
        try:
            category = await self.resolve_category(scope_list, target)
            to_remove_kinds.add(category.kind)
        except ValidationError:
            pass
        item_matches = [
            item
            for item in current
            if text in item.name.casefold() or text == item.id.casefold()
        ]
        removed = [
            item
            for item in current
            if self._kind_of(item, categories) in to_remove_kinds or item in item_matches
        ]
        if not removed:
            raise NotFoundError(f"当前穿搭里没有匹配“{target}”的衣物。")
        await self._push_history(state)
        removed_ids = {item.id for item in removed}
        state.worn_ids = [item.id for item in current if item.id not in removed_ids]
        state.mode = "manual"
        state.updated_at = now_ts()
        if any(self._kind_of(item, categories) in COMPLETE_KINDS for item in removed):
            notes.append("已脱下整套衣物。")
        await self.storage.save_wear_state(state)
        return await self._normalize_state(state, scope_list, notes=notes)

    async def swap(
        self,
        state_owner: str,
        scopes: Sequence[str],
        category_query: str,
        item_query: str,
    ) -> tuple[WearState, list[ClothingItem], list[str]]:
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        category = await self.resolve_category(scope_list, category_query)
        item = await self.resolve_item(scope_list, item_query)
        categories = await self._categories_by_key(scope_list)
        item_kind = self._kind_of(item, categories)
        if item_kind != category.kind and not (
            item_kind in COMPLETE_KINDS and category.kind in COMPLETE_KINDS
        ):
            raise ValidationError(f"《{item.name}》不属于“{category.label}”分类。")
        state, current, notes = await self.get_state(state_owner, scope_list)
        await self._push_history(state)
        state.worn_ids = [
            worn.id
            for worn in current
            if self._kind_of(worn, categories) != category.kind
            and not (
                category.kind in COMPLETE_KINDS
                and self._kind_of(worn, categories) in COMPLETE_KINDS
            )
        ]
        state.worn_ids.append(item.id)
        state.mode = "manual"
        state.updated_at = now_ts()
        await self.storage.save_wear_state(state)
        return await self._normalize_state(state, scope_list, notes=notes)

    async def restore(
        self,
        state_owner: str,
        scopes: Sequence[str],
    ) -> tuple[WearState, list[ClothingItem], list[str]]:
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        state = await self.storage.get_wear_state(state_owner)
        if not state.history:
            raise ValidationError("没有可以还原的上一次穿搭。")
        previous = state.history.pop()
        state.worn_ids = previous
        state.mode = "manual"
        state.updated_at = now_ts()
        notes = ["已还原到上一次穿搭。"]
        await self.storage.save_wear_state(state)
        return await self._normalize_state(state, scope_list, notes=notes)

    async def reset_auto(
        self,
        state_owner: str,
        scopes: Sequence[str],
        *,
        activity: str = "",
        season: str = "",
        temperature: int | None = None,
        indoor: bool | None = None,
        scene: str = "",
    ) -> tuple[WearState, WardrobePlan]:
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        state = await self.storage.get_wear_state(state_owner)
        if state.worn_ids:
            await self._push_history(state)
        state.mode = "auto"
        state.worn_ids = []
        state.updated_at = now_ts()
        await self.storage.save_wear_state(state)
        plan = await self.auto_outfit(
            state_owner,
            scope_list,
            activity=activity,
            season=season,
            temperature=temperature,
            indoor=indoor,
            scene=scene,
        )
        return state, plan

    # ---------- automatic outfit ----------

    @staticmethod
    def _target_formality(text: str) -> int:
        lowered = text.casefold()
        for keywords, value in FORMALITY_HINTS:
            if any(keyword in lowered for keyword in keywords):
                return value
        return 3

    @staticmethod
    def _scene_tokens(activity: str, scene: str, season: str) -> set[str]:
        tokens: set[str] = set()
        text = f"{activity} {scene}"
        for hints in ACTIVITY_SCENE_HINTS.values():
            if any(hint in text for hint in hints):
                tokens.update(hints)
        tokens.update(split_list(scene) if scene else [])
        if season:
            tokens.add(season)
        return {token for token in tokens if token}

    @staticmethod
    def _score_item(
        item: ClothingItem,
        *,
        scene_tokens: set[str],
        season: str,
        temperature: int | None,
        indoor: bool | None,
        target_formality: int,
    ) -> float:
        score = item.priority * 0.4
        if item.scenes:
            if scene_tokens and _contains_any(item.scenes, scene_tokens):
                score += 35
            elif scene_tokens:
                score -= 5
        if item.seasons:
            if season and _text_matches(season, item.seasons):
                score += 20
            elif season and not any(value in {"四季", "all", "any"} for value in item.seasons):
                score -= 15
        if temperature is not None and (item.temp_min is not None or item.temp_max is not None):
            score += 15 if item.matches_temperature(temperature) else -30
        if item.indoor_outdoor in {"indoor", "outdoor"} and indoor is not None:
            wants = "indoor" if indoor else "outdoor"
            score += 10 if item.indoor_outdoor == wants else -10
        distance = abs(item.formality - target_formality)
        score += max(0, 15 - distance * 5)
        return score

    @staticmethod
    def _conflicts(item: ClothingItem, selected: Sequence[ClothingItem]) -> bool:
        if not item.forbidden_with:
            return False
        for other in selected:
            if other.id == item.id:
                continue
            for blocked in item.forbidden_with:
                token = blocked.casefold()
                if token in {other.id.casefold(), other.name.casefold()}:
                    return True
                if token and token in other.name.casefold():
                    return True
        return False

    @staticmethod
    def _pair_bonus(item: ClothingItem, selected: Sequence[ClothingItem]) -> float:
        if not item.recommended_with:
            return 0.0
        bonus = 0.0
        for other in selected:
            if any(
                token.casefold() in {other.id.casefold(), other.name.casefold()}
                or token.casefold() in other.name.casefold()
                for token in item.recommended_with
            ):
                bonus += 8
        return min(16.0, bonus)

    async def _pick(
        self,
        candidates: list[ClothingItem],
        *,
        selected: list[ClothingItem],
        count: int = 1,
        **score_kwargs,
    ) -> list[ClothingItem]:
        scored = []
        for item in candidates:
            if self._conflicts(item, selected):
                continue
            score = self._score_item(item, **score_kwargs) + self._pair_bonus(item, selected)
            scored.append((score, item))
        scored.sort(key=lambda pair: (pair[0], pair[1].priority, pair[1].updated_at), reverse=True)
        return [item for _, item in scored[:count]]

    async def auto_outfit(
        self,
        state_owner: str,
        scopes: Sequence[str],
        *,
        activity: str = "",
        season: str = "",
        temperature: int | None = None,
        indoor: bool | None = None,
        scene: str = "",
        weather: str = "",
    ) -> WardrobePlan:
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        categories = await self._categories_by_key(scope_list)
        items = await self.list_items(scope_list, enabled_only=True)
        base_top, base_bottom = await self.ensure_base_layers(state_owner)
        candidates = [
            item
            for item in items
            if not item.is_base and item.category_key in categories
        ]
        by_kind: dict[str, list[ClothingItem]] = {}
        for item in candidates:
            by_kind.setdefault(self._kind_of(item, categories), []).append(item)
        context_text = f"{activity} {scene} {weather}"
        scene_tokens = self._scene_tokens(activity, scene, season)
        target_formality = self._target_formality(context_text)
        score_kwargs = {
            "scene_tokens": scene_tokens,
            "season": season,
            "temperature": temperature,
            "indoor": indoor,
            "target_formality": target_formality,
        }
        selected: list[ClothingItem] = []
        notes: list[str] = []

        complete_candidates = by_kind.get("onepiece", []) + by_kind.get("set", [])
        complete_pick = await self._pick(
            complete_candidates,
            selected=selected,
            **score_kwargs,
        )
        complete_score = (
            self._score_item(complete_pick[0], **score_kwargs) if complete_pick else -999.0
        )
        top_pick = await self._pick(by_kind.get("top", []), selected=[], **score_kwargs)
        bottom_pick = await self._pick(
            by_kind.get("bottom", []),
            selected=top_pick,
            **score_kwargs,
        )
        separates_score = -999.0
        if top_pick and bottom_pick:
            separates_score = (
                self._score_item(top_pick[0], **score_kwargs)
                + self._score_item(bottom_pick[0], **score_kwargs)
                + self._pair_bonus(top_pick[0], bottom_pick)
            ) / 2 + 6
        prefer_complete = complete_pick and complete_score + 12 >= separates_score
        if top_pick and bottom_pick and not prefer_complete:
            selected.extend([top_pick[0], bottom_pick[0]])
        elif complete_pick:
            selected.append(complete_pick[0])
            notes.append("已选择成套穿搭。")
        else:
            if top_pick:
                selected.append(top_pick[0])
            if bottom_pick:
                selected.append(bottom_pick[0])
        notes.extend(
            await self._auto_layers(
                by_kind.get("outer", []),
                selected,
                score_kwargs,
                kind="outer",
                season=season,
                temperature=temperature,
                indoor=indoor,
            )
        )
        notes.extend(
            await self._auto_layers(
                by_kind.get("shoes", []),
                selected,
                score_kwargs,
                kind="shoes",
                season=season,
                temperature=temperature,
                indoor=indoor,
            )
        )
        notes.extend(
            await self._auto_layers(
                by_kind.get("accessory", []),
                selected,
                score_kwargs,
                kind="accessory",
                season=season,
                temperature=temperature,
                indoor=indoor,
                threshold=95,
            )
        )
        notes.extend(
            await self._auto_layers(
                by_kind.get("headwear", []),
                selected,
                score_kwargs,
                kind="headwear",
                season=season,
                temperature=temperature,
                indoor=indoor,
                threshold=100,
            )
        )

        if not any(self._kind_of(item, categories) in COMPLETE_KINDS for item in selected):
            if not any(self._kind_of(item, categories) == "top" for item in selected):
                selected.append(base_top)
                notes.append("没有合适上装，已使用基础白T恤。")
            if not any(self._kind_of(item, categories) == "bottom" for item in selected):
                selected.append(base_bottom)
                notes.append("没有合适下装，已使用基础黑色短裤。")
        if not any(not item.is_base for item in selected):
            notes.append("衣柜中没有匹配的衣物，本次使用基础安全穿搭。")
        state = await self.storage.get_wear_state(state_owner)
        state.worn_ids = [item.id for item in selected]
        state.mode = "auto"
        state.updated_at = now_ts()
        await self.storage.save_wear_state(state)
        return WardrobePlan(items=selected, source="auto", notes=notes)

    async def _auto_layers(
        self,
        candidates: list[ClothingItem],
        selected: list[ClothingItem],
        score_kwargs: dict,
        *,
        kind: str,
        season: str,
        temperature: int | None,
        indoor: bool | None,
        threshold: float = 70,
    ) -> list[str]:
        if not candidates:
            return []
        if kind == "outer":
            if temperature is not None and temperature >= 26 and season in {"夏", "春"}:
                return []
            if indoor and temperature is None:
                return []
        picked = await self._pick(candidates, selected=selected, count=1, **score_kwargs)
        if not picked:
            return []
        item = picked[0]
        if self._score_item(item, **score_kwargs) < threshold:
            return []
        selected.append(item)
        return [f"搭配了《{item.name}》。"]

    async def plan_for_generation(
        self,
        state_owner: str,
        scopes: Sequence[str],
        *,
        outfit_hint: str = "",
        activity: str = "",
        season: str = "",
        temperature: int | None = None,
        indoor: bool | None = None,
        scene: str = "",
        weather: str = "",
    ) -> WardrobePlan:
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        if outfit_hint.strip():
            matches = await self.match_items(scope_list, outfit_hint, include_base=False)
            if matches:
                enabled_matches = [item for item in matches if item.enabled]
                chosen = self._dedupe_by_kind(
                    (enabled_matches or matches)[:4],
                    await self._categories_by_key(scope_list),
                )
                state, current, notes = await self.get_state(state_owner, scope_list)
                categories = await self._categories_by_key(scope_list)
                kinds = {self._kind_of(item, categories) for item in chosen}
                if not (kinds & COMPLETE_KINDS):
                    base_top, base_bottom = await self.ensure_base_layers(state_owner)
                    if "top" not in kinds:
                        chosen.append(base_top)
                        notes.append("指定穿搭缺少上装，已补基础白T恤。")
                    if "bottom" not in kinds:
                        chosen.append(base_bottom)
                        notes.append("指定穿搭缺少下装，已补基础黑色短裤。")
                return WardrobePlan(
                    items=chosen,
                    source="explicit",
                    notes=[*notes, f"按指定穿搭匹配到 {len(chosen)} 件衣物。"],
                )
            state, current, notes = await self.get_state(state_owner, scope_list)
            return WardrobePlan(
                items=current,
                source="state",
                notes=[*notes, "衣柜中未找到与指定穿搭完全匹配的条目，将结合描述与当前穿搭生成。"],
            )
        state, current, notes = await self.get_state(state_owner, scope_list)
        if state.mode == "manual":
            return WardrobePlan(items=current, source="state", notes=notes)
        return await self.auto_outfit(
            state_owner,
            scope_list,
            activity=activity,
            season=season,
            temperature=temperature,
            indoor=indoor,
            scene=scene,
            weather=weather,
        )

    # ---------- clothing CRUD ----------

    async def add_item(
        self,
        owner_id: str,
        *,
        name: str,
        category_query: str,
        scopes: Sequence[str] | None = None,
        color: str = "",
        material: str = "",
        seasons: object = None,
        scenes: object = None,
        temperature: object = None,
        formality: int = 3,
        indoor_outdoor: str = "both",
        recommended_with: object = None,
        forbidden_with: object = None,
        priority: int = 50,
        reference_id: str = "",
        note: str = "",
    ) -> ClothingItem:
        clean_name = " ".join((name or "").split())[:40]
        if not clean_name:
            raise ValidationError("衣物名称不能为空。")
        max_items = self.settings.integer("wardrobe", "max_items", 200, 1, 2000)
        if await self.storage.count_clothing(owner_id) >= max_items:
            raise ValidationError(f"当前衣柜已达上限（{max_items} 件）。")
        scope_list = list(dict.fromkeys([owner_id, *(scopes or [])]))
        category = await self.resolve_category(scope_list, category_query)
        temp_min, temp_max = parse_temperature_range(temperature)
        clean_indoors = (indoor_outdoor or "both").strip().lower()
        if clean_indoors not in {"indoor", "outdoor", "both", "室内", "室外", "皆可"}:
            raise ValidationError("场合属性应为室内、室外或皆可。")
        clean_indoors = {"室内": "indoor", "室外": "outdoor", "皆可": "both"}.get(
            clean_indoors, clean_indoors
        )
        furniture = normalize_tags(recommended_with) if recommended_with else []
        forbidden = normalize_tags(forbidden_with) if forbidden_with else []
        stamp = now_ts()
        item = ClothingItem(
            id=new_id("wear"),
            owner_id=owner_id,
            name=clean_name,
            category_key=category.key,
            color=" ".join(str(color or "").split())[:20],
            material=" ".join(str(material or "").split())[:20],
            seasons=normalize_tags(seasons) if seasons else [],
            scenes=normalize_tags(scenes) if scenes else [],
            temp_min=temp_min,
            temp_max=temp_max,
            formality=max(1, min(5, int(formality))),
            indoor_outdoor=clean_indoors,
            recommended_with=furniture,
            forbidden_with=forbidden,
            priority=max(0, min(100, int(priority))),
            enabled=True,
            reference_id=reference_id.strip(),
            note=" ".join(str(note or "").split())[:300],
            created_at=stamp,
            updated_at=stamp,
        )
        await self.storage.add_clothing(item)
        return item

    async def update_item(
        self,
        item_id: str,
        owner_id: str,
        **changes,
    ) -> ClothingItem:
        item = await self.storage.get_clothing(item_id, [owner_id])
        if item is None:
            raise NotFoundError("未找到该衣物，或它不属于当前数据空间。")
        if item.is_base and any(
            key in changes for key in ("category_key", "is_base", "enabled")
        ):
            raise ValidationError("基础安全衣物不能修改分类或停用。")
        if "name" in changes:
            name = " ".join(str(changes["name"]).split())[:40]
            if not name:
                raise ValidationError("衣物名称不能为空。")
            item.name = name
        for field in ("color", "material", "note"):
            if field in changes:
                setattr(item, field, " ".join(str(changes[field] or "").split())[:300])
        if "category_query" in changes:
            category = await self.resolve_category([owner_id, "shared"], str(changes["category_query"]))
            item.category_key = category.key
        if "seasons" in changes:
            item.seasons = normalize_tags(changes["seasons"])
        if "scenes" in changes:
            item.scenes = normalize_tags(changes["scenes"])
        if "temperature" in changes:
            item.temp_min, item.temp_max = parse_temperature_range(changes["temperature"])
        if "formality" in changes:
            item.formality = max(1, min(5, int(changes["formality"])))
        if "indoor_outdoor" in changes:
            value = str(changes["indoor_outdoor"]).strip().lower()
            value = {"室内": "indoor", "室外": "outdoor", "皆可": "both"}.get(value, value)
            if value not in {"indoor", "outdoor", "both"}:
                raise ValidationError("场合属性应为室内、室外或皆可。")
            item.indoor_outdoor = value
        if "recommended_with" in changes:
            item.recommended_with = normalize_tags(changes["recommended_with"])
        if "forbidden_with" in changes:
            item.forbidden_with = normalize_tags(changes["forbidden_with"])
        if "priority" in changes:
            item.priority = max(0, min(100, int(changes["priority"])))
        if "reference_id" in changes:
            item.reference_id = str(changes["reference_id"]).strip()
        if "enabled" in changes:
            item.enabled = parse_bool(changes["enabled"], True)
        item.updated_at = now_ts()
        if not await self.storage.update_clothing(item):
            raise NotFoundError("衣物更新失败。")
        return item

    async def delete_item(
        self,
        item_id: str,
        owner_id: str,
        *,
        state_owner: str | None = None,
        scopes: Sequence[str] | None = None,
    ) -> ClothingItem:
        item = await self.storage.delete_clothing(item_id, owner_id)
        if item is None:
            raise NotFoundError("未找到该衣物，或它是不可删除的基础安全衣物。")
        await self.storage.remove_item_from_outfits(item_id, [owner_id])
        if state_owner:
            scope_list = list(dict.fromkeys([state_owner, *(scopes or [])]))
            state = await self.storage.get_wear_state(state_owner)
            if item_id in state.worn_ids:
                state.worn_ids = [value for value in state.worn_ids if value != item_id]
                state.updated_at = now_ts()
                await self._normalize_state(state, scope_list, notes=[])
        return item

    # ---------- outfits ----------

    async def create_outfit(
        self,
        owner_id: str,
        *,
        name: str,
        queries: Sequence[str],
        scopes: Sequence[str] | None = None,
        note: str = "",
    ) -> Outfit:
        clean_name = " ".join((name or "").split())[:30]
        if not clean_name:
            raise ValidationError("套装名称不能为空。")
        scope_list = list(dict.fromkeys([owner_id, *(scopes or [])]))
        items: list[ClothingItem] = []
        for query in queries:
            items.append(await self.resolve_item(scope_list, query, include_base=False))
        if not items:
            raise ValidationError("套装至少需要包含一件衣物。")
        stamp = now_ts()
        outfit = Outfit(
            id=new_id("outfit"),
            owner_id=owner_id,
            name=clean_name,
            item_ids=[item.id for item in items],
            note=" ".join(str(note or "").split())[:300],
            created_at=stamp,
            updated_at=stamp,
        )
        await self.storage.add_outfit(outfit)
        return outfit

    async def list_outfits(self, owner_ids: Iterable[str]) -> list[Outfit]:
        return await self.storage.list_outfits(list(dict.fromkeys(owner_ids)))

    async def delete_outfit(self, owner_id: str, outfit_id: str) -> bool:
        if not await self.storage.delete_outfit(outfit_id, owner_id):
            raise NotFoundError("未找到该套装。")
        return True

    async def wear_outfit(
        self,
        state_owner: str,
        scopes: Sequence[str],
        outfit_id: str,
    ) -> tuple[WearState, list[ClothingItem], list[str]]:
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        outfit = await self.storage.get_outfit(outfit_id, scope_list)
        if outfit is None:
            raise NotFoundError("未找到该套装。")
        items: list[ClothingItem] = []
        missing: list[str] = []
        for item_id in outfit.item_ids:
            item = await self.storage.get_clothing(item_id, scope_list)
            if item is None or not item.enabled:
                missing.append(item_id)
            else:
                items.append(item)
        if not items:
            raise ValidationError("该套装中的衣物均已被删除或停用。")
        state, current, notes = await self._apply_wear(state_owner, scope_list, items)
        notes.insert(0, f"已穿上套装《{outfit.name}》。")
        if missing:
            notes.append(f"有 {len(missing)} 件衣物缺失，已跳过。")
        return state, current, notes

    async def outfit_description(
        self,
        state_owner: str,
        scopes: Sequence[str],
        items: Sequence[ClothingItem],
    ) -> str:
        if not items:
            return BASE_OUTFIT_LABEL
        scope_list = [state_owner, *scopes]
        categories = await self.categories(scope_list)
        label_by_key = {category.key: category.label for category in categories}
        parts = []
        for item in items:
            label = label_by_key.get(item.category_key, item.category_key)
            detail = f"{label}：{item.name}"
            attrs = [value for value in (item.color, item.material) if value]
            if attrs:
                detail += f"（{'、'.join(attrs)}）"
            parts.append(detail)
        return "；".join(parts)
