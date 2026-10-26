from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .models import (
    DIMENSION_LABELS,
    WEIGHT_DIMENSIONS,
    DimensionChoice,
    ReferenceImage,
    ReferencePlan,
    SelectedReference,
)
from .settings import Settings

_KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "face": ("正脸", "侧脸", "五官", "素颜", "特写", "近照", "发型", "妆造"),
    "hair": ("发型", "短发", "长发", "卷发", "直发", "马尾", "刘海", "发色", "染发"),
    "outfit": (
        "穿搭",
        "服装",
        "衣服",
        "裙子",
        "连衣裙",
        "衬衫",
        "T恤",
        "外套",
        "西装",
        "制服",
        "校服",
        "汉服",
        "和服",
        "牛仔裤",
        "短裤",
        "鞋",
        "配饰",
    ),
    "scene": (
        "海边",
        "咖啡店",
        "咖啡馆",
        "街道",
        "公园",
        "森林",
        "山",
        "室内",
        "卧室",
        "客厅",
        "教室",
        "办公室",
        "车站",
        "夜景",
        "雨天",
        "雪",
        "樱花",
        "旅行",
    ),
    "pose": (
        "自拍",
        "坐着",
        "站着",
        "回头",
        "走路",
        "躺着",
        "趴着",
        "比耶",
        "微笑",
        "侧身",
        "背影",
        "低头",
    ),
    "composition": (
        "特写",
        "半身",
        "全身",
        "逆光",
        "暖光",
        "冷光",
        "光影",
        "胶片",
        "广角",
        "近景",
        "远景",
        "俯拍",
        "仰拍",
    ),
}

_DIMENSION_BASE_PRIORITY = {
    "face": 1,
    "hair": 2,
    "outfit": 3,
    "scene": 4,
    "pose": 5,
    "composition": 6,
}


@dataclass(slots=True)
class SelectorIntent:
    description: str = ""
    activity: str = ""
    scene: str = ""
    outfit: str = ""
    selfie: bool = False
    extra_tags: list[str] = field(default_factory=list)

    def combined_text(self) -> str:
        values = [
            self.description,
            self.activity,
            self.scene,
            self.outfit,
            *self.extra_tags,
        ]
        return " ".join(value for value in values if value)


def _tokens(text: str) -> set[str]:
    raw = re.split(r"[\s,，。.;；、:：!?！？/|]+", text or "")
    tokens = {item.strip().casefold() for item in raw if len(item.strip()) >= 2}
    return tokens


def _ref_text(ref: ReferenceImage) -> str:
    return " ".join([ref.name, ref.note, *ref.tags]).casefold()


class ReferenceSelector:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def _keyword_hits(ref: ReferenceImage, keywords: Iterable[str]) -> list[str]:
        text = _ref_text(ref)
        return [keyword for keyword in keywords if keyword.casefold() in text]

    @staticmethod
    def _generic_hits(ref: ReferenceImage, tokens: set[str]) -> list[str]:
        text = _ref_text(ref)
        return [token for token in tokens if token and token in text]

    def _score_dimension(
        self,
        ref: ReferenceImage,
        dimension: str,
        *,
        intent: SelectorIntent,
        global_tokens: set[str],
        linked_ids: set[str],
    ) -> tuple[float, list[str]]:
        weight = ref.dimension_weight(dimension)
        if weight <= 0:
            return 0.0, []
        reasons: list[str] = []
        score = weight * 2.0
        reasons.append(f"{DIMENSION_LABELS[dimension]}权重{weight}")
        if ref.is_primary:
            bonus = 35.0 if dimension in {"face", "hair"} else 16.0
            score += bonus
            reasons.append("主参考加成")
        if intent.selfie and dimension in {"face", "pose", "composition"}:
            score += 8
            reasons.append("自拍偏好")
        keywords = _KEYWORD_GROUPS.get(dimension, ())
        if dimension == "outfit":
            outfit_tokens = _tokens(intent.outfit)
            hits = [
                token
                for token in outfit_tokens
                if token and token in _ref_text(ref)
            ]
            if hits:
                score += 22
                reasons.append("穿搭描述匹配")
        elif dimension == "scene":
            scene_tokens = _tokens(f"{intent.scene} {intent.activity}")
            hits = [token for token in scene_tokens if token in _ref_text(ref)]
            if hits:
                score += 24
                reasons.append(f"场景标签匹配：{'、'.join(hits[:3])}")
        if dimension == "outfit" and ref.id in linked_ids:
            score += 40
            reasons.append("衣柜衣物关联参考")
        hits = self._keyword_hits(ref, keywords)
        if hits:
            score += min(24, 8 * len(hits))
            reasons.append(f"标签匹配：{'、'.join(hits[:3])}")
        generic = self._generic_hits(ref, global_tokens)
        if generic:
            score += min(18, 6 * len(generic))
            reasons.append(f"描述匹配：{'、'.join(generic[:3])}")
        if not ref.file_path:
            score -= 1000
        return score, reasons[:4]

    def _aggregate_score(
        self,
        ref: ReferenceImage,
        *,
        intent: SelectorIntent,
        global_tokens: set[str],
        linked_ids: set[str],
    ) -> tuple[float, list[str]]:
        scores: list[float] = []
        reasons: list[str] = []
        for dimension in WEIGHT_DIMENSIONS:
            score, dim_reasons = self._score_dimension(
                ref,
                dimension,
                intent=intent,
                global_tokens=global_tokens,
                linked_ids=linked_ids,
            )
            if score > 0:
                scores.append(score)
                reasons.extend(dim_reasons[:1])
        aggregate = sum(scores) / max(1, len(scores))
        aggregate += ref.dimension_weight("face") * 0.6
        if ref.id in linked_ids:
            aggregate += 12
            reasons.insert(0, "衣物关联")
        if ref.is_primary:
            aggregate += 20
            reasons.insert(0, "主参考")
        return aggregate, reasons

    def select(
        self,
        references: Sequence[ReferenceImage],
        *,
        intent: SelectorIntent | None = None,
        primary_id: str = "",
        max_images: int | None = None,
        linked_ids: Sequence[str] = (),
    ) -> ReferencePlan:
        intent = intent or SelectorIntent()
        limit = max_images or self.settings.max_reference_images()
        enabled = [ref for ref in references if ref.enabled]
        linked_set = {str(item) for item in linked_ids if str(item).strip()}
        plan = ReferencePlan()
        if not enabled:
            plan.warnings.append("图库中没有可用的参考图，本次将仅依据文字描述生成。")
            return plan

        global_tokens = _tokens(
            f"{intent.description} {intent.outfit} {intent.scene} {intent.activity}"
        )
        selected_map: dict[str, SelectedReference] = {}
        for ref in enabled:
            score, reasons = self._aggregate_score(
                ref,
                intent=intent,
                global_tokens=global_tokens,
                linked_ids=linked_set,
            )
            selected_map[ref.id] = SelectedReference(ref, score, reasons)

        dimension_choices: dict[str, DimensionChoice] = {}
        for dimension in WEIGHT_DIMENSIONS:
            best_ref: ReferenceImage | None = None
            best_score = -1.0
            best_reasons: list[str] = []
            for ref in enabled:
                score, reasons = self._score_dimension(
                    ref,
                    dimension,
                    intent=intent,
                    global_tokens=global_tokens,
                    linked_ids=linked_set,
                )
                if ref.dimension_weight(dimension) <= 0:
                    continue
                if score > best_score:
                    best_ref, best_score, best_reasons = ref, score, reasons
            if best_ref is None:
                plan.warnings.append(
                    f"没有参考图设置有效的{DIMENSION_LABELS[dimension]}权重，该维度不指定权威参考。"
                )
                continue
            dimension_choices[dimension] = DimensionChoice(
                dimension=dimension,
                reference_id=best_ref.id,
                reference_name=best_ref.name,
                weight=best_ref.dimension_weight(dimension),
                score=round(best_score, 2),
                reason="；".join(best_reasons),
            )

        ordered_ids: list[str] = []
        if "face" in dimension_choices:
            ordered_ids.append(dimension_choices["face"].reference_id)
        if primary_id and primary_id in selected_map:
            ordered_ids.append(primary_id)
        else:
            for ref in enabled:
                if ref.is_primary:
                    ordered_ids.append(ref.id)
                    break
        ordered_ids.extend(
            ref.id
            for ref in enabled
            if ref.id in linked_set and ref.dimension_weight("outfit") > 0
        )
        for dimension in WEIGHT_DIMENSIONS:
            choice = dimension_choices.get(dimension)
            if choice:
                ordered_ids.append(choice.reference_id)
        ranked = sorted(
            selected_map.values(),
            key=lambda entry: (entry.score, entry.reference.created_at),
            reverse=True,
        )
        ordered_ids.extend(entry.reference.id for entry in ranked)
        ordered_ids = list(dict.fromkeys(ordered_ids))

        chosen = [selected_map[ref_id].reference for ref_id in ordered_ids if ref_id in selected_map]
        if len(chosen) > limit:
            plan.warnings.append(
                f"参考图数量超过上限（{limit}），已按维度优先级保留最关键的图片。"
            )
            chosen = chosen[:limit]
        plan.references = chosen

        chosen_ids = {ref.id for ref in chosen}
        for dimension, choice in dimension_choices.items():
            if choice.reference_id in chosen_ids:
                plan.dimensions[dimension] = choice
            else:
                plan.warnings.append(
                    "由于参考图数量上限，"
                    f"{DIMENSION_LABELS[dimension]}权威参考《{choice.reference_name}》未能纳入本次请求。"
                )
        if "face" not in dimension_choices and plan.references:
            plan.warnings.append("没有面部权重参考，生成时不强制保证五官与参考图一致。")
        if linked_set:
            available_ids = {ref.id for ref in enabled}
            if linked_set - available_ids:
                plan.warnings.append(
                    "衣柜关联的部分参考图已删除或停用，本次不会提交这些图片。"
                )
            zero_weight = [
                ref.name
                for ref in enabled
                if ref.id in linked_set and ref.dimension_weight("outfit") <= 0
            ]
            if zero_weight:
                plan.warnings.append(
                    "衣物关联参考图的服装权重为 0，本次不作为服装依据："
                    + "、".join(zero_weight[:3])
                )
        plan.references.sort(
            key=lambda ref: (
                0
                if plan.dimensions.get("face") and plan.dimensions["face"].reference_id == ref.id
                else 1,
                0 if ref.is_primary else 1,
                _DIMENSION_BASE_PRIORITY.get(
                    next(
                        (
                            dimension
                            for dimension, choice in plan.dimensions.items()
                            if choice.reference_id == ref.id
                        ),
                        "composition",
                    ),
                    9,
                ),
                -selected_map[ref.id].score,
            )
        )
        return plan
