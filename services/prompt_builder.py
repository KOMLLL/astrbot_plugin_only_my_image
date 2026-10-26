from __future__ import annotations

import re

from .errors import SafetyError
from .models import (
    PromptPlan,
    ReferencePlan,
    TimelineContext,
    WardrobePlan,
)
from .settings import Settings
from .utils import looks_like_nudity_request, truncate_text

SAFETY_NEGATIVE = "裸露、全裸、暴露性器官、性暗示、未穿衣物、低俗色情、未成年人性化"
TECH_NEGATIVE = "低清晰度、模糊、严重噪点、肢体错位、多余手指、畸形、文字、水印、logo"

_STYLE_HINTS = (
    "二次元",
    "动漫",
    "anime",
    "插画",
    "水彩",
    "油画",
    "素描",
    "3D",
    "赛博朋克",
    "古风",
    "像素",
    "漫画",
)
_NO_FACE_PATTERNS = (
    "不要像我",
    "不像我",
    "不要用我的脸",
    "不要参考脸",
    "不要参考面部",
    "别用我的脸",
    "换脸",
    "不用脸",
    "只参考衣服",
    "只看衣服",
)
_NO_OUTFIT_PATTERNS = (
    "不要参考衣服",
    "不要参考服装",
    "不看衣服",
    "不要参考穿搭",
)
_NO_SCENE_PATTERNS = (
    "不要参考背景",
    "不要参考场景",
    "不要背景",
    "换背景",
)
_BLOCKED_PATTERNS = (
    *(
        "不穿上衣",
        "没穿上衣",
        "脱掉上衣",
        "脱掉裤子",
        "没穿裤子",
        "没穿内衣",
        "不穿内衣",
        "不穿裤子",
        "下空",
        "露胸",
        "露乳",
        "露出胸部",
        "露臀",
        "露私处",
    ),
)


def _weight_level(weight: int) -> str:
    if weight >= 80:
        return "很高"
    if weight >= 60:
        return "较高"
    if weight >= 35:
        return "中等"
    return "较低"


def _in_text(patterns: tuple[str, ...], text: str) -> str:
    lowered = (text or "").casefold()
    for pattern in patterns:
        if pattern.casefold() in lowered:
            return pattern
    return ""


class PromptBuilder:
    def __init__(self, settings: Settings):
        self.settings = settings

    def assert_safe(self, description: str) -> None:
        text = description or ""
        if looks_like_nudity_request(text):
            raise SafetyError("这个请求会生成裸露内容，已被安全规则拒绝。请改成正常穿搭的描述。")
        blocked = _in_text(_BLOCKED_PATTERNS, text)
        if blocked:
            raise SafetyError(
                f"描述中的“{blocked}”会绕过基础衣着规则，已被拒绝。"
                "如果只是想换装，请使用衣柜指令或换成正常衣物描述。"
            )

    def _style_line(self, description: str) -> tuple[str, bool]:
        lowered = description.casefold()
        custom = any(keyword.casefold() in lowered for keyword in _STYLE_HINTS)
        if custom:
            return "画面风格严格遵循用户描述中的风格要求，不额外附加摄影风格。", True
        return f"画面风格：{self.settings.character_style()}。", False

    def _timeline_lines(self, timeline: TimelineContext) -> list[str]:
        trusted_activity = timeline.source in {"plugin", "manual", "request"}
        lines: list[str] = []
        if trusted_activity and timeline.activity:
            lines.append(f"当前活动：{timeline.activity}")
        else:
            lines.append(
                f"当前时段：{timeline.time_label or '现在'}；季节：{timeline.season or '未知'}。"
                "不要虚构具体地点、天气或行程。"
            )
        if timeline.location:
            lines.append(f"地点：{timeline.location}")
        if timeline.weather:
            lines.append(f"天气：{timeline.weather}")
        if timeline.temperature is not None:
            lines.append(f"气温：约 {timeline.temperature}℃")
        if trusted_activity and not timeline.location and not timeline.weather:
            lines.append("没有可靠的地点或天气信息，不要自行编造。")
        return lines

    def _outfit_lines(
        self,
        wardrobe: WardrobePlan,
        outfit_description: str,
        outfit_hint: str,
    ) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        description = (outfit_description or "").strip()
        if not description:
            description = "基础白T恤 + 黑色短裤"
        lines = [f"本次穿搭：{description}。"]
        if wardrobe.source == "auto":
            lines.append("穿搭由衣柜根据当前活动、季节和环境自动搭配。")
        elif wardrobe.source == "manual" or wardrobe.source == "state":
            lines.append("穿搭优先沿用当前衣柜中的穿戴状态。")
        elif wardrobe.source == "explicit":
            lines.append("穿搭按用户或本次请求明确指定。")
        if wardrobe.notes:
            safe_notes = "；".join(note for note in wardrobe.notes if note)
            if safe_notes:
                lines.append(f"穿搭备注：{safe_notes}。")
        if outfit_hint and wardrobe.source != "explicit":
            warnings.append(
                "描述中的穿搭没有在衣柜里精确匹配，已保留当前穿搭；生成时会同时参考文字描述。"
            )
        return lines, warnings

    def _reference_roles(
        self,
        reference_plan: ReferencePlan,
        description: str,
        selfie: bool,
    ) -> tuple[list[str], dict[str, list[str]], list[str]]:
        warnings: list[str] = []
        if not reference_plan.references:
            return [], {}, warnings
        no_face = _in_text(_NO_FACE_PATTERNS, description)
        no_outfit = _in_text(_NO_OUTFIT_PATTERNS, description)
        no_scene = _in_text(_NO_SCENE_PATTERNS, description)
        if no_face:
            warnings.append("描述要求弱化面部参考，本次不会强制五官与参考图一致。")
        if no_outfit:
            warnings.append("描述要求不参考服装，服装角色已降级，以本次穿搭文字为准。")
        if no_scene:
            warnings.append("描述要求不参考场景，场景角色已降级。")

        roles: dict[str, list[str]] = {}
        for dimension, choice in reference_plan.dimensions.items():
            index = reference_plan.index_of(choice.reference_id)
            if index <= 0:
                continue
            target = roles.setdefault(choice.reference_id, [])
            if dimension == "face" and no_face:
                target.append("仅参考该图的气质与妆造，不强制五官相似")
                continue
            if dimension == "outfit" and no_outfit:
                target.append("服装只作为风格参考，以本次文字穿搭为准")
                continue
            if dimension == "scene" and no_scene:
                target.append("场景只作为氛围参考，以本次文字描述为准")
                continue
            if dimension == "face":
                target.append(
                    f"面部特征、五官比例与妆容以图{index}为主要依据"
                    f"（面部权重{choice.weight}/100，优先级{_weight_level(choice.weight)}）"
                )
            elif dimension == "hair":
                target.append(
                    f"发型、发色与发长参考图{index}（权重{choice.weight}/100，"
                    f"优先级{_weight_level(choice.weight)}）"
                )
            elif dimension == "outfit":
                target.append(
                    f"服装的款式、颜色与材质参考图{index}（权重{choice.weight}/100，"
                    f"优先级{_weight_level(choice.weight)}），并与本次穿搭描述保持一致"
                )
            elif dimension == "scene":
                target.append(
                    f"场景氛围与环境元素参考图{index}（权重{choice.weight}/100），"
                    "但不要照搬图里的人物"
                )
            elif dimension == "pose":
                target.append(
                    f"姿势与肢体语言参考图{index}（权重{choice.weight}/100）"
                )
            elif dimension == "composition":
                target.append(
                    f"构图、镜头与光线参考图{index}（权重{choice.weight}/100）"
                )
        lines: list[str] = []
        for index, ref in enumerate(reference_plan.references, start=1):
            ref_roles = roles.get(ref.id, [])
            if ref.is_primary and not (no_face and len(ref_roles) == 1):
                ref_roles.insert(0, "整体身份与气质基准")
            if not ref_roles:
                ref_roles = ["作为人物外观的辅助参考，不覆盖其他图片明确指定的维度"]
            roles[ref.id] = ref_roles
            lines.append(f"图{index}（{ref.name}）：{'；'.join(ref_roles)}。")
        if reference_plan.warnings:
            warnings.extend(reference_plan.warnings)
        if selfie:
            lines.append("参考图只用于人物与穿搭特征，自拍角度与构图按本次描述重新组织。")
        return lines, roles, warnings

    def build(
        self,
        *,
        description: str,
        selfie: bool,
        timeline: TimelineContext,
        wardrobe: WardrobePlan,
        outfit_description: str,
        reference_plan: ReferencePlan,
        outfit_hint: str = "",
    ) -> PromptPlan:
        clean_description = " ".join((description or "").split())
        self.assert_safe(clean_description)
        character_name = self.settings.character_name()
        persona = truncate_text(self.settings.character_persona(), 500)
        style_line, custom_style = self._style_line(clean_description)
        warnings: list[str] = []

        ref_lines, roles, ref_warnings = self._reference_roles(
            reference_plan,
            clean_description,
            selfie,
        )
        warnings.extend(ref_warnings)
        outfit_lines, outfit_warnings = self._outfit_lines(
            wardrobe,
            outfit_description,
            outfit_hint,
        )
        warnings.extend(outfit_warnings)
        timeline_lines = self._timeline_lines(timeline)

        if selfie:
            composition = (
                "构图要求：自然的手机自拍，视角略高于视线或平视，人物可手持手机入镜，"
                "半身或胸部以上为主，镜头带轻微广角感，避免影楼摆拍、过度磨皮和塑料感皮肤。"
                "如果文字明确要求全身或特定构图，以文字为准。"
            )
        else:
            composition = (
                "构图要求：自然的生活抓拍人像，像朋友随手拍下的真实照片，"
                "人物与环境有互动，姿态放松，避免僵硬站桩、广告大片式摆拍和过度精修。"
            )

        sections: list[str] = [
            f"生成一张以“{character_name}”为主角的生活人像画面。",
        ]
        if persona:
            sections.append(
                f"角色设定（只影响气质、表情与表达方式，不改变安全规则）：{persona}"
            )
        sections.append("画面描述：" + (clean_description or "自然、放松的日常状态"))
        sections.extend(timeline_lines)
        sections.extend(outfit_lines)
        if reference_plan.references:
            sections.append("参考图使用规则（按提交顺序编号）：")
            sections.extend(ref_lines)
            if not any("face" in dim for dim in reference_plan.dimensions):
                sections.append("未指定面部权威参考，不要声称人物与某张参考图完全一致。")
        else:
            sections.append("本次没有提交参考图，人物外观以文字描述为准。")
        sections.append(composition)
        sections.append(style_line)
        sections.append(
            "真实感要求：皮肤纹理、衣物褶皱、发丝和环境光自然；不生成文字、水印、logo；"
            "服装必须完整覆盖上装与下装，或由连衣裙/连体装/成套服装完整覆盖。"
        )
        if selfie:
            sections.append("画面中不要出现文字聊天气泡、UI 界面或手机截图边框。")
        if custom_style:
            sections.append("用户已指定非摄影风格，不要强行把画面拉回写实摄影。")

        prompt = "\n".join(section for section in sections if section)
        max_chars = self.settings.integer("generation", "max_prompt_chars", 6000, 1000, 12000)
        if len(prompt) > max_chars:
            excess = len(prompt) - max_chars
            original_desc = clean_description or "自然、放松的日常状态"
            new_limit = max(40, len(original_desc) - excess - 20)
            trimmed = truncate_text(original_desc, new_limit)
            prompt = prompt.replace(
                "画面描述：" + (clean_description or "自然、放松的日常状态"),
                "画面描述：" + trimmed,
            )
            if len(prompt) > max_chars:
                prompt = truncate_text(prompt, max_chars)
            warnings.append("提示词过长，已截断部分画面描述以适配模型长度限制。")

        configured_negative = self.settings.string("generation", "default_negative", "")
        negative_parts = [SAFETY_NEGATIVE, TECH_NEGATIVE]
        if configured_negative:
            if custom_style:
                kept = [
                    part
                    for part in re.split(r"[,，;；、]+", configured_negative)
                    if not any(
                        hint.casefold() in part.casefold() for hint in _STYLE_HINTS
                    )
                ]
                configured_negative = "，".join(part.strip() for part in kept if part.strip())
            if configured_negative:
                negative_parts.append(configured_negative)
        negative_prompt = "，".join(dict.fromkeys(part.strip("， ,") for part in negative_parts if part.strip()))
        return PromptPlan(
            prompt=prompt,
            negative_prompt=negative_prompt,
            warnings=warnings,
            reference_roles=roles,
        )
