from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

WEIGHT_DIMENSIONS: tuple[str, ...] = (
    "face",
    "hair",
    "outfit",
    "scene",
    "pose",
    "composition",
)

DIMENSION_LABELS: dict[str, str] = {
    "face": "面部",
    "hair": "发型",
    "outfit": "服装",
    "scene": "场景",
    "pose": "姿势",
    "composition": "构图光影",
}

DIMENSION_ALIASES: dict[str, str] = {
    "脸": "face",
    "脸部": "face",
    "面部": "face",
    "五官": "face",
    "脸型": "face",
    "头发": "hair",
    "发型": "hair",
    "发色": "hair",
    "服装": "outfit",
    "衣服": "outfit",
    "穿搭": "outfit",
    "服饰": "outfit",
    "场景": "scene",
    "背景": "scene",
    "环境": "scene",
    "姿势": "pose",
    "动作": "pose",
    "姿态": "pose",
    "构图": "composition",
    "光影": "composition",
    "运镜": "composition",
    "镜头": "composition",
    "全部": "all",
}

DEFAULT_WEIGHTS: dict[str, int] = {
    "face": 80,
    "hair": 60,
    "outfit": 60,
    "scene": 45,
    "pose": 40,
    "composition": 40,
}

QUICK_WEIGHTS: dict[str, int] = {
    "高": 85,
    "中": 60,
    "低": 35,
    "关": 0,
    "关闭": 0,
    "无": 0,
}

BASE_OUTFIT_LABEL = "基础白T恤 + 黑色短裤"


def empty_weights() -> dict[str, int]:
    return {key: DEFAULT_WEIGHTS[key] for key in WEIGHT_DIMENSIONS}


def normalize_dimension(raw: str) -> str | None:
    text = (raw or "").strip().lower()
    if text in WEIGHT_DIMENSIONS:
        return text
    return DIMENSION_ALIASES.get(text)


def coerce_weight(raw: Any) -> int:
    if isinstance(raw, bool):
        return 100 if raw else 0
    if isinstance(raw, (int, float)):
        return max(0, min(100, int(raw)))
    text = str(raw).strip()
    if text in QUICK_WEIGHTS:
        return QUICK_WEIGHTS[text]
    text = text.removesuffix("%")
    try:
        return max(0, min(100, int(float(text))))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法识别权重：{raw}") from exc


def normalize_weights(raw: Any) -> dict[str, int]:
    weights = empty_weights()
    if isinstance(raw, dict):
        for key, value in raw.items():
            dim = normalize_dimension(str(key))
            if dim in WEIGHT_DIMENSIONS:
                try:
                    weights[dim] = coerce_weight(value)
                except ValueError:
                    continue
    return weights


@dataclass(slots=True)
class ReferenceImage:
    id: str
    owner_id: str
    name: str
    file_path: str
    sha256: str = ""
    mime: str = ""
    note: str = ""
    tags: list[str] = field(default_factory=list)
    weights: dict[str, int] = field(default_factory=empty_weights)
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    is_primary: bool = False
    enabled: bool = True
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def display_scope(self) -> str:
        return "共享" if self.owner_id == "shared" else "个人"

    def dimension_weight(self, dimension: str) -> int:
        return int(self.weights.get(dimension, 0))


@dataclass(slots=True)
class WardrobeCategory:
    key: str
    label: str
    kind: str
    owner_id: str = "shared"
    custom: bool = False
    sort_order: int = 100


@dataclass(slots=True)
class ClothingItem:
    id: str
    owner_id: str
    name: str
    category_key: str
    color: str = ""
    material: str = ""
    seasons: list[str] = field(default_factory=list)
    scenes: list[str] = field(default_factory=list)
    temp_min: int | None = None
    temp_max: int | None = None
    formality: int = 3
    indoor_outdoor: str = "both"
    recommended_with: list[str] = field(default_factory=list)
    forbidden_with: list[str] = field(default_factory=list)
    priority: int = 50
    enabled: bool = True
    reference_id: str = ""
    note: str = ""
    is_base: bool = False
    base_role: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def matches_temperature(self, temperature: int | None) -> bool:
        if temperature is None or (self.temp_min is None and self.temp_max is None):
            return True
        low = self.temp_min if self.temp_min is not None else -99
        high = self.temp_max if self.temp_max is not None else 99
        return low <= temperature <= high


@dataclass(slots=True)
class Outfit:
    id: str
    owner_id: str
    name: str
    item_ids: list[str] = field(default_factory=list)
    note: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(slots=True)
class WearState:
    owner_id: str
    worn_ids: list[str] = field(default_factory=list)
    history: list[list[str]] = field(default_factory=list)
    mode: str = "auto"
    updated_at: float = 0.0


@dataclass(slots=True)
class ApiEndpoint:
    id: str
    name: str
    base_url: str
    model: str
    api_key: str = ""
    enabled: bool = True
    priority: int = 100
    timeout: float = 180.0
    edit_field: str = "image[]"
    extra: dict[str, Any] = field(default_factory=dict)
    is_primary: bool = False
    last_ok_at: float | None = None
    last_error: str = ""
    fail_count: int = 0
    created_at: float = 0.0


@dataclass(slots=True)
class RequestLog:
    id: str
    endpoint_name: str
    task_id: str = ""
    ok: bool = False
    status_code: int | None = None
    message: str = ""
    created_at: float = 0.0


@dataclass(slots=True)
class GenerationTask:
    id: str
    owner_id: str
    request_key: str
    status: str
    mode: str = "photo"
    description: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    size: str = ""
    endpoint_name: str = ""
    result_path: str = ""
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(slots=True)
class SelectedReference:
    reference: ReferenceImage
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DimensionChoice:
    dimension: str
    reference_id: str = ""
    reference_name: str = ""
    weight: int = 0
    score: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class ReferencePlan:
    references: list[ReferenceImage] = field(default_factory=list)
    dimensions: dict[str, DimensionChoice] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def index_of(self, reference_id: str) -> int:
        for index, reference in enumerate(self.references, start=1):
            if reference.id == reference_id:
                return index
        return 0


@dataclass(slots=True)
class WardrobePlan:
    items: list[ClothingItem] = field(default_factory=list)
    source: str = "auto"
    notes: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    onepiece: ClothingItem | None = None

    def describe(self) -> str:
        if not self.items:
            return BASE_OUTFIT_LABEL
        parts = []
        for item in self.items:
            detail = item.name
            if item.color:
                detail = f"{item.color}{detail}"
            parts.append(detail)
        return "、".join(parts)


@dataclass(slots=True)
class TimelineContext:
    activity: str
    location: str = ""
    weather: str = ""
    temperature: int | None = None
    indoor: bool | None = None
    season: str = ""
    time_label: str = ""
    source: str = "fallback"
    note: str = ""
    now_label: str = ""

    def describe(self, include_location: bool = True) -> str:
        parts = [self.activity] if self.activity else []
        if include_location and self.location:
            parts.append(f"地点：{self.location}")
        if self.weather:
            parts.append(f"天气：{self.weather}")
        if self.temperature is not None:
            parts.append(f"约{self.temperature}℃")
        if self.time_label:
            parts.append(self.time_label)
        if self.season:
            parts.append(self.season)
        return "，".join(part for part in parts if part)


@dataclass(slots=True)
class PromptPlan:
    prompt: str
    negative_prompt: str
    warnings: list[str] = field(default_factory=list)
    reference_roles: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    task_id: str
    image_path: str
    reply_text: str
    endpoint_name: str = ""
    deduplicated: bool = False
