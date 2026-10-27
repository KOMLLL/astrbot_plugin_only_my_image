from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .utils import clamp_int, parse_bool, safe_float, split_list


class Settings:
    """对 AstrBot 插件配置的只读、防御性访问封装。"""

    def __init__(self, raw: Mapping[str, Any] | None):
        self.raw: dict[str, Any] = dict(raw or {})

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        return dict(value) if isinstance(value, Mapping) else {}

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.section(section).get(key, default)

    def string(self, section: str, key: str, default: str = "") -> str:
        value = self.get(section, key, default)
        return value.strip() if isinstance(value, str) else default

    def boolean(self, section: str, key: str, default: bool = False) -> bool:
        return parse_bool(self.get(section, key, default), default)

    def integer(
        self,
        section: str,
        key: str,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        return clamp_int(self.get(section, key, default), default, minimum, maximum)

    def number(
        self,
        section: str,
        key: str,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return safe_float(self.get(section, key, default), default, minimum, maximum)

    def string_list(self, section: str, key: str) -> list[str]:
        value = self.get(section, key, [])
        if isinstance(value, str):
            return split_list(value)
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @property
    def api(self) -> dict[str, Any]:
        return self.section("api")

    @property
    def gallery(self) -> dict[str, Any]:
        return self.section("gallery")

    @property
    def wardrobe(self) -> dict[str, Any]:
        return self.section("wardrobe")

    @property
    def generation(self) -> dict[str, Any]:
        return self.section("generation")

    @property
    def timeline(self) -> dict[str, Any]:
        return self.section("timeline")

    @property
    def character(self) -> dict[str, Any]:
        return self.section("character")

    @property
    def permissions(self) -> dict[str, Any]:
        return self.section("permissions")

    @property
    def features(self) -> dict[str, Any]:
        return self.section("features")

    def api_base_url(self) -> str:
        return self.string("api", "base_url", "https://api1.weilai.chat/v1")

    def api_model(self) -> str:
        return self.string("api", "model", "gpt-image-2")

    def api_key(self) -> str:
        return self.string("api", "api_key", "")

    def api_timeout(self) -> float:
        return self.number("api", "timeout", 180.0, 10.0, 600.0)

    def api_max_retries(self) -> int:
        return self.integer("api", "max_retries", 2, 0, 5)

    def default_size(self) -> str:
        return self.string("generation", "size", "1536x1024")

    def selfie_size(self) -> str:
        return self.string("generation", "selfie_size", "1024x1536")

    def max_concurrency(self) -> int:
        return self.integer("generation", "max_concurrency", 2, 1, 10)

    def max_reference_images(self) -> int:
        return self.integer("generation", "max_reference_images", 6, 1, 16)

    def require_reference(self) -> bool:
        return self.boolean("api", "require_reference", False)

    def dedupe_window(self) -> float:
        return self.number("generation", "dedupe_window", 20.0, 0.0, 600.0)

    def allow_user_generation(self) -> bool:
        return self.boolean("generation", "allow_user_generation", True)

    def allow_user_upload(self) -> bool:
        return self.boolean("gallery", "allow_user_upload", True)

    def allow_user_wardrobe_edit(self) -> bool:
        return self.boolean("wardrobe", "allow_user_edit", True)

    def on_llm_request_hint(self) -> bool:
        return self.boolean("features", "on_llm_request_hint", True)

    def preview_by_default(self) -> bool:
        return self.boolean("features", "preview_by_default", False)

    def admin_ids(self) -> set[str]:
        values = self.string_list("permissions", "admin_ids")
        also = self.get("permissions", "admin_ids", "")
        if isinstance(also, str):
            values.extend(split_list(also))
        return {str(value).strip() for value in values if str(value).strip()}

    def timezone_offset(self) -> int:
        return self.integer("timeline", "timezone_offset", 8, -12, 14)

    def southern_hemisphere(self) -> bool:
        return self.boolean("timeline", "southern_hemisphere", False)

    def allowed_image_extensions(self) -> set[str]:
        from .utils import normalize_extensions

        value = self.get("gallery", "allowed_formats", "png,jpg,jpeg,webp")
        allowed = normalize_extensions(value)
        return allowed or {"png", "jpg", "webp"}

    def character_name(self) -> str:
        return self.string("character", "name", "我") or "我"

    def character_persona(self) -> str:
        return self.string("character", "persona", "")

    def character_style(self) -> str:
        return self.string("character", "style", "自然、真实、有生活感的人像照片")
