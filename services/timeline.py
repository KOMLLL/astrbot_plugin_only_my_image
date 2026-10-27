from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from .log import logger
from .models import TimelineContext
from .settings import Settings
from .storage import Storage
from .utils import parse_bool, parse_temperature_range

TimelineResolver = Callable[[str, str], Awaitable[Any] | Any]

_MANUAL_KEY = "timeline_manual"


def _season_for_month(month: int, southern: bool) -> str:
    if month in (3, 4, 5):
        season = "春"
    elif month in (6, 7, 8):
        season = "夏"
    elif month in (9, 10, 11):
        season = "秋"
    else:
        season = "冬"
    if southern:
        season = {"春": "秋", "夏": "冬", "秋": "春", "冬": "夏"}[season]
    return season


def _time_label(hour: int) -> str:
    if hour < 5:
        return "凌晨"
    if hour < 8:
        return "清晨"
    if hour < 11:
        return "上午"
    if hour < 13:
        return "中午"
    if hour < 17:
        return "下午"
    if hour < 19:
        return "傍晚"
    if hour < 23:
        return "晚上"
    return "深夜"


class TimelineService:
    def __init__(
        self,
        storage: Storage,
        settings: Settings,
        *,
        resolver: TimelineResolver | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.storage = storage
        self.settings = settings
        self.resolver = resolver
        self.clock = clock or time.time

    def _now(self) -> datetime:
        offset = self.settings.timezone_offset()
        return datetime.fromtimestamp(self.clock(), tz=timezone(timedelta(hours=offset)))

    async def get_manual(self, scope: str) -> dict[str, Any] | None:
        raw = await self.storage.get_meta(f"{_MANUAL_KEY}:{scope}", "")
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    async def set_manual(
        self,
        scope: str,
        *,
        activity: str,
        location: str = "",
        weather: str = "",
        temperature: object = None,
        indoor: object = None,
        season: str = "",
    ) -> dict[str, Any]:
        data = {
            "activity": " ".join((activity or "").split())[:80],
            "location": " ".join((location or "").split())[:60],
            "weather": " ".join((weather or "").split())[:40],
            "temperature": temperature,
            "indoor": indoor,
            "season": (season or "").strip()[:10],
            "updated_at": time.time(),
        }
        if not data["activity"]:
            data["activity"] = "日常状态"
        await self.storage.set_meta(
            f"{_MANUAL_KEY}:{scope}",
            json.dumps(data, ensure_ascii=False),
        )
        return data

    async def clear_manual(self, scope: str) -> None:
        await self.storage.delete_meta(f"{_MANUAL_KEY}:{scope}")

    async def _call_adapter(self) -> tuple[dict[str, Any] | None, str]:
        plugin_name = self.settings.string("timeline", "plugin_name", "")
        function_name = self.settings.string("timeline", "function_name", "")
        mode = self.settings.string("timeline", "mode", "auto").lower()
        if mode == "off":
            return None, "时间轴适配已关闭。"
        if not plugin_name or not function_name or self.resolver is None:
            return None, "未配置可用的时间轴适配器。"
        try:
            result = self.resolver(plugin_name, function_name)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=3.0)
        except TimeoutError:
            return None, "时间轴适配器超时，已使用本地时间回退。"
        except Exception as exc:  # noqa: BLE001 - 外部插件适配器不可信，失败必须回退
            logger.warning("时间轴适配器调用失败：%s", type(exc).__name__)
            return None, "时间轴适配器调用失败，已使用本地时间回退。"
        if isinstance(result, Mapping):
            return dict(result), ""
        if isinstance(result, str) and result.strip():
            return {"activity": result.strip()}, ""
        return None, "时间轴适配器未返回有效数据，已使用本地时间回退。"

    @staticmethod
    def _parse_temperature(value: Any) -> int | None:
        if value is None or value == "":
            return None
        low, high = parse_temperature_range(value)
        if low is None or high is None:
            return None
        value_int = round((low + high) / 2)
        return value_int if -60 <= value_int <= 60 else None

    @staticmethod
    def _parse_optional_bool(value: Any) -> bool | None:
        if value is None or value == "":
            return None
        return parse_bool(value, False)

    def _fallback_context(self, now: datetime) -> TimelineContext:
        season = _season_for_month(now.month, self.settings.southern_hemisphere())
        return TimelineContext(
            activity="日常状态",
            location="",
            weather="",
            temperature=None,
            indoor=None,
            season=season,
            time_label=_time_label(now.hour),
            source="time",
            note="未接入时间轴插件，仅依据本地时间推测时段，未虚构具体地点、天气或行程。",
            now_label=now.strftime("%Y-%m-%d %H:%M"),
        )

    def _from_mapping(
        self,
        data: Mapping[str, Any],
        *,
        now: datetime,
        source: str,
        note: str = "",
    ) -> TimelineContext:
        fallback = self._fallback_context(now)
        activity = ""
        for key in ("activity", "current_activity", "doing", "task"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                activity = " ".join(value.split())[:80]
                break
        location = ""
        for key in ("location", "place", "where"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                location = " ".join(value.split())[:60]
                break
        weather = ""
        for key in ("weather", "condition"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                weather = " ".join(value.split())[:40]
                break
        temperature = None
        for key in ("temperature", "temp"):
            if key in data:
                temperature = self._parse_temperature(data.get(key))
                if temperature is not None:
                    break
        indoor = None
        for key in ("indoor", "is_indoor"):
            if key in data:
                indoor = self._parse_optional_bool(data.get(key))
                if indoor is not None:
                    break
        season = str(data.get("season") or "").strip()[:10] or fallback.season
        time_label = str(data.get("time_label") or "").strip()[:20] or fallback.time_label
        return TimelineContext(
            activity=activity or fallback.activity,
            location=location,
            weather=weather,
            temperature=temperature,
            indoor=indoor,
            season=season,
            time_label=time_label,
            source=source,
            note=note,
            now_label=now.strftime("%Y-%m-%d %H:%M"),
        )

    def _config_manual(self) -> dict[str, Any] | None:
        cfg = self.settings.timeline
        activity = str(cfg.get("manual_activity") or "").strip()
        location = str(cfg.get("manual_location") or "").strip()
        weather = str(cfg.get("manual_weather") or "").strip()
        temperature = cfg.get("manual_temperature")
        if not any((activity, location, weather, temperature not in (None, ""))):
            return None
        return {
            "activity": activity,
            "location": location,
            "weather": weather,
            "temperature": temperature,
            "indoor": cfg.get("manual_indoor", ""),
            "season": cfg.get("manual_season", ""),
        }

    async def get_context(
        self,
        scope: str,
        *,
        override_activity: str = "",
    ) -> TimelineContext:
        now = self._now()
        if override_activity.strip():
            context = self._fallback_context(now)
            context.activity = " ".join(override_activity.split())[:80]
            context.source = "request"
            context.note = "活动来自本次请求描述。"
            return context

        manual = await self.get_manual(scope)
        if not manual:
            manual = self._config_manual()
        mode = self.settings.string("timeline", "mode", "auto").lower()
        if manual:
            context = self._from_mapping(
                manual,
                now=now,
                source="manual",
                note="活动来自手动设置。",
            )
            return context

        if mode in {"auto", "adapter"}:
            data, note = await self._call_adapter()
            if data:
                return self._from_mapping(
                    data,
                    now=now,
                    source="plugin",
                    note="活动来自已配置的时间轴插件。",
                )
            fallback = self._fallback_context(now)
            fallback.note = note or fallback.note
            return fallback

        return self._fallback_context(now)

    async def status(self, scope: str) -> str:
        manual = await self.get_manual(scope)
        if manual:
            return "手动时间轴：" + str(manual.get("activity") or "日常状态")
        mode = self.settings.string("timeline", "mode", "auto").lower()
        if mode in {"auto", "adapter"} and self.settings.string("timeline", "plugin_name", ""):
            return "自动时间轴适配（失败时回退本地时段）"
        return "本地时段回退（不虚构地点与天气）"
