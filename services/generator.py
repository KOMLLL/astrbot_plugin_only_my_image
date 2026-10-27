from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from .errors import BusyError, ConfigError, PluginError, ValidationError
from .gallery import GalleryService
from .image_api import ImageApiManager, ImagePayload, ImageRequest, ReferenceInput
from .log import logger
from .models import (
    DIMENSION_LABELS,
    WEIGHT_DIMENSIONS,
    GenerationResult,
    GenerationTask,
    PromptPlan,
    ReferencePlan,
    TimelineContext,
    WardrobePlan,
)
from .prompt_builder import PromptBuilder
from .reference_selector import ReferenceSelector, SelectorIntent
from .settings import Settings
from .storage import Storage
from .timeline import TimelineService
from .utils import (
    atomic_write_bytes,
    detect_image,
    new_id,
    now_ts,
    resolve_relative_path,
    safe_relative_path,
    sha256_bytes,
    truncate_text,
)
from .wardrobe import WardrobeService

_SIZE_RE = re.compile(r"^(\d{2,5})\s*[xX×]\s*(\d{2,5})$")
_OUTFIT_RE = re.compile(
    r"(?:穿着|穿了|换上|换成|搭配|穿衣|穿(?!搭))\s*([^，。,.!?！？；;\n]{1,30})"
)


@dataclass(slots=True)
class PreparedGeneration:
    actor_id: str
    state_owner: str
    scopes: list[str]
    description: str
    selfie: bool
    size: str
    mode: str
    outfit_hint: str
    timeline: TimelineContext
    wardrobe: WardrobePlan
    outfit_description: str
    reference_plan: ReferencePlan
    prompt_plan: PromptPlan
    request_key: str
    warnings: list[str] = field(default_factory=list)


class GenerationCoordinator:
    def __init__(
        self,
        storage: Storage,
        gallery: GalleryService,
        wardrobe: WardrobeService,
        selector: ReferenceSelector,
        timeline: TimelineService,
        api: ImageApiManager,
        prompts: PromptBuilder,
        settings: Settings,
        data_dir: Path,
    ):
        self.storage = storage
        self.gallery = gallery
        self.wardrobe = wardrobe
        self.selector = selector
        self.timeline = timeline
        self.api = api
        self.prompts = prompts
        self.settings = settings
        self.data_dir = data_dir
        self.generated_dir = data_dir / "generated"
        self._semaphore = asyncio.Semaphore(settings.max_concurrency())
        self._dedupe_lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Task[GenerationResult]] = {}
        self._workers: set[asyncio.Task[GenerationResult]] = set()
        self._closed = False

    @staticmethod
    def _extract_outfit_hint(description: str) -> str:
        match = _OUTFIT_RE.search(description or "")
        if not match:
            return ""
        value = match.group(1).strip()
        value = re.split(r"(?:在|去|到|拍|自拍|然后|并|还)", value, maxsplit=1)[0].strip()
        return value[:30]

    @staticmethod
    def _normalize_size(value: str | None, default: str) -> str:
        text = (value or default or "").strip().lower().replace("×", "x")
        match = _SIZE_RE.fullmatch(text)
        if not match:
            raise ValidationError("图片尺寸格式应为“宽x高”，例如 1024x1024。")
        width, height = int(match.group(1)), int(match.group(2))
        if not (64 <= width <= 8192 and 64 <= height <= 8192):
            raise ValidationError("图片尺寸超出允许范围（64～8192）。")
        return f"{width}x{height}"

    @staticmethod
    def _normalize_output_format(value: str) -> str:
        text = (value or "").strip().lower()
        if text in {"", "none", "off"}:
            return ""
        if text in {"jpg", "jpeg"}:
            return "jpeg"
        if text not in {"png", "webp", "jpeg"}:
            raise ValidationError("输出格式仅支持 png、jpeg 或 webp。")
        return text

    def _request_key(
        self,
        *,
        actor_id: str,
        description: str,
        selfie: bool,
        size: str,
        outfit_hint: str,
        prepared_refs: Sequence,
        wardrobe: WardrobePlan,
        timeline: TimelineContext,
    ) -> str:
        pieces = [
            actor_id,
            description,
            "selfie" if selfie else "photo",
            size,
            outfit_hint,
            timeline.activity,
            timeline.season,
            timeline.location,
            "|".join(item.id for item in wardrobe.items),
            "|".join(f"{ref.id}:{ref.updated_at}" for ref in prepared_refs),
        ]
        return hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()

    async def prepare(
        self,
        *,
        actor_id: str,
        state_owner: str,
        scopes: Sequence[str],
        description: str,
        selfie: bool = False,
        size: str | None = None,
        outfit_hint: str = "",
        scene: str = "",
    ) -> PreparedGeneration:
        scope_list = list(dict.fromkeys([state_owner, *scopes]))
        clean_description = " ".join((description or "").split())[:1200]
        clean_description = clean_description or (
            "自然放松的自拍，记录当下的日常状态" if selfie else "自然放松的日常人像照片"
        )
        self.prompts.assert_safe(clean_description)
        if outfit_hint:
            self.prompts.assert_safe(outfit_hint)
        explicit_outfit = (outfit_hint or "").strip()
        if not explicit_outfit:
            explicit_outfit = self._extract_outfit_hint(clean_description)
        if explicit_outfit:
            self.prompts.assert_safe(explicit_outfit)

        timeline = await self.timeline.get_context(state_owner)
        resolved_size = self._normalize_size(
            size,
            self.settings.selfie_size() if selfie else self.settings.default_size(),
        )
        wardrobe_plan = await self.wardrobe.plan_for_generation(
            state_owner,
            scope_list,
            outfit_hint=explicit_outfit,
            activity=timeline.activity if timeline.source in {"plugin", "manual", "request"} else "",
            season=timeline.season,
            temperature=timeline.temperature,
            indoor=timeline.indoor,
            scene=scene or clean_description,
            weather=timeline.weather,
        )
        outfit_description = await self.wardrobe.outfit_description(
            state_owner,
            scope_list,
            wardrobe_plan.items,
        )

        references, gallery_warnings = await self.gallery.list_available(scope_list)
        if not references and self.settings.require_reference():
            raise ConfigError("图库中没有可用的参考图，而当前配置要求必须使用参考图。")
        primary = await self.gallery.primary_reference(scope_list)
        linked_ids = [
            item.reference_id for item in wardrobe_plan.items if item.reference_id
        ]
        reference_plan = self.selector.select(
            references,
            intent=SelectorIntent(
                description=clean_description,
                activity=timeline.activity if timeline.source in {"plugin", "manual", "request"} else "",
                scene=scene,
                outfit=explicit_outfit,
                selfie=selfie,
            ),
            primary_id=primary.id if primary else "",
            linked_ids=linked_ids,
        )
        prompt_plan = self.prompts.build(
            description=clean_description,
            selfie=selfie,
            timeline=timeline,
            wardrobe=wardrobe_plan,
            outfit_description=outfit_description,
            reference_plan=reference_plan,
            outfit_hint=explicit_outfit,
        )

        warnings: list[str] = []
        for message in [
            *gallery_warnings,
            *wardrobe_plan.notes,
            *reference_plan.warnings,
            *prompt_plan.warnings,
        ]:
            if message and message not in warnings:
                warnings.append(message)
        endpoints = await self.api.enabled_endpoints()
        if not any(endpoint.api_key for endpoint in endpoints):
            warnings.append("当前没有配置 API Key；如果生图接口需要鉴权，调用会失败。")

        request_key = self._request_key(
            actor_id=actor_id,
            description=clean_description,
            selfie=selfie,
            size=resolved_size,
            outfit_hint=explicit_outfit,
            prepared_refs=reference_plan.references,
            wardrobe=wardrobe_plan,
            timeline=timeline,
        )
        return PreparedGeneration(
            actor_id=actor_id,
            state_owner=state_owner,
            scopes=scope_list,
            description=clean_description,
            selfie=selfie,
            size=resolved_size,
            mode="selfie" if selfie else "photo",
            outfit_hint=explicit_outfit,
            timeline=timeline,
            wardrobe=wardrobe_plan,
            outfit_description=outfit_description,
            reference_plan=reference_plan,
            prompt_plan=prompt_plan,
            request_key=request_key,
            warnings=warnings,
        )

    async def generate(self, prepared: PreparedGeneration) -> GenerationResult:
        if self._closed:
            raise BusyError("插件正在卸载，暂时不能生成图片。")
        key = prepared.request_key
        deduplicated = False
        async with self._dedupe_lock:
            existing_future = self._inflight.get(key)
            if existing_future is not None:
                future = existing_future
                deduplicated = True
            else:
                recent = await self._find_recent(prepared)
                if recent is not None:
                    _task, result = recent
                    if result is not None:
                        return replace(result, deduplicated=True)
                    raise BusyError("相同的生图请求正在处理中，请稍候。")
                task_id = new_id("task")
                future = asyncio.create_task(self._run_worker(prepared, task_id))
                self._inflight[key] = future
                self._workers.add(future)
                future.add_done_callback(lambda _fut: self._forget(key, _fut))
        result = await asyncio.shield(future)
        return replace(result, deduplicated=deduplicated) if not result.deduplicated else result

    def _forget(self, key: str, future: asyncio.Task[GenerationResult]) -> None:
        self._workers.discard(future)
        if self._inflight.get(key) is future:
            self._inflight.pop(key, None)

    async def _find_recent(
        self,
        prepared: PreparedGeneration,
    ) -> tuple[GenerationTask, GenerationResult | None] | None:
        window = self.settings.dedupe_window()
        if window <= 0:
            return None
        task = await self.storage.find_recent_task(
            prepared.actor_id,
            prepared.request_key,
            since=now_ts() - window,
            statuses=("succeeded", "running", "pending"),
        )
        if task is None:
            return None
        if task.status == "succeeded" and task.result_path:
            try:
                path = resolve_relative_path(self.data_dir, task.result_path)
            except ValidationError:
                return None
            if path.is_file():
                return task, GenerationResult(
                    task_id=task.id,
                    image_path=str(path),
                    reply_text=self._duplicate_reply(prepared),
                    endpoint_name=task.endpoint_name,
                    deduplicated=True,
                )
        return task, None

    def _duplicate_reply(self, prepared: PreparedGeneration) -> str:
        return (
            f"刚刚已经生成过相同的画面，这次直接复用上一张，避免重复调用和计费。\n"
            f"穿搭：{prepared.outfit_description}"
        )

    async def _run_worker(
        self,
        prepared: PreparedGeneration,
        task_id: str,
    ) -> GenerationResult:
        stamp = now_ts()
        task = GenerationTask(
            id=task_id,
            owner_id=prepared.actor_id,
            request_key=prepared.request_key,
            status="pending",
            mode=prepared.mode,
            description=truncate_text(prepared.description, 500),
            prompt=truncate_text(prepared.prompt_plan.prompt, 4000),
            negative_prompt=truncate_text(prepared.prompt_plan.negative_prompt, 1000),
            size=prepared.size,
            created_at=stamp,
            updated_at=stamp,
        )
        await self.storage.create_task(task)
        acquired = False
        saved_path: Path | None = None
        try:
            queue_timeout = self.settings.integer("generation", "queue_timeout", 120, 5, 600)
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=queue_timeout)
                acquired = True
            except TimeoutError as exc:
                raise BusyError("生图队列繁忙，请稍后再试。") from exc
            await self.storage.update_task(task_id, status="running")
            payload = await self._call_api(prepared)
            saved_path = await self._save_image(task_id, prepared.actor_id, payload.content, payload.ext)
            relative = safe_relative_path(self.data_dir, saved_path)
            try:
                updated = await self.storage.update_task(
                    task_id,
                    status="succeeded",
                    endpoint_name=payload.endpoint_name,
                    result_path=relative,
                    error="",
                )
            except BaseException:
                try:
                    saved_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            if not updated:
                try:
                    saved_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise PluginError("生成结果保存失败。")
            reply = self._reply_text(prepared)
            logger.info(
                "生图成功 task=%s endpoint=%s images=%s",
                task_id,
                payload.endpoint_name,
                len(prepared.reference_plan.references),
            )
            return GenerationResult(
                task_id=task_id,
                image_path=str(saved_path),
                reply_text=reply,
                endpoint_name=payload.endpoint_name,
            )
        except asyncio.CancelledError:
            await self._mark_failed(task_id, "任务已取消（插件卸载或关闭）")
            if saved_path is not None:
                try:
                    saved_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        except PluginError as exc:
            await self._mark_failed(task_id, exc.user_message)
            raise
        except Exception as exc:
            logger.exception("生图任务异常 task=%s error=%s", task_id, type(exc).__name__)
            await self._mark_failed(task_id, "生成失败，请稍后重试。")
            raise PluginError("生成失败，请稍后重试。") from exc
        finally:
            if acquired:
                self._semaphore.release()

    async def _mark_failed(self, task_id: str, message: str) -> None:
        try:
            await self.storage.update_task(
                task_id,
                status="failed",
                error=truncate_text(message, 300),
            )
        except Exception:  # noqa: BLE001 - 失败状态写入失败不能掩盖原始异常
            logger.warning("更新生图任务失败状态时出错 task=%s", task_id)

    async def _call_api(self, prepared: PreparedGeneration) -> ImagePayload:
        refs: list[ReferenceInput] = []
        for index, reference in enumerate(prepared.reference_plan.references, start=1):
            try:
                path = self.gallery.resolve_file(reference)
            except ValidationError as exc:
                raise PluginError(f"参考图《{reference.name}》路径异常，已停止生成。") from exc
            if not path.is_file():
                raise PluginError(f"参考图《{reference.name}》文件缺失，已停止生成。")
            data = await asyncio.to_thread(path.read_bytes)
            try:
                info = detect_image(data)
            except ValidationError as exc:
                raise PluginError(f"参考图《{reference.name}》内容损坏，已停止生成。") from exc
            if len(data) > self.settings.integer("gallery", "max_image_mb", 10, 1, 50) * 1024 * 1024:
                raise PluginError(f"参考图《{reference.name}》超过大小限制。")
            refs.append(
                ReferenceInput(
                    filename=f"ref-{index}.{info.ext}",
                    data=data,
                    mime=info.mime,
                )
            )
        if not refs and self.settings.require_reference():
            raise PluginError("本次没有可提交的参考图，已停止生成。")
        quality = self.settings.string("api", "quality", "auto")
        output_format = self._normalize_output_format(
            self.settings.string("api", "output_format", "png")
        )
        return await self.api.generate(
            ImageRequest(
                prompt=prepared.prompt_plan.prompt,
                negative_prompt=prepared.prompt_plan.negative_prompt,
                size=prepared.size,
                quality=quality,
                output_format=output_format,
                references=refs,
            )
        )

    async def _save_image(
        self,
        task_id: str,
        actor_id: str,
        data: bytes,
        ext: str,
    ) -> Path:
        max_bytes = self.settings.integer("generation", "max_result_mb", 30, 1, 100) * 1024 * 1024
        if len(data) > max_bytes:
            raise PluginError("生成图片超过大小限制。")
        scope = "shared" if actor_id == "shared" else f"u_{sha256_bytes(actor_id.encode('utf-8'))[:12]}"
        directory = self.generated_dir / scope
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{task_id}.{ext}"
        resolve_relative_path(self.data_dir, safe_relative_path(self.data_dir, target))
        try:
            await asyncio.to_thread(atomic_write_bytes, target, data)
        except OSError as exc:
            raise PluginError("保存生成图片失败，请检查磁盘空间。") from exc
        return target

    def _reply_text(self, prepared: PreparedGeneration) -> str:
        name = self.settings.character_name()
        activity_line = ""
        if prepared.timeline.source in {"plugin", "manual", "request"} and prepared.timeline.activity:
            activity_line = f"画面按“{prepared.timeline.activity}”的当下状态来拍。"
        elif prepared.timeline.time_label:
            activity_line = f"用了{prepared.timeline.time_label}的自然光线。"
        if prepared.selfie:
            opening = f"{name}的自拍生成好啦。"
        else:
            opening = f"{name}的照片生成好啦。"
        reference_line = ""
        if prepared.reference_plan.references:
            dimensions = []
            for dimension in WEIGHT_DIMENSIONS:
                choice = prepared.reference_plan.dimensions.get(dimension)
                if choice:
                    dimensions.append(f"{DIMENSION_LABELS[dimension]} {choice.weight}")
            details = "、".join(dimensions[:4]) if dimensions else "整体气质"
            reference_line = f"本次提交了 {len(prepared.reference_plan.references)} 张参考图（{details}）。"
        lines = [opening, activity_line, f"穿搭：{prepared.outfit_description}。", reference_line]
        visible_warnings = [
            warning
            for warning in prepared.warnings
            if "API Key" not in warning and len(warning) < 80
        ]
        if visible_warnings:
            lines.append("提示：" + "；".join(visible_warnings[:2]))
        return "\n".join(line for line in lines if line).strip()

    def format_preview(self, prepared: PreparedGeneration) -> str:
        timeline = prepared.timeline
        lines = [
            "【生图预览】（尚未调用生图接口）",
            f"模式：{'自拍' if prepared.selfie else '照片'}｜尺寸：{prepared.size}",
            f"活动：{timeline.describe()}",
            f"时间轴来源：{timeline.source}" + (f"（{timeline.note}）" if timeline.note else ""),
            f"穿搭：{prepared.outfit_description}",
            f"穿搭来源：{prepared.wardrobe.source}",
        ]
        if prepared.reference_plan.references:
            lines.append("参考图顺序与用途：")
            for index, ref in enumerate(prepared.reference_plan.references, start=1):
                labels = []
                for dimension, choice in prepared.reference_plan.dimensions.items():
                    if choice.reference_id == ref.id:
                        labels.append(f"{DIMENSION_LABELS[dimension]}={choice.weight}")
                primary = "｜主图" if ref.is_primary else ""
                lines.append(
                    f"  图{index}《{ref.name}》{primary}｜{('，'.join(labels) or '整体气质辅助')}"
                )
        else:
            lines.append("参考图：无（本次仅依据文字描述）。")
        if prepared.warnings:
            lines.append("警告：")
            lines.extend(f"  - {warning}" for warning in prepared.warnings[:8])
        lines.append("提示词：")
        lines.append(prepared.prompt_plan.prompt)
        lines.append("负向约束：")
        lines.append(prepared.prompt_plan.negative_prompt)
        return "\n".join(lines)

    async def close(self) -> None:
        self._closed = True
        workers = list(self._workers)
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()
        self._inflight.clear()

    async def recent_tasks(self, actor_id: str, limit: int = 5) -> list[GenerationTask]:
        return await self.storage.recent_tasks(actor_id, limit)
