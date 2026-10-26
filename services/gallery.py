from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

from .errors import NotFoundError, ValidationError
from .log import logger
from .models import (
    DEFAULT_WEIGHTS,
    ReferenceImage,
    normalize_weights,
)
from .settings import Settings
from .storage import Storage
from .utils import (
    atomic_write_bytes,
    new_id,
    normalize_tags,
    now_ts,
    resolve_relative_path,
    safe_relative_path,
    sha256_bytes,
    validate_image_bytes,
)


def _clean_text(value: str, *, limit: int) -> str:
    text = " ".join(str(value or "").replace("\x00", "").split())
    return text[:limit]


class GalleryService:
    def __init__(self, storage: Storage, data_dir: Path, settings: Settings):
        self.storage = storage
        self.data_dir = data_dir
        self.settings = settings
        self.references_dir = data_dir / "references"

    def _scope_dir(self, owner_id: str) -> Path:
        safe_scope = "shared" if owner_id == "shared" else f"u_{sha256_bytes(owner_id.encode('utf-8'))[:12]}"
        return self.references_dir / safe_scope

    def resolve_file(self, reference: ReferenceImage) -> Path:
        return resolve_relative_path(self.data_dir, reference.file_path)

    async def save_reference(
        self,
        owner_id: str,
        *,
        name: str,
        note: str,
        tags: object,
        data: bytes,
        weights: object = None,
        make_primary: bool = False,
    ) -> ReferenceImage:
        display_name = _clean_text(name, limit=40)
        if not display_name:
            raise ValidationError("参考图名称不能为空。")
        clean_note = _clean_text(note, limit=500)
        clean_tags = normalize_tags(tags)

        max_count = self.settings.integer("gallery", "max_images_per_scope", 80, 1, 1000)
        if await self.storage.count_references(owner_id) >= max_count:
            raise ValidationError(f"当前图库已达到上限（{max_count} 张），请先删除旧图。")

        max_mb = self.settings.integer("gallery", "max_image_mb", 10, 1, 50)
        allowed = self.settings.allowed_image_extensions()
        info = validate_image_bytes(
            data,
            max_bytes=max_mb * 1024 * 1024,
            allowed_exts=allowed,
        )
        digest = sha256_bytes(data)
        duplicate = await self.storage.find_reference_by_sha(owner_id, digest)
        if duplicate:
            raise ValidationError(f"这张图片已存在（{duplicate.name}），无需重复上传。")

        existing = await self.storage.list_references([owner_id], limit=max_count)
        is_primary = make_primary or not existing
        ref_id = new_id("ref")
        target = self._scope_dir(owner_id) / f"{ref_id}.{info.ext}"
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(atomic_write_bytes, target, data)
        stamp = now_ts()
        reference = ReferenceImage(
            id=ref_id,
            owner_id=owner_id,
            name=display_name,
            note=clean_note,
            tags=clean_tags,
            weights=normalize_weights(weights or DEFAULT_WEIGHTS),
            file_path=safe_relative_path(self.data_dir, target),
            sha256=digest,
            mime=info.mime,
            width=info.width,
            height=info.height,
            size_bytes=len(data),
            is_primary=is_primary,
            enabled=True,
            created_at=stamp,
            updated_at=stamp,
        )
        try:
            await self.storage.add_reference(reference)
        except BaseException:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                logger.warning("清理未入库的参考图文件失败：%s", target.name)
            raise
        if not is_primary:
            await self.storage.ensure_primary_reference(owner_id)
        return reference

    async def list_references(
        self,
        owner_ids: Iterable[str],
        *,
        enabled_only: bool = False,
    ) -> list[ReferenceImage]:
        scopes = list(dict.fromkeys(str(item) for item in owner_ids))
        return await self.storage.list_references(scopes, enabled_only=enabled_only)

    async def find_references(
        self,
        owner_ids: Iterable[str],
        query: str,
        *,
        enabled_only: bool = False,
    ) -> list[ReferenceImage]:
        text = (query or "").strip().casefold()
        references = await self.list_references(owner_ids, enabled_only=enabled_only)
        if not text:
            return references
        return [
            ref
            for ref in references
            if text == ref.id.casefold()
            or text == ref.name.casefold()
            or text in ref.name.casefold()
            or text in ref.note.casefold()
            or any(text in tag.casefold() for tag in ref.tags)
        ]

    async def get_owned(self, ref_id: str, owner_id: str) -> ReferenceImage:
        reference = await self.storage.get_reference(ref_id, [owner_id])
        if reference is None:
            raise NotFoundError("未找到该参考图，或它不属于当前数据空间。")
        return reference

    async def update_metadata(
        self,
        ref_id: str,
        owner_id: str,
        *,
        name: str | None = None,
        note: str | None = None,
        tags: object = None,
        weights: object = None,
        enabled: bool | None = None,
    ) -> ReferenceImage:
        reference = await self.get_owned(ref_id, owner_id)
        if name is not None:
            clean_name = _clean_text(name, limit=40)
            if not clean_name:
                raise ValidationError("参考图名称不能为空。")
            reference.name = clean_name
        if note is not None:
            reference.note = _clean_text(note, limit=500)
        if tags is not None:
            reference.tags = normalize_tags(tags)
        if weights is not None:
            reference.weights = normalize_weights(weights)
        if enabled is not None:
            reference.enabled = enabled
        reference.updated_at = now_ts()
        updated = await self.storage.update_reference_metadata(reference)
        if not updated:
            raise NotFoundError("参考图更新失败，记录可能已被删除。")
        if enabled is False:
            await self.storage.ensure_primary_reference(owner_id)
        return reference

    async def set_enabled(self, ref_id: str, owner_id: str, enabled: bool) -> ReferenceImage:
        reference = await self.get_owned(ref_id, owner_id)
        if not await self.storage.set_reference_enabled(ref_id, owner_id, enabled):
            raise NotFoundError("参考图不存在。")
        reference.enabled = enabled
        if not enabled:
            await self.storage.ensure_primary_reference(owner_id)
        return reference

    async def set_primary(self, ref_id: str, owner_id: str) -> ReferenceImage:
        reference = await self.get_owned(ref_id, owner_id)
        if not reference.enabled:
            raise ValidationError("已禁用的参考图不能设为主图，请先启用。")
        if not await self.storage.set_primary_reference(ref_id, owner_id):
            raise ValidationError("设置主图失败，请确认参考图属于当前数据空间。")
        reference.is_primary = True
        return reference

    async def delete(self, ref_id: str, owner_id: str) -> ReferenceImage:
        reference = await self.storage.delete_reference(ref_id, owner_id)
        if reference is None:
            raise NotFoundError("未找到该参考图，或它不属于当前数据空间。")
        try:
            path = resolve_relative_path(self.data_dir, reference.file_path)
            await asyncio.to_thread(path.unlink, missing_ok=True)
        except (OSError, ValidationError):
            logger.warning("删除参考图文件失败，已保留数据库记录清理结果：%s", reference.id)
        return reference

    async def list_available(self, owner_ids: Iterable[str]) -> tuple[list[ReferenceImage], list[str]]:
        references = await self.list_references(owner_ids, enabled_only=True)
        usable: list[ReferenceImage] = []
        warnings: list[str] = []
        for reference in references:
            try:
                path = self.resolve_file(reference)
            except ValidationError:
                warnings.append(f"参考图《{reference.name}》路径异常，已跳过。")
                continue
            if not path.is_file():
                warnings.append(f"参考图《{reference.name}》文件缺失，已跳过。")
                continue
            usable.append(reference)
        # 个人主图优先于共享主图，用于生成时默认身份参考。
        usable.sort(
            key=lambda item: (
                0 if item.owner_id != "shared" and item.is_primary else 1,
                0 if item.is_primary else 1,
                0 if item.owner_id != "shared" else 1,
                -item.created_at,
            )
        )
        return usable, warnings

    async def primary_reference(self, owner_ids: Iterable[str]) -> ReferenceImage | None:
        references = await self.list_references(owner_ids, enabled_only=True)
        if not references:
            return None
        for reference in references:
            if reference.owner_id != "shared" and reference.is_primary:
                return reference
        for reference in references:
            if reference.is_primary:
                return reference
        return references[0]

    async def all_tags(self, owner_ids: Iterable[str]) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for reference in await self.list_references(owner_ids):
            for tag in reference.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    async def rename_tag(self, owner_id: str, old: str, new: str) -> int:
        old_tag = (old or "").strip().lstrip("#").strip()
        new_tags = normalize_tags([new])
        if not old_tag:
            raise ValidationError("旧标签不能为空。")
        if not new_tags:
            raise ValidationError("新标签不能为空。")
        new_tag = new_tags[0]
        references = await self.storage.list_references([owner_id], limit=1000)
        changed = 0
        for reference in references:
            if old_tag not in reference.tags:
                continue
            reference.tags = normalize_tags([new_tag if tag == old_tag else tag for tag in reference.tags])
            reference.updated_at = now_ts()
            await self.storage.update_reference_metadata(reference)
            changed += 1
        if changed == 0:
            raise NotFoundError("没有参考图使用该标签。")
        return changed

    async def delete_tag(self, owner_id: str, tag: str) -> int:
        target = (tag or "").strip().lstrip("#").strip()
        if not target:
            raise ValidationError("标签不能为空。")
        references = await self.storage.list_references([owner_id], limit=1000)
        changed = 0
        for reference in references:
            if target not in reference.tags:
                continue
            reference.tags = [value for value in reference.tags if value != target]
            reference.updated_at = now_ts()
            await self.storage.update_reference_metadata(reference)
            changed += 1
        if changed == 0:
            raise NotFoundError("没有参考图使用该标签。")
        return changed

    async def count_owned(self, owner_id: str) -> int:
        return await self.storage.count_references(owner_id)
