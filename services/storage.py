from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from .errors import ConfigError
from .models import (
    ApiEndpoint,
    ClothingItem,
    GenerationTask,
    Outfit,
    ReferenceImage,
    RequestLog,
    WardrobeCategory,
    WearState,
    normalize_weights,
)
from .utils import new_id, now_ts  # noqa: F401  # now_ts re-exported for services

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reference_images (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    name TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    weights TEXT NOT NULL DEFAULT '{}',
    file_path TEXT NOT NULL,
    sha256 TEXT NOT NULL DEFAULT '',
    mime TEXT NOT NULL DEFAULT '',
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    is_primary INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reference_owner ON reference_images(owner_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reference_owner_sha
    ON reference_images(owner_id, sha256) WHERE sha256 <> '';

CREATE TABLE IF NOT EXISTS wardrobe_categories (
    owner_id TEXT NOT NULL,
    key TEXT NOT NULL,
    label TEXT NOT NULL,
    kind TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 100,
    created_at REAL NOT NULL,
    PRIMARY KEY (owner_id, key)
);

CREATE TABLE IF NOT EXISTS clothing (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category_key TEXT NOT NULL,
    color TEXT NOT NULL DEFAULT '',
    material TEXT NOT NULL DEFAULT '',
    seasons TEXT NOT NULL DEFAULT '[]',
    scenes TEXT NOT NULL DEFAULT '[]',
    temp_min INTEGER,
    temp_max INTEGER,
    formality INTEGER NOT NULL DEFAULT 3,
    indoor_outdoor TEXT NOT NULL DEFAULT 'both',
    recommended TEXT NOT NULL DEFAULT '[]',
    forbidden TEXT NOT NULL DEFAULT '[]',
    priority INTEGER NOT NULL DEFAULT 50,
    enabled INTEGER NOT NULL DEFAULT 1,
    reference_id TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    is_base INTEGER NOT NULL DEFAULT 0,
    base_role TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clothing_owner ON clothing(owner_id);

CREATE TABLE IF NOT EXISTS outfits (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    name TEXT NOT NULL,
    item_ids TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outfits_owner ON outfits(owner_id);

CREATE TABLE IF NOT EXISTS wear_state (
    owner_id TEXT PRIMARY KEY,
    worn_ids TEXT NOT NULL DEFAULT '[]',
    history TEXT NOT NULL DEFAULT '[]',
    mode TEXT NOT NULL DEFAULT 'auto',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS api_endpoints (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    base_url TEXT NOT NULL,
    model TEXT NOT NULL,
    api_key TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 100,
    timeout REAL NOT NULL DEFAULT 180,
    edit_field TEXT NOT NULL DEFAULT 'image[]',
    extra TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    last_ok_at REAL,
    last_error TEXT NOT NULL DEFAULT '',
    fail_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS api_logs (
    id TEXT PRIMARY KEY,
    endpoint_name TEXT NOT NULL,
    task_id TEXT NOT NULL DEFAULT '',
    ok INTEGER NOT NULL DEFAULT 0,
    status_code INTEGER,
    message TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_logs_created ON api_logs(created_at DESC);

CREATE TABLE IF NOT EXISTS generation_tasks (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    request_key TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'photo',
    description TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL DEFAULT '',
    negative_prompt TEXT NOT NULL DEFAULT '',
    size TEXT NOT NULL DEFAULT '',
    endpoint_name TEXT NOT NULL DEFAULT '',
    result_path TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON generation_tasks(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_request ON generation_tasks(owner_id, request_key, created_at DESC);

CREATE TABLE IF NOT EXISTS plugin_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _loads_list(raw: Any) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _loads_dict(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _row_to_reference(row: aiosqlite.Row) -> ReferenceImage:
    return ReferenceImage(
        id=row["id"],
        owner_id=row["owner_id"],
        name=row["name"],
        note=row["note"],
        tags=_loads_list(row["tags"]),
        weights=normalize_weights(_loads_dict(row["weights"])),
        file_path=row["file_path"],
        sha256=row["sha256"],
        mime=row["mime"],
        width=row["width"],
        height=row["height"],
        size_bytes=row["size_bytes"],
        is_primary=bool(row["is_primary"]),
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_clothing(row: aiosqlite.Row) -> ClothingItem:
    return ClothingItem(
        id=row["id"],
        owner_id=row["owner_id"],
        name=row["name"],
        category_key=row["category_key"],
        color=row["color"],
        material=row["material"],
        seasons=_loads_list(row["seasons"]),
        scenes=_loads_list(row["scenes"]),
        temp_min=row["temp_min"],
        temp_max=row["temp_max"],
        formality=row["formality"],
        indoor_outdoor=row["indoor_outdoor"],
        recommended_with=_loads_list(row["recommended"]),
        forbidden_with=_loads_list(row["forbidden"]),
        priority=row["priority"],
        enabled=bool(row["enabled"]),
        reference_id=row["reference_id"],
        note=row["note"],
        is_base=bool(row["is_base"]),
        base_role=row["base_role"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_outfit(row: aiosqlite.Row) -> Outfit:
    return Outfit(
        id=row["id"],
        owner_id=row["owner_id"],
        name=row["name"],
        item_ids=_loads_list(row["item_ids"]),
        note=row["note"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_api(row: aiosqlite.Row) -> ApiEndpoint:
    return ApiEndpoint(
        id=row["id"],
        name=row["name"],
        base_url=row["base_url"],
        model=row["model"],
        api_key=row["api_key"],
        enabled=bool(row["enabled"]),
        priority=row["priority"],
        timeout=row["timeout"],
        edit_field=row["edit_field"],
        extra=_loads_dict(row["extra"]),
        is_primary=False,
        last_ok_at=row["last_ok_at"],
        last_error=row["last_error"],
        fail_count=row["fail_count"],
        created_at=row["created_at"],
    )


def _row_to_task(row: aiosqlite.Row) -> GenerationTask:
    return GenerationTask(
        id=row["id"],
        owner_id=row["owner_id"],
        request_key=row["request_key"],
        status=row["status"],
        mode=row["mode"],
        description=row["description"],
        prompt=row["prompt"],
        negative_prompt=row["negative_prompt"],
        size=row["size"],
        endpoint_name=row["endpoint_name"],
        result_path=row["result_path"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(self.db_path, isolation_level=None)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=5000")
        self._conn = conn
        try:
            await self._migrate()
        except BaseException:
            await conn.close()
            self._conn = None
            raise

    async def _migrate(self) -> None:
        conn = self._require()
        cursor = await conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        await cursor.close()
        current = int(row[0]) if row else 0
        if current > SCHEMA_VERSION:
            raise ConfigError("数据库版本高于当前插件支持的版本，请升级插件。")
        if current < SCHEMA_VERSION:
            # v1 为初始结构。后续版本在此按 current 递增执行增量迁移。
            await conn.executescript(_SCHEMA)
            await conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    async def close(self) -> None:
        if self._conn is None:
            return
        async with self._lock:
            conn, self._conn = self._conn, None
            await conn.close()

    def _require(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise ConfigError("数据库尚未初始化。")
        return self._conn

    @asynccontextmanager
    async def _write(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._lock:
            conn = self._require()
            await conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                await conn.rollback()
                raise
            else:
                await conn.commit()

    async def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        async with self._lock:
            conn = self._require()
            cursor = await conn.execute(sql, params)
            rowcount = cursor.rowcount
            await cursor.close()
            await conn.commit()
            return rowcount

    async def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> aiosqlite.Row | None:
        async with self._lock:
            conn = self._require()
            cursor = await conn.execute(sql, params)
            row = await cursor.fetchone()
            await cursor.close()
            return row

    async def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        async with self._lock:
            conn = self._require()
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            await cursor.close()
            return list(rows)

    @staticmethod
    def _scope_sql(owner_ids: list[str]) -> tuple[str, tuple[Any, ...]]:
        scopes = list(dict.fromkeys(owner_ids)) or ["__none__"]
        placeholders = ",".join("?" for _ in scopes)
        return placeholders, tuple(scopes)

    # ---------- references ----------

    async def add_reference(self, ref: ReferenceImage) -> None:
        async with self._write() as conn:
            if ref.is_primary:
                await conn.execute(
                    "UPDATE reference_images SET is_primary=0 WHERE owner_id=? AND is_primary=1",
                    (ref.owner_id,),
                )
            await conn.execute(
                """
                INSERT INTO reference_images(
                    id, owner_id, name, note, tags, weights, file_path, sha256, mime,
                    width, height, size_bytes, is_primary, enabled, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ref.id,
                    ref.owner_id,
                    ref.name,
                    ref.note,
                    _dumps(ref.tags),
                    _dumps(ref.weights),
                    ref.file_path,
                    ref.sha256,
                    ref.mime,
                    ref.width,
                    ref.height,
                    ref.size_bytes,
                    int(ref.is_primary),
                    int(ref.enabled),
                    ref.created_at,
                    ref.updated_at,
                ),
            )

    async def get_reference(self, ref_id: str, owner_ids: list[str]) -> ReferenceImage | None:
        placeholders, params = self._scope_sql(owner_ids)
        row = await self._fetchone(
            f"SELECT * FROM reference_images WHERE id=? AND owner_id IN ({placeholders})",
            (ref_id, *params),
        )
        return _row_to_reference(row) if row else None

    async def find_reference_by_sha(self, owner_id: str, sha256: str) -> ReferenceImage | None:
        row = await self._fetchone(
            "SELECT * FROM reference_images WHERE owner_id=? AND sha256=?",
            (owner_id, sha256),
        )
        return _row_to_reference(row) if row else None

    async def list_references(
        self,
        owner_ids: list[str],
        *,
        enabled_only: bool = False,
        limit: int = 200,
    ) -> list[ReferenceImage]:
        placeholders, params = self._scope_sql(owner_ids)
        sql = f"SELECT * FROM reference_images WHERE owner_id IN ({placeholders})"
        if enabled_only:
            sql += " AND enabled=1"
        sql += " ORDER BY is_primary DESC, created_at DESC LIMIT ?"
        rows = await self._fetchall(sql, (*params, max(1, limit)))
        return [_row_to_reference(row) for row in rows]

    async def count_references(self, owner_id: str) -> int:
        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM reference_images WHERE owner_id=?",
            (owner_id,),
        )
        return int(row["n"]) if row else 0

    async def update_reference_metadata(self, ref: ReferenceImage) -> bool:
        rowcount = await self._execute(
            """
            UPDATE reference_images
            SET name=?, note=?, tags=?, weights=?, enabled=?, updated_at=?
            WHERE id=? AND owner_id=?
            """,
            (
                ref.name,
                ref.note,
                _dumps(ref.tags),
                _dumps(ref.weights),
                int(ref.enabled),
                ref.updated_at,
                ref.id,
                ref.owner_id,
            ),
        )
        return rowcount > 0

    async def set_reference_enabled(self, ref_id: str, owner_id: str, enabled: bool) -> bool:
        rowcount = await self._execute(
            "UPDATE reference_images SET enabled=?, updated_at=? WHERE id=? AND owner_id=?",
            (int(enabled), now_ts(), ref_id, owner_id),
        )
        if not enabled:
            await self._execute(
                "UPDATE reference_images SET is_primary=0 WHERE id=? AND owner_id=?",
                (ref_id, owner_id),
            )
        return rowcount > 0

    async def set_primary_reference(self, ref_id: str, owner_id: str) -> bool:
        async with self._write() as conn:
            cursor = await conn.execute(
                "SELECT enabled FROM reference_images WHERE id=? AND owner_id=?",
                (ref_id, owner_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None or not row["enabled"]:
                return False
            await conn.execute(
                "UPDATE reference_images SET is_primary=0 WHERE owner_id=? AND is_primary=1",
                (owner_id,),
            )
            await conn.execute(
                "UPDATE reference_images SET is_primary=1, updated_at=? WHERE id=? AND owner_id=?",
                (now_ts(), ref_id, owner_id),
            )
            return True

    async def ensure_primary_reference(self, owner_id: str) -> None:
        row = await self._fetchone(
            "SELECT 1 AS x FROM reference_images WHERE owner_id=? AND is_primary=1 AND enabled=1 LIMIT 1",
            (owner_id,),
        )
        if row:
            return
        candidate = await self._fetchone(
            """
            SELECT id FROM reference_images
            WHERE owner_id=? AND enabled=1
            ORDER BY created_at DESC LIMIT 1
            """,
            (owner_id,),
        )
        if candidate:
            await self.set_primary_reference(candidate["id"], owner_id)

    async def delete_reference(self, ref_id: str, owner_id: str) -> ReferenceImage | None:
        async with self._write() as conn:
            cursor = await conn.execute(
                "SELECT * FROM reference_images WHERE id=? AND owner_id=?",
                (ref_id, owner_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            ref = _row_to_reference(row)
            await conn.execute(
                "DELETE FROM reference_images WHERE id=? AND owner_id=?",
                (ref_id, owner_id),
            )
            await conn.execute(
                "UPDATE clothing SET reference_id='' WHERE reference_id=?",
                (ref_id,),
            )
            if ref.is_primary:
                next_row = await conn.execute(
                    """
                    SELECT id FROM reference_images
                    WHERE owner_id=? AND enabled=1
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (owner_id,),
                )
                next_candidate = await next_row.fetchone()
                await next_row.close()
                if next_candidate:
                    await conn.execute(
                        "UPDATE reference_images SET is_primary=1 WHERE id=? AND owner_id=?",
                        (next_candidate["id"], owner_id),
                    )
            return ref

    # ---------- custom categories ----------

    async def list_custom_categories(self, owner_id: str) -> list[WardrobeCategory]:
        rows = await self._fetchall(
            "SELECT * FROM wardrobe_categories WHERE owner_id=? ORDER BY sort_order, label",
            (owner_id,),
        )
        return [
            WardrobeCategory(
                key=row["key"],
                label=row["label"],
                kind=row["kind"],
                owner_id=row["owner_id"],
                custom=True,
                sort_order=row["sort_order"],
            )
            for row in rows
        ]

    async def get_custom_category(self, owner_id: str, key: str) -> WardrobeCategory | None:
        row = await self._fetchone(
            "SELECT * FROM wardrobe_categories WHERE owner_id=? AND key=?",
            (owner_id, key),
        )
        if row is None:
            return None
        return WardrobeCategory(
            key=row["key"],
            label=row["label"],
            kind=row["kind"],
            owner_id=row["owner_id"],
            custom=True,
            sort_order=row["sort_order"],
        )

    async def upsert_custom_category(self, category: WardrobeCategory) -> None:
        await self._execute(
            """
            INSERT INTO wardrobe_categories(owner_id, key, label, kind, sort_order, created_at)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(owner_id, key) DO UPDATE SET
                label=excluded.label, kind=excluded.kind, sort_order=excluded.sort_order
            """,
            (
                category.owner_id,
                category.key,
                category.label,
                category.kind,
                category.sort_order,
                now_ts(),
            ),
        )

    async def delete_custom_category(self, owner_id: str, key: str) -> bool:
        rowcount = await self._execute(
            "DELETE FROM wardrobe_categories WHERE owner_id=? AND key=?",
            (owner_id, key),
        )
        return rowcount > 0

    async def category_in_use(self, owner_id: str, key: str) -> bool:
        row = await self._fetchone(
            "SELECT 1 AS x FROM clothing WHERE owner_id=? AND category_key=? LIMIT 1",
            (owner_id, key),
        )
        return row is not None

    # ---------- clothing ----------

    async def add_clothing(self, item: ClothingItem) -> None:
        await self._execute(
            """
            INSERT INTO clothing(
                id, owner_id, name, category_key, color, material, seasons, scenes,
                temp_min, temp_max, formality, indoor_outdoor, recommended, forbidden,
                priority, enabled, reference_id, note, is_base, base_role, created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                item.id,
                item.owner_id,
                item.name,
                item.category_key,
                item.color,
                item.material,
                _dumps(item.seasons),
                _dumps(item.scenes),
                item.temp_min,
                item.temp_max,
                item.formality,
                item.indoor_outdoor,
                _dumps(item.recommended_with),
                _dumps(item.forbidden_with),
                item.priority,
                int(item.enabled),
                item.reference_id,
                item.note,
                int(item.is_base),
                item.base_role,
                item.created_at,
                item.updated_at,
            ),
        )

    async def update_clothing(self, item: ClothingItem) -> bool:
        rowcount = await self._execute(
            """
            UPDATE clothing SET
                name=?, category_key=?, color=?, material=?, seasons=?, scenes=?,
                temp_min=?, temp_max=?, formality=?, indoor_outdoor=?, recommended=?,
                forbidden=?, priority=?, enabled=?, reference_id=?, note=?, updated_at=?
            WHERE id=? AND owner_id=?
            """,
            (
                item.name,
                item.category_key,
                item.color,
                item.material,
                _dumps(item.seasons),
                _dumps(item.scenes),
                item.temp_min,
                item.temp_max,
                item.formality,
                item.indoor_outdoor,
                _dumps(item.recommended_with),
                _dumps(item.forbidden_with),
                item.priority,
                int(item.enabled),
                item.reference_id,
                item.note,
                item.updated_at,
                item.id,
                item.owner_id,
            ),
        )
        return rowcount > 0

    async def get_clothing(self, item_id: str, owner_ids: list[str]) -> ClothingItem | None:
        placeholders, params = self._scope_sql(owner_ids)
        row = await self._fetchone(
            f"SELECT * FROM clothing WHERE id=? AND owner_id IN ({placeholders})",
            (item_id, *params),
        )
        return _row_to_clothing(row) if row else None

    async def list_clothing(
        self,
        owner_ids: list[str],
        *,
        enabled_only: bool = False,
        limit: int = 500,
    ) -> list[ClothingItem]:
        placeholders, params = self._scope_sql(owner_ids)
        sql = f"SELECT * FROM clothing WHERE owner_id IN ({placeholders})"
        if enabled_only:
            sql += " AND enabled=1"
        sql += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        rows = await self._fetchall(sql, (*params, max(1, limit)))
        return [_row_to_clothing(row) for row in rows]

    async def find_base_clothing(self, owner_id: str, base_role: str) -> ClothingItem | None:
        row = await self._fetchone(
            "SELECT * FROM clothing WHERE owner_id=? AND is_base=1 AND base_role=? LIMIT 1",
            (owner_id, base_role),
        )
        return _row_to_clothing(row) if row else None

    async def count_clothing(self, owner_id: str) -> int:
        row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM clothing WHERE owner_id=?",
            (owner_id,),
        )
        return int(row["n"]) if row else 0

    async def delete_clothing(self, item_id: str, owner_id: str) -> ClothingItem | None:
        async with self._write() as conn:
            cursor = await conn.execute(
                "SELECT * FROM clothing WHERE id=? AND owner_id=? AND is_base=0",
                (item_id, owner_id),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            item = _row_to_clothing(row)
            await conn.execute(
                "DELETE FROM clothing WHERE id=? AND owner_id=?",
                (item_id, owner_id),
            )
            return item

    async def remove_item_from_outfits(self, item_id: str, owner_ids: list[str]) -> None:
        placeholders, params = self._scope_sql(owner_ids)
        rows = await self._fetchall(
            f"SELECT * FROM outfits WHERE owner_id IN ({placeholders})",
            params,
        )
        async with self._write() as conn:
            for row in rows:
                item_ids = _loads_list(row["item_ids"])
                if item_id not in item_ids:
                    continue
                item_ids = [value for value in item_ids if value != item_id]
                await conn.execute(
                    "UPDATE outfits SET item_ids=?, updated_at=? WHERE id=?",
                    (_dumps(item_ids), now_ts(), row["id"]),
                )

    # ---------- outfits ----------

    async def add_outfit(self, outfit: Outfit) -> None:
        await self._execute(
            """
            INSERT INTO outfits(id, owner_id, name, item_ids, note, created_at, updated_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                outfit.id,
                outfit.owner_id,
                outfit.name,
                _dumps(outfit.item_ids),
                outfit.note,
                outfit.created_at,
                outfit.updated_at,
            ),
        )

    async def list_outfits(self, owner_ids: list[str], limit: int = 100) -> list[Outfit]:
        placeholders, params = self._scope_sql(owner_ids)
        rows = await self._fetchall(
            f"SELECT * FROM outfits WHERE owner_id IN ({placeholders}) ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        )
        return [_row_to_outfit(row) for row in rows]

    async def get_outfit(self, outfit_id: str, owner_ids: list[str]) -> Outfit | None:
        placeholders, params = self._scope_sql(owner_ids)
        row = await self._fetchone(
            f"SELECT * FROM outfits WHERE id=? AND owner_id IN ({placeholders})",
            (outfit_id, *params),
        )
        return _row_to_outfit(row) if row else None

    async def delete_outfit(self, outfit_id: str, owner_id: str) -> bool:
        rowcount = await self._execute(
            "DELETE FROM outfits WHERE id=? AND owner_id=?",
            (outfit_id, owner_id),
        )
        return rowcount > 0

    # ---------- wear state ----------

    async def get_wear_state(self, owner_id: str) -> WearState:
        row = await self._fetchone(
            "SELECT * FROM wear_state WHERE owner_id=?",
            (owner_id,),
        )
        if row is None:
            return WearState(owner_id=owner_id)
        try:
            raw_history = json.loads(row["history"] or "[]")
        except (TypeError, ValueError):
            raw_history = []
        history = [
            [str(item) for item in entry]
            for entry in raw_history
            if isinstance(entry, list)
        ]
        return WearState(
            owner_id=row["owner_id"],
            worn_ids=_loads_list(row["worn_ids"]),
            history=history,
            mode=row["mode"],
            updated_at=row["updated_at"],
        )

    async def save_wear_state(self, state: WearState) -> None:
        await self._execute(
            """
            INSERT INTO wear_state(owner_id, worn_ids, history, mode, updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(owner_id) DO UPDATE SET
                worn_ids=excluded.worn_ids,
                history=excluded.history,
                mode=excluded.mode,
                updated_at=excluded.updated_at
            """,
            (
                state.owner_id,
                _dumps(state.worn_ids),
                _dumps(state.history[-20:]),
                state.mode,
                state.updated_at,
            ),
        )

    # ---------- API endpoints ----------

    async def add_api_endpoint(self, endpoint: ApiEndpoint) -> None:
        await self._execute(
            """
            INSERT INTO api_endpoints(
                id, name, base_url, model, api_key, enabled, priority, timeout,
                edit_field, extra, created_at, last_ok_at, last_error, fail_count
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                endpoint.id,
                endpoint.name,
                endpoint.base_url,
                endpoint.model,
                endpoint.api_key,
                int(endpoint.enabled),
                endpoint.priority,
                endpoint.timeout,
                endpoint.edit_field,
                _dumps(endpoint.extra),
                endpoint.created_at,
                endpoint.last_ok_at,
                endpoint.last_error,
                endpoint.fail_count,
            ),
        )

    async def update_api_endpoint(self, endpoint: ApiEndpoint) -> bool:
        rowcount = await self._execute(
            """
            UPDATE api_endpoints SET
                base_url=?, model=?, api_key=?, enabled=?, priority=?, timeout=?,
                edit_field=?, extra=?
            WHERE id=?
            """,
            (
                endpoint.base_url,
                endpoint.model,
                endpoint.api_key,
                int(endpoint.enabled),
                endpoint.priority,
                endpoint.timeout,
                endpoint.edit_field,
                _dumps(endpoint.extra),
                endpoint.id,
            ),
        )
        return rowcount > 0

    async def list_api_endpoints(self, *, enabled_only: bool = False) -> list[ApiEndpoint]:
        sql = "SELECT * FROM api_endpoints"
        if enabled_only:
            sql += " WHERE enabled=1"
        sql += " ORDER BY priority ASC, created_at ASC"
        rows = await self._fetchall(sql)
        return [_row_to_api(row) for row in rows]

    async def get_api_endpoint(self, endpoint_id: str) -> ApiEndpoint | None:
        row = await self._fetchone(
            "SELECT * FROM api_endpoints WHERE id=?",
            (endpoint_id,),
        )
        return _row_to_api(row) if row else None

    async def find_api_endpoint_by_name(self, name: str) -> ApiEndpoint | None:
        row = await self._fetchone(
            "SELECT * FROM api_endpoints WHERE name=?",
            (name,),
        )
        return _row_to_api(row) if row else None

    async def delete_api_endpoint(self, endpoint_id: str) -> bool:
        rowcount = await self._execute(
            "DELETE FROM api_endpoints WHERE id=?",
            (endpoint_id,),
        )
        return rowcount > 0

    async def record_api_result(
        self,
        endpoint_id: str,
        *,
        ok: bool,
        error: str = "",
        status_code: int | None = None,
    ) -> None:
        stamp = now_ts()
        if ok:
            await self._execute(
                """
                UPDATE api_endpoints
                SET last_ok_at=?, last_error='', fail_count=0
                WHERE id=?
                """,
                (stamp, endpoint_id),
            )
        else:
            await self._execute(
                """
                UPDATE api_endpoints
                SET last_error=?, fail_count=fail_count+1
                WHERE id=?
                """,
                (error, endpoint_id),
            )

    async def add_api_log(self, log: RequestLog) -> None:
        async with self._write() as conn:
            await conn.execute(
                """
                INSERT INTO api_logs(id, endpoint_name, task_id, ok, status_code, message, created_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    log.id,
                    log.endpoint_name,
                    log.task_id,
                    int(log.ok),
                    log.status_code,
                    log.message,
                    log.created_at,
                ),
            )
            await conn.execute(
                """
                DELETE FROM api_logs WHERE id NOT IN (
                    SELECT id FROM api_logs ORDER BY created_at DESC LIMIT 200
                )
                """
            )

    async def recent_api_errors(self, limit: int = 10) -> list[RequestLog]:
        rows = await self._fetchall(
            """
            SELECT * FROM api_logs
            WHERE ok=0
            ORDER BY created_at DESC LIMIT ?
            """,
            (max(1, limit),),
        )
        return [
            RequestLog(
                id=row["id"],
                endpoint_name=row["endpoint_name"],
                task_id=row["task_id"],
                ok=bool(row["ok"]),
                status_code=row["status_code"],
                message=row["message"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # ---------- generation tasks ----------

    async def create_task(self, task: GenerationTask) -> None:
        await self._execute(
            """
            INSERT INTO generation_tasks(
                id, owner_id, request_key, status, mode, description, prompt,
                negative_prompt, size, endpoint_name, result_path, error, created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                task.id,
                task.owner_id,
                task.request_key,
                task.status,
                task.mode,
                task.description,
                task.prompt,
                task.negative_prompt,
                task.size,
                task.endpoint_name,
                task.result_path,
                task.error,
                task.created_at,
                task.updated_at,
            ),
        )

    async def update_task(self, task_id: str, **fields: Any) -> bool:
        allowed = {
            "status",
            "endpoint_name",
            "result_path",
            "error",
            "prompt",
            "negative_prompt",
            "size",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return False
        updates["updated_at"] = now_ts()
        assignments = ", ".join(f"{key}=?" for key in updates)
        params = (*updates.values(), task_id)
        rowcount = await self._execute(
            f"UPDATE generation_tasks SET {assignments} WHERE id=?",
            params,
        )
        return rowcount > 0

    async def get_task(self, task_id: str) -> GenerationTask | None:
        row = await self._fetchone(
            "SELECT * FROM generation_tasks WHERE id=?",
            (task_id,),
        )
        return _row_to_task(row) if row else None

    async def find_recent_task(
        self,
        owner_id: str,
        request_key: str,
        *,
        since: float,
        statuses: tuple[str, ...] = ("succeeded", "running"),
    ) -> GenerationTask | None:
        placeholders = ",".join("?" for _ in statuses)
        row = await self._fetchone(
            f"""
            SELECT * FROM generation_tasks
            WHERE owner_id=? AND request_key=? AND created_at>=? AND status IN ({placeholders})
            ORDER BY created_at DESC LIMIT 1
            """,
            (owner_id, request_key, since, *statuses),
        )
        return _row_to_task(row) if row else None

    async def recent_tasks(self, owner_id: str, limit: int = 10) -> list[GenerationTask]:
        rows = await self._fetchall(
            """
            SELECT * FROM generation_tasks
            WHERE owner_id=?
            ORDER BY created_at DESC LIMIT ?
            """,
            (owner_id, max(1, limit)),
        )
        return [_row_to_task(row) for row in rows]

    async def mark_stale_tasks(self, older_than: float) -> int:
        rowcount = await self._execute(
            """
            UPDATE generation_tasks
            SET status='failed', error='插件重启或任务中断', updated_at=?
            WHERE status IN ('pending', 'running') AND created_at<?
            """,
            (now_ts(), older_than),
        )
        return rowcount

    # ---------- meta ----------

    async def get_meta(self, key: str, default: str = "") -> str:
        row = await self._fetchone(
            "SELECT value FROM plugin_meta WHERE key=?",
            (key,),
        )
        return row["value"] if row else default

    async def set_meta(self, key: str, value: str) -> None:
        await self._execute(
            """
            INSERT INTO plugin_meta(key, value) VALUES(?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )

    async def delete_meta(self, key: str) -> None:
        await self._execute("DELETE FROM plugin_meta WHERE key=?", (key,))

    async def ping(self) -> bool:
        row = await self._fetchone("SELECT 1 AS ok")
        return bool(row and row["ok"] == 1)
