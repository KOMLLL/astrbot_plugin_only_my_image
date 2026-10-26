from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.session_waiter import (
    SessionController,
    SessionFilter,
    session_waiter,
)

from .services.errors import (
    NotFoundError,
    PermissionDeniedError,
    PluginError,
    ValidationError,
)
from .services.gallery import GalleryService
from .services.generator import GenerationCoordinator
from .services.image_api import ImageApiManager
from .services.models import (
    DEFAULT_WEIGHTS,
    DIMENSION_LABELS,
    WEIGHT_DIMENSIONS,
    ClothingItem,
    ReferenceImage,
    coerce_weight,
    normalize_dimension,
)
from .services.prompt_builder import PromptBuilder
from .services.reference_selector import ReferenceSelector
from .services.settings import Settings
from .services.storage import Storage
from .services.timeline import TimelineService
from .services.utils import new_id, split_list, truncate_text
from .services.wardrobe import WardrobeService

PLUGIN_NAME = "astrbot_plugin_only_my_image"
TOOL_NAME = "generate_character_image"

_HELP_TEXT = """【角色生图插件指令】
生图
  /生图 <画面描述> [--尺寸 1024x1024] [--穿搭 白色连衣裙] [--场景 咖啡店]
  /生图 <描述> --预览      只查看时间轴、穿搭、参考图与提示词，不调用接口
  /自拍 <画面描述>          使用竖版自拍尺寸
  /预览生图 <描述>          同 /生图 --预览

参考图库
  /图库                    查看图库概览
  /图库 列表 [--共享]
  /图库 搜索 <关键词>
  /图库 详情 <名称或ID>
  /图库 添加 [名称|备注|标签]  进入连续上传会话；发送“完成”或“取消”结束
  /添加参考图 [名称|备注|标签]
  /图库 权重 <名称或ID> [面部=高 发型=60 服装=低 ...] [重置]
  /图库 主图 <名称或ID>
  /图库 启用/停用 <名称或ID>
  /图库 重命名 <名称或ID> <新名称>
  /图库 说明 <名称或ID> <新备注>
  /图库 标签 [添加 <名称> <标签...> | 删除 <标签> | 重命名 <旧> <新>]
  /图库 删除 <名称或ID> --确认
  管理员可加 --共享 管理共享图库。

衣柜
  /衣柜                    当前穿搭概览
  /衣柜 列表 [分类] [--全部]
  /衣柜 添加 <名称> <分类> [颜色=... 材质=... 季节=春,秋 场景=... 温度=5-22
                          正式度=1-5 场合=室内 推荐=... 禁忌=... 优先级=80
                          参考图=<名称或ID> 备注=...]
  /衣柜 启用/停用/删除 <名称或ID>（删除需 --确认）
  /衣柜 穿上 <衣物...>｜脱下 <衣物或分类>｜换 <分类> <衣物>
  /衣柜 还原｜当前｜自动搭配
  /衣柜 分类 [添加 <名称> [类型=...] | 删除 <名称> --确认]
  /衣柜 套装 [添加 <名称> <衣物...> | 删除 <名称> --确认 | 穿上 <名称>]
  管理员可加 --共享 管理共享衣柜。

形象与时间轴
  /形象                    形象概览
  /形象 活动               当前时间轴
  /形象 活动 设置 <活动> [地点=... 天气=... 温度=... 室内=是 季节=秋]
  /形象 活动 清除
  /形象 状态               运行状态

接口管理（仅管理员）
  /接口                    接口列表（密钥已隐藏）
  /接口 测试 [名称或ID]     访问模型列表，不生成图片、不产生生图费用
  /接口 添加 <名称> 地址=... 模型=... 密钥=... [优先级=] [超时=] [字段=]
  /接口 修改 <名称或ID> [地址= 模型= 密钥= 优先级= 超时= 字段=]
  /接口 启用/停用 <名称或ID>
  /接口 删除 <名称或ID> --确认
  /接口 错误               最近错误

说明
  普通用户默认只能操作自己的图库和衣柜；共享内容仅管理员可改。
  任何脱衣后缺少上装/下装的情况都会回退到基础白T恤 + 黑色短裤。
  生成请求不会因为普通聊天自动触发；只有明确要求照片/自拍或使用 /生图 才会调用生图接口。"""


class OnlyMyImagePlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context, config)
        self.config = config or {}
        self.settings = Settings(self.config)
        try:
            self.data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        except Exception:  # noqa: BLE001 - 数据目录获取失败必须回退，不能让插件加载失败
            logger.exception("获取插件数据目录失败，使用相对路径回退。")
            self.data_dir = Path.cwd() / "data" / "plugin_data" / PLUGIN_NAME
        self.storage = Storage(self.data_dir / "only_my_image.sqlite3")
        self.gallery = GalleryService(self.storage, self.data_dir, self.settings)
        self.wardrobe = WardrobeService(self.storage, self.settings)
        self.timeline = TimelineService(
            self.storage,
            self.settings,
            resolver=self._resolve_timeline_adapter,
        )
        self.selector = ReferenceSelector(self.settings)
        self.prompts = PromptBuilder(self.settings)
        self.api = ImageApiManager(self.storage, self.settings)
        self.coordinator = GenerationCoordinator(
            self.storage,
            self.gallery,
            self.wardrobe,
            self.selector,
            self.timeline,
            self.api,
            self.prompts,
            self.settings,
            self.data_dir,
        )
        self._ready = False
        self._closed = False
        self._startup_error = ""
        self._tool_registered = False

    # ---------- lifecycle ----------

    async def initialize(self) -> None:
        try:
            await asyncio.to_thread(self.data_dir.mkdir, parents=True, exist_ok=True)
            await self.storage.open()
            await self.storage.mark_stale_tasks(3600)
            self._ready = True
        except Exception as exc:  # noqa: BLE001 - 启动失败必须以错误提示继续加载
            self._startup_error = f"{type(exc).__name__}: {exc}"
            logger.exception("生图插件数据层初始化失败，插件将以只读错误提示方式继续加载。")
            return
        self._tool_registered = (
            self.context.get_llm_tool_manager().get_func(TOOL_NAME) is not None
        )
        if self._tool_registered:
            logger.info("LLM 工具 %s 已注册。", TOOL_NAME)
        else:
            logger.error(
                "LLM 工具 %s 未注册；自然语言生图不可用，但显式指令仍可使用。",
                TOOL_NAME,
            )
        logger.info("生图插件已就绪，数据目录：%s", self.data_dir)

    async def terminate(self) -> None:
        self._closed = True
        self._stop_upload_sessions()
        try:
            await self.coordinator.close()
        finally:
            try:
                await self.api.close()
            finally:
                await self.storage.close()
        logger.info("生图插件已卸载。")

    @staticmethod
    def _stop_upload_sessions() -> None:
        try:
            from astrbot.core.utils import session_waiter as waiter_module
        except ImportError:
            return
        sessions = getattr(waiter_module, "USER_SESSIONS", {})
        filters = getattr(waiter_module, "FILTERS", [])
        for session in list(sessions.values()):
            session_filter = getattr(session, "session_filter", None)
            if getattr(session_filter, "plugin_name", "") != PLUGIN_NAME:
                continue
            controller = getattr(session, "session_controller", None)
            if controller is not None:
                controller.stop()
            sessions.pop(getattr(session, "session_id", ""), None)
            try:
                filters.remove(session_filter)
            except ValueError:
                pass

    def _startup_problem(self) -> str:
        if self._closed:
            return "插件正在卸载，暂时不可用。"
        if not self._ready:
            return "插件数据层未初始化成功，请管理员检查日志中的数据库错误。"
        return ""

    def _resolve_timeline_adapter(self, plugin_name: str, function_name: str) -> Any:
        metadata = self.context.get_registered_star(plugin_name)
        if metadata is None:
            raise RuntimeError("时间轴插件未加载")
        target = getattr(metadata, "star_cls", None) or getattr(metadata, "module", None)
        if target is None:
            raise RuntimeError("时间轴插件没有可调用的实例")
        func = getattr(target, function_name, None)
        if not callable(func):
            raise TypeError("时间轴适配函数不存在")
        return func()

    # ---------- identity and permissions ----------

    @staticmethod
    def _sender(event: AstrMessageEvent) -> str:
        return str(event.get_sender_id() or "unknown")

    @staticmethod
    def _actor_id(event: AstrMessageEvent) -> str:
        platform = str(event.get_platform_name() or "").strip()
        sender = str(event.get_sender_id() or "unknown")
        return f"{platform}:{sender}" if platform else sender

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        try:
            value = event.is_admin
            is_admin = bool(value() if callable(value) else value)
        except Exception:  # noqa: BLE001 - 兼容不同 AstrBot 版本的事件方法
            is_admin = False
        if is_admin:
            return True
        sender = self._sender(event)
        actor = self._actor_id(event)
        admins = self.settings.admin_ids()
        return sender in admins or actor in admins

    def _require_generation(self, event: AstrMessageEvent) -> None:
        if self.settings.allow_user_generation() or self._is_admin(event):
            return
        raise PermissionDeniedError("管理员已关闭普通用户的生图权限。")

    def _require_gallery_edit(self, event: AstrMessageEvent) -> None:
        if self._is_admin(event) or self.settings.allow_user_upload():
            return
        raise PermissionDeniedError("管理员已关闭普通用户的参考图修改权限。")

    def _require_wardrobe_edit(self, event: AstrMessageEvent) -> None:
        if self._is_admin(event) or self.settings.allow_user_wardrobe_edit():
            return
        raise PermissionDeniedError("管理员已关闭普通用户的衣柜修改权限。")

    def _edit_scope(self, event: AstrMessageEvent, shared: bool) -> str:
        if shared:
            if not self._is_admin(event):
                raise PermissionDeniedError("只有管理员可以操作共享数据空间。")
            return "shared"
        return self._actor_id(event)

    # ---------- text helpers ----------

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _strip_quotes(text: str) -> str:
        value = (text or "").strip()
        for start, end in (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")):
            if len(value) >= 2 and value.startswith(start) and value.endswith(end):
                return value[1:-1].strip()
        return value

    @classmethod
    def _extract_flag(cls, text: str, *names: str) -> tuple[bool, str]:
        found = False
        for name in names:
            pattern = re.compile(rf"(?<!\S)--{re.escape(name)}(?![^\s])")
            text, count = pattern.subn(" ", text)
            if count:
                found = True
        return found, cls._clean(text)

    @classmethod
    def _take_option(cls, text: str, *names: str) -> tuple[str, str]:
        for name in names:
            pattern = re.compile(
                rf"(?:^|\s)--{re.escape(name)}(?:=|\s+)"
                r'(?:"([^"]*)"|“([^”]*)”|\'([^\']*)\'|(\S+))'
            )
            match = pattern.search(text)
            if not match:
                continue
            value = next(group for group in match.groups() if group is not None)
            cleaned = text[: match.start()] + " " + text[match.end() :]
            return value.strip(), cls._clean(cleaned)
        return "", cls._clean(text)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        pattern = re.compile(r'"([^"]*)"|“([^”]*)”|\'([^\']*)\'|(\S+)')
        tokens: list[str] = []
        for match in pattern.finditer(text or ""):
            value = next((group for group in match.groups() if group is not None), "")
            value = value.strip()
            if value:
                tokens.append(value)
        return tokens

    @classmethod
    def _pairs(cls, tokens: list[str]) -> tuple[list[str], dict[str, str]]:
        free: list[str] = []
        pairs: dict[str, str] = {}
        for token in tokens:
            if "=" in token or "＝" in token:
                key, value = re.split(r"[=＝]", token, maxsplit=1)
                key = key.strip().casefold()
                if key:
                    pairs[key] = value.strip()
            else:
                free.append(token)
        return free, pairs

    @staticmethod
    def _pair(pairs: dict[str, str], *keys: str, default: str = "") -> str:
        for key in keys:
            value = pairs.get(key.casefold())
            if value is not None:
                return value
        return default

    @staticmethod
    def _to_int(value: str, label: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label}必须是整数。") from exc

    @staticmethod
    def _to_float(value: str, label: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label}必须是数字。") from exc

    @classmethod
    def _split_action(cls, text: str) -> tuple[str, str, list[str]]:
        chunks = (text or "").strip().split(maxsplit=1)
        action = chunks[0] if chunks else ""
        rest = chunks[1] if len(chunks) > 1 else ""
        return action, rest, cls._tokenize(rest)

    @staticmethod
    def _error_text(error: Exception) -> str:
        if isinstance(error, PluginError):
            return error.user_message
        logger.exception("生图插件命令执行失败。")
        return "操作失败，请稍后重试或查看 AstrBot 日志。"

    @staticmethod
    def _format_time(timestamp: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp or 0))

    @staticmethod
    def _weights_brief(reference: ReferenceImage) -> str:
        parts = [
            f"{DIMENSION_LABELS[dimension]}{reference.dimension_weight(dimension)}"
            for dimension in WEIGHT_DIMENSIONS
            if reference.dimension_weight(dimension) > 0
        ]
        return " ".join(parts) or "全部关闭"

    def _format_reference(self, reference: ReferenceImage) -> str:
        owner = "共享" if reference.owner_id == "shared" else "个人"
        flags = []
        if reference.is_primary:
            flags.append("主图")
        flags.append("启用" if reference.enabled else "停用")
        tags = "、".join(reference.tags) if reference.tags else "无"
        lines = [
            f"《{reference.name}》 [{reference.id}]",
            f"空间：{owner}｜状态：{'、'.join(flags)}｜尺寸：{reference.width}x{reference.height}",
            f"权重：{self._weights_brief(reference)}",
            f"标签：{tags}｜创建：{self._format_time(reference.created_at)}",
        ]
        if reference.note:
            lines.append(f"说明：{truncate_text(reference.note, 300)}")
        return "\n".join(lines)

    def _format_reference_list(self, references: list[ReferenceImage]) -> str:
        if not references:
            return "没有匹配的参考图。"
        lines = [f"共 {len(references)} 张参考图："]
        for reference in references[:25]:
            owner = "共享" if reference.owner_id == "shared" else "个人"
            flags = []
            if reference.is_primary:
                flags.append("主图")
            if not reference.enabled:
                flags.append("停用")
            suffix = f"（{'、'.join(flags)}）" if flags else ""
            tags = f"｜{'、'.join(reference.tags)}" if reference.tags else ""
            lines.append(
                f"- 《{reference.name}》 [{reference.id}] {owner}{suffix}{tags}"
            )
        if len(references) > 25:
            lines.append(f"…… 还有 {len(references) - 25} 张，请用“搜索”缩小范围。")
        return "\n".join(lines)

    async def _resolve_reference(
        self,
        scopes: list[str],
        query: str,
    ) -> ReferenceImage:
        text = (query or "").strip()
        if not text:
            raise ValidationError("请提供参考图名称或 ID。")
        references = await self.gallery.find_references(scopes, text)
        if not references:
            raise NotFoundError(f"没有找到参考图“{text}”。")
        exact = [
            reference
            for reference in references
            if text.casefold() in {reference.id.casefold(), reference.name.casefold()}
        ]
        if len(exact) == 1:
            return exact[0]
        candidates = exact or references
        if len(candidates) == 1:
            return candidates[0]
        names = "、".join(
            f"《{reference.name}》({reference.id})" for reference in candidates[:8]
        )
        raise ValidationError(f"“{text}”匹配到多张参考图：{names}。请使用完整名称或 ID。")

    async def _resolve_item(
        self,
        scopes: list[str],
        query: str,
        *,
        include_base: bool = True,
    ) -> ClothingItem:
        return await self.wardrobe.resolve_item(
            scopes,
            query,
            include_base=include_base,
        )

    async def _state_lines(self, actor: str, scopes: list[str]) -> list[str]:
        state, items, notes = await self.wardrobe.get_state(actor, scopes)
        description = await self.wardrobe.outfit_description(actor, scopes, items)
        lines = [
            f"当前穿搭（{state.mode}）：{description}",
        ]
        if notes:
            lines.append("提示：" + "；".join(notes))
        return lines

    # ---------- natural language tool ----------

    @filter.llm_tool(name=TOOL_NAME)
    async def generate_character_image(
        self,
        event: AstrMessageEvent,
        description: str,
        selfie: bool = False,
        outfit: str = "",
        scene: str = "",
        size: str = "",
    ):
        """当用户明确要求生成自己的照片、自拍、换装照或“拍一张/画一张”时，调用本工具生成并直接发送图片。
        只在用户明确表达生成图片的意图时调用。用户只是普通聊天、询问“在干嘛”或让你描述状态时，绝对不要调用本工具。

        Args:
            description(string): 用户希望出现的人物、动作、表情与画面氛围；保留用户原始意图，不要编造用户没有提到的地点或天气。
            selfie(boolean): 用户明确要自拍时设为 true，否则为 false。
            outfit(string): 用户明确指定的穿搭，例如“白色连衣裙”；未指定时留空，由智能衣柜决定。
            scene(string): 用户明确指定的场景，例如“咖啡店门口”；未指定时留空。
            size(string): 用户明确要求的图片尺寸，例如 1024x1024；未指定时留空。
        """
        problem = self._startup_problem()
        if problem:
            yield event.plain_result(problem)
            event.stop_event()
            return
        try:
            self._require_generation(event)
            text = self._clean(description)
            if not text:
                yield event.plain_result("请补充想生成的画面描述。")
                event.stop_event()
                return
            actor = self._actor_id(event)
            scopes = [actor, "shared"]
            prepared = await self.coordinator.prepare(
                actor_id=actor,
                state_owner=actor,
                scopes=scopes,
                description=text,
                selfie=bool(selfie),
                size=size,
                outfit_hint=outfit,
                scene=scene,
            )
            logger.info(
                "LLM 工具触发生图 actor=%s selfie=%s refs=%s",
                actor,
                bool(selfie),
                len(prepared.reference_plan.references),
            )
            result = await self.coordinator.generate(prepared)
            components = [Plain(result.reply_text)]
            image_path = Path(result.image_path)
            if image_path.is_file():
                components.append(Image.fromFileSystem(str(image_path)))
            yield event.chain_result(components)
        except PluginError as exc:
            yield event.plain_result(self._error_text(exc))
        except Exception as exc:  # noqa: BLE001 - LLM 工具边界必须兜底
            logger.exception("LLM 工具生图失败。")
            yield event.plain_result(self._error_text(exc))
        event.stop_event()

    @filter.on_llm_request()
    async def add_generation_hint(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self._ready or not self.settings.on_llm_request_hint():
            return
        system_prompt = getattr(req, "system_prompt", "") or ""
        marker = f"[{PLUGIN_NAME}]"
        if marker in system_prompt:
            return
        hint = (
            f"{marker} 本插件提供角色照片与自拍生成能力。只有用户明确提出要照片、自拍、"
            "换装照或“拍/画一张”时，才调用工具 generate_character_image。"
            "普通聊天、询问“在干嘛/今天怎么样”时不要调用任何生图工具，也不要自行把聊天改写成生图请求。"
            "用户可以用 /生图、/自拍、/预览生图 显式操作图库与衣柜。"
        )
        req.system_prompt = f"{system_prompt}\n\n{hint}".strip()

    # ---------- generation commands ----------

    @filter.command("生图", alias={"画图", "出图", "拍照"})
    async def generate_command(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """根据描述生成一张角色照片。"""
        yield await self._generate_reply(event, str(text), selfie=False, preview_only=False)

    @filter.command("自拍", alias={"自拍照"})
    async def selfie_command(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """根据描述生成一张角色自拍。"""
        yield await self._generate_reply(event, str(text), selfie=True, preview_only=False)

    @filter.command("预览生图", alias={"生图预览"})
    async def preview_command(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """预览生成参数，不调用图片接口。"""
        yield await self._generate_reply(event, str(text), selfie=False, preview_only=True)

    async def _generate_reply(
        self,
        event: AstrMessageEvent,
        raw_text: str,
        *,
        selfie: bool,
        preview_only: bool,
    ):
        problem = self._startup_problem()
        if problem:
            return event.plain_result(problem)
        try:
            self._require_generation(event)
            text = raw_text or ""
            explicit, text = self._extract_flag(text, "直接")
            preview, text = self._extract_flag(text, "预览")
            size, text = self._take_option(text, "尺寸", "size")
            outfit, text = self._take_option(text, "穿搭", "服装", "outfit")
            scene, text = self._take_option(text, "场景", "scene")
            description = self._clean(text)
            if not description:
                description = (
                    "自然放松的自拍，记录当下的日常状态"
                    if selfie
                    else "自然放松的日常人像照片"
                )
            actor = self._actor_id(event)
            scopes = [actor, "shared"]
            prepared = await self.coordinator.prepare(
                actor_id=actor,
                state_owner=actor,
                scopes=scopes,
                description=description,
                selfie=selfie,
                size=size,
                outfit_hint=outfit,
                scene=scene,
            )
            if preview_only or preview or (
                self.settings.preview_by_default() and not explicit
            ):
                return event.plain_result(self.coordinator.format_preview(prepared))
            await event.send(MessageChain([Plain("正在生成图片，请稍候…")]))
            result = await self.coordinator.generate(prepared)
            components = [Plain(result.reply_text)]
            image_path = Path(result.image_path)
            if image_path.is_file():
                components.append(Image.fromFileSystem(str(image_path)))
            return event.chain_result(components)
        except PluginError as exc:
            return event.plain_result(self._error_text(exc))
        except Exception as exc:  # noqa: BLE001 - 生图指令边界必须兜底
            logger.exception("生图指令失败。")
            return event.plain_result(self._error_text(exc))

    # ---------- gallery ----------

    @filter.command("图库", alias={"参考图"})
    async def gallery_command(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """管理角色参考图库。"""
        problem = self._startup_problem()
        if problem:
            yield event.plain_result(problem)
            return
        try:
            result = await self._handle_gallery(event, str(text))
        except PluginError as exc:
            result = self._error_text(exc)
        except Exception as exc:  # noqa: BLE001 - 图库指令边界必须兜底
            result = self._error_text(exc)
        yield event.plain_result(result)

    @filter.command("添加参考图", alias={"上传参考图"})
    async def add_reference_command(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """连续上传角色参考图。"""
        problem = self._startup_problem()
        if problem:
            yield event.plain_result(problem)
            return
        try:
            result = await self._handle_gallery(event, f"添加 {text}".strip())
        except PluginError as exc:
            result = self._error_text(exc)
        except Exception as exc:  # noqa: BLE001 - 上传指令边界必须兜底
            result = self._error_text(exc)
        yield event.plain_result(result)

    async def _handle_gallery(self, event: AstrMessageEvent, raw: str) -> str:
        shared, text = self._extract_flag(raw, "共享")
        confirm, text = self._extract_flag(text, "确认")
        action, rest, tokens = self._split_action(text)
        actor = self._actor_id(event)
        view_scopes = ["shared"] if shared else [actor, "shared"]

        edit_actions = {
            "添加",
            "上传",
            "add",
            "权重",
            "weight",
            "weights",
            "主图",
            "primary",
            "启用",
            "停用",
            "enable",
            "disable",
            "重命名",
            "改名",
            "rename",
            "说明",
            "备注",
            "note",
            "标签",
            "tag",
            "tags",
            "删除",
            "delete",
        }
        read_only = (
            action in {"标签", "tag", "tags"}
            and (not tokens or tokens[0] in {"列表", "list"})
        ) or (
            action in {"权重", "weight", "weights"}
            and len(tokens) < 2
        )
        if action in edit_actions and not read_only:
            self._require_gallery_edit(event)
            edit_scope = self._edit_scope(event, shared)
        else:
            edit_scope = "shared" if shared else actor
        modify_scopes = [edit_scope]

        if action in {"", "列表", "list", "ls"}:
            references = await self.gallery.list_references(view_scopes)
            if not references:
                return "参考图库为空。发送 /添加参考图 开始上传第一张参考图。"
            return self._format_reference_list(references)

        if action in {"搜索", "查找", "search"}:
            query = " ".join(tokens) or rest.strip()
            if not query:
                raise ValidationError("请输入搜索关键词。")
            found: dict[str, ReferenceImage] = {}
            terms = self._tokenize(query) or [query]
            for term in terms:
                for reference in await self.gallery.find_references(view_scopes, term):
                    found[reference.id] = reference
            references = list(found.values())
            if not references:
                references = await self.gallery.find_references(view_scopes, query)
            return self._format_reference_list(references)

        if action in {"详情", "查看", "info"}:
            query = " ".join(tokens) or rest.strip()
            reference = await self._resolve_reference(view_scopes, query)
            return self._format_reference(reference)

        if action in {"添加", "上传", "add"}:
            caption = rest.strip()
            return await self._run_upload_session(event, edit_scope, caption)

        if action in {"权重", "weight", "weights"}:
            if len(tokens) < 1:
                raise ValidationError("用法：/图库 权重 <名称或ID> [面部=高 发型=60 ...]")
            reference = await self._resolve_reference(modify_scopes, tokens[0])
            changes = tokens[1:]
            if not changes:
                return (
                    f"《{reference.name}》当前权重：{self._weights_brief(reference)}\n"
                    "修改示例：/图库 权重 主参考 面部=高 发型=60 场景=关\n"
                    "发送“重置”可恢复默认权重。"
                )
            weights = dict(reference.weights)
            applied: list[str] = []
            index = 0
            while index < len(changes):
                token = changes[index]
                if token in {"重置", "默认", "reset"}:
                    weights = dict(DEFAULT_WEIGHTS)
                    applied.append("已重置为默认权重")
                    index += 1
                    continue
                if "=" in token or "＝" in token:
                    key, value = re.split(r"[=＝]", token, maxsplit=1)
                else:
                    if index + 1 >= len(changes):
                        raise ValidationError(f"“{token}”缺少权重值。")
                    key, value = token, changes[index + 1]
                    index += 1
                dimension = normalize_dimension(key)
                if dimension is None:
                    raise ValidationError(
                        f"未知维度“{key}”。可用维度：面部、发型、服装、场景、姿势、构图光影。"
                    )
                try:
                    weight = coerce_weight(value)
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                if dimension == "all":
                    for key_name in WEIGHT_DIMENSIONS:
                        weights[key_name] = weight
                    applied.append(f"全部={weight}")
                else:
                    weights[dimension] = weight
                    applied.append(f"{DIMENSION_LABELS[dimension]}={weight}")
                index += 1
            updated = await self.gallery.update_metadata(
                reference.id,
                edit_scope,
                weights=weights,
            )
            return (
                f"已更新《{updated.name}》的权重：{'，'.join(applied)}。\n"
                f"当前：{self._weights_brief(updated)}"
            )

        if action in {"主图", "primary"}:
            if not tokens:
                raise ValidationError("请提供要设为主图的参考图名称或 ID。")
            reference = await self._resolve_reference(modify_scopes, " ".join(tokens))
            updated = await self.gallery.set_primary(reference.id, edit_scope)
            return f"已将《{updated.name}》设为主图。"

        if action in {"启用", "停用", "enable", "disable"}:
            if not tokens:
                raise ValidationError("请提供参考图名称或 ID。")
            enabled = action in {"启用", "enable"}
            reference = await self._resolve_reference(modify_scopes, " ".join(tokens))
            updated = await self.gallery.set_enabled(reference.id, edit_scope, enabled)
            return f"已{'启用' if enabled else '停用'}《{updated.name}》。"

        if action in {"重命名", "改名", "rename"}:
            if len(tokens) < 2:
                raise ValidationError("用法：/图库 重命名 <名称或ID> <新名称>")
            reference = await self._resolve_reference(modify_scopes, tokens[0])
            updated = await self.gallery.update_metadata(
                reference.id,
                edit_scope,
                name=" ".join(tokens[1:]),
            )
            return f"已将参考图重命名为《{updated.name}》。"

        if action in {"说明", "备注", "note"}:
            if len(tokens) < 2:
                raise ValidationError("用法：/图库 说明 <名称或ID> <新备注>")
            reference = await self._resolve_reference(modify_scopes, tokens[0])
            updated = await self.gallery.update_metadata(
                reference.id,
                edit_scope,
                note=" ".join(tokens[1:]),
            )
            return f"已更新《{updated.name}》的说明。"

        if action in {"标签", "tag", "tags"}:
            return await self._handle_gallery_tags(
                view_scopes,
                edit_scope,
                tokens,
            )

        if action in {"删除", "delete"}:
            if not confirm:
                raise ValidationError("删除操作需要确认：/图库 删除 <名称或ID> --确认")
            if not tokens:
                raise ValidationError("请提供要删除的参考图名称或 ID。")
            reference = await self._resolve_reference(modify_scopes, " ".join(tokens))
            deleted = await self.gallery.delete(reference.id, edit_scope)
            return f"已删除参考图《{deleted.name}》。"

        return (
            "未知的图库操作。可用：列表、搜索、详情、添加、权重、主图、启用、停用、"
            "重命名、说明、标签、删除。"
        )

    async def _handle_gallery_tags(
        self,
        view_scopes: list[str],
        edit_scope: str,
        tokens: list[str],
    ) -> str:
        sub = tokens[0] if tokens else "列表"
        if sub in {"列表", "list", ""}:
            tags = await self.gallery.all_tags(view_scopes)
            if not tags:
                return "当前没有标签。"
            return "标签统计：" + "、".join(f"#{tag}({count})" for tag, count in tags[:30])
        if sub in {"添加", "add"}:
            if len(tokens) < 3:
                raise ValidationError("用法：/图库 标签 添加 <名称或ID> <标签...>")
            reference = await self._resolve_reference([edit_scope], tokens[1])
            new_tags = list(dict.fromkeys([*reference.tags, *tokens[2:]]))
            updated = await self.gallery.update_metadata(
                reference.id,
                edit_scope,
                tags=new_tags,
            )
            return f"已更新《{updated.name}》的标签：{'、'.join(updated.tags)}"
        if sub in {"删除", "delete"}:
            if len(tokens) < 2:
                raise ValidationError("用法：/图库 标签 删除 <标签>")
            changed = await self.gallery.delete_tag(edit_scope, tokens[1])
            return f"已从 {changed} 张参考图中删除标签“{tokens[1]}”。"
        if sub in {"重命名", "改名", "rename"}:
            if len(tokens) < 3:
                raise ValidationError("用法：/图库 标签 重命名 <旧标签> <新标签>")
            changed = await self.gallery.rename_tag(edit_scope, tokens[1], tokens[2])
            return f"已将标签“{tokens[1]}”重命名为“{tokens[2]}”，涉及 {changed} 张参考图。"
        raise ValidationError("标签操作支持：列表、添加、删除、重命名。")

    # ---------- upload session ----------

    async def _run_upload_session(
        self,
        event: AstrMessageEvent,
        scope: str,
        caption: str,
    ) -> str:
        self._require_gallery_edit(event)
        timeout = self.settings.integer("gallery", "upload_timeout", 120, 30, 600)
        max_images = self.settings.integer("gallery", "upload_max_images", 10, 1, 50)
        max_mb = self.settings.integer("gallery", "max_image_mb", 10, 1, 50)
        origin = event.unified_msg_origin
        sender = self._sender(event)
        session_id = f"{PLUGIN_NAME}:upload:{origin}:{sender}:{new_id('session')}"
        summary: dict[str, Any] = {
            "saved": 0,
            "failed": 0,
            "cancelled": False,
            "timed_out": False,
        }
        current_caption = caption.strip()

        class UploadFilter(SessionFilter):
            plugin_name = PLUGIN_NAME

            def filter(self, other: AstrMessageEvent) -> str:
                if other.unified_msg_origin != origin:
                    return ""
                if str(other.get_sender_id() or "unknown") != sender:
                    return ""
                return session_id

        scope_label = "共享图库" if scope == "shared" else "个人图库"
        await event.send(
            MessageChain(
                [
                    Plain(
                        f"开始向{scope_label}上传参考图（最多 {max_images} 张，{timeout} 秒内有效）。\n"
                        "直接发送图片即可；可以先发送“名称|备注|标签1,标签2”作为说明。\n"
                        "发送“完成”结束，发送“取消”放弃本次上传。"
                    )
                ]
            )
        )

        async def handler(controller: SessionController, reply_event: AstrMessageEvent) -> None:
            nonlocal current_caption
            if self._closed:
                controller.stop()
                return
            text = self._clean(reply_event.get_message_str())
            images = [
                component
                for component in reply_event.get_messages()
                if isinstance(component, Image)
            ]
            if not images:
                if text in {"完成", "结束", "done", "finish"}:
                    controller.stop()
                    return
                if text in {"取消", "cancel"}:
                    summary["cancelled"] = True
                    controller.stop()
                    return
                if text.startswith("/"):
                    await reply_event.send(
                        MessageChain(
                            [Plain("上传会话尚未结束：请发送图片，或发送“完成”/“取消”。")]
                        )
                    )
                    controller.keep(timeout=timeout)
                    return
                if text:
                    current_caption = text
                    await reply_event.send(
                        MessageChain([Plain("已记录这条文字作为后续图片的说明。")])
                    )
                else:
                    await reply_event.send(
                        MessageChain([Plain("请发送图片，或发送“完成”/“取消”。")])
                    )
                controller.keep(timeout=timeout)
                return

            remaining = max_images - summary["saved"]
            if remaining <= 0:
                await reply_event.send(
                    MessageChain([Plain(f"本次已达到 {max_images} 张上限，上传结束。")])
                )
                controller.stop()
                return

            caption_text = current_caption or text
            for index, component in enumerate(images[:remaining], start=1):
                if self._closed:
                    controller.stop()
                    return
                try:
                    image_path = Path(await component.convert_to_file_path())
                    size_bytes = await asyncio.to_thread(lambda p=image_path: p.stat().st_size)
                    if size_bytes > max_mb * 1024 * 1024 * 2:
                        raise ValidationError("图片文件过大。")
                    data = await asyncio.to_thread(image_path.read_bytes)
                    name, note, tags = self._parse_upload_caption(caption_text, index)
                    reference = await self.gallery.save_reference(
                        scope,
                        name=name,
                        note=note,
                        tags=tags,
                        data=data,
                    )
                    summary["saved"] += 1
                    await reply_event.send(
                        MessageChain(
                            [
                                Plain(
                                    f"已保存《{reference.name}》"
                                    f"（{reference.width}x{reference.height}）。"
                                )
                            ]
                        )
                    )
                except (PluginError, OSError, ValueError) as exc:
                    summary["failed"] += 1
                    message = (
                        exc.user_message
                        if isinstance(exc, PluginError)
                        else "读取或保存图片失败。"
                    )
                    await reply_event.send(
                        MessageChain([Plain(f"这张图片未能保存：{message}")])
                    )
            if summary["saved"] >= max_images:
                await reply_event.send(
                    MessageChain([Plain(f"已达到 {max_images} 张上限，上传结束。")])
                )
                controller.stop()
                return
            controller.keep(timeout=timeout)

        wrapped = session_waiter(timeout=timeout)(handler)
        try:
            await wrapped(event, session_filter=UploadFilter())
        except TimeoutError:
            summary["timed_out"] = True

        if summary["cancelled"]:
            prefix = "已取消上传。"
        elif summary["timed_out"]:
            prefix = "上传会话超时结束。"
        else:
            prefix = "参考图上传完成。"
        lines = [
            prefix,
            f"本次保存 {summary['saved']} 张，失败 {summary['failed']} 张。",
        ]
        if summary["saved"]:
            lines.append("可以用 /图库 列表 或 /图库 权重 <名称> 继续调整。")
        return "\n".join(lines)

    @staticmethod
    def _parse_upload_caption(
        caption: str,
        index: int,
    ) -> tuple[str, str, list[str]]:
        parts = [
            OnlyMyImagePlugin._strip_quotes(part.strip())
            for part in (caption or "").split("|", 2)
        ]
        name = parts[0] if parts and parts[0] else ""
        note = parts[1] if len(parts) > 1 else ""
        tags = split_list(parts[2]) if len(parts) > 2 else []
        if not name:
            name = time.strftime("参考图 %m-%d %H:%M:%S")
        if index > 1:
            name = f"{name} {index}"
        return name[:40], note[:500], tags

    # ---------- wardrobe ----------

    @filter.command("衣柜")
    async def wardrobe_command(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """管理智能衣柜与当前穿搭。"""
        problem = self._startup_problem()
        if problem:
            yield event.plain_result(problem)
            return
        try:
            result = await self._handle_wardrobe(event, str(text))
        except PluginError as exc:
            result = self._error_text(exc)
        except Exception as exc:  # noqa: BLE001 - 衣柜指令边界必须兜底
            result = self._error_text(exc)
        yield event.plain_result(result)

    async def _handle_wardrobe(self, event: AstrMessageEvent, raw: str) -> str:
        shared, text = self._extract_flag(raw, "共享")
        confirm, text = self._extract_flag(text, "确认")
        show_all, text = self._extract_flag(text, "全部")
        action, rest, tokens = self._split_action(text)
        actor = self._actor_id(event)
        view_scopes = ["shared"] if shared else [actor, "shared"]
        sub_action = tokens[0] if tokens else ""
        mutating = action in {
            "添加",
            "新增",
            "add",
            "删除",
            "delete",
            "启用",
            "停用",
            "enable",
            "disable",
            "穿上",
            "穿",
            "wear",
            "脱下",
            "脱",
            "remove",
            "换",
            "swap",
            "还原",
            "restore",
            "撤销",
            "自动搭配",
            "自动",
            "auto",
        }
        if action in {"分类", "category", "categories", "套装", "outfit", "outfits"}:
            mutating = mutating or sub_action in {"添加", "新增", "add", "删除", "delete"}
        edit_scope = (
            self._edit_scope(event, shared)
            if shared and mutating
            else ("shared" if shared else actor)
        )
        category_scopes = [edit_scope, "shared"] if not shared else ["shared"]

        if action in {"", "当前", "状态"}:
            lines = await self._state_lines(actor, [actor, "shared"])
            categories = await self.wardrobe.categories([actor, "shared"])
            items = await self.wardrobe.list_items([actor, "shared"])
            lines.append(f"衣柜包含 {len(items)} 件衣物、{len(categories)} 个分类。")
            lines.append("用 /衣柜 列表 查看明细；/衣柜 自动搭配 让衣柜按活动自行选择。")
            return "\n".join(lines)

        if action in {"列表", "list", "ls"}:
            category = None
            if tokens:
                category = await self.wardrobe.resolve_category(view_scopes, tokens[0])
            items = await self.wardrobe.list_items(view_scopes, enabled_only=not show_all)
            categories = {
                entry.key: entry for entry in await self.wardrobe.categories(view_scopes)
            }
            if category is not None:
                items = [
                    item
                    for item in items
                    if categories.get(item.category_key)
                    and categories[item.category_key].kind == category.kind
                ]
            if not items:
                return "衣柜中没有匹配的衣物。"
            lines = [f"共 {len(items)} 件衣物："]
            for item in items[:30]:
                category_label = (
                    categories[item.category_key].label
                    if item.category_key in categories
                    else item.category_key
                )
                flags = []
                if item.is_base:
                    flags.append("基础")
                if not item.enabled:
                    flags.append("停用")
                detail = "、".join(value for value in (item.color, item.material) if value)
                lines.append(
                    f"- {item.name} [{item.id}]｜{category_label}"
                    f"｜{detail or '无颜色材质'}｜优先级{item.priority}"
                    + (f"（{'、'.join(flags)}）" if flags else "")
                )
            if len(items) > 30:
                lines.append(f"…… 还有 {len(items) - 30} 件，请用分类筛选。")
            return "\n".join(lines)

        if action in {"添加", "新增", "add"}:
            self._require_wardrobe_edit(event)
            return await self._add_wardrobe_item(
                edit_scope,
                category_scopes,
                tokens,
            )

        if action in {"删除", "delete"}:
            self._require_wardrobe_edit(event)
            if not confirm:
                raise ValidationError("删除衣物需要确认：/衣柜 删除 <名称或ID> --确认")
            if not tokens:
                raise ValidationError("请提供要删除的衣物名称或 ID。")
            item = await self._resolve_item(
                [edit_scope],
                " ".join(tokens),
                include_base=False,
            )
            deleted = await self.wardrobe.delete_item(
                item.id,
                edit_scope,
                state_owner=actor,
                scopes=[actor, "shared"],
            )
            return f"已删除衣物《{deleted.name}》。"

        if action in {"启用", "停用", "enable", "disable"}:
            self._require_wardrobe_edit(event)
            if not tokens:
                raise ValidationError("请提供衣物名称或 ID。")
            item = await self._resolve_item([edit_scope], " ".join(tokens))
            enabled = action in {"启用", "enable"}
            updated = await self.wardrobe.update_item(
                item.id,
                edit_scope,
                enabled=enabled,
            )
            return f"已{'启用' if enabled else '停用'}《{updated.name}》。"

        if action in {"穿上", "穿", "wear"}:
            self._require_wardrobe_edit(event)
            if not tokens:
                raise ValidationError("用法：/衣柜 穿上 <衣物...>")
            _state, items, notes = await self.wardrobe.wear(
                actor,
                [actor, "shared"],
                tokens,
            )
            description = await self.wardrobe.outfit_description(
                actor,
                [actor, "shared"],
                items,
            )
            lines = [f"已更新穿搭：{description}"]
            if notes:
                lines.append("提示：" + "；".join(notes))
            return "\n".join(lines)

        if action in {"脱下", "脱", "remove"}:
            self._require_wardrobe_edit(event)
            target = " ".join(tokens) or rest.strip()
            if not target:
                raise ValidationError("用法：/衣柜 脱下 <衣物或分类>")
            _state, items, notes = await self.wardrobe.remove(
                actor,
                [actor, "shared"],
                target,
            )
            description = await self.wardrobe.outfit_description(
                actor,
                [actor, "shared"],
                items,
            )
            lines = [f"已更新穿搭：{description}"]
            if notes:
                lines.append("提示：" + "；".join(notes))
            return "\n".join(lines)

        if action in {"换", "swap"}:
            self._require_wardrobe_edit(event)
            if len(tokens) < 2:
                raise ValidationError("用法：/衣柜 换 <分类> <衣物>")
            _state, items, notes = await self.wardrobe.swap(
                actor,
                [actor, "shared"],
                tokens[0],
                " ".join(tokens[1:]),
            )
            description = await self.wardrobe.outfit_description(
                actor,
                [actor, "shared"],
                items,
            )
            lines = [f"已更换衣物：{description}"]
            if notes:
                lines.append("提示：" + "；".join(notes))
            return "\n".join(lines)

        if action in {"还原", "restore", "撤销"}:
            self._require_wardrobe_edit(event)
            _state, items, notes = await self.wardrobe.restore(
                actor,
                [actor, "shared"],
            )
            description = await self.wardrobe.outfit_description(
                actor,
                [actor, "shared"],
                items,
            )
            return "\n".join(
                [f"已还原上一次穿搭：{description}", *([f"提示：{'；'.join(notes)}"] if notes else [])]
            )

        if action in {"自动搭配", "自动", "auto"}:
            self._require_wardrobe_edit(event)
            context = await self.timeline.get_context(actor)
            trusted = context.source in {"plugin", "manual", "request"}
            _state, plan = await self.wardrobe.reset_auto(
                actor,
                [actor, "shared"],
                activity=context.activity if trusted else "",
                season=context.season,
                temperature=context.temperature,
                indoor=context.indoor,
                scene=" ".join(tokens),
            )
            description = await self.wardrobe.outfit_description(
                actor,
                [actor, "shared"],
                plan.items,
            )
            lines = [
                f"已按“{context.describe()}”自动搭配：{description}",
            ]
            if plan.notes:
                lines.append("提示：" + "；".join(plan.notes))
            return "\n".join(lines)

        if action in {"分类", "category", "categories"}:
            return await self._handle_wardrobe_categories(
                event,
                edit_scope,
                tokens,
                confirm,
            )

        if action in {"套装", "outfit", "outfits"}:
            return await self._handle_wardrobe_outfits(
                event,
                actor,
                edit_scope,
                category_scopes,
                tokens,
                confirm,
            )

        return (
            "未知的衣柜操作。可用：列表、添加、删除、启用、停用、穿上、脱下、换、"
            "还原、当前、自动搭配、分类、套装。"
        )

    async def _add_wardrobe_item(
        self,
        edit_scope: str,
        category_scopes: list[str],
        tokens: list[str],
    ) -> str:
        if len(tokens) < 2:
            raise ValidationError(
                "用法：/衣柜 添加 <名称> <分类> [颜色=... 材质=... 季节=春,秋 ...]"
            )
        name = tokens[0]
        category = tokens[1]
        _free, pairs = self._pairs(tokens[2:])
        known = {
            "颜色",
            "color",
            "材质",
            "material",
            "季节",
            "season",
            "seasons",
            "场景",
            "scene",
            "scenes",
            "温度",
            "temp",
            "temperature",
            "正式度",
            "formality",
            "场合",
            "indoor",
            "推荐",
            "推荐搭配",
            "recommended",
            "禁忌",
            "禁忌搭配",
            "forbidden",
            "优先级",
            "priority",
            "参考图",
            "ref",
            "reference",
            "备注",
            "note",
        }
        unknown = [key for key in pairs if key not in known]
        if unknown:
            raise ValidationError(f"未知属性：{'、'.join(unknown)}。")
        reference_id = ""
        reference_query = self._pair(pairs, "参考图", "ref", "reference")
        if reference_query:
            reference = await self._resolve_reference(
                [edit_scope, "shared"] if edit_scope != "shared" else ["shared"],
                reference_query,
            )
            reference_id = reference.id
        try:
            formality = int(self._pair(pairs, "正式度", "formality", default="3"))
            priority = int(self._pair(pairs, "优先级", "priority", default="50"))
        except ValueError as exc:
            raise ValidationError("正式度和优先级必须是数字。") from exc
        item = await self.wardrobe.add_item(
            edit_scope,
            name=name,
            category_query=category,
            scopes=category_scopes,
            color=self._pair(pairs, "颜色", "color"),
            material=self._pair(pairs, "材质", "material"),
            seasons=self._pair(pairs, "季节", "season", "seasons"),
            scenes=self._pair(pairs, "场景", "scene", "scenes"),
            temperature=self._pair(pairs, "温度", "temp", "temperature"),
            formality=formality,
            indoor_outdoor=self._pair(pairs, "场合", "indoor", default="both"),
            recommended_with=self._pair(pairs, "推荐", "推荐搭配", "recommended"),
            forbidden_with=self._pair(pairs, "禁忌", "禁忌搭配", "forbidden"),
            priority=priority,
            reference_id=reference_id,
            note=self._pair(pairs, "备注", "note"),
        )
        return (
            f"已添加衣物《{item.name}》[{item.id}]，分类：{category}。\n"
            "发送 /衣柜 列表 查看，或 /衣柜 穿上 立即换上。"
        )

    async def _handle_wardrobe_categories(
        self,
        event: AstrMessageEvent,
        edit_scope: str,
        tokens: list[str],
        confirm: bool,
    ) -> str:
        sub = tokens[0] if tokens else "列表"
        view_scopes = [edit_scope, "shared"] if edit_scope != "shared" else ["shared"]
        if sub in {"列表", "list", ""}:
            categories = await self.wardrobe.categories(view_scopes)
            lines = ["衣柜分类："]
            for category in categories:
                marker = "自定义" if category.custom else "内置"
                lines.append(
                    f"- {category.label}（key={category.key}，类型={category.kind}，{marker}）"
                )
            return "\n".join(lines)
        if sub in {"添加", "新增", "add"}:
            self._require_wardrobe_edit(event)
            if len(tokens) < 2:
                raise ValidationError("用法：/衣柜 分类 添加 <名称> [类型=其他]")
            _free, pairs = self._pairs(tokens[2:])
            kind = self._pair(pairs, "类型", "kind", default="other")
            category = await self.wardrobe.add_custom_category(
                edit_scope,
                tokens[1],
                kind,
            )
            return f"已添加自定义分类《{category.label}》（key={category.key}）。"
        if sub in {"删除", "delete"}:
            self._require_wardrobe_edit(event)
            if not confirm:
                raise ValidationError("删除分类需要确认：/衣柜 分类 删除 <名称> --确认")
            if len(tokens) < 2:
                raise ValidationError("请提供自定义分类名称。")
            category = await self.wardrobe.delete_custom_category(
                edit_scope,
                " ".join(tokens[1:]),
            )
            return f"已删除自定义分类《{category.label}》。"
        raise ValidationError("分类操作支持：列表、添加、删除。")

    async def _handle_wardrobe_outfits(
        self,
        event: AstrMessageEvent,
        actor: str,
        edit_scope: str,
        category_scopes: list[str],
        tokens: list[str],
        confirm: bool,
    ) -> str:
        sub = tokens[0] if tokens else "列表"
        view_scopes = (
            ["shared"] if edit_scope == "shared" else [actor, "shared"]
        )
        if sub in {"列表", "list", ""}:
            outfits = await self.wardrobe.list_outfits(view_scopes)
            if not outfits:
                return "还没有保存的套装。用 /衣柜 套装 添加 <名称> <衣物...> 创建。"
            lines = [f"共 {len(outfits)} 套："]
            for outfit in outfits:
                owner = "共享" if outfit.owner_id == "shared" else "个人"
                lines.append(
                    f"- 《{outfit.name}》 [{outfit.id}]｜{owner}｜{len(outfit.item_ids)} 件"
                )
            return "\n".join(lines)
        if sub in {"添加", "新增", "add"}:
            self._require_wardrobe_edit(event)
            if len(tokens) < 3:
                raise ValidationError("用法：/衣柜 套装 添加 <名称> <衣物1> <衣物2> ...")
            outfit = await self.wardrobe.create_outfit(
                edit_scope,
                name=tokens[1],
                queries=tokens[2:],
                scopes=category_scopes,
            )
            return f"已创建套装《{outfit.name}》[{outfit.id}]，包含 {len(outfit.item_ids)} 件衣物。"
        if sub in {"删除", "delete"}:
            self._require_wardrobe_edit(event)
            if not confirm:
                raise ValidationError("删除套装需要确认：/衣柜 套装 删除 <名称> --确认")
            if len(tokens) < 2:
                raise ValidationError("请提供套装名称或 ID。")
            outfit = await self._resolve_outfit([edit_scope], " ".join(tokens[1:]))
            await self.wardrobe.delete_outfit(edit_scope, outfit.id)
            return f"已删除套装《{outfit.name}》。"
        if sub in {"穿上", "wear"}:
            self._require_wardrobe_edit(event)
            if len(tokens) < 2:
                raise ValidationError("请提供套装名称或 ID。")
            outfit = await self._resolve_outfit(view_scopes, " ".join(tokens[1:]))
            _state, items, notes = await self.wardrobe.wear_outfit(
                actor,
                view_scopes,
                outfit.id,
            )
            description = await self.wardrobe.outfit_description(
                actor,
                view_scopes,
                items,
            )
            lines = [f"已穿上套装《{outfit.name}》：{description}"]
            if notes:
                lines.append("提示：" + "；".join(notes))
            return "\n".join(lines)
        raise ValidationError("套装操作支持：列表、添加、删除、穿上。")

    async def _resolve_outfit(self, scopes: list[str], query: str):
        outfits = await self.wardrobe.list_outfits(scopes)
        text = (query or "").strip().casefold()
        exact = [
            outfit
            for outfit in outfits
            if text in {outfit.id.casefold(), outfit.name.casefold()}
        ]
        if len(exact) == 1:
            return exact[0]
        partial = [
            outfit
            for outfit in outfits
            if text in outfit.name.casefold() or text in outfit.id.casefold()
        ]
        candidates = exact or partial
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise NotFoundError(f"没有找到套装“{query}”。")
        names = "、".join(f"《{outfit.name}》({outfit.id})" for outfit in candidates[:8])
        raise ValidationError(f"“{query}”匹配到多套服装：{names}。请使用完整名称或 ID。")

    # ---------- API management ----------

    @filter.command("接口", alias={"生图接口"})
    async def api_command(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """管理生图接口（仅管理员）。"""
        problem = self._startup_problem()
        if problem:
            yield event.plain_result(problem)
            return
        try:
            result = await self._handle_api(event, str(text))
        except PluginError as exc:
            result = self._error_text(exc)
        except Exception as exc:  # noqa: BLE001 - 接口指令边界必须兜底
            result = self._error_text(exc)
        yield event.plain_result(result)

    async def _handle_api(self, event: AstrMessageEvent, raw: str) -> str:
        if not self._is_admin(event):
            raise PermissionDeniedError("接口管理仅限管理员。")
        confirm, text = self._extract_flag(raw, "确认")
        action, rest, tokens = self._split_action(text)

        if action in {"", "列表", "list", "ls"}:
            endpoints = await self.api.list_endpoint_info()
            lines = ["生图接口（密钥已隐藏）："]
            for endpoint in endpoints:
                flags = []
                if endpoint.primary:
                    flags.append("默认")
                if endpoint.enabled:
                    flags.append("启用")
                else:
                    flags.append("停用")
                lines.append(
                    f"- {endpoint.name}｜{endpoint.model}｜{endpoint.base_url}"
                    f"｜密钥：{endpoint.masked_key}｜优先级{endpoint.priority}"
                    f"｜{'、'.join(flags)}"
                )
                if endpoint.last_error:
                    lines.append(f"  最近错误：{endpoint.last_error}")
            return "\n".join(lines)

        if action in {"测试", "test"}:
            query = " ".join(tokens) or rest.strip()
            endpoint = await self.api.resolve_endpoint_for_test(query)
            result = await self.api.test_endpoint(endpoint)
            return f"接口《{endpoint.name}》测试结果：{result.message}"

        if action in {"添加", "add"}:
            free, pairs = self._pairs(tokens)
            name = self._pair(pairs, "名称", "name") or " ".join(free)
            if not name:
                raise ValidationError("用法：/接口 添加 <名称> 地址=... 模型=... 密钥=...")
            timeout_text = self._pair(pairs, "超时", "timeout")
            endpoint = await self.api.add_backup(
                name=name,
                base_url=self._pair(pairs, "地址", "url", "base_url", "base"),
                model=self._pair(pairs, "模型", "model"),
                api_key=self._pair(pairs, "密钥", "key", "api_key"),
                priority=self._to_int(
                    self._pair(pairs, "优先级", "priority", default="100"),
                    "优先级",
                ),
                timeout=self._to_float(timeout_text, "超时") if timeout_text else None,
                edit_field=self._pair(
                    pairs,
                    "字段",
                    "field",
                    "edit_field",
                    default="image[]",
                ),
            )
            return (
                f"已添加备用接口《{endpoint.name}》[{endpoint.id}]。\n"
                "密钥已保存且不会回显；建议优先在插件配置中管理默认接口。"
            )

        if action in {"修改", "update", "edit"}:
            query, pairs = self._resolve_api_query(tokens)
            changes: dict[str, Any] = {}
            mapping = {
                "地址": "base_url",
                "url": "base_url",
                "base": "base_url",
                "base_url": "base_url",
                "模型": "model",
                "model": "model",
                "密钥": "api_key",
                "key": "api_key",
                "api_key": "api_key",
                "优先级": "priority",
                "priority": "priority",
                "超时": "timeout",
                "timeout": "timeout",
                "字段": "edit_field",
                "field": "edit_field",
                "edit_field": "edit_field",
            }
            for key, value in pairs.items():
                if key not in mapping:
                    continue
                target = mapping[key]
                if target == "priority":
                    changes[target] = self._to_int(value, "优先级")
                elif target == "timeout":
                    changes[target] = self._to_float(value, "超时")
                else:
                    changes[target] = value
            if not changes:
                raise ValidationError("没有提供要修改的字段。")
            endpoint = await self.api.update_backup(query, **changes)
            return f"已更新备用接口《{endpoint.name}》。密钥不会回显。"

        if action in {"启用", "停用", "enable", "disable"}:
            if not tokens:
                raise ValidationError("请提供备用接口名称或 ID。")
            endpoint = await self.api.update_backup(
                " ".join(tokens),
                enabled=action in {"启用", "enable"},
            )
            return f"已{'启用' if endpoint.enabled else '停用'}接口《{endpoint.name}》。"

        if action in {"删除", "delete"}:
            if not confirm:
                raise ValidationError("删除接口需要确认：/接口 删除 <名称或ID> --确认")
            if not tokens:
                raise ValidationError("请提供备用接口名称或 ID。")
            endpoint = await self.api.delete_backup(" ".join(tokens))
            return f"已删除备用接口《{endpoint.name}》。"

        if action in {"错误", "最近错误", "errors"}:
            logs = await self.storage.recent_api_errors(10)
            if not logs:
                return "最近没有接口错误记录。"
            lines = ["最近接口错误："]
            for log in logs:
                status = f" HTTP {log.status_code}" if log.status_code else ""
                lines.append(
                    f"- {self._format_time(log.created_at)}｜{log.endpoint_name}{status}"
                    f"｜{log.message or '无详情'}"
                )
            return "\n".join(lines)

        if action in {"帮助", "help"}:
            return (
                "接口操作：列表、测试、添加、修改、启用、停用、删除、错误。\n"
                "示例：/接口 测试 默认；/接口 添加 备用一 地址=https://api.example/v1 "
                "模型=gpt-image-2 密钥=sk-xxx"
            )
        return "未知的接口操作。可用：列表、测试、添加、修改、启用、停用、删除、错误。"

    @staticmethod
    def _resolve_api_query(tokens: list[str]) -> tuple[str, dict[str, str]]:
        free, pairs = OnlyMyImagePlugin._pairs(tokens)
        name = OnlyMyImagePlugin._pair(pairs, "名称", "name")
        pairs = {key: value for key, value in pairs.items() if key not in {"名称", "name"}}
        query = name or " ".join(free)
        if not query:
            raise ValidationError("请提供备用接口名称或 ID。")
        return query, pairs

    # ---------- character / timeline ----------

    @filter.command("形象", alias={"角色"})
    async def image_profile_command(self, event: AstrMessageEvent, text: GreedyStr = ""):
        """查看形象、时间轴与插件运行状态。"""
        problem = self._startup_problem()
        if problem:
            yield event.plain_result(problem)
            return
        try:
            result = await self._handle_profile(event, str(text))
        except PluginError as exc:
            result = self._error_text(exc)
        except Exception as exc:  # noqa: BLE001 - 形象指令边界必须兜底
            result = self._error_text(exc)
        yield event.plain_result(result)

    @filter.command("生图帮助", alias={"我的形象"})
    async def help_command(self, event: AstrMessageEvent):
        """查看角色生图插件的完整指令帮助。"""
        yield event.plain_result(_HELP_TEXT)

    async def _handle_profile(self, event: AstrMessageEvent, raw: str) -> str:
        action, _rest, tokens = self._split_action(raw)
        actor = self._actor_id(event)
        view_scopes = [actor, "shared"]
        if not action:
            references, _warnings = await self.gallery.list_available(view_scopes)
            items = await self.wardrobe.list_items(view_scopes)
            state, worn, _notes = await self.wardrobe.get_state(actor, view_scopes)
            description = await self.wardrobe.outfit_description(actor, view_scopes, worn)
            context = await self.timeline.get_context(actor)
            endpoints = await self.api.list_endpoint_info()
            lines = [
                f"【{self.settings.character_name()}】",
                f"风格：{self.settings.character_style()}",
                f"可用参考图：{len(references)} 张｜衣物：{len(items)} 件",
                f"当前穿搭（{state.mode}）：{description}",
                f"时间轴：{context.describe()}",
                f"接口：{len(endpoints)} 个（{sum(1 for item in endpoints if item.enabled)} 个启用）",
                "发送 /生图帮助 查看全部指令。",
            ]
            return "\n".join(lines)

        if action in {"活动", "时间轴", "timeline"}:
            sub = tokens[0] if tokens else "查看"
            if sub in {"设置", "set"}:
                _free, pairs = self._pairs(tokens[1:])
                activity_text = " ".join(
                    token for token in tokens[1:] if "=" not in token and "＝" not in token
                ).strip()
                if not activity_text:
                    raise ValidationError(
                        "用法：/形象 活动 设置 <活动> [地点=... 天气=... 温度=... 室内=是 季节=秋]"
                    )
                data = await self.timeline.set_manual(
                    actor,
                    activity=activity_text,
                    location=self._pair(pairs, "地点", "location"),
                    weather=self._pair(pairs, "天气", "weather"),
                    temperature=self._pair(pairs, "温度", "temperature") or None,
                    indoor=self._pair(pairs, "室内", "indoor") or None,
                    season=self._pair(pairs, "季节", "season"),
                )
                return (
                    "已更新手动时间轴："
                    + str(data.get("activity") or "日常状态")
                    + "。发送 /形象 活动 查看，或 /形象 活动 清除 恢复自动。"
                )
            if sub in {"清除", "重置", "clear"}:
                await self.timeline.clear_manual(actor)
                return "已清除手动时间轴，将回退到插件适配或本地时段。"
            context = await self.timeline.get_context(actor)
            status = await self.timeline.status(actor)
            lines = [
                f"当前时间轴：{context.describe()}",
                f"来源：{context.source}",
                f"状态：{status}",
            ]
            if context.note:
                lines.append(f"说明：{context.note}")
            return "\n".join(lines)

        if action in {"状态", "status"}:
            references, warnings = await self.gallery.list_available(view_scopes)
            items = await self.wardrobe.list_items(view_scopes)
            endpoints = await self.api.list_endpoint_info()
            tasks = await self.coordinator.recent_tasks(actor, 3)
            lines = [
                "插件状态：正常运行",
                f"数据层：{'已连接' if self._ready else '未就绪'}",
                f"参考图：{len(references)} 张可用" + (f"，{len(warnings)} 条警告" if warnings else ""),
                f"衣物：{len(items)} 件",
                f"生图接口：{len(endpoints)} 个",
                f"API Key：{'已配置' if any(item.masked_key != '未配置' for item in endpoints) else '未配置'}",
                f"LLM 工具：{'已注册' if self._tool_registered else '未注册'}",
            ]
            if tasks:
                lines.append("最近任务：")
                for task in tasks:
                    lines.append(
                        f"- {task.id}｜{task.status}｜{self._format_time(task.updated_at)}"
                    )
            if self._is_admin(event):
                lines.append(f"数据目录：{self.data_dir}")
                if self._startup_error:
                    lines.append(f"启动错误：{self._startup_error}")
            return "\n".join(lines)

        if action in {"帮助", "help"}:
            return _HELP_TEXT
        return "形象操作支持：活动、状态、帮助。"


__all__ = ["OnlyMyImagePlugin"]
