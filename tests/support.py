from __future__ import annotations

import base64
import importlib.util
import logging
import struct
import sys
import types
import zlib
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


class Plain:
    def __init__(self, text: str):
        self.text = str(text)


class Image:
    def __init__(self, path: str = ""):
        self.path = str(path)

    @classmethod
    def fromFileSystem(cls, path: str) -> Image:
        return cls(path)

    async def convert_to_file_path(self) -> str:
        return self.path


class MessageChain:
    def __init__(self, chain: list[Any] | None = None):
        self.chain = list(chain or [])


def chain_text(chain: Any) -> str:
    components = getattr(chain, "chain", chain)
    if isinstance(components, str):
        return components
    parts: list[str] = []
    for component in components or []:
        if isinstance(component, Plain):
            parts.append(component.text)
        elif isinstance(component, Image):
            parts.append(f"[image:{component.path}]")
        else:
            parts.append(str(component))
    return "\n".join(parts)


class FakeResult:
    def __init__(self, kind: str, value: Any):
        self.kind = kind
        self.value = value

    @property
    def text(self) -> str:
        if self.kind == "plain":
            return str(self.value)
        if self.kind == "chain":
            return chain_text(self.value)
        return str(self.value)


class FakeEvent:
    def __init__(
        self,
        *,
        sender: str = "user-a",
        platform: str = "test",
        origin: str = "test:private:user-a",
        admin: bool = False,
        message_str: str = "",
        messages: list[Any] | None = None,
        upload_replies: list[FakeEvent] | None = None,
    ):
        self.sender = sender
        self.platform = platform
        self.origin = origin
        self.admin = admin
        self.message_str = message_str
        self.messages = list(messages or [])
        self.upload_replies = upload_replies or []
        self.sent: list[Any] = []
        self.stopped = False
        self.last_session_filter: Any = None

    @property
    def unified_msg_origin(self) -> str:
        return self.origin

    @property
    def is_admin(self) -> bool:
        return self.admin

    def get_sender_id(self) -> str:
        return self.sender

    def get_sender_name(self) -> str:
        return self.sender

    def get_platform_name(self) -> str:
        return self.platform

    def get_message_str(self) -> str:
        return self.message_str

    def get_messages(self) -> list[Any]:
        return self.messages

    def plain_result(self, text: str) -> FakeResult:
        return FakeResult("plain", text)

    def chain_result(self, components: list[Any]) -> FakeResult:
        return FakeResult("chain", components)

    async def send(self, chain: Any) -> None:
        self.sent.append(chain)

    def stop_event(self) -> None:
        self.stopped = True

    def sent_text(self) -> str:
        return "\n".join(chain_text(chain) for chain in self.sent)


class ProviderRequest:
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt


class SessionFilter:
    plugin_name = ""

    def filter(self, event: Any) -> str:
        return ""


class SessionController:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def keep(self, timeout: float | None = None) -> None:
        self.stopped = False


def session_waiter(timeout: float | None = None):
    def decorator(handler):
        async def wrapper(event, session_filter=None):
            event.last_session_filter = session_filter
            controller = SessionController()
            replies = list(getattr(event, "upload_replies", []) or [])
            if not replies:
                raise TimeoutError
            for reply in replies:
                await handler(controller, reply)
                if controller.stopped:
                    break

        return wrapper

    return decorator


class _Filter:
    @staticmethod
    def command(name: str, alias: set[str] | None = None):
        def decorator(func):
            func._astrbot_command_name = name
            func._astrbot_command_aliases = set(alias or set())
            return func

        return decorator

    @staticmethod
    def llm_tool(name: str | None = None):
        def decorator(func):
            func._astrbot_llm_tool_name = name
            return func

        return decorator

    @staticmethod
    def on_llm_request():
        def decorator(func):
            return func

        return decorator


class _ToolManager:
    def __init__(self, registered: bool):
        self.registered = registered
        self._tools = {"generate_character_image": object()} if registered else {}

    def get_func(self, name: str):
        return self._tools.get(name)


class _StarTools:
    data_root: Path | None = None

    @classmethod
    def get_data_dir(cls, name: str = "") -> Path:
        root = cls.data_root or ROOT / "data" / "plugin_data"
        return Path(root) / name


class _Context:
    def __init__(self, tool_registered: bool = True, stars: dict[str, Any] | None = None):
        self._tool_manager = _ToolManager(tool_registered)
        self._stars = dict(stars or {})

    def get_llm_tool_manager(self) -> _ToolManager:
        return self._tool_manager

    def get_registered_star(self, name: str):
        return self._stars.get(name)


class _Star:
    def __init__(self, context: Any, config: dict | None = None):
        self.context = context
        self.config = config or {}


FakeContext = _Context


def _make_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def install_astrbot_stubs() -> None:
    logger = logging.getLogger("astrbot.stub")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    astrbot = _make_module("astrbot")
    api = _make_module("astrbot.api", logger=logger)
    event = _make_module(
        "astrbot.api.event",
        AstrMessageEvent=FakeEvent,
        MessageChain=MessageChain,
        filter=_Filter(),
    )
    components = _make_module("astrbot.api.message_components", Image=Image, Plain=Plain)
    provider = _make_module("astrbot.api.provider", ProviderRequest=ProviderRequest)
    star = _make_module(
        "astrbot.api.star",
        Context=_Context,
        Star=_Star,
        StarTools=_StarTools,
    )
    core = _make_module("astrbot.core")
    core_star = _make_module("astrbot.core.star")
    core_filter = _make_module("astrbot.core.star.filter")
    command = _make_module("astrbot.core.star.filter.command", GreedyStr=str)
    utils = _make_module("astrbot.core.utils")
    waiter = _make_module(
        "astrbot.core.utils.session_waiter",
        SessionController=SessionController,
        SessionFilter=SessionFilter,
        session_waiter=session_waiter,
        USER_SESSIONS={},
        FILTERS=[],
    )
    for parent, child in [
        (astrbot, api),
        (astrbot, core),
        (api, event),
        (api, components),
        (api, provider),
        (api, star),
        (core, core_star),
        (core_star, core_filter),
        (core_filter, command),
        (core, utils),
        (utils, waiter),
    ]:
        attr = child.__name__.rsplit(".", 1)[-1]
        setattr(parent, attr, child)


def set_data_root(path: Path) -> None:
    _StarTools.data_root = Path(path)


def load_plugin_module():
    """以真实包名加载 main.py，保证相对导入与测试使用同一批服务模块。"""

    name = "astrbot_plugin_only_my_image"
    main_name = f"{name}.main"
    if main_name in sys.modules:
        return sys.modules[main_name]

    import services
    import services.errors
    import services.gallery
    import services.generator
    import services.image_api
    import services.models
    import services.prompt_builder
    import services.reference_selector
    import services.settings
    import services.storage
    import services.timeline
    import services.utils
    import services.wardrobe  # noqa: F401

    package = _make_module(name)
    package.__path__ = [str(ROOT)]  # type: ignore[attr-defined]
    for module_name, module in list(sys.modules.items()):
        if module_name == "services" or module_name.startswith("services."):
            sys.modules[f"{name}.{module_name}"] = module

    spec = importlib.util.spec_from_file_location(main_name, ROOT / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[main_name] = module
    spec.loader.exec_module(module)
    return module


def make_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "api": {
            "enabled": True,
            "base_url": "https://primary.test/v1",
            "model": "gpt-image-2",
            "api_key": "sk-primary-test-secret",
            "timeout": 10,
            "max_retries": 0,
            "require_reference": False,
            "quality": "auto",
            "output_format": "png",
            "edit_field": "image[]",
            "allow_private_download_urls": False,
        },
        "character": {
            "name": "小只",
            "persona": "温和、自然、有生活感",
            "style": "自然真实的生活人像照片",
        },
        "gallery": {
            "allow_user_upload": True,
            "max_images_per_scope": 20,
            "max_image_mb": 5,
            "allowed_formats": "png,jpg,jpeg,webp",
            "upload_timeout": 30,
            "upload_max_images": 5,
        },
        "wardrobe": {"allow_user_edit": True, "max_items": 100},
        "generation": {
            "size": "1024x1024",
            "selfie_size": "768x1024",
            "max_concurrency": 2,
            "queue_timeout": 10,
            "max_reference_images": 6,
            "dedupe_window": 0,
            "allow_user_generation": True,
            "max_result_mb": 10,
            "max_prompt_chars": 6000,
            "default_negative": "",
        },
        "timeline": {
            "mode": "off",
            "plugin_name": "",
            "function_name": "",
            "timezone_offset": 8,
            "southern_hemisphere": False,
        },
        "permissions": {"admin_ids": []},
        "features": {"on_llm_request_hint": True, "preview_by_default": False},
    }
    _deep_merge(config, overrides)
    return config


def _deep_merge(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def make_png(width: int = 2, height: int = 2, color: tuple[int, int, int] = (180, 30, 30)) -> bytes:
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes(color) * width
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )


def b64_image_response(data: bytes, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"data": [{"b64_json": base64.b64encode(data).decode("ascii")}]},
        headers={"content-type": "application/json"},
    )


async def noop_sleep(_seconds: float) -> None:
    return None


async def collect_results(agen) -> list[FakeResult]:
    return [item async for item in agen]


class AppHarness:
    def __init__(self, data_dir: Path, **parts: Any):
        self.data_dir = data_dir
        for key, value in parts.items():
            setattr(self, key, value)
        self._closed = False

    @classmethod
    async def create(
        cls,
        data_dir: Path,
        *,
        config: dict[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver=None,
        clock=None,
    ) -> AppHarness:
        from services.gallery import GalleryService
        from services.generator import GenerationCoordinator
        from services.image_api import ImageApiManager
        from services.prompt_builder import PromptBuilder
        from services.reference_selector import ReferenceSelector
        from services.settings import Settings
        from services.storage import Storage
        from services.timeline import TimelineService
        from services.wardrobe import WardrobeService

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        raw_config = config or make_config()
        settings = Settings(raw_config)
        storage = Storage(data_dir / "only_my_image.sqlite3")
        await storage.open()
        gallery = GalleryService(storage, data_dir, settings)
        wardrobe = WardrobeService(storage, settings)
        timeline = TimelineService(storage, settings, resolver=resolver, clock=clock)
        selector = ReferenceSelector(settings)
        prompts = PromptBuilder(settings)
        api = ImageApiManager(storage, settings, transport=transport, sleep=noop_sleep)
        coordinator = GenerationCoordinator(
            storage,
            gallery,
            wardrobe,
            selector,
            timeline,
            api,
            prompts,
            settings,
            data_dir,
        )
        return cls(
            data_dir,
            config=raw_config,
            settings=settings,
            storage=storage,
            gallery=gallery,
            wardrobe=wardrobe,
            timeline=timeline,
            selector=selector,
            prompts=prompts,
            api=api,
            coordinator=coordinator,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.coordinator.close()
        await self.api.close()
        await self.storage.close()


install_astrbot_stubs()
plugin_main = load_plugin_module()
