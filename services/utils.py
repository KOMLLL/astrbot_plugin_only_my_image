from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .errors import ValidationError

_EXT_ALIASES = {"jpeg": "jpg", "jpe": "jpg"}
_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}
_EXT_BY_MIME = {value: key for key, value in _MIME_BY_EXT.items()}
_VERSION_SUFFIX = re.compile(r"/v\d+(?:beta\d*)?$", re.IGNORECASE)


@dataclass(slots=True)
class ImageInfo:
    mime: str
    ext: str
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


def now_ts() -> float:
    return time.time()


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex
    return f"{prefix}_{value}" if prefix else value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clamp_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def safe_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def parse_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "是", "开", "开启", "启用"}:
        return True
    if text in {"0", "false", "no", "n", "off", "否", "关", "关闭", "禁用"}:
        return False
    return default


def split_list(value: str, *, limit: int = 30) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[,，;；、\n]+", value)
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = part.strip()
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def normalize_tags(tags: object, *, limit: int = 12) -> list[str]:
    if isinstance(tags, str):
        raw = split_list(tags, limit=limit * 2)
    elif isinstance(tags, (list, tuple, set)):
        raw = [str(item) for item in tags]
    else:
        raw = []
    result: list[str] = []
    seen: set[str] = set()
    for tag in raw:
        text = str(tag).strip().lstrip("#").strip()
        if not text:
            continue
        text = text[:20]
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def normalize_extensions(values: object) -> set[str]:
    if isinstance(values, str):
        raw = split_list(values)
    elif isinstance(values, (list, tuple, set)):
        raw = [str(item) for item in values]
    else:
        raw = []
    normalized = {_EXT_ALIASES.get(item.strip().lower(), item.strip().lower()) for item in raw}
    return {item for item in normalized if item in _MIME_BY_EXT}


def parse_temperature_range(value: object) -> tuple[int | None, int | None]:
    if value is None or isinstance(value, bool):
        return None, None
    if isinstance(value, (int, float)):
        number = int(value)
        return number, number
    text = str(value).strip().replace("℃", "").replace("°C", "").replace("°c", "")
    if not text:
        return None, None
    match = re.fullmatch(r"(-?\d+)\s*(?:-|~|～|至|到)\s*(-?\d+)", text)
    if match:
        low, high = int(match.group(1)), int(match.group(2))
        return min(low, high), max(low, high)
    match = re.fullmatch(r"-?\d+", text)
    if match:
        number = int(text)
        return number, number
    return None, None


def normalize_base_url(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise ValidationError("接口地址不能为空。")
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValidationError("接口地址必须是完整的 http/https URL。")
    if parsed.username or parsed.password:
        raise ValidationError("接口地址中不允许包含用户名或密码。")
    return text.rstrip("/")


def build_images_url(base_url: str, operation: str) -> str:
    if operation not in {"generations", "edits"}:
        raise ValidationError("未知的图片接口操作。")
    normalized = normalize_base_url(base_url)
    parsed = urlsplit(normalized)
    path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/")
    lowered = path.lower()

    for suffix in ("/images/generations", "/images/edits"):
        if lowered.endswith(suffix):
            path = path[: -len(suffix)]
            lowered = path.lower()
            break

    if lowered.endswith("/images"):
        new_path = f"{path}/{operation}"
    elif not path or path == "/":
        new_path = f"/v1/images/{operation}"
    elif _VERSION_SUFFIX.search(path):
        new_path = f"{path}/images/{operation}"
    else:
        new_path = f"{path}/images/{operation}"

    return urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, ""))


def build_models_url(base_url: str) -> str:
    normalized = normalize_base_url(base_url)
    parsed = urlsplit(normalized)
    path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/")
    lowered = path.lower()
    for suffix in ("/images/generations", "/images/edits", "/images"):
        if lowered.endswith(suffix):
            path = path[: -len(suffix)]
            lowered = path.lower()
            break
    if not path or path == "/":
        new_path = "/v1/models"
    elif _VERSION_SUFFIX.search(path):
        new_path = f"{path}/models"
    else:
        new_path = f"{path}/models"
    return urlunsplit((parsed.scheme, parsed.netloc, new_path, parsed.query, ""))


def mask_url(raw: str) -> str:
    try:
        parsed = urlsplit((raw or "").strip())
    except ValueError:
        return "无效地址"
    if not parsed.scheme or not parsed.netloc:
        return "无效地址"
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}{parsed.path}"


def mask_secret(secret: str) -> str:
    text = (secret or "").strip()
    if not text:
        return "未配置"
    if len(text) <= 8:
        return "****"
    return f"****{text[-4:]}"


def sanitize_error_text(text: object, *, api_key: str = "", limit: int = 300) -> str:
    value = str(text or "")
    if api_key:
        value = value.replace(api_key, "[已隐藏密钥]")
    value = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"',;}]+",
        r"\1[已隐藏]",
        value,
    )
    value = re.sub(
        r"(?i)((?:api[_-]?key|token|secret)\s*[\"']?\s*[:=]\s*[\"']?)[^\s\"',;}]+",
        r"\1[已隐藏]",
        value,
    )
    value = re.sub(r"sk-[A-Za-z0-9._-]{6,}", "[已隐藏密钥]", value)
    value = re.sub(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+", "[图片数据]", value)
    value = " ".join(value.split())
    if len(value) > limit:
        value = value[:limit] + "…"
    return value


def ensure_within(base: Path, target: Path) -> Path:
    try:
        base_resolved = base.resolve()
        target_resolved = target.resolve()
        common = os.path.commonpath([str(base_resolved), str(target_resolved)])
    except (OSError, ValueError) as exc:
        raise ValidationError("文件路径无效。") from exc
    if common != str(base_resolved):
        raise ValidationError("文件路径越界，已拒绝访问。")
    return target_resolved


def safe_relative_path(base: Path, target: Path) -> str:
    resolved = ensure_within(base, target)
    return resolved.relative_to(base.resolve()).as_posix()


def resolve_relative_path(base: Path, relative: str) -> Path:
    rel = (relative or "").strip().replace("\\", "/")
    if not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        raise ValidationError("文件路径无效。")
    candidate = base / rel
    return ensure_within(base, candidate)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _png_size(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None


def _gif_size(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    index = 2
    sof_markers = set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = int.from_bytes(data[index + 2 : index + 4], "big")
        if length < 2:
            return None
        if marker in sof_markers and index + 9 <= len(data):
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            if width and height:
                return width, height
        index += 2 + length
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8L" and data[20:21] == b"\x2f":
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    if chunk == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    return None


def detect_image(data: bytes) -> ImageInfo:
    if not data:
        raise ValidationError("图片内容为空。")
    png = _png_size(data)
    if png:
        return ImageInfo("image/png", "png", png[0], png[1])
    jpeg = _jpeg_size(data)
    if jpeg:
        return ImageInfo("image/jpeg", "jpg", jpeg[0], jpeg[1])
    webp = _webp_size(data)
    if webp:
        return ImageInfo("image/webp", "webp", webp[0], webp[1])
    gif = _gif_size(data)
    if gif:
        return ImageInfo("image/gif", "gif", gif[0], gif[1])
    raise ValidationError("无法识别图片格式，仅支持 PNG、JPEG、WebP 或 GIF。")


def validate_image_bytes(
    data: bytes,
    *,
    max_bytes: int,
    allowed_exts: set[str] | None = None,
    max_pixels: int = 40_000_000,
) -> ImageInfo:
    if not data:
        raise ValidationError("图片内容为空。")
    if len(data) > max_bytes:
        raise ValidationError(f"图片过大，最大允许 {max_bytes // (1024 * 1024)} MB。")
    info = detect_image(data)
    allowed = allowed_exts or {"png", "jpg", "webp"}
    if info.ext not in allowed:
        names = "、".join(sorted(ext.upper() for ext in allowed))
        raise ValidationError(f"不支持的图片格式，仅允许 {names}。")
    if info.width <= 0 or info.height <= 0:
        raise ValidationError("图片尺寸无效。")
    if info.width * info.height > max_pixels:
        raise ValidationError("图片分辨率过高，请压缩后再上传。")
    return info


def mime_for_ext(ext: str) -> str:
    return _MIME_BY_EXT.get(_EXT_ALIASES.get(ext.lower(), ext.lower()), "application/octet-stream")


def ext_for_mime(mime: str) -> str:
    normalized = (mime or "").split(";", 1)[0].strip().lower()
    return _EXT_BY_MIME.get(normalized, "png")


def format_file_size(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def format_local_time(timestamp: float | None, offset_hours: int = 8) -> str:
    try:
        zone = timezone(timedelta(hours=offset_hours))
        moment = datetime.fromtimestamp(timestamp or 0, tz=zone)
    except (OSError, OverflowError, ValueError):
        return "-"
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, limit: int, suffix: str = "…") -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - len(suffix))] + suffix


def looks_like_nudity_request(text: str) -> bool:
    value = (text or "").casefold()
    patterns = (
        "裸体",
        "裸照",
        "裸露",
        "裸身",
        "裸奔",
        "全裸",
        "赤裸",
        "露点",
        "色情",
        "涩图",
        "黄图",
        "不穿衣服",
        "不穿任何",
        "没穿衣服",
        "脱光",
        "光着身子",
        "一丝不挂",
        "nude",
        "naked",
        "topless",
        "no clothes",
    )
    return any(pattern in value for pattern in patterns)
