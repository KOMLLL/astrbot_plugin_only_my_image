from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from .errors import ConfigError, ImageApiError, NotFoundError, ValidationError
from .log import logger
from .models import ApiEndpoint, RequestLog
from .settings import Settings
from .storage import Storage
from .utils import (
    build_images_url,
    build_models_url,
    detect_image,
    ext_for_mime,
    mask_secret,
    mask_url,
    new_id,
    normalize_base_url,
    sanitize_error_text,
)

MAX_RESPONSE_BYTES = 30 * 1024 * 1024
MAX_ERROR_BYTES = 8 * 1024
_EDIT_FIELD_RE = re.compile(r"^[A-Za-z0-9_.\[\]-]{1,40}$")
_RESERVED_FORM_FIELDS = {
    "model",
    "prompt",
    "image",
    "image[]",
    "n",
    "size",
    "quality",
    "output_format",
}


@dataclass(slots=True)
class ReferenceInput:
    filename: str
    data: bytes
    mime: str


@dataclass(slots=True)
class ImageRequest:
    prompt: str
    negative_prompt: str = ""
    size: str = ""
    quality: str = "auto"
    output_format: str = "png"
    references: list[ReferenceInput] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def combined_prompt(self) -> str:
        if not self.negative_prompt.strip():
            return self.prompt
        return f"{self.prompt}\n\n需要避免出现：{self.negative_prompt}"


@dataclass(slots=True)
class ImagePayload:
    content: bytes
    mime: str
    ext: str
    revised_prompt: str = ""
    endpoint_name: str = ""


@dataclass(slots=True)
class EndpointTestResult:
    ok: bool
    message: str
    status_code: int | None = None
    latency_ms: int = 0


@dataclass(slots=True)
class EndpointInfo:
    id: str
    name: str
    base_url: str
    model: str
    masked_key: str
    enabled: bool
    priority: int
    timeout: float
    primary: bool
    last_ok_at: float | None = None
    last_error: str = ""
    fail_count: int = 0


class ImageApiManager:
    def __init__(
        self,
        storage: Storage,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ):
        self.storage = storage
        self.settings = settings
        self._transport = transport
        self._sleep = sleep or asyncio.sleep
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                follow_redirects=True,
                headers={"User-Agent": "astrbot_plugin_only_my_image/1.0"},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()

    def primary_endpoint(self) -> ApiEndpoint:
        return ApiEndpoint(
            id="primary",
            name="默认接口",
            base_url=self.settings.api_base_url(),
            model=self.settings.api_model(),
            api_key=self.settings.api_key(),
            enabled=self.settings.boolean("api", "enabled", True),
            priority=0,
            timeout=self.settings.api_timeout(),
            edit_field=self.settings.string("api", "edit_field", "image[]") or "image[]",
            is_primary=True,
            created_at=0.0,
        )

    async def enabled_endpoints(self) -> list[ApiEndpoint]:
        endpoints: list[ApiEndpoint] = []
        primary = self.primary_endpoint()
        if primary.enabled:
            endpoints.append(primary)
        endpoints.extend(await self.storage.list_api_endpoints(enabled_only=True))
        return endpoints

    async def _resolve_backup(self, query: str) -> ApiEndpoint:
        text = (query or "").strip()
        if not text:
            raise ValidationError("请提供备用接口名称或 ID。")
        endpoint = await self.storage.get_api_endpoint(text)
        if endpoint is None:
            endpoint = await self.storage.find_api_endpoint_by_name(text)
        if endpoint is None:
            raise NotFoundError("未找到该备用接口。")
        return endpoint

    async def add_backup(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str = "",
        priority: int = 100,
        timeout: float | None = None,
        edit_field: str = "image[]",
    ) -> ApiEndpoint:
        clean_name = " ".join((name or "").split())[:30]
        if not clean_name:
            raise ValidationError("接口名称不能为空。")
        if clean_name == "默认接口":
            raise ValidationError("“默认接口”是保留名称，请换一个名称。")
        if await self.storage.find_api_endpoint_by_name(clean_name):
            raise ValidationError("已存在同名备用接口。")
        if not (model or "").strip():
            raise ValidationError("模型名称不能为空。")
        if not _EDIT_FIELD_RE.fullmatch(edit_field or ""):
            raise ValidationError("图片字段名只能包含字母、数字、点、下划线和中括号。")
        endpoint = ApiEndpoint(
            id=new_id("api"),
            name=clean_name,
            base_url=normalize_base_url(base_url),
            model=model.strip(),
            api_key=(api_key or "").strip(),
            enabled=True,
            priority=max(0, min(9999, int(priority))),
            timeout=max(10.0, min(600.0, float(timeout or self.settings.api_timeout()))),
            edit_field=edit_field,
            created_at=time.time(),
        )
        await self.storage.add_api_endpoint(endpoint)
        return endpoint

    async def update_backup(
        self,
        query: str,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        priority: int | None = None,
        timeout: float | None = None,
        edit_field: str | None = None,
        enabled: bool | None = None,
    ) -> ApiEndpoint:
        endpoint = await self._resolve_backup(query)
        if endpoint.is_primary:
            raise ValidationError("默认接口请在插件配置中修改。")
        if base_url is not None:
            endpoint.base_url = normalize_base_url(base_url)
        if model is not None:
            if not model.strip():
                raise ValidationError("模型名称不能为空。")
            endpoint.model = model.strip()
        if api_key is not None:
            endpoint.api_key = api_key.strip()
        if priority is not None:
            endpoint.priority = max(0, min(9999, int(priority)))
        if timeout is not None:
            endpoint.timeout = max(10.0, min(600.0, float(timeout)))
        if edit_field is not None:
            if not _EDIT_FIELD_RE.fullmatch(edit_field):
                raise ValidationError("图片字段名格式不合法。")
            endpoint.edit_field = edit_field
        if enabled is not None:
            endpoint.enabled = enabled
        if not await self.storage.update_api_endpoint(endpoint):
            raise NotFoundError("接口更新失败，可能已被删除。")
        return endpoint

    async def delete_backup(self, query: str) -> ApiEndpoint:
        endpoint = await self._resolve_backup(query)
        if not await self.storage.delete_api_endpoint(endpoint.id):
            raise NotFoundError("接口删除失败，可能已被删除。")
        return endpoint

    async def list_endpoint_info(self) -> list[EndpointInfo]:
        all_endpoints = [self.primary_endpoint(), *await self.storage.list_api_endpoints()]
        return [
            EndpointInfo(
                id=endpoint.id,
                name=endpoint.name,
                base_url=mask_url(endpoint.base_url),
                model=endpoint.model,
                masked_key=mask_secret(endpoint.api_key),
                enabled=endpoint.enabled,
                priority=endpoint.priority,
                timeout=endpoint.timeout,
                primary=endpoint.is_primary,
                last_ok_at=endpoint.last_ok_at,
                last_error=sanitize_error_text(endpoint.last_error, api_key=endpoint.api_key),
                fail_count=endpoint.fail_count,
            )
            for endpoint in all_endpoints
        ]

    async def resolve_endpoint_for_test(self, query: str) -> ApiEndpoint:
        text = (query or "").strip()
        if not text or text in {"默认", "默认接口", "primary"}:
            return self.primary_endpoint()
        return await self._resolve_backup(text)

    async def test_endpoint(self, endpoint: ApiEndpoint) -> EndpointTestResult:
        started = time.perf_counter()
        url = build_models_url(endpoint.base_url)
        headers = self._auth_headers(endpoint, json_body=False)
        client = await self._ensure_client()
        try:
            async with client.stream(
                "GET",
                url,
                headers=headers,
                timeout=httpx.Timeout(min(30.0, endpoint.timeout), connect=10.0),
            ) as response:
                body = await self._read_limited(response, MAX_ERROR_BYTES)
                latency = int((time.perf_counter() - started) * 1000)
                if response.status_code < 400:
                    message = f"连接成功（HTTP {response.status_code}，{latency}ms）。"
                    result = EndpointTestResult(True, message, response.status_code, latency)
                elif response.status_code in {401, 403}:
                    result = EndpointTestResult(
                        False,
                        f"接口可达，但鉴权失败（HTTP {response.status_code}）。请检查 API Key。",
                        response.status_code,
                        latency,
                    )
                elif response.status_code == 404:
                    result = EndpointTestResult(
                        False,
                        "接口可达，但模型列表地址返回 404；地址或版本路径可能不正确。",
                        response.status_code,
                        latency,
                    )
                else:
                    detail = sanitize_error_text(
                        body.decode("utf-8", "ignore"),
                        api_key=endpoint.api_key,
                        limit=120,
                    )
                    result = EndpointTestResult(
                        False,
                        f"接口返回 HTTP {response.status_code}：{detail}",
                        response.status_code,
                        latency,
                    )
        except httpx.TimeoutException:
            result = EndpointTestResult(
                False,
                "连接超时，请检查地址和网络。",
                None,
                int((time.perf_counter() - started) * 1000),
            )
        except httpx.HTTPError as exc:
            result = EndpointTestResult(
                False,
                f"连接失败：{sanitize_error_text(type(exc).__name__, api_key=endpoint.api_key)}",
                None,
                int((time.perf_counter() - started) * 1000),
            )
        except ImageApiError as exc:
            result = EndpointTestResult(
                False,
                exc.user_message,
                exc.status_code,
                int((time.perf_counter() - started) * 1000),
            )
        if not endpoint.is_primary:
            await self.storage.record_api_result(
                endpoint.id,
                ok=result.ok,
                error="" if result.ok else result.message,
                status_code=result.status_code,
            )
        return result

    async def generate(self, request: ImageRequest) -> ImagePayload:
        endpoints = await self.enabled_endpoints()
        if not endpoints:
            raise ConfigError("没有启用任何生图接口，请在配置或接口管理中启用。")
        failures: list[str] = []
        for endpoint in endpoints:
            try:
                payload = await self._generate_with_retries(endpoint, request)
                await self._record_success(endpoint)
                payload.endpoint_name = endpoint.name
                return payload
            except ImageApiError as exc:
                failures.append(f"{endpoint.name}：{exc.user_message}")
                await self._record_failure(endpoint, exc)
                logger.warning(
                    "生图接口失败 endpoint=%s status=%s retryable=%s",
                    endpoint.name,
                    exc.status_code,
                    exc.retryable,
                )
                continue
        message = "所有生图接口均失败。"
        if failures:
            message += " " + "；".join(failures[:4])
        raise ImageApiError(sanitize_error_text(message, limit=500))

    async def _record_success(self, endpoint: ApiEndpoint) -> None:
        if endpoint.is_primary:
            await self.storage.add_api_log(
                RequestLog(
                    id=new_id("log"),
                    endpoint_name=endpoint.name,
                    ok=True,
                    message="生图成功",
                    created_at=time.time(),
                )
            )
            return
        await self.storage.record_api_result(endpoint.id, ok=True)
        await self.storage.add_api_log(
            RequestLog(
                id=new_id("log"),
                endpoint_name=endpoint.name,
                ok=True,
                message="生图成功",
                created_at=time.time(),
            )
        )

    async def _record_failure(self, endpoint: ApiEndpoint, error: ImageApiError) -> None:
        safe_message = sanitize_error_text(error.user_message, api_key=endpoint.api_key, limit=200)
        if not endpoint.is_primary:
            await self.storage.record_api_result(
                endpoint.id,
                ok=False,
                error=safe_message,
                status_code=error.status_code,
            )
        await self.storage.add_api_log(
            RequestLog(
                id=new_id("log"),
                endpoint_name=endpoint.name,
                ok=False,
                status_code=error.status_code,
                message=safe_message,
                created_at=time.time(),
            )
        )

    async def _generate_with_retries(
        self,
        endpoint: ApiEndpoint,
        request: ImageRequest,
    ) -> ImagePayload:
        attempts = self.settings.api_max_retries() + 1
        last_error: ImageApiError | None = None
        for attempt in range(attempts):
            try:
                return await self._single_request(endpoint, request)
            except ImageApiError as exc:
                last_error = exc
                if not exc.retryable or attempt >= attempts - 1:
                    raise
                delay = min(5.0, 0.8 * (2**attempt))
                logger.info(
                    "重试生图接口 endpoint=%s attempt=%s delay=%.1fs",
                    endpoint.name,
                    attempt + 1,
                    delay,
                )
                await self._sleep(delay)
        if last_error:
            raise last_error
        raise ImageApiError("接口请求失败。")

    async def _single_request(
        self,
        endpoint: ApiEndpoint,
        request: ImageRequest,
    ) -> ImagePayload:
        client = await self._ensure_client()
        is_edit = bool(request.references)
        url = build_images_url(endpoint.base_url, "edits" if is_edit else "generations")
        prompt = request.combined_prompt()
        headers = self._auth_headers(endpoint, json_body=not is_edit)
        timeout = httpx.Timeout(endpoint.timeout, connect=min(15.0, endpoint.timeout))
        try:
            if is_edit:
                form, files = self._multipart_payload(endpoint, request, prompt)
                response_ctx = client.stream(
                    "POST",
                    url,
                    headers=headers,
                    data=form,
                    files=files,
                    timeout=timeout,
                )
            else:
                payload = self._generation_payload(endpoint, request, prompt)
                response_ctx = client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            async with response_ctx as response:
                if response.status_code >= 400:
                    error_body = await self._read_limited(response, MAX_ERROR_BYTES)
                    raise self._api_error_from_response(
                        endpoint,
                        response.status_code,
                        error_body,
                        response.headers.get("retry-after", ""),
                    )
                body = await self._read_limited(response, MAX_RESPONSE_BYTES)
                content_type = response.headers.get("content-type", "")
        except httpx.TimeoutException as exc:
            raise ImageApiError(
                "接口请求超时。",
                retryable=True,
                endpoint_name=endpoint.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ImageApiError(
                f"接口连接失败（{type(exc).__name__}）。",
                retryable=True,
                endpoint_name=endpoint.name,
            ) from exc
        return await self._parse_response(endpoint, body, content_type)

    def _generation_payload(
        self,
        endpoint: ApiEndpoint,
        request: ImageRequest,
        prompt: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": endpoint.model,
            "prompt": prompt,
            "n": 1,
        }
        if request.size:
            payload["size"] = request.size
        if request.quality:
            payload["quality"] = request.quality
        if request.output_format:
            payload["output_format"] = request.output_format
        self._merge_extra(payload, endpoint.extra)
        self._merge_extra(payload, request.extra)
        return payload

    def _multipart_payload(
        self,
        endpoint: ApiEndpoint,
        request: ImageRequest,
        prompt: str,
    ) -> tuple[dict[str, str], list[tuple[str, tuple[str, bytes, str]]]]:
        field_name = endpoint.edit_field if _EDIT_FIELD_RE.fullmatch(endpoint.edit_field or "") else "image[]"
        form: dict[str, str] = {
            "model": endpoint.model,
            "prompt": prompt,
        }
        if request.size:
            form["size"] = request.size
        if request.quality:
            form["quality"] = request.quality
        if request.output_format:
            form["output_format"] = request.output_format
        form["n"] = "1"
        extra: dict[str, Any] = {}
        self._merge_extra(extra, endpoint.extra)
        self._merge_extra(extra, request.extra)
        for key, value in extra.items():
            if key in _RESERVED_FORM_FIELDS:
                continue
            if isinstance(value, (str, int, float, bool)):
                form[key] = str(value)
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, reference in enumerate(request.references, start=1):
            ext = ext_for_mime(reference.mime)
            filename = f"ref-{index}.{ext}"
            files.append((field_name, (filename, reference.data, reference.mime)))
        return form, files

    @staticmethod
    def _merge_extra(target: dict[str, Any], extra: Mapping[str, Any] | None) -> None:
        if not isinstance(extra, Mapping):
            return
        for key, value in extra.items():
            if key in target or not isinstance(key, str):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                target[key] = value

    def _api_error_from_response(
        self,
        endpoint: ApiEndpoint,
        status_code: int,
        body: bytes,
        retry_after: str,
    ) -> ImageApiError:
        raw = body.decode("utf-8", "ignore")
        detail = ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                error = parsed.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or error.get("code") or "")
                elif isinstance(error, str):
                    detail = error
                if not detail:
                    detail = str(parsed.get("message") or parsed.get("detail") or "")
        except (TypeError, ValueError):
            detail = raw
        detail = sanitize_error_text(detail, api_key=endpoint.api_key, limit=160)
        retryable = status_code in {408, 425, 429} or status_code >= 500
        if status_code in {401, 403}:
            message = "接口鉴权失败，请检查 API Key 是否有效。"
        elif status_code == 404:
            message = "接口地址或模型不存在（HTTP 404）。"
        elif status_code == 429:
            message = "接口限流（HTTP 429），请稍后再试。"
        elif status_code >= 500:
            message = f"接口服务异常（HTTP {status_code}）。"
        else:
            message = f"接口拒绝了请求（HTTP {status_code}）。"
        if detail and "<html" not in detail.casefold():
            message += f" {detail}"
        if retry_after and retry_after.isdigit():
            message += f" 建议等待 {retry_after}s。"
        return ImageApiError(
            message,
            retryable=retryable,
            status_code=status_code,
            endpoint_name=endpoint.name,
        )

    async def _read_limited(self, response: httpx.Response, limit: int) -> bytes:
        buffer = bytearray()
        async for chunk in response.aiter_bytes():
            buffer.extend(chunk)
            if len(buffer) > limit:
                raise ImageApiError("接口返回内容过大，已中止读取。", retryable=False)
        return bytes(buffer)

    async def _parse_response(
        self,
        endpoint: ApiEndpoint,
        body: bytes,
        content_type: str,
    ) -> ImagePayload:
        try:
            return await self._parse_response_inner(endpoint, body, content_type)
        except ImageApiError:
            raise
        except Exception as exc:
            raise ImageApiError(
                "接口返回的图片数据无法识别。",
                retryable=False,
                endpoint_name=endpoint.name,
            ) from exc

    async def _parse_response_inner(
        self,
        endpoint: ApiEndpoint,
        body: bytes,
        content_type: str,
    ) -> ImagePayload:
        if content_type.startswith("image/"):
            info = detect_image(body)
            return ImagePayload(body, info.mime, info.ext)
        try:
            data = json.loads(body.decode("utf-8", "ignore"))
        except (TypeError, ValueError) as exc:
            raise ImageApiError(
                "接口返回的不是有效图片或 JSON。",
                retryable=False,
                endpoint_name=endpoint.name,
            ) from exc
        revised_prompt = ""
        if isinstance(data, dict):
            revised_prompt = str(data.get("revised_prompt") or "")
        b64_value = self._find_b64(data)
        if b64_value:
            raw = self._decode_b64(b64_value, endpoint)
            info = detect_image(raw)
            return ImagePayload(raw, info.mime, info.ext, revised_prompt)
        url = self._find_url(data)
        if url:
            raw, mime = await self._download_image(endpoint, url)
            ext = ext_for_mime(mime)
            return ImagePayload(raw, mime, ext, revised_prompt)
        raise ImageApiError(
            "接口响应中没有图片数据（b64_json/url 均缺失）。",
            retryable=False,
            endpoint_name=endpoint.name,
        )

    @staticmethod
    def _find_b64(data: Any) -> str:
        candidates: list[Any] = []
        if isinstance(data, dict):
            candidates.extend(
                [
                    data.get("b64_json"),
                    data.get("image"),
                    data.get("image_base64"),
                ]
            )
            for key in ("data", "images", "output", "result"):
                value = data.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            candidates.extend(
                                [
                                    item.get("b64_json"),
                                    item.get("image"),
                                    item.get("image_base64"),
                                ]
                            )
                elif isinstance(value, dict):
                    candidates.extend(
                        [
                            value.get("b64_json"),
                            value.get("image"),
                            value.get("image_base64"),
                        ]
                    )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                stripped = candidate.strip()
                if stripped.startswith(("http://", "https://")):
                    continue
                return stripped
        return ""

    @staticmethod
    def _find_url(data: Any) -> str:
        candidates: list[Any] = []

        def add_value(value: Any) -> None:
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, dict):
                candidates.extend([value.get("url"), value.get("href")])

        if isinstance(data, dict):
            candidates.extend([data.get("url"), data.get("image_url")])
            for key in ("data", "images", "output", "result"):
                value = data.get(key)
                if isinstance(value, list):
                    for item in value:
                        add_value(item)
                elif isinstance(value, dict):
                    add_value(value)
                elif isinstance(value, str):
                    candidates.append(value)
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                stripped = candidate.strip()
                if stripped.startswith("data:image/"):
                    return stripped
                if stripped.startswith(("http://", "https://")):
                    return stripped
        return ""

    def _decode_b64(self, value: str, endpoint: ApiEndpoint) -> bytes:
        payload = value
        if payload.startswith("data:image/"):
            try:
                payload = payload.split(",", 1)[1]
            except IndexError as exc:
                raise ImageApiError("图片 data URL 无效。", endpoint_name=endpoint.name) from exc
        try:
            raw = base64.b64decode(payload, validate=False)
        except (ValueError, TypeError) as exc:
            raise ImageApiError("图片 Base64 解码失败。", endpoint_name=endpoint.name) from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ImageApiError("接口返回图片过大。", endpoint_name=endpoint.name)
        return raw

    async def _download_image(self, endpoint: ApiEndpoint, url: str) -> tuple[bytes, str]:
        if url.startswith("data:image/"):
            try:
                header, payload = url.split(",", 1)
                mime = header.split(";", 1)[0].split(":", 1)[1]
            except (IndexError, ValueError) as exc:
                raise ImageApiError("图片 data URL 无效。", endpoint_name=endpoint.name) from exc
            raw = self._decode_b64(payload, endpoint)
            info = detect_image(raw)
            return raw, info.mime
        client = await self._ensure_client()
        current = url
        allow_private = self.settings.boolean("api", "allow_private_download_urls", False)
        endpoint_host = (urlsplit(endpoint.base_url).hostname or "").casefold()
        for _ in range(4):
            parsed = urlsplit(current)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ImageApiError("接口返回的图片 URL 无效。", endpoint_name=endpoint.name)
            same_host = parsed.hostname.casefold() == endpoint_host
            if not same_host and not allow_private and self._is_private_host(parsed.hostname):
                raise ImageApiError(
                    "接口返回了指向内网/本机的图片地址，已按安全策略拒绝下载。",
                    endpoint_name=endpoint.name,
                )
            headers = self._auth_headers(endpoint, json_body=False) if same_host else {
                "User-Agent": "astrbot_plugin_only_my_image/1.0"
            }
            try:
                async with client.stream(
                    "GET",
                    current,
                    headers=headers,
                    timeout=httpx.Timeout(endpoint.timeout, connect=min(15.0, endpoint.timeout)),
                    follow_redirects=False,
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location", "").strip()
                        if not location:
                            raise ImageApiError(
                                "接口返回的图片重定向缺少目标地址。",
                                endpoint_name=endpoint.name,
                            )
                        current = str(response.url.join(location))
                        continue
                    if response.status_code >= 400:
                        raise ImageApiError(
                            f"下载生成图片失败（HTTP {response.status_code}）。",
                            retryable=response.status_code >= 500,
                            status_code=response.status_code,
                            endpoint_name=endpoint.name,
                        )
                    body = await self._read_limited(response, MAX_RESPONSE_BYTES)
                    mime = response.headers.get("content-type", "").split(";", 1)[0].strip()
            except httpx.TimeoutException as exc:
                raise ImageApiError(
                    "下载生成图片超时。",
                    retryable=True,
                    endpoint_name=endpoint.name,
                ) from exc
            except httpx.HTTPError as exc:
                raise ImageApiError(
                    "下载生成图片失败。",
                    retryable=True,
                    endpoint_name=endpoint.name,
                ) from exc
            try:
                info = detect_image(body)
            except ValidationError as exc:
                raise ImageApiError(
                    "接口返回的图片内容无法识别。",
                    retryable=False,
                    endpoint_name=endpoint.name,
                ) from exc
            return body, info.mime or mime
        raise ImageApiError(
            "下载生成图片重定向次数过多。",
            endpoint_name=endpoint.name,
        )

    @staticmethod
    def _is_private_host(host: str) -> bool:
        lowered = (host or "").casefold()
        if lowered in {"localhost", "localhost.localdomain"} or lowered.endswith(".local"):
            return True
        try:
            address = ipaddress.ip_address(lowered)
        except ValueError:
            return False
        return (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        )

    @staticmethod
    def _auth_headers(endpoint: ApiEndpoint, *, json_body: bool) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        return headers

    def validate_edit_field(self, value: str) -> str:
        if not _EDIT_FIELD_RE.fullmatch(value or ""):
            raise ValidationError("图片字段名格式不合法。")
        return value
