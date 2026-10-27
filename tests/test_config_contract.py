from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _metadata_fields() -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in (ROOT / "metadata.yaml").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith((" ", "\t", "-", "#")) or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_metadata_is_complete_and_reviewer_friendly():
    fields = _metadata_fields()
    assert fields["name"] == "astrbot_plugin_only_my_image"
    assert fields["display_name"]
    assert fields["author"] == "KOMLLL"
    assert re.fullmatch(r"\d+\.\d+\.\d+", fields["version"]), fields["version"]
    assert not fields["version"].startswith("v")
    assert fields["desc"]
    assert "description" not in fields
    assert fields["repo"] == "https://github.com/KOMLLL/astrbot_plugin_only_my_image"
    assert re.fullmatch(r">=4\.\d+(?:\.\d+)?,\s*<5", fields["astrbot_version"])
    assert "aiocqhttp" in fields.get("support_platforms", "") or "aiocqhttp" in (
        ROOT / "metadata.yaml"
    ).read_text(encoding="utf-8")


def test_config_schema_covers_all_runtime_sections():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    assert set(schema) == {
        "api",
        "character",
        "gallery",
        "wardrobe",
        "generation",
        "timeline",
        "permissions",
        "features",
    }
    for name, section in schema.items():
        assert section["type"] == "object", name
        assert isinstance(section["items"], dict), name

    api = schema["api"]["items"]
    assert api["api_key"]["secret"] is True
    assert api["base_url"]["default"] == "https://api1.weilai.chat/v1"
    assert api["model"]["default"] == "gpt-image-2"
    assert api["edit_field"]["default"] == "image[]"
    assert "reference_weight" not in api

    timeline = schema["timeline"]["items"]
    assert timeline["plugin_name"]["default"] == ""
    assert timeline["function_name"]["default"] == ""
    assert "适配" in timeline["plugin_name"]["hint"]

    assert {"mode", "timezone_offset", "southern_hemisphere"} <= set(timeline)
    assert "allow_user_upload" in schema["gallery"]["items"]
    assert "allow_user_edit" in schema["wardrobe"]["items"]
    assert "on_llm_request_hint" in schema["features"]["items"]
    assert "admin_ids" in schema["permissions"]["items"]

    all_keys = set(_walk_keys(schema))
    assert "reference_weight" not in all_keys
    assert "response_format" not in all_keys


def test_requirements_use_async_http_and_sqlite_only():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    packages = [
        line.strip().split(";")[0].split("=")[0].split(">")[0].split("<")[0]
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert "httpx" in packages
    assert "aiosqlite" in packages
    assert "requests" not in packages
    assert "aiohttp" not in packages


def test_plugin_source_has_no_helloworld_or_stdlib_logging():
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "helloworld" not in main_text.lower()
    assert "from astrbot.api import logger" in main_text
    assert "from astrbot.api.event import AstrMessageEvent" in main_text
    assert "filter" in main_text

    sources = [ROOT / "main.py", *sorted((ROOT / "services").glob("*.py"))]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*import logging\b", text, re.MULTILINE), path
        assert "logging.getLogger" not in text, path
        assert "import requests" not in text, path


def test_services_package_is_complete():
    expected = {
        "__init__.py",
        "errors.py",
        "gallery.py",
        "generator.py",
        "image_api.py",
        "log.py",
        "models.py",
        "prompt_builder.py",
        "reference_selector.py",
        "settings.py",
        "storage.py",
        "timeline.py",
        "utils.py",
        "wardrobe.py",
    }
    assert {path.name for path in (ROOT / "services").glob("*.py")} == expected
