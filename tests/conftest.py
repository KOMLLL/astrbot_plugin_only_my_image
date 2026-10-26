from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support import (
    AppHarness,
    FakeContext,
    FakeEvent,
    make_config,
    plugin_main,
    set_data_root,
)


@pytest.fixture
def config_factory():
    return make_config


@pytest.fixture
def event_factory():
    def factory(**kwargs):
        return FakeEvent(**kwargs)

    return factory


@pytest.fixture
async def app_factory(tmp_path):
    created: list[AppHarness] = []
    counter = 0

    async def factory(*, config=None, transport=None, resolver=None, clock=None):
        nonlocal counter
        counter += 1
        app = await AppHarness.create(
            tmp_path / f"app-{counter}",
            config=config or make_config(),
            transport=transport,
            resolver=resolver,
            clock=clock,
        )
        created.append(app)
        return app

    yield factory
    for app in reversed(created):
        await app.close()


@pytest.fixture
async def plugin_factory(tmp_path):
    created = []
    counter = 0

    async def factory(
        *,
        config=None,
        transport=None,
        tool_registered: bool = True,
        stars: dict | None = None,
    ):
        nonlocal counter
        counter += 1
        set_data_root(tmp_path / f"plugin-data-{counter}")
        context = FakeContext(tool_registered=tool_registered, stars=stars)
        plugin = plugin_main.OnlyMyImagePlugin(context, config or make_config())
        if transport is not None:
            plugin.api._transport = transport
        from tests.support import noop_sleep

        plugin.api._sleep = noop_sleep
        await plugin.initialize()
        created.append(plugin)
        return plugin

    yield factory
    for plugin in reversed(created):
        await plugin.terminate()
