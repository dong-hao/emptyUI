"""Shared fixtures for emptyui framework tests."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from emptyui.registry import build_default_registry
from emptyui.server.app import create_app


@pytest.fixture(autouse=True)
def _no_ambient_emptyui_env(monkeypatch):
    """Clear EMPTYUI_* from the developer's shell for every test, so the suite
    sees the defaults. A test that needs a setting sets it explicitly (M16d)."""
    for name in [k for k in os.environ if k.startswith("EMPTYUI_")]:
        monkeypatch.delenv(name)


@pytest.fixture()
def client():
    # plugin toggles are memory-only by default (§5.1): no settings file is
    # created or read; the autouse fixture above removes the environment
    # overrides so the default app is the same on every machine
    return TestClient(create_app(build_default_registry()))
