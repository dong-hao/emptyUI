"""Shared fixtures for emptyui framework tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from emptyui.registry import build_default_registry
from emptyui.server.app import create_app


@pytest.fixture()
def client():
    return TestClient(create_app(build_default_registry()))
