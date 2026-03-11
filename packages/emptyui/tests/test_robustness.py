"""Phase 6 robustness & API hygiene regressions (issues #30-35).

Each test targets one latent defect from the review; they were all
uncovered by the pre-existing suite.
"""

import json

import pytest
from fastapi.testclient import TestClient

from emptyui.registry import Registry
from emptyui.server.app import create_app
from emptyui.settings import SettingsStore

# --- M-C1/L-C11: settings PUT validation ------------------------------------


def test_settings_put_rejects_non_finite():
    """NaN slipped through the `< 0` check (poisoning the TTL runtime),
    ±inf exploded `round()` with OverflowError (500 instead of 422). NB the
    browser's JSON.stringify maps NaN→null (already rejected), but Python's
    json decoder ACCEPTS NaN/Infinity literals — a raw API client sends
    them; exercise that decode path directly."""
    client = TestClient(create_app(Registry()))
    for literal in ("NaN", "Infinity", "-Infinity"):
        body = json.dumps(
            {"values": {"workspace_ttl_seconds": literal}}).replace(
            f'"{literal}"', literal)
        resp = client.put("/api/settings", content=body,
                          headers={"content-type": "application/json"})
        assert resp.status_code == 422, literal
        # int knob: was a 500 (OverflowError) before the finite guard
        body = json.dumps(
            {"values": {"cache_max_entries": literal}}).replace(
            f'"{literal}"', literal)
        resp = client.put("/api/settings", content=body,
                          headers={"content-type": "application/json"})
        assert resp.status_code == 422, literal
    # the poisoned NaN never reached the runtime: TTL stays off
    view = client.get("/api/settings").json()
    assert view["live"]["workspace_ttl_seconds"]["value"] is None


def test_settings_put_rejects_non_integer():
    """An int knob silently rounded 256.7 → 257 (banker's rounding on top:
    2.5 → 2) — a non-integral value must be rejected, not coerced."""
    client = TestClient(create_app(Registry()))
    resp = client.put(
        "/api/settings", json={"values": {"cache_max_entries": 256.7}})
    assert resp.status_code == 422
    assert "integer" in resp.json()["detail"]


def test_settings_store_rolls_back_failed_save(tmp_path):
    """L-C8: a failed settings-file write must not leave memory claiming a
    value the file never received (the live knobs stay consistent with
    disk; the caller surfaces the error)."""
    store = SettingsStore(path=tmp_path / "s.json")

    class FailingSave(SettingsStore):
        def save(self):
            raise OSError("disk full")

    store.__class__ = FailingSave
    with pytest.raises(OSError):
        store.set_settings({"cache_max_entries": 7})
    assert "cache_max_entries" not in store._settings

    with pytest.raises(OSError):
        store.set_disabled("some-plugin", True)
    assert "some-plugin" not in store._disabled
