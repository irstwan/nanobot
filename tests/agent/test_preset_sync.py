"""Tests for the dynamic preset sync (nanobot/agent/preset_sync.py)."""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error

import pytest

from nanobot.agent import preset_sync
from nanobot.agent.preset_sync import (
    PresetSyncRunner,
    build_synced_presets,
    compute_preset_diff,
    fetch_catalog_ids,
    pinned_preset_names,
)


def _preset(provider: str = "ninerouter", model: str = "m") -> dict:
    return {
        "extraHeaders": None,
        "extraQuery": None,
        "extraBody": None,
        "provider": provider,
        "maxTokens": 8192,
        "contextWindowTokens": 200000,
        "temperature": 0.1,
        "reasoningEffort": None,
        "api": None,
        "model": model,
    }


def _write_config(path, presets, default="keep-me", fallbacks=None):
    raw = {
        "agents": {
            "defaults": {"modelPreset": default, "fallbackModels": fallbacks or []}
        },
        "providers": {
            "ninerouter": {
                "apiBase": "http://example.invalid/v1",
                "apiKey": "test-key",
            }
        },
        "modelPresets": presets,
    }
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return raw


def test_compute_preset_diff_scoped_to_provider_and_pins():
    current = {
        "keep": _preset(model="keep"),
        "gone": _preset(model="gone"),
        "pinned-gone": _preset(model="pinned-gone"),
        "foreign": _preset(provider="other", model="foreign"),
    }
    added, removed = compute_preset_diff(
        current,
        ["keep", "new-one"],
        provider="ninerouter",
        pinned={"pinned-gone"},
    )
    assert added == {"new-one"}
    assert removed == {"gone"}
    assert compute_preset_diff({}, ["x"], provider="ninerouter", pinned=set()) == (
        {"x"},
        set(),
    )


def test_build_synced_presets_preserves_entries_and_adds_template():
    entry = _preset(model="keep")
    entry["temperature"] = 0.7
    synced = build_synced_presets(
        {"keep": entry, "gone": _preset(model="gone")},
        ["keep", "fresh"],
        provider="ninerouter",
        pinned=set(),
    )
    assert synced["keep"] is entry
    assert "gone" not in synced
    assert synced["fresh"] == {
        "extraHeaders": None,
        "extraQuery": None,
        "extraBody": None,
        "provider": "ninerouter",
        "maxTokens": 8192,
        "contextWindowTokens": 200000,
        "temperature": 0.1,
        "reasoningEffort": None,
        "api": None,
        "model": "fresh",
    }
    assert list(synced) == ["keep", "fresh"]


def test_pinned_preset_names_reads_both_key_styles():
    raw = {
        "agents": {
            "defaults": {"modelPreset": "a", "fallback_models": ["b", 3, ""]},
        }
    }
    assert pinned_preset_names(raw) == {"a", "b"}
    assert pinned_preset_names({}) == set()
    assert pinned_preset_names({"agents": {}}) == set()


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_catalog_ids_sends_bearer_and_parses_ids(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["request"] = request
        seen["timeout"] = timeout
        payload = {"data": [{"id": "a/b"}, {"id": "c"}, {"owned_by": "x"}]}
        return _FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert fetch_catalog_ids("http://example.invalid/v1/", "secret-key") == ["a/b", "c"]
    assert seen["request"].get_header("Authorization") == "Bearer secret-key"
    assert seen["request"].full_url == "http://example.invalid/v1/models"
    assert seen["timeout"] == 30.0


def test_fetch_catalog_ids_without_key_sends_no_auth(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["request"] = request
        return _FakeResponse(b'{"data": []}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert fetch_catalog_ids("http://example.invalid/v1", None) == []
    assert seen["request"].get_header("Authorization") is None


def test_fetch_catalog_ids_rejects_http_errors(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="HTTP 401"):
        fetch_catalog_ids("http://example.invalid/v1", "k")


def _runner(tmp_path, fetch, *, cycles=2):
    config_path = tmp_path / "config.json"
    invalidated = []
    runner = PresetSyncRunner(
        config_path=config_path,
        provider="ninerouter",
        interval_seconds=3600.0,
        require_stable_cycles=cycles,
        fetch_ids=fetch,
        invalidate=lambda: invalidated.append(1),
    )
    return config_path, runner, invalidated


def test_runner_applies_after_stable_cycles(tmp_path):
    config_path, runner, invalidated = _runner(
        tmp_path, lambda base, key: ["keep-me", "brand-new"]
    )
    _write_config(
        config_path,
        {"keep-me": _preset(model="keep-me"), "stale": _preset(model="stale")},
    )
    assert asyncio.run(runner.run_once()) is False
    assert set(json.loads(config_path.read_text())["modelPresets"]) == {
        "keep-me",
        "stale",
    }
    assert invalidated == []
    assert asyncio.run(runner.run_once()) is True
    raw = json.loads(config_path.read_text())
    assert set(raw["modelPresets"]) == {"keep-me", "brand-new"}
    assert raw["providers"]["ninerouter"]["apiBase"] == "http://example.invalid/v1"
    assert raw["agents"]["defaults"]["modelPreset"] == "keep-me"
    assert invalidated == [1]
    assert len(list(tmp_path.glob("config.json.bak-preset-sync-*"))) == 1
    assert asyncio.run(runner.run_once()) is False
    assert len(list(tmp_path.glob("config.json.bak-preset-sync-*"))) == 1
    assert invalidated == [1]


def test_runner_flapping_diff_never_applies(tmp_path):
    calls = {"n": 0}

    def fetch(base, key):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return ["keep-me", "model-a"]
        return ["keep-me", "model-b"]

    config_path, runner, invalidated = _runner(tmp_path, fetch)
    _write_config(config_path, {"keep-me": _preset(model="keep-me")})
    for _ in range(4):
        assert asyncio.run(runner.run_once()) is False
    assert invalidated == []
    assert set(json.loads(config_path.read_text())["modelPresets"]) == {"keep-me"}


def test_runner_fetch_failure_resets_pending(tmp_path):
    state = {"fail": False}

    def fetch(base, key):
        if state["fail"]:
            raise ConnectionError("boom")
        return ["keep-me", "brand-new"]

    config_path, runner, invalidated = _runner(tmp_path, fetch)
    _write_config(
        config_path,
        {"keep-me": _preset(model="keep-me"), "stale": _preset(model="stale")},
    )
    assert asyncio.run(runner.run_once()) is False
    state["fail"] = True
    assert asyncio.run(runner.run_once()) is False
    state["fail"] = False
    assert asyncio.run(runner.run_once()) is False
    assert asyncio.run(runner.run_once()) is True
    assert invalidated == [1]


def test_runner_never_removes_default_preset(tmp_path):
    config_path, runner, invalidated = _runner(tmp_path, lambda base, key: ["other"])
    _write_config(config_path, {"keep-me": _preset(model="keep-me")}, default="keep-me")
    assert asyncio.run(runner.run_once()) is False
    assert asyncio.run(runner.run_once()) is True
    raw = json.loads(config_path.read_text())
    assert set(raw["modelPresets"]) == {"keep-me", "other"}


def test_runner_skips_when_provider_has_no_endpoint(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"modelPresets": {}}) + "\n", encoding="utf-8")

    def fetch(base, key):
        raise AssertionError("must not fetch without an endpoint")

    runner = PresetSyncRunner(config_path=config_path, fetch_ids=fetch)
    assert asyncio.run(runner.run_once()) is False
