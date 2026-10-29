"""Follow a provider model catalog for model presets, without restarts.

A background gateway task periodically fetches the provider's
OpenAI-compatible ``/models`` listing, rewrites only the ``modelPresets``
section of the config file, and relies on the existing config watcher to
invalidate the runtime resolver, so new presets appear without a restart.

Presets belonging to other providers are never touched. The configured
default preset and fallback presets are pinned: they are never removed even
when the catalog drops them. A change is applied only after the same diff is
observed on ``require_stable_cycles`` consecutive successful checks, so
short-lived catalog flapping does not churn the preset list.

The provider API key is only ever read from the resolved configuration and
is never logged, stored, or written anywhere by this module.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

_BACKUP_PREFIX = "config.json.bak-preset-sync-"
_BACKUP_KEEP = 10


def preset_template(provider: str, model_id: str) -> dict[str, Any]:
    """Default preset entry for a catalog model id, matching the sync script."""
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
        "model": model_id,
    }


def _entry_provider(entry: Any) -> str | None:
    if isinstance(entry, Mapping):
        provider = entry.get("provider")
        return provider if isinstance(provider, str) else None
    return None


def compute_preset_diff(
    current: Mapping[str, Any],
    live_ids: Collection[str],
    *,
    provider: str,
    pinned: Collection[str],
) -> tuple[set[str], set[str]]:
    """Return ``(added, removed)`` preset names for one catalog snapshot."""
    live = set(live_ids)
    pinned_names = set(pinned)
    added = {model_id for model_id in live if model_id not in current}
    removed = {
        name
        for name, entry in current.items()
        if _entry_provider(entry) == provider
        and name not in live
        and name not in pinned_names
    }
    return added, removed


def build_synced_presets(
    current: Mapping[str, Any],
    live_ids: Collection[str],
    *,
    provider: str,
    pinned: Collection[str],
) -> dict[str, Any]:
    """Presets after applying one catalog snapshot, preserving custom entries."""
    added, removed = compute_preset_diff(
        current, live_ids, provider=provider, pinned=pinned
    )
    synced: dict[str, Any] = {
        name: entry for name, entry in current.items() if name not in removed
    }
    for model_id in sorted(added):
        synced[model_id] = preset_template(provider, model_id)
    return synced


def pinned_preset_names(raw_config: Mapping[str, Any]) -> set[str]:
    """Default + fallback preset names from the raw config; never removed."""
    pinned: set[str] = set()
    agents = raw_config.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, Mapping) else None
    if not isinstance(defaults, Mapping):
        return pinned
    for key in ("modelPreset", "model_preset"):
        name = defaults.get(key)
        if isinstance(name, str) and name:
            pinned.add(name)
    for key in ("fallbackModels", "fallback_models"):
        fallbacks = defaults.get(key)
        if isinstance(fallbacks, list):
            pinned.update(n for n in fallbacks if isinstance(n, str) and n)
    return pinned


def fetch_catalog_ids(
    api_base: str, api_key: str | None, *, timeout: float = 30.0
) -> list[str]:
    """Fetch model ids from an OpenAI-compatible ``/models`` endpoint."""
    url = api_base.rstrip("/") + "/models"
    request = urllib.request.Request(url, method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"catalog endpoint answered HTTP {exc.code}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError("catalog endpoint returned an unexpected payload")
    ids: list[str] = []
    for entry in data:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and model_id:
            ids.append(model_id)
    return ids


def load_provider_endpoint(
    config_path: Path, provider_name: str
) -> tuple[str | None, str | None]:
    """Resolve a provider's ``(api_base, api_key)`` from the config file."""
    from nanobot.config.loader import load_config, resolve_config_env_vars

    config = resolve_config_env_vars(
        load_config(config_path), config_path=config_path
    )
    providers = getattr(config, "providers", None)
    entry = getattr(providers, provider_name, None) if providers is not None else None
    if entry is None:
        return None, None
    api_base = getattr(entry, "api_base", None)
    api_key = getattr(entry, "api_key", None)
    return (
        api_base if isinstance(api_base, str) and api_base else None,
        api_key if isinstance(api_key, str) and api_key else None,
    )


def read_raw_config(config_path: Path) -> dict[str, Any]:
    """Read the config file as raw JSON, preserving every unrelated key."""
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"config file {config_path} does not contain an object")
    return raw


def write_config_backup(config_path: Path) -> Path:
    """Copy the current config file to a dated backup; prune old ones."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_path.with_name(f"{_BACKUP_PREFIX}{stamp}")
    backup.write_bytes(config_path.read_bytes())
    backups = sorted(config_path.parent.glob(_BACKUP_PREFIX + "*"))
    for stale in backups[: max(0, len(backups) - _BACKUP_KEEP)]:
        stale.unlink(missing_ok=True)
    return backup


def write_raw_config(config_path: Path, raw: dict[str, Any]) -> None:
    """Atomically replace the config file, preserving its JSON shape."""
    text = json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
    tmp_path = config_path.with_name(config_path.name + ".tmp-preset-sync")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, config_path)


@dataclass
class PresetSyncRunner:
    """Periodically sync model presets from a provider catalog."""

    config_path: Path
    provider: str = "ninerouter"
    interval_seconds: float = 1800.0
    require_stable_cycles: int = 2
    fetch_ids: Callable[[str, str | None], list[str]] = fetch_catalog_ids
    invalidate: Callable[[], None] = lambda: None
    _pending: tuple[frozenset[str], frozenset[str]] | None = field(
        default=None, init=False, repr=False
    )
    _pending_count: int = field(default=0, init=False, repr=False)

    async def run_forever(self) -> None:
        """Check forever until cancelled; a failed check never kills the loop."""
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("preset sync check failed unexpectedly")
            await asyncio.sleep(self.interval_seconds)

    async def run_once(self) -> bool:
        """Run one check; return True when presets were rewritten."""
        try:
            api_base, api_key = load_provider_endpoint(
                self.config_path, self.provider
            )
        except Exception:
            logger.warning(
                "preset sync: could not load provider '{}' from config; skipping",
                self.provider,
            )
            self._clear_pending()
            return False
        if not api_base:
            logger.warning(
                "preset sync: provider '{}' has no api base; skipping", self.provider
            )
            self._clear_pending()
            return False
        try:
            live_ids = await asyncio.to_thread(self.fetch_ids, api_base, api_key)
        except Exception as exc:
            logger.warning(
                "preset sync: catalog fetch failed ({}); skipping",
                type(exc).__name__,
            )
            self._clear_pending()
            return False
        try:
            raw = read_raw_config(self.config_path)
        except Exception:
            logger.warning("preset sync: could not read config file; skipping")
            self._clear_pending()
            return False
        presets = raw.get("modelPresets")
        if not isinstance(presets, dict):
            presets = {}
        pinned = pinned_preset_names(raw)
        added, removed = compute_preset_diff(
            presets, live_ids, provider=self.provider, pinned=pinned
        )
        signature = (frozenset(added), frozenset(removed))
        if not added and not removed:
            self._clear_pending()
            logger.info(
                "preset sync: no preset changes for provider '{}'", self.provider
            )
            return False
        if signature == self._pending:
            self._pending_count += 1
        else:
            self._pending = signature
            self._pending_count = 1
        if self._pending_count < self.require_stable_cycles:
            logger.info(
                "preset sync: change observed for provider '{}' (+{} -{}), "
                "stable {}/{}",
                self.provider,
                len(added),
                len(removed),
                self._pending_count,
                self.require_stable_cycles,
            )
            return False
        try:
            fresh = read_raw_config(self.config_path)
        except Exception:
            logger.warning("preset sync: could not re-read config file; skipping")
            self._clear_pending()
            return False
        fresh_presets = fresh.get("modelPresets")
        if not isinstance(fresh_presets, dict):
            fresh_presets = {}
        fresh_pinned = pinned_preset_names(fresh)
        added_now, removed_now = compute_preset_diff(
            fresh_presets, live_ids, provider=self.provider, pinned=fresh_pinned
        )
        if (frozenset(added_now), frozenset(removed_now)) != signature:
            logger.info("preset sync: diff changed before apply; re-recording")
            if added_now or removed_now:
                self._pending = (frozenset(added_now), frozenset(removed_now))
                self._pending_count = 1
            else:
                self._clear_pending()
            return False
        synced = build_synced_presets(
            fresh_presets, live_ids, provider=self.provider, pinned=fresh_pinned
        )
        backup = write_config_backup(self.config_path)
        fresh["modelPresets"] = synced
        write_raw_config(self.config_path, fresh)
        self._clear_pending()
        logger.info(
            "preset sync: applied +{} -{} for provider '{}' (backup {})",
            len(added),
            len(removed),
            self.provider,
            backup.name,
        )
        self.invalidate()
        return True

    def _clear_pending(self) -> None:
        self._pending = None
        self._pending_count = 0
