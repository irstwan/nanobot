"""Shared session key constants and helpers."""

from __future__ import annotations

from collections.abc import Container, Mapping, MutableMapping
from typing import Any

UNIFIED_SESSION_KEY = "unified:default"
HEARTBEAT_SESSION_KEY = "heartbeat"
LAST_CHANNEL_METADATA_KEY = "last_channel"


def is_dream_session(key: str) -> bool:
    """Identify the per-run session namespace owned by Dream."""
    return key.startswith("dream:")


def is_internal_session(key: str) -> bool:
    """Identify maintenance sessions whose context is not a user conversation."""
    return key == HEARTBEAT_SESSION_KEY or is_dream_session(key)


def session_key_for_channel(channel: str, chat_id: str, *, unified_session: bool = False) -> str:
    """Return the session key for a channel/chat pair."""
    if unified_session:
        return UNIFIED_SESSION_KEY
    return f"{channel}:{chat_id}"


def session_key_under_base(key: str, base: str) -> bool:
    """Whether *key* is *base* or a session derived from it.

    Derived keys are topic threads (``base:topic:N``) and rotated sessions
    started by ``/new`` (``base#N``). The separator check keeps ``173`` from
    matching ``1730``.
    """
    return key == base or key.startswith(f"{base}:") or key.startswith(f"{base}#")


def session_base_key(key: str) -> str:
    """Return *key* without any ``/new`` rotation suffix (``...#N``)."""
    base, sep, suffix = key.rpartition("#")
    return base if sep and suffix.isdigit() else key


def next_successor_session_key(base: str, taken: Container[str]) -> str:
    """Return the first free ``base#N`` successor key at or after ``base#2``."""
    index = 2
    while f"{base}#{index}" in taken:
        index += 1
    return f"{base}#{index}"


def remember_last_channel(
    metadata: MutableMapping[str, Any],
    channel: str,
    chat_id: str,
) -> None:
    """Persist the latest concrete delivery route in session metadata."""
    if not channel or not chat_id:
        return
    metadata[LAST_CHANNEL_METADATA_KEY] = f"{channel}:{chat_id}"


def last_channel_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> tuple[str, str] | None:
    """Return a concrete delivery route from persisted session metadata."""
    if not isinstance(metadata, Mapping):
        return None
    route = metadata.get(LAST_CHANNEL_METADATA_KEY)
    if not isinstance(route, str) or ":" not in route:
        return None
    channel, chat_id = route.split(":", 1)
    if not channel or not chat_id:
        return None
    return channel, chat_id
