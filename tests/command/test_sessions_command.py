"""Tests for the /sessions built-in command (list + switch)."""

import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.builtin import (
    _sessions_relative_time,
    cmd_sessions,
    register_builtin_commands,
)
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.config.schema import ModelPresetConfig


def _provider(default_model: str = "test-model") -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    provider.generation = MagicMock()
    provider.generation.max_tokens = 4096
    provider.generation.temperature = 0.1
    provider.generation.reasoning_effort = None
    return provider


def _make_loop(tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(),
        provider=_provider(),
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=8000,
        model_presets={
            "default": ModelPresetConfig(
                model="test-model",
                max_tokens=4096,
                context_window_tokens=8000,
            ),
        },
    )


def _ctx(loop: AgentLoop, raw: str, args: str = "") -> CommandContext:
    msg = InboundMessage(
        channel="telegram", sender_id="173210699", chat_id="173210699", content=raw
    )
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


def _seed(loop: AgentLoop, key: str, preview: str, minutes_ago: int) -> None:
    session = loop.sessions.get_or_create(key)
    session.add_message("user", preview)
    session.updated_at = datetime.now() - timedelta(minutes=minutes_ago)
    loop.sessions.save(session)


def _seed_chat(loop: AgentLoop) -> None:
    _seed(loop, "telegram:173210699", "Project A in progress", 30)
    _seed(loop, "telegram:999", "Project B just started", 5)
    _seed(loop, "cli:direct", "Fizzbuzz from the terminal", 60)
    _seed(loop, "dream:today", "Dream consolidation payload", 1)
    _seed(loop, "subagent:abc", "Subagent verification payload", 1)


@pytest.mark.asyncio
async def test_sessions_lists_and_excludes_internal(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    router = CommandRouter()
    register_builtin_commands(router)

    out = await router.dispatch(_ctx(loop, "/sessions"))

    assert out is not None
    assert "Project B just started" in out.content
    assert "Project A in progress" in out.content
    assert "Fizzbuzz from the terminal" in out.content
    assert "this session" in out.content
    assert "Dream consolidation payload" not in out.content
    assert "Subagent verification payload" not in out.content
    assert len(out.buttons) == 3
    assert len(out.metadata["telegram_callback_data"]) == 3
    payloads = [row[0] for row in out.metadata["telegram_callback_data"]]
    assert all(payload.startswith("/sessions pick ") for payload in payloads)
    list_ids = {payload.split(":", 1)[0] for payload in payloads}
    assert len(list_ids) == 1


@pytest.mark.asyncio
async def test_sessions_pick_by_number_switches(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    router = CommandRouter()
    register_builtin_commands(router)

    await router.dispatch(_ctx(loop, "/sessions"))
    out = await router.dispatch(_ctx(loop, "/sessions 1"))

    assert out is not None
    assert "Switched to session" in out.content
    assert "Project B just started" in out.content
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {("telegram", "173210699"): "telegram:999"}
    msg = InboundMessage(
        channel="telegram", sender_id="173210699", chat_id="173210699", content="halo"
    )
    assert loop._effective_session_key(msg) == "telegram:999"


def _payload_for(listed_content: str, callback_data: list[list[str]], needle: str) -> str:
    """Find the button payload for the list row whose text contains needle."""
    for line in listed_content.splitlines():
        if needle in line:
            number = line.split(".", 1)[0].strip()
            return callback_data[int(number) - 1][0]
    raise AssertionError(f"no list row contains {needle!r}")


@pytest.mark.asyncio
async def test_sessions_pick_by_button_data(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await cmd_sessions(_ctx(loop, "/sessions", args=""))
    payload = _payload_for(
        listed.content, listed.metadata["telegram_callback_data"], "Project B just started"
    )
    out = await cmd_sessions(_ctx(loop, payload, args=payload.split(" ", 1)[1]))

    assert "Switched to session" in out.content
    assert "Project B just started" in out.content
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {("telegram", "173210699"): "telegram:999"}


@pytest.mark.asyncio
async def test_sessions_pick_current_is_noop(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await cmd_sessions(_ctx(loop, "/sessions", args=""))
    payload = _payload_for(
        listed.content,
        listed.metadata["telegram_callback_data"],
        "Project A in progress",
    )
    out = await cmd_sessions(_ctx(loop, payload, args=payload.split(" ", 1)[1]))

    assert "already on that session" in out.content
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {}


@pytest.mark.asyncio
async def test_sessions_pick_expired_list(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await cmd_sessions(_ctx(loop, "/sessions", args=""))
    payload = listed.metadata["telegram_callback_data"][0][0]
    list_id = payload.split(" ", 2)[2].split(":", 1)[0]
    loop._session_picker_lists[list_id]["created"] = time.monotonic() - 99999

    out = await cmd_sessions(_ctx(loop, payload, args=payload.split(" ", 1)[1]))

    assert "expired" in out.content
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {}


@pytest.mark.asyncio
async def test_sessions_unknown_number(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed(loop, "telegram:173210699", "The only session", 5)

    await cmd_sessions(_ctx(loop, "/sessions", args=""))
    out = await cmd_sessions(_ctx(loop, "/sessions 5", args="5"))

    assert "out of range" in out.content


@pytest.mark.asyncio
async def test_sessions_clear_returns_to_default(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    await loop.set_chat_session_binding("telegram", "173210699", "telegram:999")

    out = await cmd_sessions(_ctx(loop, "/sessions clear", args="clear"))

    assert "default session" in out.content
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {}
    msg = InboundMessage(
        channel="telegram", sender_id="173210699", chat_id="173210699", content="halo"
    )
    assert loop._effective_session_key(msg) == "telegram:173210699"


@pytest.mark.asyncio
async def test_sessions_binding_survives_new_loop(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    await loop.set_chat_session_binding("telegram", "173210699", "telegram:999")

    loop2 = _make_loop(tmp_path)
    msg = InboundMessage(
        channel="telegram", sender_id="173210699", chat_id="173210699", content="halo"
    )
    assert loop2._effective_session_key(msg) == "telegram:999"


@pytest.mark.asyncio
async def test_sessions_override_wins_over_binding(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    await loop.set_chat_session_binding("telegram", "173210699", "telegram:999")

    msg = InboundMessage(
        channel="telegram",
        sender_id="173210699",
        chat_id="173210699",
        content="halo",
        session_key_override="telegram:173210699:topic:5",
    )
    assert loop._effective_session_key(msg) == "telegram:173210699:topic:5"


@pytest.mark.asyncio
async def test_sessions_empty_store(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    out = await cmd_sessions(_ctx(loop, "/sessions", args=""))

    assert out.content == "No saved sessions yet."


def test_sessions_relative_time() -> None:
    now = datetime.now().astimezone()
    assert _sessions_relative_time(now.isoformat()) == "just now"
    assert (
        _sessions_relative_time((now - timedelta(seconds=90)).isoformat())
        == "1 min ago"
    )
    assert (
        _sessions_relative_time((now - timedelta(seconds=3700)).isoformat())
        == "1 h ago"
    )
    assert (
        _sessions_relative_time((now - timedelta(days=2)).isoformat()) == "2 d ago"
    )
    assert _sessions_relative_time("bukan-tanggal") == ""
    assert _sessions_relative_time(None) == ""


def _row_number(content: str, needle: str) -> int:
    """Leading list number of the row whose text contains needle."""
    for line in content.splitlines():
        if needle in line:
            return int(line.split(".", 1)[0].strip())
    raise AssertionError(f"no list row contains {needle!r}")


async def _listed(loop: AgentLoop):
    return await cmd_sessions(_ctx(loop, "/sessions", args=""))


async def _delete_prompt(loop: AgentLoop, number: int):
    return await cmd_sessions(
        _ctx(loop, f"/sessions delete {number}", args=f"delete {number}")
    )


@pytest.mark.asyncio
async def test_sessions_delete_asks_confirmation(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await _listed(loop)
    number = _row_number(listed.content, "Project B just started")
    out = await _delete_prompt(loop, number)

    assert "permanently" in out.content
    assert "Project B just started" in out.content
    assert out.buttons == [["Yes, delete"], ["Cancel"]]
    assert out.metadata["telegram_callback_data"][0][0].startswith(
        "/sessions delete-yes "
    )
    assert out.metadata["telegram_callback_data"][1][0] == "/sessions delete-no"
    keys = [row["key"] for row in await loop.sessions.state.list_sessions()]
    assert "telegram:999" in keys


@pytest.mark.asyncio
async def test_sessions_delete_confirm_removes(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await _listed(loop)
    number = _row_number(listed.content, "Project B just started")
    prompt = await _delete_prompt(loop, number)
    payload = prompt.metadata["telegram_callback_data"][0][0]
    out = await cmd_sessions(_ctx(loop, payload, args=payload.split(" ", 1)[1]))

    assert "deleted" in out.content
    assert "Project B just started" in out.content
    keys = [row["key"] for row in await loop.sessions.state.list_sessions()]
    assert "telegram:999" not in keys
    assert "telegram:173210699" in keys
    assert "cli:direct" in keys


@pytest.mark.asyncio
async def test_sessions_delete_current_refused(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await _listed(loop)
    number = _row_number(listed.content, "this session")
    out = await _delete_prompt(loop, number)

    assert "in use" in out.content
    keys = [row["key"] for row in await loop.sessions.state.list_sessions()]
    assert "telegram:173210699" in keys


@pytest.mark.asyncio
async def test_sessions_delete_cancel(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await _listed(loop)
    number = _row_number(listed.content, "Project B just started")
    prompt = await _delete_prompt(loop, number)
    payload = prompt.metadata["telegram_callback_data"][1][0]
    out = await cmd_sessions(_ctx(loop, payload, args=payload.split(" ", 1)[1]))

    assert "cancelled" in out.content
    keys = [row["key"] for row in await loop.sessions.state.list_sessions()]
    assert "telegram:999" in keys


@pytest.mark.asyncio
async def test_sessions_delete_expired_confirmation(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await _listed(loop)
    number = _row_number(listed.content, "Project B just started")
    prompt = await _delete_prompt(loop, number)
    payload = prompt.metadata["telegram_callback_data"][0][0]
    list_id = payload.split(" ", 2)[2].split(":", 1)[0]
    loop._session_picker_lists[list_id]["created"] = time.monotonic() - 99999
    out = await cmd_sessions(_ctx(loop, payload, args=payload.split(" ", 1)[1]))

    assert "expired" in out.content
    keys = [row["key"] for row in await loop.sessions.state.list_sessions()]
    assert "telegram:999" in keys


@pytest.mark.asyncio
async def test_sessions_delete_unknown_number(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed(loop, "telegram:173210699", "The only session", 5)

    await _listed(loop)
    out = await _delete_prompt(loop, 5)

    assert "out of range" in out.content


@pytest.mark.asyncio
async def test_sessions_delete_refused_when_bound(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    await loop.set_chat_session_binding("telegram", "173210699", "telegram:999")

    listed = await _listed(loop)
    number = _row_number(listed.content, "Project B just started")
    out = await _delete_prompt(loop, number)

    assert "in use" in out.content
    keys = [row["key"] for row in await loop.sessions.state.list_sessions()]
    assert "telegram:999" in keys
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {("telegram", "173210699"): "telegram:999"}


@pytest.mark.asyncio
async def test_sessions_delete_confirm_missing_session(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await _listed(loop)
    number = _row_number(listed.content, "Project B just started")
    prompt = await _delete_prompt(loop, number)
    await loop.sessions.state.delete("telegram:999")
    payload = prompt.metadata["telegram_callback_data"][0][0]
    out = await cmd_sessions(_ctx(loop, payload, args=payload.split(" ", 1)[1]))

    assert "no longer exists" in out.content
