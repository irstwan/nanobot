"""Tests for the /sessions built-in command (list + switch + delete)."""

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
    cmd_new,
    cmd_sessions,
    register_builtin_commands,
)
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.config.schema import ModelPresetConfig

CHAT_ID = "173210699"
CHAT_KEY = f"telegram:{CHAT_ID}"


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


def _ctx(loop: AgentLoop, raw: str, args: str = "", key: str | None = None) -> CommandContext:
    msg = InboundMessage(
        channel="telegram", sender_id=CHAT_ID, chat_id=CHAT_ID, content=raw
    )
    return CommandContext(
        msg=msg, session=None, key=key or msg.session_key, raw=raw, args=args, loop=loop
    )


def _route_key(loop: AgentLoop) -> str:
    msg = InboundMessage(channel="telegram", sender_id=CHAT_ID, chat_id=CHAT_ID, content="halo")
    return loop._effective_session_key(msg)


def _seed(loop: AgentLoop, key: str, preview: str, minutes_ago: int) -> None:
    session = loop.sessions.get_or_create(key)
    session.add_message("user", preview)
    session.updated_at = datetime.now() - timedelta(minutes=minutes_ago)
    loop.sessions.save(session)


def _seed_chat(loop: AgentLoop) -> None:
    """Three sessions for this chat plus noise from other chats and internals."""
    _seed(loop, CHAT_KEY, "Project A in progress", 30)
    _seed(loop, f"{CHAT_KEY}#2", "Project B just started", 5)
    _seed(loop, f"{CHAT_KEY}#3", "Project C archived", 60)
    _seed(loop, "telegram:999", "Other chat conversation", 5)
    _seed(loop, "cli:direct", "Fizzbuzz from the terminal", 60)
    _seed(loop, "dream:today", "Dream consolidation payload", 1)
    _seed(loop, "subagent:abc", "Subagent verification payload", 1)


async def _listed(loop: AgentLoop, key: str | None = None):
    return await cmd_sessions(_ctx(loop, "/sessions", args="", key=key))


def _payload_for(listed_content: str, callback_data: list[list[str]], needle: str) -> str:
    """Find the button payload for the list row whose text contains needle."""
    for line in listed_content.splitlines():
        if needle in line:
            number = line.split(".", 1)[0].strip()
            return callback_data[int(number) - 1][0]
    raise AssertionError(f"no list row contains {needle!r}")


def _row_number(content: str, needle: str) -> int:
    """Leading list number of the row whose text contains needle."""
    for line in content.splitlines():
        if needle in line:
            return int(line.split(".", 1)[0].strip())
    raise AssertionError(f"no list row contains {needle!r}")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sessions_lists_only_this_chat(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    router = CommandRouter()
    register_builtin_commands(router)

    out = await router.dispatch(_ctx(loop, "/sessions"))

    assert out is not None
    assert "Project B just started" in out.content
    assert "Project A in progress" in out.content
    assert "Project C archived" in out.content
    assert "this session" in out.content
    # Other chats and internal namespaces stay out of the picker.
    assert "Other chat conversation" not in out.content
    assert "Fizzbuzz from the terminal" not in out.content
    assert "Dream consolidation payload" not in out.content
    assert "Subagent verification payload" not in out.content
    assert len(out.buttons) == 3
    assert len(out.metadata["telegram_callback_data"]) == 3
    payloads = [row[0] for row in out.metadata["telegram_callback_data"]]
    assert all(payload.startswith("/sessions pick ") for payload in payloads)
    list_ids = {payload.split(":", 1)[0] for payload in payloads}
    assert len(list_ids) == 1


@pytest.mark.asyncio
async def test_sessions_list_is_newest_first(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    out = await _listed(loop)

    assert _row_number(out.content, "Project B just started") == 1
    assert _row_number(out.content, "Project A in progress") == 2
    assert _row_number(out.content, "Project C archived") == 3


@pytest.mark.asyncio
async def test_sessions_excludes_similar_chat_ids(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed(loop, CHAT_KEY, "Mine", 5)
    _seed(loop, f"telegram:{CHAT_ID}0", "Neighbour chat", 4)
    _seed(loop, f"telegram:{CHAT_ID}#2", "Mine successor", 3)

    out = await _listed(loop)

    assert "Mine" in out.content
    assert "Mine successor" in out.content
    assert "Neighbour chat" not in out.content


@pytest.mark.asyncio
async def test_sessions_shows_bound_session(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    await loop.set_chat_session_binding("telegram", CHAT_ID, f"{CHAT_KEY}#2")

    out = await _listed(loop, key=f"{CHAT_KEY}#2")

    assert "Project B just started" in out.content
    marked = [line for line in out.content.splitlines() if "this session" in line]
    assert len(marked) == 1
    assert "Project B just started" in marked[0]


# ---------------------------------------------------------------------------
# Switching
# ---------------------------------------------------------------------------


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
    assert bindings == {("telegram", CHAT_ID): f"{CHAT_KEY}#2"}
    assert _route_key(loop) == f"{CHAT_KEY}#2"


@pytest.mark.asyncio
async def test_sessions_pick_by_button_data(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await _listed(loop)
    payload = _payload_for(
        listed.content, listed.metadata["telegram_callback_data"], "Project B just started"
    )
    out = await cmd_sessions(_ctx(loop, payload, args=payload.split(" ", 1)[1]))

    assert "Switched to session" in out.content
    assert "Project B just started" in out.content
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {("telegram", CHAT_ID): f"{CHAT_KEY}#2"}


@pytest.mark.asyncio
async def test_sessions_pick_current_is_noop(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await _listed(loop)
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

    listed = await _listed(loop)
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
    _seed(loop, CHAT_KEY, "The only session", 5)

    await _listed(loop)
    out = await cmd_sessions(_ctx(loop, "/sessions 5", args="5"))

    assert "out of range" in out.content


@pytest.mark.asyncio
async def test_sessions_clear_returns_to_default(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    await loop.set_chat_session_binding("telegram", CHAT_ID, f"{CHAT_KEY}#2")

    out = await cmd_sessions(_ctx(loop, "/sessions clear", args="clear"))

    assert "default session" in out.content
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {}
    assert _route_key(loop) == CHAT_KEY


@pytest.mark.asyncio
async def test_sessions_binding_survives_new_loop(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    await loop.set_chat_session_binding("telegram", CHAT_ID, f"{CHAT_KEY}#2")

    loop2 = _make_loop(tmp_path)
    assert _route_key(loop2) == f"{CHAT_KEY}#2"


@pytest.mark.asyncio
async def test_sessions_override_wins_over_binding(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    await loop.set_chat_session_binding("telegram", CHAT_ID, f"{CHAT_KEY}#2")

    msg = InboundMessage(
        channel="telegram",
        sender_id=CHAT_ID,
        chat_id=CHAT_ID,
        content="halo",
        session_key_override=f"{CHAT_KEY}:topic:5",
    )
    assert loop._effective_session_key(msg) == f"{CHAT_KEY}:topic:5"


@pytest.mark.asyncio
async def test_sessions_empty_store(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)

    out = await _listed(loop)

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


# ---------------------------------------------------------------------------
# /new rotates to a fresh session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_keeps_previous_session_listed(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed(loop, CHAT_KEY, "Percakapan lama", 5)

    out = await cmd_new(_ctx(loop, "/new", args=""))

    assert "New session started" in out.content
    keys = [row["key"] for row in await loop.sessions.state.list_sessions()]
    # The old conversation survives as its own session.
    assert CHAT_KEY in keys
    previous = await loop.sessions.state.get(CHAT_KEY)
    assert [m["content"] for m in previous.messages] == ["Percakapan lama"]

    listed = await _listed(loop, key=_route_key(loop))
    assert "Percakapan lama" in listed.content
    assert "(empty session)" in listed.content
    assert len(listed.buttons) == 2


@pytest.mark.asyncio
async def test_new_binds_chat_to_fresh_session(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed(loop, CHAT_KEY, "Percakapan lama", 5)

    await cmd_new(_ctx(loop, "/new", args=""))

    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {("telegram", CHAT_ID): f"{CHAT_KEY}#2"}
    fresh = await loop.sessions.state.get(f"{CHAT_KEY}#2")
    assert fresh.messages == []


@pytest.mark.asyncio
async def test_new_reuses_empty_successor(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed(loop, CHAT_KEY, "Percakapan lama", 5)

    await cmd_new(_ctx(loop, "/new", args=""))
    await cmd_new(_ctx(loop, "/new", args="", key=_route_key(loop)))

    keys = sorted(row["key"] for row in await loop.sessions.state.list_sessions())
    assert keys == [CHAT_KEY, f"{CHAT_KEY}#2"]


@pytest.mark.asyncio
async def test_new_in_topic_thread_resets_in_place(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    topic_key = f"{CHAT_KEY}:topic:5"
    _seed(loop, topic_key, "Topik lima", 5)
    msg = InboundMessage(
        channel="telegram",
        sender_id=CHAT_ID,
        chat_id=CHAT_ID,
        content="/new",
        session_key_override=topic_key,
    )
    ctx = CommandContext(
        msg=msg, session=None, key=topic_key, raw="/new", args="", loop=loop
    )

    await cmd_new(ctx)

    reset = await loop.sessions.state.get(topic_key)
    assert reset.messages == []
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {}


# ---------------------------------------------------------------------------
# Deleting
# ---------------------------------------------------------------------------


async def _delete_prompt(loop: AgentLoop, number: int, key: str | None = None):
    return await cmd_sessions(
        _ctx(loop, f"/sessions delete {number}", args=f"delete {number}", key=key)
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
    assert f"{CHAT_KEY}#2" in keys


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
    assert f"{CHAT_KEY}#2" not in keys
    assert CHAT_KEY in keys
    assert f"{CHAT_KEY}#3" in keys
    # Other chats are never touched.
    assert "telegram:999" in keys
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
    assert CHAT_KEY in keys


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
    assert f"{CHAT_KEY}#2" in keys


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
    assert f"{CHAT_KEY}#2" in keys


@pytest.mark.asyncio
async def test_sessions_delete_unknown_number(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed(loop, CHAT_KEY, "The only session", 5)

    await _listed(loop)
    out = await _delete_prompt(loop, 5)

    assert "out of range" in out.content


@pytest.mark.asyncio
async def test_sessions_delete_refused_when_bound(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)
    await loop.set_chat_session_binding("telegram", CHAT_ID, f"{CHAT_KEY}#2")

    listed = await _listed(loop, key=f"{CHAT_KEY}#2")
    number = _row_number(listed.content, "Project B just started")
    out = await _delete_prompt(loop, number, key=f"{CHAT_KEY}#2")

    assert "in use" in out.content
    keys = [row["key"] for row in await loop.sessions.state.list_sessions()]
    assert f"{CHAT_KEY}#2" in keys
    bindings = await loop.sessions.state.list_chat_session_bindings()
    assert bindings == {("telegram", CHAT_ID): f"{CHAT_KEY}#2"}


@pytest.mark.asyncio
async def test_sessions_delete_confirm_missing_session(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    _seed_chat(loop)

    listed = await _listed(loop)
    number = _row_number(listed.content, "Project B just started")
    prompt = await _delete_prompt(loop, number)
    await loop.sessions.state.delete(f"{CHAT_KEY}#2")
    payload = prompt.metadata["telegram_callback_data"][0][0]
    out = await cmd_sessions(_ctx(loop, payload, args=payload.split(" ", 1)[1]))

    assert "no longer exists" in out.content
