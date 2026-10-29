from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.builtin import cmd_model
from nanobot.command.router import CommandContext
from nanobot.config.schema import ModelPresetConfig
from nanobot.session.model_selection import (
    SESSION_MODEL_PRESET_METADATA_KEY,
    model_preset_from_metadata,
)


def _provider(default_model: str, max_tokens: int = 123) -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = default_model
    provider.generation = SimpleNamespace(
        max_tokens=max_tokens,
        temperature=0.1,
        reasoning_effort=None,
    )
    return provider


def _picker_presets() -> dict[str, ModelPresetConfig]:
    presets = {
        "default": ModelPresetConfig(
            model="base-model",
            max_tokens=123,
            context_window_tokens=1000,
        ),
        "fast-pilihan": ModelPresetConfig(
            model="combo-model",
            max_tokens=123,
            context_window_tokens=1000,
        ),
        "slow-pilihan": ModelPresetConfig(
            model="combo-slow",
            max_tokens=123,
            context_window_tokens=1000,
        ),
        "acme/alpha": ModelPresetConfig(
            model="acme-alpha",
            max_tokens=123,
            context_window_tokens=1000,
        ),
        "acme/beta": ModelPresetConfig(
            model="acme-beta",
            max_tokens=123,
            context_window_tokens=1000,
        ),
        "other/gamma": ModelPresetConfig(
            model="other-gamma",
            max_tokens=123,
            context_window_tokens=1000,
        ),
    }
    for i in range(30):
        presets[f"big/model-{i:02d}"] = ModelPresetConfig(
            model=f"big-model-{i:02d}",
            max_tokens=123,
            context_window_tokens=1000,
        )
    return presets


def _make_loop(tmp_path, *, preset_snapshot_loader=None) -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(),
        provider=_provider("base-model", max_tokens=123),
        workspace=tmp_path,
        model="base-model",
        context_window_tokens=1000,
        model_presets=_picker_presets(),
        preset_snapshot_loader=preset_snapshot_loader,
    )


def _ctx(loop: AgentLoop, raw: str, args: str = "") -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content=raw)
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


def _tap(loop: AgentLoop, payload: str) -> CommandContext:
    """Build a context as if a Telegram button tap sent `payload`."""
    return _ctx(loop, payload, args=payload.removeprefix("/model "))


def _saved_model_preset(loop: AgentLoop, session_key: str = "cli:direct") -> str | None:
    session = loop.sessions.get_or_create(session_key)
    return model_preset_from_metadata(session.metadata)


def _payloads(out) -> list[list[str]]:
    return out.metadata["telegram_callback_data"]


def _first_list_id(out) -> str:
    return _payloads(out)[0][0].split(" ")[2].split(":")[0]


@pytest.mark.asyncio
async def test_model_picker_shows_provider_step(tmp_path) -> None:
    loop = _make_loop(tmp_path)

    out = await cmd_model(_ctx(loop, "/model"))

    assert "Current model: `base-model`" in out.content
    assert "Current preset: `default`" in out.content
    assert "Choose a provider:" in out.content
    assert [row[0] for row in out.buttons] == [
        "Combos (2)",
        "✓ Default",
        "acme (2)",
        "big (30)",
        "other (1)",
    ]
    data = _payloads(out)
    assert data[0][0].endswith(":combos")
    assert data[1][0].split(":")[-1] == "3"
    assert data[2][0].endswith(":acme")
    assert out.metadata["render_as"] == "text"


@pytest.mark.asyncio
async def test_model_picker_group_paginates_and_picks(tmp_path) -> None:
    loop = _make_loop(tmp_path)
    providers = await cmd_model(_ctx(loop, "/model"))
    list_id = _first_list_id(providers)

    group = await cmd_model(_tap(loop, f"/model group {list_id}:big"))

    assert "## Model — big" in group.content
    assert "- Page 1/2 — 30 presets" in group.content
    assert len(group.buttons) == 27  # 25 picks + nav + back
    assert group.buttons[-2] == ["1/2", "Next »"]
    assert group.buttons[-1] == ["« Providers"]
    assert group.buttons[0] == ["big/model-00"]

    group_id = _first_list_id(group)
    page2 = await cmd_model(_tap(loop, f"/model page {group_id}:1"))

    assert "- Page 2/2 — 30 presets" in page2.content
    assert len(page2.buttons) == 7  # 5 picks + nav + back
    assert page2.buttons[-2] == ["« Previous", "2/2"]
    assert page2.buttons[-1] == ["« Providers"]
    assert page2.buttons[0] == ["big/model-25"]

    pick_payload = _payloads(page2)[4][0]
    out = await cmd_model(_tap(loop, pick_payload))

    assert "Switched model preset to `big/model-29`." in out.content
    assert out.metadata == {"render_as": "text"}
    assert _saved_model_preset(loop) == "big/model-29"


@pytest.mark.asyncio
async def test_model_picker_marks_current_preset(tmp_path) -> None:
    loop = _make_loop(tmp_path)
    await cmd_model(_ctx(loop, "/model acme/beta", args="acme/beta"))

    providers = await cmd_model(_ctx(loop, "/model"))
    list_id = _first_list_id(providers)

    assert [row[0] for row in providers.buttons] == [
        "Combos (2)",
        "Default",
        "✓ acme (2)",
        "big (30)",
        "other (1)",
    ]

    group = await cmd_model(_tap(loop, f"/model group {list_id}:acme"))
    assert [row[0] for row in group.buttons] == [
        "acme/alpha",
        "✓ acme/beta",
        "« Providers",
    ]


@pytest.mark.asyncio
async def test_model_picker_back_to_providers(tmp_path) -> None:
    loop = _make_loop(tmp_path)
    providers = await cmd_model(_ctx(loop, "/model"))
    list_id = _first_list_id(providers)

    group = await cmd_model(_tap(loop, f"/model group {list_id}:acme"))
    back_payload = _payloads(group)[-1][0]
    assert back_payload.startswith("/model providers ")

    again = await cmd_model(_tap(loop, back_payload))

    assert "Choose a provider:" in again.content
    assert [row[0] for row in again.buttons] == [
        "Combos (2)",
        "✓ Default",
        "acme (2)",
        "big (30)",
        "other (1)",
    ]


@pytest.mark.asyncio
async def test_model_picker_default_pick_switches_to_default(tmp_path) -> None:
    loop = _make_loop(tmp_path)
    await cmd_model(_ctx(loop, "/model acme/alpha", args="acme/alpha"))
    providers = await cmd_model(_ctx(loop, "/model"))
    pick_payload = _payloads(providers)[1][0]

    out = await cmd_model(_tap(loop, pick_payload))

    assert "Switched model preset to `default`." in out.content
    assert _saved_model_preset(loop) == "default"


@pytest.mark.asyncio
async def test_model_picker_expired_and_unknown_choices(tmp_path) -> None:
    loop = _make_loop(tmp_path)

    for payload in (
        "/model group abcdef:big",
        "/model page abcdef:1",
        "/model pick abcdef:1",
        "/model providers abcdef",
    ):
        out = await cmd_model(_tap(loop, payload))
        assert "has expired" in out.content

    providers = await cmd_model(_ctx(loop, "/model"))
    list_id = _first_list_id(providers)

    for payload in (
        f"/model group {list_id}:nope",
        f"/model pick {list_id}:99",
        f"/model pick {list_id}:x",
        f"/model page {list_id}:x",
    ):
        out = await cmd_model(_tap(loop, payload))
        assert "no longer available" in out.content


@pytest.mark.asyncio
async def test_model_picker_page_clamps_to_last_page(tmp_path) -> None:
    loop = _make_loop(tmp_path)
    providers = await cmd_model(_ctx(loop, "/model"))
    list_id = _first_list_id(providers)
    group = await cmd_model(_tap(loop, f"/model group {list_id}:big"))
    group_id = _first_list_id(group)

    out = await cmd_model(_tap(loop, f"/model page {group_id}:99"))

    assert "- Page 2/2 — 30 presets" in out.content
    assert out.buttons[0] == ["big/model-25"]


@pytest.mark.asyncio
async def test_model_picker_falls_back_on_runtime_error(tmp_path) -> None:
    def fail_preset(_name: str):
        raise ValueError("No API key configured for provider 'openai'.")

    loop = _make_loop(tmp_path, preset_snapshot_loader=fail_preset)
    session = loop.sessions.get_or_create("cli:direct")
    session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = "acme/alpha"
    loop.sessions.save(session)

    out = await cmd_model(_ctx(loop, "/model"))

    assert "Current selection error" in out.content
    assert "No API key configured for provider 'openai'." in out.content
    assert out.buttons == []
    assert out.metadata == {"render_as": "text"}
