"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger

from nanobot import __version__
from nanobot.bus.events import INBOUND_META_USER_SHELL, InboundMessage, OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter, normalize_command_text
from nanobot.providers.base import LLMUsage
from nanobot.session.history_visibility import HIDDEN_HISTORY_META
from nanobot.session.keys import (
    UNIFIED_SESSION_KEY,
    next_successor_session_key,
    session_base_key,
    session_key_for_channel,
    session_key_under_base,
)
from nanobot.utils.cancellation import shield_and_drain
from nanobot.utils.helpers import build_status_content
from nanobot.utils.restart import set_restart_notice_to_env
from nanobot.utils.workspace_prompts import initialize_workspace_prompt

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.session.manager import Session
    from nanobot.utils.gitstore import CommitInfo

# WebUI protocol contract for how a slash command participates in turn state:
# - side_channel: returns control text without starting or ending an agent turn.
# - finalize_active_turn: side-channel command that also closes the active UI turn.
# - stop_active_turn: cancels the active turn; WebUI may intercept exact submits.
# - agent_turn: always enters the normal agent path.
# - agent_turn_with_args: no args is side-channel usage; args enter the agent path.
CommandLifecycle = Literal[
    "side_channel",
    "finalize_active_turn",
    "stop_active_turn",
    "agent_turn",
    "agent_turn_with_args",
]

USER_SHELL_COMMAND = "/__shell"


@dataclass(frozen=True)
class BuiltinCommandSpec:
    command: str
    title: str
    description: str
    icon: str
    arg_hint: str = ""
    lifecycle: CommandLifecycle = "side_channel"
    accepts_args: bool = False

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "command": self.command,
            "title": self.title,
            "description": self.description,
            "icon": self.icon,
            "arg_hint": self.arg_hint,
            "lifecycle": self.lifecycle,
            "accepts_args": self.accepts_args,
        }


BUILTIN_COMMAND_SPECS: tuple[BuiltinCommandSpec, ...] = (
    BuiltinCommandSpec(
        "/new",
        "New chat",
        "Start a fresh session, keeping this one in /sessions.",
        "square-pen",
        lifecycle="finalize_active_turn",
    ),
    BuiltinCommandSpec(
        "/sessions",
        "Switch session",
        "List this chat's sessions, then switch or delete.",
        "history",
        "[n | delete <n> | clear]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/compact",
        "Compact context",
        "Compact this chat's context and continue the conversation.",
        "archive",
    ),
    BuiltinCommandSpec(
        "/stop",
        "Stop current task",
        "Cancel the active agent turn for this chat.",
        "square",
        lifecycle="stop_active_turn",
    ),
    BuiltinCommandSpec(
        "/restart",
        "Restart nanobot",
        "Restart the bot process.",
        "rotate-cw",
    ),
    BuiltinCommandSpec(
        "/status",
        "Show status",
        "Display runtime, provider, and channel status.",
        "activity",
    ),
    BuiltinCommandSpec(
        "/model",
        "Switch model preset",
        "Show or switch the active model preset.",
        "brain",
        "[preset]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/history",
        "Show conversation history",
        "Print the last N persisted conversation messages.",
        "history",
        "[n]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/goal",
        "Start long-running goal",
        "Tell the agent to treat the request as a long-running goal.",
        "activity",
        "<goal>",
        lifecycle="agent_turn_with_args",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/trigger",
        "Create named local trigger",
        "Create a named CLI trigger bound to this chat session.",
        "zap",
        "<name>",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/dream",
        "Run Dream",
        "Manually trigger memory consolidation.",
        "sparkles",
    ),
    BuiltinCommandSpec(
        "/dream-log",
        "Show Dream log",
        "Show what the last Dream consolidation changed.",
        "book-open",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/dream-restore",
        "Restore memory",
        "Revert memory to a previous Dream snapshot.",
        "undo-2",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/dream-prompt",
        "Dream memory",
        "Tell Dream how to organize this workspace's memory.",
        "file-text",
        "[init]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/evaluator-prompt",
        "Heartbeat evaluator",
        "Customize the heartbeat notification gate prompt for this workspace.",
        "file-text",
        "[init]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/skill",
        "List skills",
        "List all enabled skills available to the agent.",
        "wrench",
    ),
    BuiltinCommandSpec(
        "/help",
        "Show help",
        "List available slash commands.",
        "circle-help",
    ),
    BuiltinCommandSpec(
        "/pairing",
        "Manage pairing",
        "List, approve, deny or revoke pairing requests.",
        "shield",
        "[list|approve <code>|deny <code>|revoke <user_id>]",
        accepts_args=True,
    ),
)


def builtin_command_palette() -> list[dict[str, str | bool]]:
    """Return structured command metadata for UI command palettes."""
    return [spec.as_dict() for spec in BUILTIN_COMMAND_SPECS]


def builtin_command_starts_agent_turn(text: str) -> bool:
    """Return whether WebUI ingress should expect a normal agent lifecycle."""
    normalized = normalize_command_text(text)
    command, separator, args = normalized.partition(" ")
    spec = next(
        (item for item in BUILTIN_COMMAND_SPECS if item.command == command.lower()),
        None,
    )
    if spec is None or (separator and not spec.accepts_args):
        return True
    if spec.lifecycle == "agent_turn":
        return True
    return spec.lifecycle == "agent_turn_with_args" and bool(args.strip())


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    total = await loop._cancel_active_tasks(ctx.key)  # pyright: ignore[reportPrivateUsage]
    # Also drain pending queue to prevent mid-turn injection deadlock
    pending = loop._pending_queues.pop(ctx.key, None)  # pyright: ignore[reportPrivateUsage]
    if pending is not None:
        while not pending.empty():
            try:
                pending.get_nowait()
                total += 1
            except Exception:
                break
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content=content,
        metadata=dict(msg.metadata or {})
    )


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process."""
    msg = ctx.msg
    set_restart_notice_to_env(
        channel=msg.channel,
        chat_id=msg.chat_id,
        metadata=dict(msg.metadata or {}),
    )

    async def _do_restart():
        await asyncio.sleep(1)
        argv = [sys.executable, "-m", "nanobot"] + sys.argv[1:]
        mode = ctx.loop.restart_mode or "auto"
        if mode == "auto":
            mode = "spawn" if sys.platform == "win32" else "exec"
        if mode == "exec":
            os.execv(sys.executable, argv)
            return
        if mode == "spawn":
            kwargs: dict[str, Any] = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(argv, **kwargs)
        os._exit(0)

    asyncio.create_task(_do_restart())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        metadata=dict(msg.metadata or {})
    )


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or await loop.sessions.state.get(ctx.key)
    runtime = ctx.runtime or await loop.runtime_for_session(session)
    ctx_est = 0
    with suppress(Exception):
        ctx_est, _ = loop.consolidator.estimate_session_prompt_tokens(
            session,
            runtime=runtime,
        )
    last_usage = LLMUsage.from_dict(session.metadata.get("_last_usage"))
    if ctx_est <= 0:
        ctx_est = last_usage.input_tokens if last_usage is not None else 0

    # Await web search provider usage; network requests have a bounded timeout.
    search_usage_text: str | None = None
    # Never let usage fetch break /status
    with suppress(Exception):
        from nanobot.utils.searchusage import fetch_search_usage
        search_cfg = loop.web_config.search
        usage = await fetch_search_usage(
            provider=search_cfg.provider,
            api_key=search_cfg.api_key or None,
        )
        search_usage_text = usage.format()
    active_tasks = loop._active_tasks.get(ctx.key, [])  # pyright: ignore[reportPrivateUsage]
    task_count = sum(1 for t in active_tasks if not t.done())
    with suppress(Exception):
        task_count += loop.subagents.get_running_count_by_session(ctx.key)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=runtime.model,
            start_time=loop._start_time, last_usage=last_usage,  # pyright: ignore[reportPrivateUsage]
            context_window_tokens=runtime.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
            search_usage_text=search_usage_text,
            active_task_count=task_count,
            max_completion_tokens=runtime.generation.max_tokens,
        ),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def _started_session_key(loop: AgentLoop, ctx: CommandContext) -> str | None:
    """Create an empty session for this chat and point the chat at it.

    Returns the new key, or None when the chat cannot be redirected (unified
    sessions, topic threads and other explicit session overrides) and the
    caller must fall back to resetting the current session in place.
    """
    channel = ctx.msg.channel
    chat_id = ctx.msg.chat_id
    if getattr(ctx.msg, "session_key_override", None) or ctx.key == UNIFIED_SESSION_KEY:
        return None
    base = session_base_key(ctx.key)
    infos = await loop.sessions.state.list_sessions()
    keys = [str(info.get("key", "")) for info in infos]
    if ctx.key not in keys:
        keys.append(ctx.key)
    # Reuse the newest empty successor so repeated /new without chatting does
    # not pile up blank rows in the picker.
    for key in sorted((k for k in keys if k.startswith(f"{base}#")), reverse=True):
        existing = await loop.sessions.state.get(key)
        if not existing.messages:
            await loop.set_chat_session_binding(channel, chat_id, key)
            return key
    new_key = next_successor_session_key(base, set(keys))
    await loop.sessions.state.get(new_key)
    await loop.set_chat_session_binding(channel, chat_id, new_key)
    return new_key


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Stop active task and start a fresh session.

    The current conversation is kept as its own session so it stays listed in
    /sessions and can be switched back to; this chat moves to a new empty one.
    """
    loop = ctx.loop
    await loop._cancel_active_tasks(ctx.key)  # pyright: ignore[reportPrivateUsage]
    loop.discard_session_file_state(ctx.key)
    session = ctx.session or await loop.sessions.state.get(ctx.key)
    snapshot = list(session.messages)
    archive_snapshot = None
    runtime = None
    if session.last_archived < len(snapshot):
        runtime = ctx.runtime or await loop.runtime_for_session(session)
        archive_snapshot = replace(
            session,
            messages=snapshot,
            metadata=dict(session.metadata),
            provider_state=None,
        )

    async def reset_and_schedule_archive() -> None:
        if await _started_session_key(loop, ctx) is None:
            await loop.sessions.state.reset(session.key)
        if archive_snapshot is not None and runtime is not None:
            loop.schedule_background(
                loop.consolidator.archive_session(  # pyright: ignore[reportUnknownMemberType]
                    archive_snapshot,
                    archive_end=len(snapshot),
                    runtime=runtime,
                )
            )

    await shield_and_drain(reset_and_schedule_archive())
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
        metadata=dict(ctx.msg.metadata or {})
    )


async def cmd_compact(ctx: CommandContext) -> None:
    """Compact the current session without resetting the conversation."""
    loop = ctx.loop
    session = ctx.session or await loop.sessions.state.get(ctx.key)
    runtime = ctx.runtime or await loop.runtime_for_session(session)
    delivery = loop.turn_delivery_factory.create(ctx.msg, ctx.key)

    try:
        await loop.consolidator.compact_idle_session(
            ctx.key,
            runtime=runtime,
            events=delivery.events,
            notify=True,
            reason="manual",
        )
    except Exception:
        logger.exception("Manual context compaction failed for {}", ctx.key)
        return



def _format_preset_names(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "(none configured)"


def _is_combo_preset(name: str) -> bool:
    """9router combo presets use the user's *-pilihan naming.

    Nanobot presets carry no upstream owned_by flag, so combo detection
    relies on this naming convention: a future combo with a different
    name will sort with the rest until it is renamed *-pilihan.
    """
    return name.endswith("-pilihan")


def _model_preset_names(loop: AgentLoop) -> list[str]:
    names = set(loop.model_presets)
    names.add("default")
    combos = sorted(name for name in names if name != "default" and _is_combo_preset(name))
    rest = sorted(name for name in names if name != "default" and not _is_combo_preset(name))
    return [*combos, "default", *rest]


_MODEL_PICKER_PAGE_SIZE = 25


def _model_provider_key(name: str) -> str:
    """Group key for the /model provider step: combos, default, or id prefix."""
    if _is_combo_preset(name):
        return "combos"
    if name == "default":
        return "default"
    return name.split("/", 1)[0] if "/" in name else name


def _model_provider_groups(names: list[str]) -> list[tuple[str, str, list[str]]]:
    """Provider groups in picker order: combos first, then alphabetical.

    "default" is not a group; it is a direct pick on the provider step.
    """
    grouped: dict[str, list[str]] = {}
    for name in names:
        key = _model_provider_key(name)
        if key == "default":
            continue
        grouped.setdefault(key, []).append(name)
    groups = [
        (key, "Combos" if key == "combos" else key, grouped[key])
        for key in sorted(grouped)
    ]
    return [g for g in groups if g[0] == "combos"] + [
        g for g in groups if g[0] != "combos"
    ]


def _model_picker_reply(
    ctx: CommandContext,
    content: str,
    buttons: list[list[str]] | None = None,
    callback_data: list[list[str]] | None = None,
) -> OutboundMessage:
    metadata: dict[str, Any] = {**dict(ctx.msg.metadata or {}), "render_as": "text"}
    if callback_data:
        # Telegram-only: per-button tap payloads, parallel to `buttons`.
        metadata["telegram_callback_data"] = callback_data
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        buttons=buttons or [],
        metadata=metadata,
    )


def _model_switched_reply(ctx: CommandContext, runtime) -> OutboundMessage:
    max_tokens = runtime.generation.max_tokens
    lines = [
        f"Switched model preset to `{runtime.model_preset}`.",
        "- Scope: current session",
        f"- Model: `{runtime.model}`",
        f"- Context window: {runtime.context_window_tokens}",
    ]
    lines.append(f"- Max output tokens: {max_tokens}")
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def _model_picker_runtime(loop: AgentLoop, ctx: CommandContext):
    """Best-effort runtime for picker headers; None when it cannot be read."""
    try:
        session = ctx.session or await loop.sessions.state.get(ctx.key)
        return await loop.runtime_for_session(session, recover_removed=False)
    except Exception:
        # Markers are cosmetic; a picker without them is still usable.
        return None


def _model_provider_reply(
    ctx: CommandContext,
    loop: AgentLoop,
    names: list[str],
    runtime,
    list_id: str | None = None,
) -> OutboundMessage:
    if list_id is None:
        list_id = loop.register_model_picker_list(
            ctx.msg.channel, ctx.msg.chat_id, names, "providers", None
        )
    active = runtime.model_preset or "default" if runtime is not None else None
    buttons: list[list[str]] = []
    callback_data: list[list[str]] = []
    default_index = names.index("default") + 1 if "default" in names else None

    def _default_row() -> tuple[str, str]:
        label = "✓ Default" if active == "default" else "Default"
        return label, f"/model pick {list_id}:{default_index}"

    inserted_default = False
    for key, label, members in _model_provider_groups(names):
        marked = active is not None and active in members
        buttons.append([f"{'✓ ' if marked else ''}{label} ({len(members)})"])
        callback_data.append([f"/model group {list_id}:{key}"])
        if key == "combos" and default_index is not None:
            row_label, payload = _default_row()
            buttons.append([row_label])
            callback_data.append([payload])
            inserted_default = True
    if default_index is not None and not inserted_default:
        row_label, payload = _default_row()
        buttons.insert(0, [row_label])
        callback_data.insert(0, [payload])

    lines = ["## Model"]
    if runtime is not None:
        lines.append(f"- Current model: `{runtime.model}`")
    if active is not None:
        lines.append(f"- Current preset: `{active}`")
    lines.append("- Choose a provider:")
    lines.append("Or switch directly with `/model <preset>`.")
    return _model_picker_reply(ctx, "\n".join(lines), buttons, callback_data)


def _model_group_reply(
    ctx: CommandContext,
    entry: dict[str, Any],
    list_id: str,
    page: int,
    runtime,
) -> OutboundMessage:
    names: list[str] = entry["names"]
    title: str = entry["title"]
    parent_id = entry["parent_id"]
    page_size = _MODEL_PICKER_PAGE_SIZE
    n_pages = max(1, (len(names) + page_size - 1) // page_size)
    page = min(max(page, 0), n_pages - 1)
    active = runtime.model_preset or "default" if runtime is not None else None
    buttons: list[list[str]] = []
    callback_data: list[list[str]] = []
    for position, name in enumerate(
        names[page * page_size : (page + 1) * page_size],
        start=page * page_size + 1,
    ):
        buttons.append([f"{'✓ ' if name == active else ''}{name}"])
        callback_data.append([f"/model pick {list_id}:{position}"])
    if n_pages > 1:
        nav: list[str] = []
        nav_data: list[str] = []
        if page > 0:
            nav.append("« Previous")
            nav_data.append(f"/model page {list_id}:{page - 1}")
        nav.append(f"{page + 1}/{n_pages}")
        nav_data.append(f"/model page {list_id}:{page}")
        if page < n_pages - 1:
            nav.append("Next »")
            nav_data.append(f"/model page {list_id}:{page + 1}")
        buttons.append(nav)
        callback_data.append(nav_data)
    if parent_id is not None:
        buttons.append(["« Providers"])
        callback_data.append([f"/model providers {parent_id}"])

    lines = [f"## Model — {title}"]
    if runtime is not None:
        lines.append(f"- Current model: `{runtime.model}`")
    if active is not None:
        lines.append(f"- Current preset: `{active}`")
    lines.append(f"- Page {page + 1}/{n_pages} — {len(names)} presets")
    return _model_picker_reply(ctx, "\n".join(lines), buttons, callback_data)


async def _model_picker_command(
    ctx: CommandContext, verb: str, payload: str
) -> OutboundMessage:
    """Handle /model button taps: pick, group, page, and providers."""
    loop = ctx.loop
    list_id, _, tail = payload.partition(":")
    entry = (
        loop.resolve_model_picker_list(ctx.msg.channel, ctx.msg.chat_id, list_id)
        if list_id
        else None
    )
    if entry is None:
        return _model_picker_reply(
            ctx, "This model list has expired. Send /model to see it again."
        )
    unavailable = _model_picker_reply(
        ctx, "That choice is no longer available. Send /model to see the list again."
    )
    runtime = await _model_picker_runtime(loop, ctx)
    if verb == "providers":
        return _model_provider_reply(
            ctx, loop, entry["names"], runtime, list_id=list_id
        )
    if verb == "group":
        members = [
            name for name in entry["names"] if _model_provider_key(name) == tail
        ]
        if not members:
            return unavailable
        label = "Combos" if tail == "combos" else tail
        group_id = loop.register_model_picker_list(
            ctx.msg.channel, ctx.msg.chat_id, members, label, list_id
        )
        return _model_group_reply(
            ctx,
            {"names": members, "title": label, "parent_id": list_id},
            group_id,
            0,
            runtime,
        )
    if verb == "page":
        try:
            page = int(tail)
        except ValueError:
            return unavailable
        return _model_group_reply(ctx, entry, list_id, page, runtime)
    if verb == "pick":
        try:
            index = int(tail)
        except ValueError:
            return unavailable
        if not 1 <= index <= len(entry["names"]):
            return unavailable
        name = entry["names"][index - 1]
        try:
            switched = await loop.set_session_model_preset(ctx.key, name)
        except (KeyError, ValueError) as exc:
            names = _model_preset_names(loop)
            return OutboundMessage(
                channel=ctx.msg.channel,
                chat_id=ctx.msg.chat_id,
                content=(
                    f"Could not switch model preset: {_command_error_message(exc)}\n\n"
                    f"Available presets: {_format_preset_names(names)}"
                ),
                metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
            )
        return _model_switched_reply(ctx, switched)
    return unavailable


def _command_error_message(exc: Exception) -> str:
    return str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)


async def _model_command_status(loop: AgentLoop, session: Session) -> str:
    names = _model_preset_names(loop)
    try:
        runtime = await loop.runtime_for_session(session, recover_removed=False)
    except (KeyError, ValueError) as exc:
        return "\n".join([
            "## Model",
            f"- Current selection error: {_command_error_message(exc)}",
            f"- Available presets: {_format_preset_names(names)}",
            "- Switch with `/model <preset>`.",
        ])
    active = runtime.model_preset or "default"
    return "\n".join([
        "## Model",
        f"- Current model: `{runtime.model}`",
        f"- Current preset: `{active}`",
        f"- Available presets: {_format_preset_names(names)}",
    ])


async def cmd_model(ctx: CommandContext) -> OutboundMessage:
    """Show or switch model presets."""
    loop = ctx.loop
    args = ctx.args.strip()
    metadata = {**dict(ctx.msg.metadata or {}), "render_as": "text"}

    if not args:
        session = ctx.session or await loop.sessions.state.get(ctx.key)
        try:
            runtime = await loop.runtime_for_session(session, recover_removed=False)
        except (KeyError, ValueError):
            return OutboundMessage(
                channel=ctx.msg.channel,
                chat_id=ctx.msg.chat_id,
                content=await _model_command_status(loop, session),
                metadata=metadata,
            )
        return _model_provider_reply(ctx, loop, _model_preset_names(loop), runtime)

    first, _, rest = args.partition(" ")
    if first in ("pick", "group", "page", "providers") and rest.strip():
        return await _model_picker_command(ctx, first, rest.strip())

    name = args
    try:
        runtime = await loop.set_session_model_preset(ctx.key, name)
    except (KeyError, ValueError) as exc:
        names = _model_preset_names(loop)
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=(
                f"Could not switch model preset: {_command_error_message(exc)}\n\n"
                f"Available presets: {_format_preset_names(names)}"
            ),
            metadata=metadata,
        )

    return _model_switched_reply(ctx, runtime)


async def cmd_dream(ctx: CommandContext) -> OutboundMessage:
    """Manually trigger a Dream consolidation run."""
    import time

    loop = ctx.loop
    msg = ctx.msg

    async def _run_dream():
        from nanobot.agent.memory import MemoryStore

        async def _silent(*_args: Any, **_kwargs: Any) -> None:
            pass

        dream_session_key = MemoryStore.dream_session_key
        build_dream_commit_message = MemoryStore.build_dream_commit_message

        store = loop.context.memory
        content = ""
        resp = None
        diff_body = ""
        t0 = time.monotonic()
        try:
            result = store.build_dream_prompt()
            if result is None:
                await loop.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=_format_dream_no_input_message(),
                    metadata={"render_as": "text"},
                ))
                return
            prompt, last_cursor = result
            key = dream_session_key()
            dream_runtime = loop.dream_runtime()
            resp = await loop.process_direct(
                prompt,
                session_key=key,
                ephemeral=True,
                tools=store.build_dream_tools(),
                on_progress=_silent,
                runtime=dream_runtime,
            )
            elapsed = time.monotonic() - t0
            # The real file delta grounds the audit record; normal completion
            # decides whether this history batch has finished processing.
            diff_body = store.dream_content_diff()
            completed = MemoryStore.dream_run_completed(resp)
            if completed:
                store.set_last_dream_cursor(last_cursor)
                if diff_body:
                    content = f"Dream completed in {elapsed:.1f}s."
                else:
                    content = f"Dream completed in {elapsed:.1f}s; no memory changes."
            else:
                reason = MemoryStore.dream_incompletion_reason(resp)
                content = (
                    f"Dream did not complete after {elapsed:.1f}s ({reason}); "
                    "memory cursor was not advanced."
                )
        except Exception as e:
            elapsed = time.monotonic() - t0
            content = f"Dream failed after {elapsed:.1f}s: {e}"
        finally:
            if store.git.is_initialized():
                commit_msg = build_dream_commit_message("dream: manual run", diff_body)
                sha = store.git.auto_commit(commit_msg)
                if sha:
                    content += f" (commit {sha})"
            store.compact_history()
            await loop.sessions.state.prune_dream_sessions()
        await loop.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    asyncio.create_task(_run_dream())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Dreaming...",
    )


async def cmd_dream_prompt(ctx: CommandContext) -> OutboundMessage:
    """Show or set up the workspace Dream memory instructions."""
    store = ctx.loop.context.memory
    path = store.dream_prompt_file
    display_path = path.relative_to(store.workspace).as_posix()
    args = ctx.args.strip().lower()

    if args == "init":
        if not initialize_workspace_prompt(path, store.default_dream_prompt()):
            content = (
                f"Dream memory instructions already exist at `{display_path}`.\n\n"
                "Edit that file, or delete/empty it to return to nanobot's default."
            )
        else:
            content = (
                f"Created Dream memory instructions at `{display_path}`.\n\n"
                "Edit that file to teach Dream how to organize memory. "
                "This fully replaces nanobot's default Dream guide for this workspace. "
                "Delete or empty it to return to nanobot's default."
            )
    elif args:
        content = "Usage: /dream-prompt [init]"
    elif store.has_dream_prompt_override():
        content = (
            "Dream memory instructions: custom for this workspace\n\n"
            f"- Path: `{display_path}`\n"
            "- Delete or empty this file to return to nanobot's default."
        )
    else:
        content = (
            "Dream memory instructions: nanobot default\n\n"
            f"- Editable file: `{display_path}`\n"
            "- Run `/dream-prompt init` to create an editable copy."
        )

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_evaluator_prompt(ctx: CommandContext) -> OutboundMessage:
    """Show or set up the workspace heartbeat evaluator prompt."""
    from nanobot.utils.evaluator import (
        default_evaluator_prompt,
        evaluator_prompt_file,
        has_evaluator_prompt_override,
    )

    workspace = ctx.loop.context.memory.workspace
    path = evaluator_prompt_file(workspace)
    display_path = path.relative_to(workspace).as_posix()
    args = ctx.args.strip().lower()

    if args == "init":
        if not initialize_workspace_prompt(path, default_evaluator_prompt()):
            content = (
                f"Heartbeat evaluator prompt already exists at `{display_path}`.\n\n"
                "Edit that file, or delete/empty it to return to nanobot's default."
            )
        else:
            content = (
                f"Created heartbeat evaluator prompt at `{display_path}`.\n\n"
                "Edit that file to control when the heartbeat notification gate speaks. "
                "It must still instruct the model to call the `evaluate_notification` tool, "
                "otherwise the gate fails closed and stays silent. "
                "Delete or empty it to return to nanobot's default."
            )
    elif args:
        content = "Usage: /evaluator-prompt [init]"
    elif has_evaluator_prompt_override(workspace):
        content = (
            "Heartbeat evaluator prompt: custom for this workspace\n\n"
            f"- Path: `{display_path}`\n"
            "- Delete or empty this file to return to nanobot's default."
        )
    else:
        content = (
            "Heartbeat evaluator prompt: nanobot default\n\n"
            f"- Editable file: `{display_path}`\n"
            "- Run `/evaluator-prompt init` to create an editable copy."
        )

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


def _format_dream_no_input_message() -> str:
    return "\n".join([
        "Dream has no conversation history to process yet.",
        "",
        "Dream reads new entries from `memory/history.jsonl` after the current Dream cursor.",
        (
            "Short chats only reach that file after token compaction or idle auto-compact, "
            "so a fresh or short WebUI chat may leave Dream with no input."
        ),
        "",
        "Next steps:",
        "- Enable `agents.defaults.idleCompactAfterMinutes` so completed chats become Dream input automatically.",
        "- Compact the current chat into memory once that manual action is available.",
        "- If you expected history to exist, check whether `memory/history.jsonl` has new entries after the Dream cursor.",
        "- Use `/dream-prompt` to see or change how Dream organizes memory.",
    ])


def _extract_changed_files(diff: str) -> list[str]:
    """Extract changed file paths from a unified diff."""
    files: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def _format_changed_files(diff: str) -> str:
    files = _extract_changed_files(diff)
    if not files:
        return "No tracked memory files changed."
    return ", ".join(f"`{path}`" for path in files)


_DREAM_COMMIT_PREFIX = "dream:"


def _format_dream_log_content(
    commit: CommitInfo,
    diff: str,
    *,
    requested_sha: str | None = None,
) -> str:
    files_line = _format_changed_files(diff)
    lines = [
        "## Dream Update",
        "",
        "Here is the selected Dream memory change." if requested_sha else "Here is the latest Dream memory change.",
        "",
        f"- Commit: `{commit.sha}`",
        f"- Time: {commit.timestamp}",
        f"- Changed files: {files_line}",
    ]
    if diff:
        lines.extend([
            "",
            f"Use `/dream-restore {commit.sha}` to undo this change.",
            "",
            "```diff",
            diff.rstrip(),
            "```",
        ])
    else:
        lines.extend([
            "",
            "Dream recorded this version, but there is no file diff to display.",
        ])
    return "\n".join(lines)


def _format_dream_restore_list(commits: list[CommitInfo]) -> str:
    lines = [
        "## Dream Restore",
        "",
        "Choose a Dream memory version to restore. Latest first:",
        "",
    ]
    for c in commits:
        lines.append(f"- `{c.sha}` {c.timestamp} - {c.subject()}")
    lines.extend([
        "",
        "Preview a version with `/dream-log <sha>` before restoring it.",
        "Restore a version with `/dream-restore <sha>`.",
    ])
    return "\n".join(lines)


async def cmd_dream_log(ctx: CommandContext) -> OutboundMessage:
    """Show what the last Dream changed.

    Default: diff of the latest Dream commit versus its parent.
    With /dream-log <sha>: diff of that specific commit.
    """
    store = ctx.loop.consolidator.store
    git = store.git

    if not git.is_initialized():
        if store.get_last_dream_cursor() == 0:
            msg = (
                "Dream has not run yet. Run `/dream`, or wait for the next scheduled Dream cycle.\n\n"
                "Use `/dream-prompt` to see or change how Dream organizes memory."
            )
        else:
            msg = "Dream history is not available because memory versioning is not initialized."
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=msg, metadata={"render_as": "text"},
        )

    args = ctx.args.strip()

    if args:
        # Show diff of a specific commit
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        if not result:
            content = (
                f"Couldn't find Dream change `{sha}`.\n\n"
                "Use `/dream-restore` to list recent versions, "
                "or `/dream-log` to inspect the latest one."
            )
        else:
            commit, diff = result
            content = _format_dream_log_content(commit, diff, requested_sha=sha)
    else:
        # Default: show the latest Dream commit's diff
        commits = git.log(max_entries=1, message_prefix=_DREAM_COMMIT_PREFIX)
        result = (
            git.show_commit_diff(
                commits[0].sha,
                max_entries=1,
                message_prefix=_DREAM_COMMIT_PREFIX,
            )
            if commits else None
        )
        if result:
            commit, diff = result
            content = _format_dream_log_content(commit, diff)
        else:
            content = (
                "Dream memory has no saved versions yet.\n\n"
                "Use `/dream-prompt` to see or change how Dream organizes memory."
            )

    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


async def cmd_dream_restore(ctx: CommandContext) -> OutboundMessage:
    """Restore memory files from a previous dream commit.

    Usage:
        /dream-restore          — list recent commits
        /dream-restore <sha>    — revert a specific commit
    """
    store = ctx.loop.consolidator.store
    git = store.git
    if not git.is_initialized():
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="Dream history is not available because memory versioning is not initialized.",
        )

    args = ctx.args.strip()
    if not args:
        # Show recent Dream commits for the user to pick
        commits = git.log(max_entries=10, message_prefix=_DREAM_COMMIT_PREFIX)
        if not commits:
            content = "Dream memory has no saved versions to restore yet."
        else:
            content = _format_dream_restore_list(commits)
    else:
        sha = args.split()[0]
        result = git.show_commit_diff(sha, message_prefix=_DREAM_COMMIT_PREFIX)
        if not result:
            content = (
                f"Couldn't restore Dream change `{sha}`.\n\n"
                "Only Dream memory versions can be restored. "
                "Use `/dream-restore` to list recent versions."
            )
        else:
            changed_files = _format_changed_files(result[1])
            new_sha = git.revert(sha, message_prefix=_DREAM_COMMIT_PREFIX)
            if new_sha:
                content = (
                    f"Restored Dream memory to the state before `{sha}`.\n\n"
                    f"- New safety commit: `{new_sha}`\n"
                    f"- Restored files: {changed_files}\n\n"
                    f"Use `/dream-log {new_sha}` to inspect the restore diff."
                )
            else:
                content = (
                    f"Couldn't restore Dream change `{sha}`.\n\n"
                    "It may be the first saved version with no earlier state to restore."
                )
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


_HISTORY_DEFAULT_COUNT = 10
_HISTORY_MAX_COUNT = 50
_HISTORY_MAX_CONTENT_CHARS = 200


_SESSIONS_MAX_LIST = 10
_SESSIONS_INTERNAL_PREFIXES = ("dream:", "subagent:")


def _sessions_display_text(info: dict[str, Any]) -> str:
    """One-line label for a saved session: title, else first-message preview."""
    title = str(info.get("title") or "").strip()
    text = title or str(info.get("preview") or "").strip().replace("\n", " ")
    if len(text) > 60:
        text = text[:60] + "…"
    return text or "(empty session)"


def _sessions_relative_time(updated_at: object) -> str:
    """Relative time for a session's last update (UTC ISO string)."""
    if not isinstance(updated_at, str) or not updated_at:
        return ""
    try:
        moment = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - moment).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


def _sessions_reply(
    ctx: CommandContext,
    content: str,
    buttons: list[list[str]] | None = None,
    callback_data: list[list[str]] | None = None,
) -> OutboundMessage:
    metadata: dict[str, Any] = {**dict(ctx.msg.metadata or {}), "render_as": "text"}
    if callback_data:
        # Telegram-only: per-button tap payloads, parallel to `buttons`.
        metadata["telegram_callback_data"] = callback_data
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        buttons=buttons or [],
        metadata=metadata,
    )


async def cmd_sessions(ctx: CommandContext) -> OutboundMessage:
    """List saved sessions and switch this chat to one.

    Usage: /sessions — list; /sessions <n> — switch to list entry n;
    /sessions pick <list>:<n> — switch from an inline-button tap;
    /sessions delete <n> — delete list entry n (asks for confirmation first);
    /sessions clear — return to this chat's default session.
    """
    loop = ctx.loop
    args = ctx.args.strip()
    current_key = ctx.key

    if args.lower() == "clear":
        await loop.clear_chat_session_binding(ctx.msg.channel, ctx.msg.chat_id)
        return _sessions_reply(
            ctx,
            "Back to this chat's default session. The next message starts or "
            "continues the default session.",
        )

    lowered = args.lower()
    if lowered == "delete-no":
        return _sessions_reply(ctx, "Deletion cancelled. Nothing changed.")
    if lowered.startswith("delete-yes "):
        _, _, confirm_arg = args.partition(" ")
        list_id, sep, index_text = confirm_arg.strip().partition(":")
        keys = (
            loop.resolve_session_picker_list(
                ctx.msg.channel, ctx.msg.chat_id, list_id
            )
            if sep and index_text.strip().isdigit()
            else None
        )
        if keys is None:
            return _sessions_reply(
                ctx,
                "That confirmation has expired. Send /sessions again for "
                "a fresh list.",
            )
        index = int(index_text.strip()) - 1
        if not 0 <= index < len(keys):
            return _sessions_reply(
                ctx,
                f"Number out of range (1–{len(keys)}). Send /sessions "
                "to see the list again.",
            )
        target = keys[index]
        infos = await loop.sessions.state.list_sessions()
        info = next((row for row in infos if row.get("key") == target), None)
        if info is None:
            return _sessions_reply(
                ctx, "That session no longer exists. Send /sessions for a fresh list."
            )
        effective_key = loop._effective_session_key(ctx.msg)  # pyright: ignore[reportPrivateUsage]
        if target == effective_key:
            return _sessions_reply(
                ctx,
                "This session is in use by this chat. Switch to another "
                "session before deleting it.",
            )
        deleted = await loop.sessions.state.delete(target)
        if not deleted:
            return _sessions_reply(
                ctx, "That session no longer exists. Send /sessions for a fresh list."
            )
        bindings = await loop.sessions.state.list_chat_session_bindings()
        if bindings.get((ctx.msg.channel, ctx.msg.chat_id)) == target:
            await loop.clear_chat_session_binding(ctx.msg.channel, ctx.msg.chat_id)
        return _sessions_reply(
            ctx,
            f"Session deleted: {_sessions_display_text(info)}\n"
            "Send /sessions for a fresh list.",
        )
    if lowered.startswith("delete ") or lowered == "delete":
        parts = args.split(None, 1)
        number_text = parts[1].strip() if len(parts) > 1 else ""
        if not number_text.isdigit():
            return _sessions_reply(
                ctx,
                "Usage: /sessions delete <n> — delete that numbered session "
                "from the latest /sessions list (asks first).",
            )
        keys = loop.latest_session_picker_list(ctx.msg.channel, ctx.msg.chat_id)
        if keys is None:
            return _sessions_reply(
                ctx,
                "No session list yet. Send /sessions first, then reply "
                "with its number.",
            )
        index = int(number_text) - 1
        if not 0 <= index < len(keys):
            return _sessions_reply(
                ctx,
                f"Number out of range (1–{len(keys)}). Send /sessions "
                "to see the list again.",
            )
        target = keys[index]
        infos = await loop.sessions.state.list_sessions()
        info = next((row for row in infos if row.get("key") == target), None)
        if info is None:
            return _sessions_reply(
                ctx, "That session no longer exists. Send /sessions for a fresh list."
            )
        effective_key = loop._effective_session_key(ctx.msg)  # pyright: ignore[reportPrivateUsage]
        if target == effective_key:
            return _sessions_reply(
                ctx,
                "This session is in use by this chat. Switch to another "
                "session before deleting it.",
            )
        confirm_list_id = loop.register_session_picker_list(
            ctx.msg.channel, ctx.msg.chat_id, keys
        )
        return _sessions_reply(
            ctx,
            f"Delete this session permanently?\n{_sessions_display_text(info)}\n"
            "Its history cannot be recovered.",
            buttons=[["Yes, delete"], ["Cancel"]],
            callback_data=[
                [f"/sessions delete-yes {confirm_list_id}:{index + 1}"],
                ["/sessions delete-no"],
            ],
        )

    if args:
        target: str | None = None
        if args.lower().startswith("pick "):
            pick_arg = args[5:].strip()
            list_id, sep, index_text = pick_arg.partition(":")
            keys = (
                loop.resolve_session_picker_list(
                    ctx.msg.channel, ctx.msg.chat_id, list_id
                )
                if sep and index_text.strip().isdigit()
                else None
            )
            if keys is not None:
                index = int(index_text.strip()) - 1
                if 0 <= index < len(keys):
                    target = keys[index]
            if target is None:
                return _sessions_reply(
                    ctx,
                    "That session choice has expired. Send /sessions again for "
                    "a fresh list.",
                )
        elif args.isdigit():
            keys = loop.latest_session_picker_list(ctx.msg.channel, ctx.msg.chat_id)
            if keys is None:
                return _sessions_reply(
                    ctx,
                    "No session list yet. Send /sessions first, then reply "
                    "with its number.",
                )
            index = int(args) - 1
            if not 0 <= index < len(keys):
                return _sessions_reply(
                    ctx,
                    f"Number out of range (1–{len(keys)}). Send /sessions "
                    "to see the list again.",
                )
            target = keys[index]
        else:
            return _sessions_reply(
                ctx,
                "Usage: /sessions — list; /sessions <n> — switch; "
                "/sessions delete <n> — delete (asks first); "
                "/sessions clear — back to the default session.",
            )
        if target is None:
            return _sessions_reply(
                ctx, "Session not found. Send /sessions for a fresh list."
            )

        infos = await loop.sessions.state.list_sessions()
        info = next((row for row in infos if row.get("key") == target), None)
        if info is None:
            return _sessions_reply(
                ctx, "That session no longer exists. Send /sessions for a fresh list."
            )
        if target == current_key:
            return _sessions_reply(ctx, "This chat is already on that session.")
        await loop._cancel_active_tasks(current_key)  # pyright: ignore[reportPrivateUsage]
        await loop.set_chat_session_binding(ctx.msg.channel, ctx.msg.chat_id, target)
        return _sessions_reply(
            ctx,
            f"Switched to session: {_sessions_display_text(info)}\n"
            "The next message continues this session. /sessions to switch again, "
            "/new to restart this session, /sessions clear to go back "
            "to the default session.",
        )

    infos = await loop.sessions.state.list_sessions()
    chat_base = session_key_for_channel(ctx.msg.channel, ctx.msg.chat_id)
    bases = {chat_base, session_base_key(current_key)}
    candidates = [
        info
        for info in infos
        if not str(info.get("key", "")).startswith(_SESSIONS_INTERNAL_PREFIXES)
        and any(
            session_key_under_base(str(info.get("key", "")), base) for base in bases
        )
    ]
    shown = candidates[:_SESSIONS_MAX_LIST]
    if not shown:
        return _sessions_reply(ctx, "No saved sessions yet.")
    list_id = loop.register_session_picker_list(
        ctx.msg.channel, ctx.msg.chat_id, [str(info["key"]) for info in shown]
    )
    lines = ["Saved sessions (newest first):"]
    buttons: list[list[str]] = []
    callback_data: list[list[str]] = []
    for index, info in enumerate(shown, start=1):
        display = _sessions_display_text(info)
        relative = _sessions_relative_time(info.get("updated_at"))
        marker = " — this session" if info["key"] == current_key else ""
        lines.append(
            f"{index}. {display}" + (f" ({relative})" if relative else "") + marker
        )
        buttons.append([f"{index}. {display[:32]}"])
        callback_data.append([f"/sessions pick {list_id}:{index}"])
    lines.append("")
    lines.append(
        "Tap a button, or reply /sessions <n>. "
        "/sessions clear goes back to the default session. "
        "/sessions delete <n> deletes a session (asks first)."
    )
    if len(candidates) > _SESSIONS_MAX_LIST:
        lines.append(
            f"Showing the {_SESSIONS_MAX_LIST} newest; older sessions are available via the WebUI."
        )
    return _sessions_reply(
        ctx, "\n".join(lines), buttons=buttons, callback_data=callback_data
    )


def _format_history_message(msg: dict[str, Any]) -> str | None:
    """Format a single history message for display. Returns None to skip."""
    role = msg.get("role")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content") or ""
    if isinstance(content, list):
        parts = [
            text
            for block in cast(list[object], content)
            if (item := cast(dict[str, Any], block) if isinstance(block, dict) else None)
            and item.get("type") == "text"
            and isinstance(text := item.get("text"), str)
        ]
        content = " ".join(parts)
    content = str(content).strip()
    if not content:
        return None
    if len(content) > _HISTORY_MAX_CONTENT_CHARS:
        content = content[:_HISTORY_MAX_CONTENT_CHARS] + "…"
    label = "👤 You" if role == "user" else "🤖 Bot"
    return f"{label}: {content}"


async def cmd_history(ctx: CommandContext) -> OutboundMessage:
    """Show the last N messages of the current session (default 10, max 50).

    Usage: /history [count]
    """
    count = _HISTORY_DEFAULT_COUNT
    if ctx.args.strip():
        try:
            count = max(1, min(int(ctx.args.strip()), _HISTORY_MAX_COUNT))
        except ValueError:
            return OutboundMessage(
                channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
                content="Usage: /history [count] — e.g. /history 5 (default: 10, max: 50)",
                metadata=dict(ctx.msg.metadata or {}),
            )

    session = ctx.session or await ctx.loop.sessions.state.get(ctx.key)
    history = session.get_history(max_messages=0, include_runtime_context=False)
    visible = [_format_history_message(m) for m in history]
    visible = [m for m in visible if m is not None]
    recent = visible[-count:]

    if not recent:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="No conversation history yet.",
            metadata=dict(ctx.msg.metadata or {}),
        )

    header = f"Last {len(recent)} message(s):\n"
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=header + "\n".join(recent),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_goal(ctx: CommandContext) -> InboundMessage | OutboundMessage:
    """Expand an explicit goal command into model-only input."""
    goal = ctx.args.strip()
    if not goal:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="Usage: /goal <long-running task description>",
            metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
        )
    if not ctx.is_user_turn:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="Goal mode can only be started by a user `/goal <task>` command.",
            metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
        )

    session = ctx.session or await ctx.loop.sessions.state.get(ctx.key)
    session.add_message("user", ctx.msg.content, _command=True, media=list(ctx.msg.media))
    await ctx.loop.sessions.state.prepare_input(session)
    return replace(
        ctx.msg,
        content=goal,
        metadata={
            **ctx.msg.metadata,
            "original_command": "/goal",
            "original_content": ctx.msg.content,
            "goal_requested": True,
            "goal_started_at": time.time(),
            HIDDEN_HISTORY_META: {"kind": "goal_request"},
        },
    )


async def cmd_pairing(ctx: CommandContext) -> OutboundMessage:
    """List, approve, deny or revoke pairing requests."""
    from nanobot.pairing import PAIRING_COMMAND_META_KEY, handle_pairing_command
    from nanobot.pairing.events import PairingChanged

    def changed() -> None:
        ctx.loop.bus.publish_nowait(PairingChanged())

    reply = handle_pairing_command(ctx.msg.channel, ctx.args, on_change=changed)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=reply,
        metadata={PAIRING_COMMAND_META_KEY: True},
    )


async def cmd_skill(ctx: CommandContext) -> OutboundMessage:
    """List all enabled skills (name and description only)."""
    loop = ctx.loop
    skills = loop.context.skills.list_skills(filter_unavailable=False)
    if not skills:
        content = "No skills available."
    else:
        lines = [f"Available skills ({len(skills)}):", ""]
        for entry in skills:
            desc = loop.context.skills.get_skill_description(entry["name"])
            lines.append(f"- **{entry['name']}** — {desc}")
        content = "\n".join(lines)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata=dict(ctx.msg.metadata or {}),
    )


async def cmd_trigger(ctx: CommandContext) -> OutboundMessage:
    """Create a local trigger bound to the current session."""
    name = ctx.args.strip()
    if not name:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=(
                "Usage: /trigger <name>\n\n"
                "Create a named local trigger bound to this chat session."
            ),
            metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
        )

    from nanobot.triggers.local_store import LocalTriggerStore

    loop = ctx.loop
    store = loop.local_trigger_store
    if store is None:
        store = LocalTriggerStore(loop.workspace)

    from nanobot.session.keys import UNIFIED_SESSION_KEY

    session_key = (
        ctx.msg.session_key
        if ctx.key == UNIFIED_SESSION_KEY
        else ctx.key
    )
    trigger = store.create(
        name=name,
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        session_key=session_key,
        sender_id="trigger",
        origin_metadata=dict(ctx.msg.metadata or {}),
    )
    command = f'nanobot trigger {trigger.id} "message"'
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=(
            f"Trigger created: {trigger.name}\n"
            f"ID: {trigger.id}\n\n"
            f"Command:\n{command}"
        ),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )

async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_user_shell(ctx: CommandContext) -> OutboundMessage:
    """Run a trusted local ``!command`` through nanobot's exec policy."""
    metadata = dict(ctx.msg.metadata or {})
    if (
        ctx.msg.channel != "websocket"
        or metadata.get("webui") is not True
        or metadata.get(INBOUND_META_USER_SHELL) is not True
    ):
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="Shell commands are only available from a trusted local client.",
            metadata={**metadata, "render_as": "text"},
        )
    if not ctx.args.strip():
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="Type a command after `!`, for example `!pwd`.",
            metadata={**metadata, "render_as": "text"},
        )
    return await ctx.loop.execute_user_shell_command(ctx)


def build_help_text() -> str:
    """Build canonical help text shared across channels."""
    lines = ["🐈 nanobot commands:"]
    for spec in BUILTIN_COMMAND_SPECS:
        command = spec.command
        if spec.arg_hint:
            command = f"{command} {spec.arg_hint}"
        lines.append(f"{command} — {spec.description}")
    return "\n".join(lines)


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/sessions", cmd_sessions)
    router.prefix("/sessions ", cmd_sessions)
    router.exact("/compact", cmd_compact, session_queue=True)
    router.exact("/status", cmd_status)
    router.exact("/model", cmd_model)
    router.prefix("/model ", cmd_model)
    router.exact("/history", cmd_history)
    router.prefix("/history ", cmd_history)
    router.exact("/goal", cmd_goal)
    router.prefix("/goal ", cmd_goal)
    router.exact("/trigger", cmd_trigger)
    router.prefix("/trigger ", cmd_trigger)
    router.exact("/dream", cmd_dream)
    router.exact("/dream-log", cmd_dream_log)
    router.prefix("/dream-log ", cmd_dream_log)
    router.exact("/dream-restore", cmd_dream_restore)
    router.prefix("/dream-restore ", cmd_dream_restore)
    router.exact("/dream-prompt", cmd_dream_prompt)
    router.prefix("/dream-prompt ", cmd_dream_prompt)
    router.exact("/evaluator-prompt", cmd_evaluator_prompt)
    router.prefix("/evaluator-prompt ", cmd_evaluator_prompt)
    router.exact("/skill", cmd_skill)
    router.exact("/help", cmd_help)
    router.exact("/pairing", cmd_pairing)
    router.prefix("/pairing ", cmd_pairing)
    router.exact(USER_SHELL_COMMAND, cmd_user_shell)
    router.prefix(f"{USER_SHELL_COMMAND} ", cmd_user_shell)
