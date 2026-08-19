"""A ``tool:pre`` hook that rewrites arguments must actually change what runs.

Both dispatch paths emitted ``tool:pre``, passed the result through
``coordinator.process_hook_result``, branched on ``action == "deny"`` -- and
then executed the ORIGINAL arguments regardless. Every hook that rewrites input
(argument normalization, path jailing, secret scrubbing) was a silent no-op:
it ran, returned its correction, and was ignored.

The subtlety that made this easy to get wrong: the kernel NORMALIZES ``modify``
away. ``emit()`` returns ``action="continue"`` with the modified payload in
``data`` (see ``hooks.rs``), so the pattern documented in
``ORCHESTRATOR_CONTRACT.md`` -- ``if result.action == "modify"`` -- is
unreachable code and can never fire. Reading ``data`` unconditionally is the
only consumption that works.

``tool:post`` already honored modifications. This makes ``tool:pre`` symmetric.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import HookRegistry, ToolResult
from amplifier_module_loop_streaming import StreamingOrchestrator

PADDED = {"action": " create ", "path": " /tmp/x "}
CLEANED = {"action": "create", "path": "/tmp/x"}


class _RecordingTool:
    """Captures exactly what arguments reached ``execute``."""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    @property
    def name(self) -> str:
        return "todo"

    async def execute(self, arguments: Any) -> ToolResult:
        self.seen.append(arguments)
        return ToolResult(success=True, data={"ok": True})


def _tool_call(arguments: dict[str, Any] | None = None) -> Any:
    call = MagicMock()
    call.id = "call-1"
    call.name = "todo"
    call.arguments = dict(PADDED if arguments is None else arguments)
    return call


def _coordinator(hook_data: Any) -> Any:
    """A coordinator whose ``tool:pre`` result carries *hook_data* as ``.data``."""
    coordinator = MagicMock()
    coordinator._tool_dispatch_contexts = {}
    coordinator.cancellation.register_tool_start = MagicMock()
    coordinator.cancellation.register_tool_complete = MagicMock()
    result = MagicMock()
    result.action = "continue"
    result.data = hook_data
    coordinator.process_hook_result = AsyncMock(return_value=result)
    return coordinator


class _Context:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def add_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_parallel_path_adopts_rewritten_arguments() -> None:
    tool = _RecordingTool()
    call = _tool_call()

    await StreamingOrchestrator(config={})._execute_tool_only(
        call,
        {"todo": tool},
        HookRegistry(),
        "group-1",
        _coordinator({"tool_input": CLEANED}),
    )

    assert tool.seen == [CLEANED], (
        f"the tool ran with {tool.seen!r}; a tool:pre hook's correction was discarded"
    )


@pytest.mark.asyncio
async def test_sequential_path_adopts_rewritten_arguments() -> None:
    """The second dispatch site -- the one that runs when tools are not batched."""
    tool = _RecordingTool()
    call = _tool_call()

    await StreamingOrchestrator(config={})._execute_tool_with_result(
        call,
        {"todo": tool},
        _Context(),
        HookRegistry(),
        _coordinator({"tool_input": CLEANED}),
    )

    assert tool.seen == [CLEANED]


@pytest.mark.asyncio
async def test_an_unmodified_payload_is_not_treated_as_a_rewrite() -> None:
    """The kernel round-trips the payload, so ``data`` is always a NEW object.

    Equal content must not count as a modification, or every call would look
    like it had been rewritten.
    """
    tool = _RecordingTool()
    call = _tool_call()
    original = call.arguments

    await StreamingOrchestrator(config={})._execute_tool_only(
        call,
        {"todo": tool},
        HookRegistry(),
        "group-1",
        _coordinator({"tool_input": dict(PADDED)}),  # equal content, different object
    )

    assert tool.seen == [PADDED]
    assert call.arguments is original, "arguments were replaced by an equal copy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook_data",
    [
        pytest.param(None, id="no-data"),
        pytest.param({}, id="empty-data"),
        pytest.param({"tool_name": "todo"}, id="data-without-tool-input"),
        pytest.param(MagicMock(), id="non-dict-data"),
        pytest.param("not a mapping", id="string-data"),
    ],
)
async def test_a_partial_result_never_corrupts_the_arguments(hook_data: Any) -> None:
    """Adopt only from a real mapping.

    A partial or mocked hook result can carry a non-dict ``data``; reading it
    loosely would replace real arguments with nonsense, which is a worse
    failure than the no-op this fix removes.
    """
    tool = _RecordingTool()
    call = _tool_call()

    await StreamingOrchestrator(config={})._execute_tool_only(
        call, {"todo": tool}, HookRegistry(), "group-1", _coordinator(hook_data)
    )

    assert tool.seen == [PADDED], f"arguments were corrupted to {tool.seen!r}"
