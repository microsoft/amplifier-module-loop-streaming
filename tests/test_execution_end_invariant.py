"""``execution:end`` must fire even when the turn exits early.

Regression cover for session ``eec9ae98``: **27 ``execution:start`` events, 15
``execution:end``**. The 12 missing ends matched the 12 cancellations exactly,
on both the kernel event log and the UI event stream, leaving the turn state
machine stuck in "executing" and never unwinding.

Cause: ``_execute_stream`` emitted the end event on its last line, and several
paths returned before reaching it -- the graceful-cancellation exit, immediate
cancellation between chunks, a denied ``provider:request``, and "no providers
available". A consumer breaking out of its ``async for`` skipped it too.

The fix is a ``finally`` inside the generator rather than an emit bolted onto
each early return: it covers the paths that exist, the paths nobody has written
yet, and ``GeneratorExit``.
"""

from __future__ import annotations

from typing import Any

import pytest
from amplifier_core import HookRegistry


def _orchestrator() -> Any:
    from amplifier_module_loop_streaming import StreamingOrchestrator

    return StreamingOrchestrator(config={})


class _Context:
    """Enough context surface to reach the early return under test."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def add_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    async def get_messages(self) -> list[dict[str, Any]]:
        return list(self.messages)

    async def get_messages_for_request(self) -> list[dict[str, Any]]:
        return list(self.messages)


def _recording_hooks() -> tuple[HookRegistry, list[str]]:
    hooks = HookRegistry()
    seen: list[str] = []

    async def record(event: str, data: Any) -> None:
        del data
        seen.append(event)

    hooks.register("execution:start", record)
    hooks.register("execution:end", record)
    return hooks, seen


@pytest.mark.asyncio
async def test_execution_end_fires_on_the_no_provider_early_return() -> None:
    """The simplest early exit that lands after ``execution:start``.

    Before the fix this path emitted a start with no matching end, which is the
    shape that left the turn state machine stuck.
    """
    hooks, seen = _recording_hooks()
    orchestrator = _orchestrator()

    tokens = [
        token
        async for token, _iteration in orchestrator._execute_stream(
            "do the thing", _Context(), {}, {}, hooks
        )
    ]

    assert any("No providers available" in token for token in tokens), (
        f"fixture did not reach the intended early return; got {tokens!r}"
    )
    assert seen.count("execution:start") == 1
    assert seen.count("execution:end") == 1, (
        f"execution:end did not fire on an early return: {seen}"
    )


@pytest.mark.asyncio
async def test_execution_end_fires_when_the_consumer_stops_reading() -> None:
    """A consumer that breaks out of ``async for`` must still close the turn.

    This is the cancellation shape from the incident: the turn stops because
    something upstream stopped listening, not because the loop ran to
    completion. Python raises ``GeneratorExit`` at the suspended yield, so the
    ``finally`` runs -- an emit bolted onto each ``return`` would not have.
    """
    hooks, seen = _recording_hooks()
    orchestrator = _orchestrator()

    stream = orchestrator._execute_stream("do the thing", _Context(), {}, {}, hooks)
    async for _token, _iteration in stream:
        break  # stop reading after the first token
    await stream.aclose()

    assert seen.count("execution:start") == 1
    assert seen.count("execution:end") == 1, (
        f"execution:end did not fire when the consumer stopped reading: {seen}"
    )


@pytest.mark.asyncio
async def test_every_start_has_exactly_one_end_across_repeated_turns() -> None:
    """The invariant the incident violated, stated directly: 27 starts, 15 ends."""
    hooks, seen = _recording_hooks()
    orchestrator = _orchestrator()

    for _ in range(5):
        async for _token, _iteration in orchestrator._execute_stream(
            "do the thing", _Context(), {}, {}, hooks
        ):
            pass

    assert seen.count("execution:start") == 5
    assert seen.count("execution:end") == 5, (
        f"starts and ends are unbalanced across turns: {seen}"
    )
