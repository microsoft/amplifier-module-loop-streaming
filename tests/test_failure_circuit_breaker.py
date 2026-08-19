"""Stop an agent re-issuing a call that keeps failing the exact same way.

In session ``eec9ae98`` the SAME failing ``read_file`` call was issued **13
times** with nothing intervening. Whitespace-malformed arguments fail
deterministically -- same input, same error, forever -- so an agent that cannot
notice the repetition burns tokens and wall-clock producing nothing. 37 distinct
tool inputs were issued more than once; 73 calls were redundant.

The breaker keys on **(tool, arguments, error)** and deliberately NOT on
arguments alone. Legitimate repeats exist: polling a file being written,
``git status`` in a loop, retrying after fixing something externally. Only a
call that fails the SAME way counts toward the trip.

The trip is SURFACED to the model, never silently dropped -- a silent breaker is
the same class of bug as a silent argument rewrite.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import HookRegistry, ToolResult
from amplifier_module_loop_streaming import StreamingOrchestrator

THRESHOLD = StreamingOrchestrator._FAILURE_BREAKER_THRESHOLD


def _orchestrator() -> StreamingOrchestrator:
    return StreamingOrchestrator(config={})


def _call(arguments: dict[str, Any] | None = None, name: str = "read_file") -> Any:
    tool_call = MagicMock()
    tool_call.id = "call-1"
    tool_call.name = name
    tool_call.arguments = arguments if arguments is not None else {"path": " /tmp/x "}
    return tool_call


def _failure(message: str = "Path not found:  /tmp/x ") -> ToolResult:
    return ToolResult(success=False, error={"message": message})


def _message(result: ToolResult) -> str:
    return (result.error or {}).get("message", "")


def test_the_first_failures_pass_through_untouched() -> None:
    """Below the threshold the model sees exactly what the tool said."""
    orchestrator = _orchestrator()
    call = _call()

    for _ in range(THRESHOLD - 1):
        out = orchestrator._apply_failure_breaker(call, _failure())
        assert _message(out) == "Path not found:  /tmp/x "
        assert "has now failed" not in _message(out)


def test_the_same_failure_repeated_trips_the_breaker() -> None:
    """The defect: 13 identical failures with nothing intervening."""
    orchestrator = _orchestrator()
    call = _call()

    for _ in range(THRESHOLD - 1):
        orchestrator._apply_failure_breaker(call, _failure())
    tripped = orchestrator._apply_failure_breaker(call, _failure())

    note = _message(tripped)
    assert "Path not found:  /tmp/x " in note, "the tool's real error must be preserved"
    assert f"has now failed {THRESHOLD} times" in note
    assert "read_file" in note
    assert "different approach" in note, (
        "the note must tell the model what to do instead"
    )


def test_a_different_error_for_the_same_input_does_not_count() -> None:
    """Same call, different failure, is not the loop this guards against."""
    orchestrator = _orchestrator()
    call = _call()

    for i in range(THRESHOLD * 2):
        out = orchestrator._apply_failure_breaker(
            call, _failure(f"transient error {i}")
        )
        assert "has now failed" not in _message(out)


def test_different_arguments_do_not_count_toward_each_other() -> None:
    """Two paths that each fail once are not one call failing twice."""
    orchestrator = _orchestrator()

    for i in range(THRESHOLD * 2):
        call = _call({"path": f" /tmp/{i} "})
        out = orchestrator._apply_failure_breaker(call, _failure("Path not found"))
        assert "has now failed" not in _message(out)


def test_the_same_failure_from_a_different_tool_does_not_count() -> None:
    orchestrator = _orchestrator()

    for name in ("read_file", "write_file", "glob", "grep"):
        call = _call(name=name)
        out = orchestrator._apply_failure_breaker(call, _failure("Path not found"))
        assert "has now failed" not in _message(out)


def test_success_never_trips_and_is_returned_unchanged() -> None:
    """Polling a file being written must not be mistaken for a stuck loop."""
    orchestrator = _orchestrator()
    call = _call()
    success = ToolResult(success=True, data={"content": "ok"})

    for _ in range(THRESHOLD * 3):
        assert orchestrator._apply_failure_breaker(call, success) is success


def test_unhashable_arguments_do_not_break_dispatch() -> None:
    """Arguments are not guaranteed to be JSON-serialisable."""
    orchestrator = _orchestrator()
    call = _call({"path": object()})

    for _ in range(THRESHOLD):
        out = orchestrator._apply_failure_breaker(call, _failure())
    assert "has now failed" in _message(out)


class _AlwaysFailingTool:
    @property
    def name(self) -> str:
        return "read_file"

    async def execute(self, arguments: Any) -> ToolResult:
        del arguments
        return _failure()


def _coordinator() -> Any:
    coordinator = MagicMock()
    coordinator._tool_dispatch_contexts = {}
    coordinator.cancellation.register_tool_start = MagicMock()
    coordinator.cancellation.register_tool_complete = MagicMock()
    result = MagicMock()
    result.action = "continue"
    result.data = None
    coordinator.process_hook_result = AsyncMock(return_value=result)
    return coordinator


@pytest.mark.asyncio
async def test_the_breaker_reaches_the_model_through_real_dispatch() -> None:
    """End to end on the parallel path: the note lands in the tool content."""
    orchestrator = _orchestrator()
    tools = {"read_file": _AlwaysFailingTool()}

    contents: list[str] = []
    for _ in range(THRESHOLD):
        _id, _name, content = await orchestrator._execute_tool_only(
            _call(), tools, HookRegistry(), "group-1", _coordinator()
        )
        contents.append(content)

    assert "has now failed" not in contents[0]
    assert "has now failed" in contents[-1], (
        f"the breaker note never reached the model: {contents[-1]!r}"
    )
