"""Tests for _tool_dispatch_context set on coordinator during tool.execute().

Verifies that StreamingOrchestrator sets coordinator._tool_dispatch_context
with the correct tool_call_id and parallel_group_id immediately before
calling tool.execute(), and clears it in a finally block afterward.

Covers:
- _execute_tool_only: context set with tool_call_id and parallel_group_id
- _execute_tool_only: context cleared after tool completes
- _execute_tool_only: context cleared even when tool raises an exception
- _execute_tool_with_result: context set with tool_call_id (parallel_group_id=None)
- _execute_tool_with_result: context cleared after tool completes
- Integration: full execute() path sets dispatch context during tool call
"""

import pytest
from amplifier_core import ToolResult
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_streaming import StreamingOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockCancellation:
    """Minimal cancellation token stub."""

    is_cancelled: bool = False
    is_immediate: bool = False

    def register_tool_start(self, tool_call_id: str, tool_name: str) -> None:
        pass

    def register_tool_complete(self, tool_call_id: str) -> None:
        pass


class MockCoordinator:
    """Minimal coordinator stub that supports _tool_dispatch_context."""

    def __init__(self) -> None:
        self.cancellation = MockCancellation()

    async def process_hook_result(self, result: object, event_name: str, source: str) -> object:
        return result


def _orch(**overrides: object) -> StreamingOrchestrator:
    config = {"max_iterations": 5, "stream_delay": 0, **overrides}
    return StreamingOrchestrator(config)


# ---------------------------------------------------------------------------
# _execute_tool_only: dispatch context is set during tool execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_only_sets_tool_call_id_in_dispatch_context() -> None:
    """_execute_tool_only sets tool_call_id on coordinator._tool_dispatch_context
    before calling tool.execute().
    """
    captured: dict = {}
    coordinator = MockCoordinator()

    class CapturingTool:
        name = "capture_tool"
        description = "Captures dispatch context"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            captured.update(getattr(coordinator, "_tool_dispatch_context", {}))
            return ToolResult(success=True, output="done")

    orch = _orch()
    tc = ToolCall(id="call-abc-123", name="capture_tool", arguments={})
    hooks = EventRecorder()

    await orch._execute_tool_only(  # type: ignore[arg-type]
        tc,
        {"capture_tool": CapturingTool()},
        hooks,
        "group-xyz-456",
        coordinator,  # type: ignore[arg-type]
    )

    assert captured.get("tool_call_id") == "call-abc-123", (
        "_tool_dispatch_context must have tool_call_id set during tool.execute()"
    )


@pytest.mark.asyncio
async def test_execute_tool_only_sets_parallel_group_id_in_dispatch_context() -> None:
    """_execute_tool_only sets parallel_group_id on coordinator._tool_dispatch_context
    before calling tool.execute().
    """
    captured: dict = {}
    coordinator = MockCoordinator()

    class CapturingTool:
        name = "capture_tool"
        description = "Captures dispatch context"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            captured.update(getattr(coordinator, "_tool_dispatch_context", {}))
            return ToolResult(success=True, output="done")

    orch = _orch()
    tc = ToolCall(id="call-abc-123", name="capture_tool", arguments={})
    hooks = EventRecorder()

    await orch._execute_tool_only(  # type: ignore[arg-type]
        tc,
        {"capture_tool": CapturingTool()},
        hooks,
        "group-xyz-456",
        coordinator,  # type: ignore[arg-type]
    )

    assert captured.get("parallel_group_id") == "group-xyz-456", (
        "_tool_dispatch_context must have parallel_group_id set during tool.execute()"
    )


@pytest.mark.asyncio
async def test_execute_tool_only_clears_dispatch_context_after_completion() -> None:
    """_execute_tool_only clears coordinator._tool_dispatch_context after tool completes."""
    coordinator = MockCoordinator()

    class SimpleTool:
        name = "simple_tool"
        description = "Returns a result"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            return ToolResult(success=True, output="done")

    orch = _orch()
    tc = ToolCall(id="call-clear-test", name="simple_tool", arguments={})
    hooks = EventRecorder()

    await orch._execute_tool_only(  # type: ignore[arg-type]
        tc,
        {"simple_tool": SimpleTool()},
        hooks,
        "group-001",
        coordinator,  # type: ignore[arg-type]
    )

    ctx_after = getattr(coordinator, "_tool_dispatch_context", None)
    assert ctx_after == {}, (
        "_tool_dispatch_context must be cleared to {} after tool.execute() completes"
    )


@pytest.mark.asyncio
async def test_execute_tool_only_clears_dispatch_context_on_tool_exception() -> None:
    """_execute_tool_only clears coordinator._tool_dispatch_context even if tool raises."""
    coordinator = MockCoordinator()

    class RaisingTool:
        name = "raising_tool"
        description = "Always raises"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            raise RuntimeError("tool exploded")

    orch = _orch()
    tc = ToolCall(id="call-raise-test", name="raising_tool", arguments={})
    hooks = EventRecorder()

    # Should not propagate — _execute_tool_only converts exceptions to error ToolResults
    await orch._execute_tool_only(  # type: ignore[arg-type]
        tc,
        {"raising_tool": RaisingTool()},
        hooks,
        "group-002",
        coordinator,  # type: ignore[arg-type]
    )

    ctx_after = getattr(coordinator, "_tool_dispatch_context", None)
    assert ctx_after == {}, (
        "_tool_dispatch_context must be cleared even when tool.execute() raises"
    )


# ---------------------------------------------------------------------------
# _execute_tool_with_result: dispatch context is set during tool execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_with_result_sets_tool_call_id_in_dispatch_context() -> None:
    """_execute_tool_with_result sets tool_call_id on coordinator._tool_dispatch_context."""
    captured: dict = {}
    coordinator = MockCoordinator()

    class CapturingTool:
        name = "capture_tool"
        description = "Captures dispatch context"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            captured.update(getattr(coordinator, "_tool_dispatch_context", {}))
            return ToolResult(success=True, output="done")

    orch = _orch()
    tc = ToolCall(id="call-with-result-123", name="capture_tool", arguments={})
    hooks = EventRecorder()
    context = MockContextManager()

    await orch._execute_tool_with_result(  # type: ignore[arg-type]
        tc,
        {"capture_tool": CapturingTool()},
        context,
        hooks,
        coordinator,  # type: ignore[arg-type]
    )

    assert captured.get("tool_call_id") == "call-with-result-123", (
        "_execute_tool_with_result must set tool_call_id in _tool_dispatch_context"
    )


@pytest.mark.asyncio
async def test_execute_tool_with_result_sets_parallel_group_id_none() -> None:
    """_execute_tool_with_result sets parallel_group_id=None (not available on this path)."""
    captured: dict = {}
    coordinator = MockCoordinator()

    class CapturingTool:
        name = "capture_tool"
        description = "Captures dispatch context"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            ctx = getattr(coordinator, "_tool_dispatch_context", {})
            captured["parallel_group_id"] = ctx.get("parallel_group_id", "MISSING")
            return ToolResult(success=True, output="done")

    orch = _orch()
    tc = ToolCall(id="call-legacy-path", name="capture_tool", arguments={})
    hooks = EventRecorder()
    context = MockContextManager()

    await orch._execute_tool_with_result(  # type: ignore[arg-type]
        tc,
        {"capture_tool": CapturingTool()},
        context,
        hooks,
        coordinator,  # type: ignore[arg-type]
    )

    assert captured.get("parallel_group_id") is None, (
        "_execute_tool_with_result sets parallel_group_id=None (not available on legacy path)"
    )


@pytest.mark.asyncio
async def test_execute_tool_with_result_clears_dispatch_context_after_completion() -> None:
    """_execute_tool_with_result clears coordinator._tool_dispatch_context after completing."""
    coordinator = MockCoordinator()

    class SimpleTool:
        name = "simple_tool"
        description = "Returns a result"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            return ToolResult(success=True, output="done")

    orch = _orch()
    tc = ToolCall(id="call-clear-legacy", name="simple_tool", arguments={})
    hooks = EventRecorder()
    context = MockContextManager()

    await orch._execute_tool_with_result(  # type: ignore[arg-type]
        tc,
        {"simple_tool": SimpleTool()},
        context,
        hooks,
        coordinator,  # type: ignore[arg-type]
    )

    ctx_after = getattr(coordinator, "_tool_dispatch_context", None)
    assert ctx_after == {}, (
        "_tool_dispatch_context must be cleared after _execute_tool_with_result completes"
    )


# ---------------------------------------------------------------------------
# Integration: full execute() path sets dispatch context during tool call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_execute_sets_tool_call_id_during_tool_execution() -> None:
    """Integration: orchestrator.execute() sets _tool_dispatch_context during tool execution."""
    captured: dict = {}
    coordinator = MockCoordinator()

    class CapturingTool:
        name = "test_tool"
        description = "Captures dispatch context"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            captured.update(getattr(coordinator, "_tool_dispatch_context", {}))
            return ToolResult(success=True, output="done")

    class ToolThenTextProvider:
        name = "mock"

        def __init__(self) -> None:
            self._call_count = 0

        async def complete(self, request: object, **kwargs: object) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="Using tool")],
                    tool_calls=[
                        ToolCall(id="full-exec-call-id", name="test_tool", arguments={})
                    ],
                )
            return ChatResponse(content=[TextBlock(text="Done!")])

        def parse_tool_calls(self, response: object) -> list:
            return getattr(response, "tool_calls", []) or []

    orch = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    await orch.execute(
        prompt="Run the tool",
        context=context,
        providers={"default": ToolThenTextProvider()},  # type: ignore[dict-item]
        tools={"test_tool": CapturingTool()},  # type: ignore[dict-item]
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert captured.get("tool_call_id") == "full-exec-call-id", (
        "Full execute() path must set tool_call_id in _tool_dispatch_context"
    )


@pytest.mark.asyncio
async def test_full_execute_clears_dispatch_context_after_tool_execution() -> None:
    """Integration: orchestrator.execute() clears _tool_dispatch_context after tool."""
    coordinator = MockCoordinator()

    class SimpleTool:
        name = "test_tool"
        description = "Simple test tool"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, args: dict) -> ToolResult:
            return ToolResult(success=True, output="done")

    class ToolThenTextProvider:
        name = "mock"

        def __init__(self) -> None:
            self._call_count = 0

        async def complete(self, request: object, **kwargs: object) -> ChatResponse:
            self._call_count += 1
            if self._call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="Using tool")],
                    tool_calls=[
                        ToolCall(id="clear-test-call-id", name="test_tool", arguments={})
                    ],
                )
            return ChatResponse(content=[TextBlock(text="Done!")])

        def parse_tool_calls(self, response: object) -> list:
            return getattr(response, "tool_calls", []) or []

    orch = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    await orch.execute(
        prompt="Run the tool",
        context=context,
        providers={"default": ToolThenTextProvider()},  # type: ignore[dict-item]
        tools={"test_tool": SimpleTool()},  # type: ignore[dict-item]
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    ctx_after = getattr(coordinator, "_tool_dispatch_context", None)
    assert ctx_after == {}, (
        "_tool_dispatch_context must be cleared to {} after execute() completes"
    )
