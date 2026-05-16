"""Tests for per-task dispatch context injection in _execute_tool_only().

The orchestrator must set coordinator._tool_dispatch_contexts[current_task]
before calling tool.execute() so that delegate tools can read the calling
tool_call_id and parallel_group_id during their execution.

RED / GREEN notes:
  Run against the *old* code (no _tool_dispatch_contexts set) to see tests 1
  and 3 FAIL, test 2 FAIL (raises AssertionError because entry is never popped
  from a dict that was never populated with the right data).
  Run against the fixed code to see all three tests PASS.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_core import HookRegistry, ToolResult
from amplifier_module_loop_streaming import StreamingOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(tool_call_id: str = "call-abc", name: str = "delegate"):
    """Create a minimal tool_call mock."""
    tc = MagicMock()
    tc.id = tool_call_id
    tc.name = name
    tc.arguments = {"agent": "test-agent", "instruction": "do something"}
    return tc


def _make_coordinator():
    """Create a minimal coordinator mock safe for _execute_tool_only().

    Critically: _tool_dispatch_contexts is initialized as a real empty dict
    (not a MagicMock auto-attribute) so we can detect when the orchestrator
    populates it vs when it never does.
    """
    coord = MagicMock()
    # Use a real dict so dict operations work deterministically
    coord._tool_dispatch_contexts = {}
    # cancellation token methods
    coord.cancellation.register_tool_start = MagicMock()
    coord.cancellation.register_tool_complete = MagicMock()
    # process_hook_result returns an AsyncMock result with action="continue"
    hook_result = MagicMock()
    hook_result.action = "continue"
    coord.process_hook_result = AsyncMock(return_value=hook_result)
    return coord


def _make_orchestrator():
    return StreamingOrchestrator(config={})


# ---------------------------------------------------------------------------
# Test 1: dispatch context is set during tool.execute()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_context_set_during_tool_execute():
    """_execute_tool_only() must set coordinator._tool_dispatch_contexts[task]
    before awaiting tool.execute(), so delegates can read it.

    RED: before fix, the dict is never populated so captured_context is None.
    GREEN: after fix, the dict contains tool_call_id and parallel_group_id.
    """
    orch = _make_orchestrator()
    coordinator = _make_coordinator()
    hooks = HookRegistry()

    captured_context: dict | None = None

    async def capturing_execute(arguments):
        """Capture the dispatch context visible during execute()."""
        nonlocal captured_context
        task = asyncio.current_task()
        # Read from the real dict we pre-initialized on the coordinator
        captured_context = coordinator._tool_dispatch_contexts.get(task)
        return ToolResult(success=True, output="ok")

    mock_tool = MagicMock()
    mock_tool.execute = capturing_execute

    tool_call = _make_tool_call(tool_call_id="call-xyz-123")
    parallel_group_id = "group-aaa"

    await orch._execute_tool_only(
        tool_call=tool_call,
        tools={"delegate": mock_tool},
        hooks=hooks,
        parallel_group_id=parallel_group_id,
        coordinator=coordinator,
    )

    assert captured_context is not None, (
        "coordinator._tool_dispatch_contexts[task] was None during tool.execute(); "
        "the orchestrator must set it before calling execute()."
    )
    assert captured_context["tool_call_id"] == "call-xyz-123", (
        f"Expected tool_call_id='call-xyz-123', got {captured_context.get('tool_call_id')!r}"
    )
    assert captured_context["parallel_group_id"] == "group-aaa", (
        f"Expected parallel_group_id='group-aaa', got {captured_context.get('parallel_group_id')!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: dispatch context is cleared after tool.execute()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_context_cleared_after_tool_execute():
    """After _execute_tool_only() returns, the task's entry must be removed
    from coordinator._tool_dispatch_contexts so leaked tasks don't accumulate.

    This test verifies the finally-block cleanup in the fix.
    If the orchestrator sets the context but forgets to pop it, this fails.
    """
    orch = _make_orchestrator()
    coordinator = _make_coordinator()
    hooks = HookRegistry()

    # Track the task used inside execute() so we can check it afterward
    task_used_inside: list[asyncio.Task] = []

    async def tracking_execute(arguments):
        task_used_inside.append(asyncio.current_task())
        return ToolResult(success=True, output="ok")

    mock_tool = MagicMock()
    mock_tool.execute = tracking_execute

    tool_call = _make_tool_call(tool_call_id="call-cleanup")
    parallel_group_id = "group-cleanup"

    await orch._execute_tool_only(
        tool_call=tool_call,
        tools={"delegate": mock_tool},
        hooks=hooks,
        parallel_group_id=parallel_group_id,
        coordinator=coordinator,
    )

    # The execute() must have run
    assert len(task_used_inside) == 1, "execute() never ran, can't verify cleanup"

    # After return, the task's entry should be gone from the real dict
    assert task_used_inside[0] not in coordinator._tool_dispatch_contexts, (
        f"Task entry was not cleaned up from _tool_dispatch_contexts after execute() "
        f"returned. Remaining entries: {coordinator._tool_dispatch_contexts}"
    )


# ---------------------------------------------------------------------------
# Test 3: dispatch contexts are isolated across concurrent tool executions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_context_isolated_across_concurrent_tools():
    """When two _execute_tool_only() calls run concurrently via asyncio.gather(),
    each tool must see only ITS OWN (tool_call_id, parallel_group_id), not
    the other's values — proving per-task isolation.

    RED: before fix, the shared real dict is never populated so both tools
    capture None → AssertionError on the tool_call_id checks.
    GREEN: after fix, each task key maps to the correct context.
    """
    orch = _make_orchestrator()
    coordinator = _make_coordinator()
    hooks = HookRegistry()

    results: dict[str, dict | None] = {}
    pause = asyncio.Event()

    async def capturing_execute_a(arguments):
        # Pause to allow both tasks to be "inside execute()" simultaneously
        await asyncio.wait_for(asyncio.shield(pause.wait()), timeout=2.0)
        task = asyncio.current_task()
        results["tool_a"] = coordinator._tool_dispatch_contexts.get(task)
        return ToolResult(success=True, output="a")

    async def capturing_execute_b(arguments):
        await asyncio.wait_for(asyncio.shield(pause.wait()), timeout=2.0)
        task = asyncio.current_task()
        results["tool_b"] = coordinator._tool_dispatch_contexts.get(task)
        return ToolResult(success=True, output="b")

    tool_a = MagicMock()
    tool_a.execute = capturing_execute_a

    tool_b = MagicMock()
    tool_b.execute = capturing_execute_b

    tc_a = _make_tool_call(tool_call_id="call-AAA", name="tool_a")
    tc_b = _make_tool_call(tool_call_id="call-BBB", name="tool_b")

    async def run_a():
        return await orch._execute_tool_only(
            tool_call=tc_a,
            tools={"tool_a": tool_a},
            hooks=hooks,
            parallel_group_id="group-AAAA",
            coordinator=coordinator,
        )

    async def run_b():
        return await orch._execute_tool_only(
            tool_call=tc_b,
            tools={"tool_b": tool_b},
            hooks=hooks,
            parallel_group_id="group-BBBB",
            coordinator=coordinator,
        )

    # Run both concurrently, pause them inside execute(), then release
    async def run_concurrent():
        gather_task = asyncio.ensure_future(asyncio.gather(run_a(), run_b()))
        # Give both coroutines time to reach their pause point
        await asyncio.sleep(0.05)
        pause.set()
        await gather_task

    await run_concurrent()

    assert "tool_a" in results, "tool_a never captured its context (did execute() run?)"
    assert "tool_b" in results, "tool_b never captured its context (did execute() run?)"

    ctx_a = results["tool_a"]
    ctx_b = results["tool_b"]

    assert ctx_a is not None, (
        "tool_a saw None context — dispatch context was not set before execute()."
    )
    assert ctx_b is not None, (
        "tool_b saw None context — dispatch context was not set before execute()."
    )

    assert ctx_a["tool_call_id"] == "call-AAA", (
        f"tool_a expected tool_call_id='call-AAA', got {ctx_a.get('tool_call_id')!r}. "
        "Context is leaking between concurrent tasks."
    )
    assert ctx_a["parallel_group_id"] == "group-AAAA", (
        f"tool_a expected parallel_group_id='group-AAAA', got {ctx_a.get('parallel_group_id')!r}"
    )

    assert ctx_b["tool_call_id"] == "call-BBB", (
        f"tool_b expected tool_call_id='call-BBB', got {ctx_b.get('tool_call_id')!r}. "
        "Context is leaking between concurrent tasks."
    )
    assert ctx_b["parallel_group_id"] == "group-BBBB", (
        f"tool_b expected parallel_group_id='group-BBBB', got {ctx_b.get('parallel_group_id')!r}"
    )
