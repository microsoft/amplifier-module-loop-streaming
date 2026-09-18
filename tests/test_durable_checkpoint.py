"""The host checkpoints only a complete, settled, canonically ordered batch."""

from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest
from amplifier_core import ContextLengthError, ToolResult
from amplifier_core.models import ProviderInfo

from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockContext as RequestContext,
)
from tests.test_ephemeral_cache_persist_mode import (
    MockCoordinator,
    MockResponse,
    ScriptedHooks,
    ToolCallStub,
)


class MockContext(RequestContext):
    async def get_messages(self):
        return list(self._messages)


class BatchProvider:
    def __init__(self, count=2, *, overflow=False):
        self.count = count
        self.calls = 0
        self.overflow = overflow

    def get_info(self):
        return ProviderInfo(
            id="fixture",
            display_name="Fixture",
            credential_env_vars=[],
            defaults={"model": "fixture"},
        )

    async def complete(self, request, **kwargs):
        self.calls += 1
        if self.calls == 2 and self.overflow:
            raise ContextLengthError("synthetic overflow after completed tools")
        return MockResponse("done")

    def parse_tool_calls(self, response):
        calls = []
        if self.calls == 1:
            for i in range(self.count):
                call = ToolCallStub(f"call-{i}")
                call.arguments = {"index": i}
                calls.append(call)
        return calls


class OrderedTool:
    """Finish the second call first, without clock-dependent sleeps."""

    name = "mock_tool"
    description = "deterministic concurrent tool"
    input_schema: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {"index": {"type": "integer"}},
    }

    def __init__(self, count=2):
        self.count = count
        self.second_finished = asyncio.Event()
        self.completed = []

    async def execute(self, arguments):
        index = arguments["index"]
        if index == 0 and self.count == 2:
            await self.second_finished.wait()
        self.completed.append(index)
        if index == 1:
            self.second_finished.set()
        return ToolResult(success=True, output=f"result-{index}")


def _receipts(context):
    return [
        (m["tool_call_id"], m["content"])
        for m in context._messages
        if m["role"] == "tool"
    ]


async def _execute(provider, context, coordinator, tool, *, max_iterations=3):
    return await StreamingOrchestrator({"max_iterations": max_iterations}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": tool},
        ScriptedHooks({}),
        coordinator,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("max_iterations", [1, 3])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_checkpoint_precedes_next_dispatch_and_budget_finalization(
    count, max_iterations, asynchronous
):
    context, coordinator = MockContext(), MockCoordinator()
    provider, tool = BatchProvider(count, overflow=True), OrderedTool(count)
    checkpoints = []

    def checkpoint():
        assert provider.calls == 1
        assert tool.completed == ([0] if count == 1 else [1, 0])
        checkpoints.append(_receipts(context))

    async def async_checkpoint():
        checkpoint()

    coordinator.register_capability(
        "session.durable_checkpoint", async_checkpoint if asynchronous else checkpoint
    )
    with pytest.raises(ContextLengthError, match="synthetic overflow"):
        await _execute(
            provider, context, coordinator, tool, max_iterations=max_iterations
        )

    assert checkpoints == [[(f"call-{i}", f"result-{i}") for i in range(count)]]
    assert provider.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("max_iterations", [1, 3])
@pytest.mark.parametrize(
    "failure", ["malformed", "false", "exception", "async-false", "async-exception"]
)
async def test_checkpoint_failure_prevents_any_next_provider_dispatch(
    failure, max_iterations
):
    context, coordinator = MockContext(), MockCoordinator()
    provider, tool = BatchProvider(), OrderedTool()

    def checkpoint():
        if "exception" in failure:
            raise OSError("disk unavailable")
        return False

    async def async_checkpoint():
        return checkpoint()

    callback = (
        "invalid"
        if failure == "malformed"
        else (async_checkpoint if failure.startswith("async") else checkpoint)
    )
    coordinator.register_capability("session.durable_checkpoint", callback)
    with pytest.raises(
        RuntimeError, match="Durable session checkpoint failed"
    ) as caught:
        await _execute(
            provider, context, coordinator, tool, max_iterations=max_iterations
        )
    assert caught.value.__cause__ is not None
    assert provider.calls == 1
    assert _receipts(context) == [("call-0", "result-0"), ("call-1", "result-1")]
    assert tool.completed == [1, 0]


@pytest.mark.asyncio
async def test_checkpoint_cancellation_propagates_with_appended_results_intact():
    context, coordinator = MockContext(), MockCoordinator()
    provider, tool = BatchProvider(), OrderedTool()

    async def checkpoint():
        raise asyncio.CancelledError()

    coordinator.register_capability("session.durable_checkpoint", checkpoint)
    with pytest.raises(asyncio.CancelledError):
        await _execute(provider, context, coordinator, tool)
    assert provider.calls == 1
    assert _receipts(context) == [("call-0", "result-0"), ("call-1", "result-1")]


@pytest.mark.asyncio
async def test_absent_checkpoint_preserves_old_host_behavior():
    context, coordinator = MockContext(), MockCoordinator()
    provider, tool = BatchProvider(), OrderedTool()
    assert await _execute(provider, context, coordinator, tool) == "done"
    assert provider.calls == 2
    assert _receipts(context) == [("call-0", "result-0"), ("call-1", "result-1")]


@pytest.mark.asyncio
async def test_no_tool_turn_does_not_checkpoint():
    context, coordinator = MockContext(), MockCoordinator()
    coordinator.register_capability(
        "session.durable_checkpoint", lambda: pytest.fail("no batch to checkpoint")
    )
    assert (
        await _execute(BatchProvider(count=0), context, coordinator, OrderedTool())
        == "done"
    )
