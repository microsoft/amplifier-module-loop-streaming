"""Regression tests for provider-marked sequential native-toolset batches."""

from __future__ import annotations

import asyncio

import pytest

from amplifier_core import ToolResult
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_streaming import StreamingOrchestrator


class _Provider:
    """Return one tool response followed by a normal completion."""

    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self._responses = [
            ChatResponse(content=[TextBlock(text="running tools")], tool_calls=tool_calls),
            ChatResponse(content=[TextBlock(text="done")]),
        ]

    async def complete(self, request, **kwargs):  # noqa: ANN001, ANN201
        return self._responses.pop(0)

    def parse_tool_calls(self, response):  # noqa: ANN001, ANN201
        return response.tool_calls or []


class _Tool:
    description = "test tool"
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, name: str, execute) -> None:  # noqa: ANN001
        self.name = name
        self._execute = execute

    async def execute(self, arguments):  # noqa: ANN001, ANN201
        return await self._execute(arguments)


async def _run_batch(
    tool_calls: list[ToolCall], tools: dict[str, _Tool]
) -> MockContextManager:
    context = MockContextManager()
    await StreamingOrchestrator({"stream_delay": 0}).execute(
        prompt="run tools",
        context=context,
        providers={"default": _Provider(tool_calls)},
        tools=tools,
        hooks=EventRecorder(),
    )
    return context


def _tool_messages(context: MockContextManager) -> list[dict]:
    return [message for message in context.messages if message["role"] == "tool"]


@pytest.mark.asyncio
async def test_marked_toolcalls_use_pydantic_extra_and_execute_in_response_order() -> None:
    """A core ToolCall extra makes the whole response batch ordered."""

    events: list[str] = []

    async def first(arguments):  # noqa: ANN001
        events.append("first:start")
        await asyncio.sleep(0)
        events.append("first:finish")
        return ToolResult(success=True, output="first complete")

    async def second(arguments):  # noqa: ANN001
        # This assertion fails if the loop starts this action before the first
        # native action has settled.
        assert events == ["first:start", "first:finish"]
        events.append("second:start")
        return ToolResult(success=True, output="second complete")

    calls = [
        ToolCall(
            id="call-first",
            name="first",
            arguments={},
            _amplifier_execution_mode="sequential",
        ),
        ToolCall(
            id="call-second",
            name="second",
            arguments={},
            _amplifier_execution_mode="sequential",
        ),
    ]

    assert getattr(calls[0], "_amplifier_execution_mode") == "sequential"
    context = await _run_batch(
        calls,
        {
            "first": _Tool("first", first),
            "second": _Tool("second", second),
        },
    )

    assert events == ["first:start", "first:finish", "second:start"]
    assert [message["tool_call_id"] for message in _tool_messages(context)] == [
        "call-first",
        "call-second",
    ]


@pytest.mark.asyncio
async def test_sequential_failure_marks_failure_and_skips_later_actions() -> None:
    """A failed native action stops the batch and preserves all result pairs."""

    executed: list[str] = []

    async def succeeds(arguments):  # noqa: ANN001
        executed.append("succeeds")
        return ToolResult(success=True, output="ok")

    async def fails(arguments):  # noqa: ANN001
        executed.append("fails")
        return ToolResult(success=False, error={"message": "native action failed"})

    async def must_not_run(arguments):  # noqa: ANN001
        executed.append("must_not_run")
        return ToolResult(success=True, output="unexpected")

    calls = [
        ToolCall(
            id="call-ok",
            name="succeeds",
            arguments={},
            _amplifier_execution_mode="sequential",
        ),
        ToolCall(
            id="call-fail",
            name="fails",
            arguments={},
            _amplifier_execution_mode="sequential",
        ),
        ToolCall(
            id="call-skipped",
            name="must_not_run",
            arguments={},
            _amplifier_execution_mode="sequential",
        ),
    ]
    context = await _run_batch(
        calls,
        {
            "succeeds": _Tool("succeeds", succeeds),
            "fails": _Tool("fails", fails),
            "must_not_run": _Tool("must_not_run", must_not_run),
        },
    )

    messages = _tool_messages(context)
    assert executed == ["succeeds", "fails"]
    assert [message["tool_call_id"] for message in messages] == [
        "call-ok",
        "call-fail",
        "call-skipped",
    ]
    assert "is_error" not in messages[0]
    assert messages[1]["is_error"] is True
    assert messages[2]["is_error"] is True
    assert "native action failed" in messages[1]["content"]
    assert "Skipped because a prior sequential tool call failed" in messages[2]["content"]
    assert '"failed_tool_call_id": "call-fail"' in messages[2]["content"]


@pytest.mark.asyncio
async def test_sequential_cancellation_preserves_settled_actions_and_pairs_rest() -> None:
    """Cancellation keeps completed work and pairs active/unstarted calls."""

    second_started = asyncio.Event()
    third_executed = False

    async def first(arguments):  # noqa: ANN001
        return ToolResult(success=True, output="first complete")

    async def waits_for_cancellation(arguments):  # noqa: ANN001
        second_started.set()
        await asyncio.Event().wait()

    async def must_not_run(arguments):  # noqa: ANN001
        nonlocal third_executed
        third_executed = True
        return ToolResult(success=True, output="unexpected")

    calls = [
        ToolCall(
            id="call-first",
            name="first",
            arguments={},
            _amplifier_execution_mode="sequential",
        ),
        ToolCall(
            id="call-active",
            name="waits",
            arguments={},
            _amplifier_execution_mode="sequential",
        ),
        ToolCall(
            id="call-never-started",
            name="third",
            arguments={},
            _amplifier_execution_mode="sequential",
        ),
    ]
    context = MockContextManager()
    task = asyncio.create_task(
        StreamingOrchestrator({"stream_delay": 0}).execute(
            prompt="run tools",
            context=context,
            providers={"default": _Provider(calls)},
            tools={
                "first": _Tool("first", first),
                "waits": _Tool("waits", waits_for_cancellation),
                "third": _Tool("third", must_not_run),
            },
            hooks=EventRecorder(),
        )
    )
    await asyncio.wait_for(second_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    messages = _tool_messages(context)
    assert third_executed is False
    assert [message["tool_call_id"] for message in messages] == [
        "call-first",
        "call-active",
        "call-never-started",
    ]
    assert messages[0]["content"] == "first complete"
    assert '"cancelled": true' in messages[1]["content"]
    assert '"cancelled": true' in messages[2]["content"]


@pytest.mark.asyncio
async def test_unmarked_batch_retains_concurrent_execution() -> None:
    """The marker is opt-in; ordinary batches still start all tools together."""

    started: set[str] = set()
    both_started = asyncio.Event()

    async def concurrent_tool(name: str, arguments):  # noqa: ANN001
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        return ToolResult(success=True, output=name)

    calls = [
        ToolCall(id="call-a", name="a", arguments={}),
        ToolCall(id="call-b", name="b", arguments={}),
    ]
    context = await asyncio.wait_for(
        _run_batch(
            calls,
            {
                "a": _Tool("a", lambda arguments: concurrent_tool("a", arguments)),
                "b": _Tool("b", lambda arguments: concurrent_tool("b", arguments)),
            },
        ),
        timeout=1,
    )

    assert started == {"a", "b"}
    assert [message["tool_call_id"] for message in _tool_messages(context)] == [
        "call-a",
        "call-b",
    ]