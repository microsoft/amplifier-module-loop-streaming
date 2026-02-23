"""Tests for dict-based tool_call safety (normalize_tool_call fix).

Providers may return tool_calls as plain dicts instead of ToolCall Pydantic
objects. All tool_call field accesses (.id, .name, .arguments) must work
regardless of whether the tool_call is a dict or a ToolCall object.

This is the streaming-orchestrator counterpart to the fix merged in
amplifier-module-loop-basic PR #6.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ToolResult
from amplifier_core.hooks import HookRegistry
from amplifier_core.message_models import ChatResponse, TextBlock

from amplifier_module_loop_streaming import StreamingOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orch(**overrides):
    config = {"max_iterations": 5, "stream_delay": 0, **overrides}
    return StreamingOrchestrator(config)


def _make_tool_result(output="done", success=True):
    """Create a mock tool result with get_serialized_output() and model_dump()."""
    result = MagicMock()
    result.success = success
    result.output = output
    result.error = None
    result.get_serialized_output = lambda: (
        json.dumps(output) if isinstance(output, (dict, list)) else str(output)
    )
    result.model_dump = lambda: {"success": success, "output": output, "error": None}
    return result


class SimpleTool:
    name = "test_tool"
    description = "test"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, args):
        return ToolResult(success=True, output="done")


class MockContext:
    """Minimal context mock."""

    def __init__(self):
        self.messages = []

    async def get_messages_for_request(self, provider=None):
        return [{"role": "user", "content": "test"}]

    async def add_message(self, msg):
        self.messages.append(msg)


# ---------------------------------------------------------------------------
# The dict tool_call that triggers the bug
# ---------------------------------------------------------------------------

DICT_TOOL_CALL = {"id": "tc_dict_1", "name": "test_tool", "arguments": {"key": "value"}}


# ---------------------------------------------------------------------------
# Test 1: _execute_tool_only with a dict tool_call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_only_with_dict_tool_call():
    """_execute_tool_only must handle dict tool_calls without AttributeError.

    Before the fix, `tool_call.name` on a dict raises:
        AttributeError: 'dict' object has no attribute 'name'
    """
    orchestrator = _orch()
    hooks = HookRegistry()

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value=_make_tool_result("tool output"))
    tools = {"test_tool": mock_tool}

    # Pass a plain dict as the tool_call
    tool_call_id, tool_name, content = await orchestrator._execute_tool_only(
        DICT_TOOL_CALL, tools, hooks, "group_1"
    )

    assert tool_call_id == "tc_dict_1"
    assert tool_name == "test_tool"
    assert content is not None


# ---------------------------------------------------------------------------
# Test 2: _execute_tool_with_result with a dict tool_call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_with_result_with_dict_tool_call():
    """_execute_tool_with_result must handle dict tool_calls without AttributeError."""
    orchestrator = _orch()
    hooks = HookRegistry()
    context = MockContext()

    tools = {"test_tool": SimpleTool()}

    result = await orchestrator._execute_tool_with_result(
        DICT_TOOL_CALL, tools, context, hooks
    )

    assert result["success"] is True
    # Verify a tool response message was added to context
    tool_msgs = [m for m in context.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "tc_dict_1"
    assert tool_msgs[0]["name"] == "test_tool"


# ---------------------------------------------------------------------------
# Test 3: Error-handling path in _execute_tool_only with dict tool_calls
#
# _execute_tool_only has a catch-all `except (Exception, asyncio.CancelledError)`
# that accesses tool_call.name and tool_call.id to build error messages and
# emit TOOL_ERROR events. This is the HIGHEST RISK path — it's the same
# pattern that crashed amplifier-module-loop-basic.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_only_error_path_with_dict_tool_call():
    """_execute_tool_only error handler must not crash with dict tool_calls.

    When a tool raises an exception, the except block at ~line 1075 accesses
    tool_call.name and tool_call.id for logging and TOOL_ERROR emission.
    Before the fix, dict tool_calls crash here with AttributeError.
    """
    orchestrator = _orch()
    hooks = HookRegistry()

    # Tool that raises an exception
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(side_effect=RuntimeError("Simulated failure"))
    tools = {"test_tool": mock_tool}

    # Pass a plain dict as the tool_call — should not crash in the except block
    tool_call_id, tool_name, content = await orchestrator._execute_tool_only(
        DICT_TOOL_CALL, tools, hooks, "group_1"
    )

    assert tool_call_id == "tc_dict_1"
    assert tool_name == "test_tool"
    assert "error" in content.lower() or "Simulated failure" in content


# ---------------------------------------------------------------------------
# Test 4: Full execute() flow with dict tool_calls through error path
#
# End-to-end: provider returns dict tool_calls, tools fail, error results
# flow through the system. Exercises the CancelledError handler's tool_calls
# iteration (lines ~620-637) since tool_calls are normalized before that block.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_execute_with_failing_tools_and_dict_tool_calls():
    """Full execute() with dict tool_calls and failing tools must not crash.

    This exercises the entire flow: assistant msg construction (tc.id, tc.name,
    tc.arguments), task creation, error handling, and context message writing.
    """
    call_count = 0

    class _BaseProvider:
        name = "mock"

    class ToolCallProvider(_BaseProvider):
        """Returns dict-based tool_calls on first call, text on subsequent."""

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = MagicMock()
                resp.content = [TextBlock(text="Calling tool")]
                resp.content_blocks = None
                resp.text = "Calling tool"
                resp.usage = None
                resp.metadata = None
                resp.tool_calls = [
                    {"id": "tc_fail_1", "name": "test_tool", "arguments": {"x": 1}},
                    {"id": "tc_fail_2", "name": "test_tool", "arguments": {"x": 2}},
                ]
                return resp
            return ChatResponse(content=[TextBlock(text="Final")])

        def parse_tool_calls(self, response):
            return getattr(response, "tool_calls", None) or []

    class FailingTool:
        """Tool that raises an exception to exercise the error-handling path."""

        name = "test_tool"
        description = "test"
        input_schema = {"type": "object", "properties": {}}

        async def execute(self, args):
            raise RuntimeError("Simulated tool failure")

    orchestrator = _orch(max_iterations=3)
    context = MockContext()
    hooks = HookRegistry()

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": ToolCallProvider()},
        tools={"test_tool": FailingTool()},
        hooks=hooks,
    )

    # Should complete without AttributeError
    assert result is not None
    # Tool error results should have been written to context
    tool_msgs = [m for m in context.messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 2, (
        f"Expected 2 tool result messages but got {len(tool_msgs)}"
    )


# ---------------------------------------------------------------------------
# Test 5: Assistant message construction with dict tool_calls
#
# Lines 546-567 build assistant messages accessing tc.id, tc.name, tc.arguments.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assistant_message_construction_with_dict_tool_calls():
    """Full execute() flow with dict tool_calls must build assistant messages correctly.

    The assistant message construction at lines 546-567 accesses tc.id, tc.name,
    tc.arguments in list comprehensions. Dict tool_calls crash here before the fix.
    """
    call_count = 0

    class _BaseProvider:
        name = "mock"

    class DictToolCallProvider(_BaseProvider):
        """First call returns dict-based tool_calls, second returns text only."""

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = MagicMock()
                resp.content = [TextBlock(text="Let me use the tool")]
                resp.content_blocks = None
                resp.text = "Let me use the tool"
                resp.usage = None
                resp.metadata = None
                resp.tool_calls = [
                    {"id": "tc_msg_1", "name": "test_tool", "arguments": {"q": "hello"}}
                ]
                return resp
            return ChatResponse(content=[TextBlock(text="Done")])

        def parse_tool_calls(self, response):
            return getattr(response, "tool_calls", None) or []

    orchestrator = _orch(max_iterations=5)
    context = MockContext()
    hooks = HookRegistry()

    result = await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": DictToolCallProvider()},
        tools={"test_tool": SimpleTool()},
        hooks=hooks,
    )

    # The execute should complete without AttributeError
    assert result is not None
    # Verify the assistant message was constructed with correct tool_calls
    assistant_msgs = [m for m in context.messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) >= 1
    first_assistant = assistant_msgs[0]
    assert "tool_calls" in first_assistant
    assert first_assistant["tool_calls"][0]["id"] == "tc_msg_1"
    assert first_assistant["tool_calls"][0]["tool"] == "test_tool"


# ---------------------------------------------------------------------------
# Test 6: Dict with "tool" key instead of "name" (provider variation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_only_with_dict_using_tool_key():
    """Some providers use 'tool' instead of 'name' in dict tool_calls.

    The normalizer should handle both key names.
    """
    orchestrator = _orch()
    hooks = HookRegistry()

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value=_make_tool_result("output"))
    tools = {"test_tool": mock_tool}

    # Dict with "tool" key instead of "name"
    dict_tc_alt = {"id": "tc_alt_1", "tool": "test_tool", "arguments": {"a": 1}}

    tool_call_id, tool_name, content = await orchestrator._execute_tool_only(
        dict_tc_alt, tools, hooks, "group_1"
    )

    assert tool_call_id == "tc_alt_1"
    assert tool_name == "test_tool"
