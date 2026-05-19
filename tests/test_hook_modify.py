"""Tests for hook modify action on tool:post events in streaming orchestrator.

Verifies that when a hook returns HookResult(action="modify", data={"result": ...})
on a tool:post event, both _execute_tool_only and _execute_tool_with_result
use the modified data instead of the original result.get_serialized_output().
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from amplifier_core.hooks import HookRegistry
from amplifier_core.models import HookResult


def _make_tool_result(output, success=True):
    """Create a mock tool result with get_serialized_output() and model_dump()."""
    result = MagicMock()
    result.success = success
    result.output = output
    result.error = None

    def get_serialized_output():
        if isinstance(output, (dict, list)):
            return json.dumps(output)
        return str(output)

    result.get_serialized_output = get_serialized_output

    def model_dump():
        return {"success": success, "output": output, "error": None}

    result.model_dump = model_dump
    return result


@pytest.mark.asyncio
async def test_execute_tool_only_modify_replaces_result():
    """_execute_tool_only should use modified data when hook returns modify."""
    with patch.dict("sys.modules", {"amplifier_core.llm_errors": MagicMock()}):
        from amplifier_module_loop_streaming import StreamingOrchestrator

    orchestrator = StreamingOrchestrator({"max_iterations": 5})

    # Tool with original output
    original_output = {"original": True, "big_data": "x" * 1000}
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value=_make_tool_result(original_output))
    tools = {"test_tool": mock_tool}

    # Hook that modifies tool output
    modified_content = {"modified": True, "truncated": True}
    hooks = HookRegistry()

    async def modify_hook(event: str, data: dict) -> HookResult:
        if event == "tool:post":
            return HookResult(action="modify", data={"result": modified_content})
        return HookResult()

    hooks.register("tool:post", modify_hook, priority=50, name="test_modify")

    # Mock tool call
    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.name = "test_tool"
    tool_call.arguments = {"key": "value"}

    # Call _execute_tool_only directly
    tool_call_id, tool_name, content = await orchestrator._execute_tool_only(
        tool_call, tools, hooks, "group_1"
    )

    assert tool_call_id == "tc_1"
    assert tool_name == "test_tool"
    # Content should be the MODIFIED data
    assert content == json.dumps(modified_content), (
        f"Expected modified content {json.dumps(modified_content)}, got {content}"
    )
    assert "big_data" not in content


@pytest.mark.asyncio
async def test_execute_tool_only_no_modify_uses_original():
    """_execute_tool_only should use original when no modify hook."""
    with patch.dict("sys.modules", {"amplifier_core.llm_errors": MagicMock()}):
        from amplifier_module_loop_streaming import StreamingOrchestrator

    orchestrator = StreamingOrchestrator({"max_iterations": 5})

    original_output = {"original": True}
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value=_make_tool_result(original_output))
    tools = {"test_tool": mock_tool}

    hooks = HookRegistry()

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.name = "test_tool"
    tool_call.arguments = {}

    tool_call_id, tool_name, content = await orchestrator._execute_tool_only(
        tool_call, tools, hooks, "group_1"
    )

    assert content == json.dumps(original_output)


@pytest.mark.asyncio
async def test_execute_tool_with_result_modify_replaces_context():
    """_execute_tool_with_result should use modified data in context message."""
    with patch.dict("sys.modules", {"amplifier_core.llm_errors": MagicMock()}):
        from amplifier_module_loop_streaming import StreamingOrchestrator

    orchestrator = StreamingOrchestrator({"max_iterations": 5})

    # Tool with original output
    original_output = {"original": True, "big_data": "x" * 1000}
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value=_make_tool_result(original_output))
    tools = {"test_tool": mock_tool}

    # Hook that modifies tool output
    modified_content = {"modified": True, "truncated": True}
    hooks = HookRegistry()

    async def modify_hook(event: str, data: dict) -> HookResult:
        if event == "tool:post":
            return HookResult(action="modify", data={"result": modified_content})
        return HookResult()

    hooks.register("tool:post", modify_hook, priority=50, name="test_modify")

    # Mock tool call
    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.name = "test_tool"
    tool_call.arguments = {"key": "value"}

    # Mock context to capture add_message calls
    context = AsyncMock()
    messages_added = []

    async def capture_add_message(msg):
        messages_added.append(msg)

    context.add_message = AsyncMock(side_effect=capture_add_message)

    # Call _execute_tool_with_result directly
    await orchestrator._execute_tool_with_result(tool_call, tools, context, hooks)

    # Find the tool result message
    tool_msgs = [msg for msg in messages_added if msg.get("role") == "tool"]
    assert len(tool_msgs) == 1, f"Expected 1 tool message, got {len(tool_msgs)}"

    tool_content = tool_msgs[0]["content"]
    assert tool_content == json.dumps(modified_content), (
        f"Expected modified content {json.dumps(modified_content)}, got {tool_content}"
    )
    assert "big_data" not in tool_content


@pytest.mark.asyncio
async def test_execute_tool_with_result_no_modify_uses_original():
    """_execute_tool_with_result should use original when no modify hook."""
    with patch.dict("sys.modules", {"amplifier_core.llm_errors": MagicMock()}):
        from amplifier_module_loop_streaming import StreamingOrchestrator

    orchestrator = StreamingOrchestrator({"max_iterations": 5})

    original_output = {"original": True}
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value=_make_tool_result(original_output))
    tools = {"test_tool": mock_tool}

    hooks = HookRegistry()

    tool_call = MagicMock()
    tool_call.id = "tc_1"
    tool_call.name = "test_tool"
    tool_call.arguments = {}

    context = AsyncMock()
    messages_added = []

    async def capture_add_message(msg):
        messages_added.append(msg)

    context.add_message = AsyncMock(side_effect=capture_add_message)

    await orchestrator._execute_tool_with_result(tool_call, tools, context, hooks)

    tool_msgs = [msg for msg in messages_added if msg.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == json.dumps(original_output)
