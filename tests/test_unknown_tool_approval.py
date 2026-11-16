"""Unavailable names cannot consume approval; exact known calls still can."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import HookRegistry, HookResult, ToolResult
from amplifier_core.message_models import ToolCall
from amplifier_module_loop_streaming import StreamingOrchestrator


async def dispatch(loop, parallel, call, tools, hooks, coordinator):
    if parallel:
        return await loop._execute_tool_only(call, tools, hooks, "group", coordinator)
    context = SimpleNamespace(add_message=AsyncMock())
    await loop._execute_tool_with_result(call, tools, context, hooks, coordinator)
    message = context.add_message.call_args.args[0]
    assert context.add_message.await_count == 1
    return message["tool_call_id"], message["name"], message["content"]


def coordinator():
    value = MagicMock()
    value.get_capability.return_value = None
    value._tool_dispatch_contexts = {}
    value.process_hook_result = AsyncMock(side_effect=lambda result, *_: result)
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("decision", ["allow", "deny"])
@pytest.mark.parametrize("unknown_name", ["safe", "MCP_1_safe"])
async def test_unknown_name_skips_approval_but_exact_correction_keeps_it(
    parallel, decision, unknown_name
):
    loop = StreamingOrchestrator(config={})
    hooks = HookRegistry()
    owner = coordinator()
    seen, approvals = [], []
    tool = SimpleNamespace(
        execute=AsyncMock(return_value=ToolResult(success=True, output="ok"))
    )

    async def policy(event, data):
        seen.append((event, dict(data)))
        if event == "tool:pre":
            return HookResult(action="ask_user", approval_prompt=data["tool_name"])
        return HookResult()

    async def approve(result, event, name):
        if result.action == "ask_user":
            approvals.append((event, name))
            return HookResult(
                action="continue" if decision == "allow" else "deny",
                reason="Fixture decision",
            )
        return result

    owner.process_hook_result.side_effect = approve
    for event in ("tool:pre", "tool:post", "tool:error"):
        hooks.register(event, policy)
    tools = {"mcp_1_safe": tool}
    missing = await dispatch(
        loop, parallel, ToolCall(id="missing", name=unknown_name, arguments={}),
        tools, hooks, owner,
    )
    assert missing == ("missing", unknown_name, f"Error: Tool '{unknown_name}' not found")
    assert approvals == []
    assert loop._tool_calls_this_turn == 0
    tool.execute.assert_not_awaited()
    owner.cancellation.register_tool_start.assert_not_called()
    owner.process_hook_result.assert_not_awaited()
    errors = [data for event, data in seen if event == "tool:error"]
    assert len(errors) == (1 if parallel else 0)
    if parallel:
        # Core may add observational metadata such as the event timestamp.
        assert {key: errors[0][key] for key in (
            "tool_name", "tool_call_id", "error", "parallel_group_id"
        )} == {
            "tool_name": unknown_name, "tool_call_id": "missing",
            "error": {"type": "RuntimeError", "msg": missing[2]},
            "parallel_group_id": "group",
        }

    corrected = await dispatch(
        loop, parallel, ToolCall(id="corrected", name="mcp_1_safe", arguments={}),
        tools, hooks, owner,
    )
    assert corrected[:2] == ("corrected", "mcp_1_safe")
    assert approvals == [("tool:pre", "mcp_1_safe")]
    assert tool.execute.await_count == (1 if decision == "allow" else 0)
    assert loop._tool_calls_this_turn == tool.execute.await_count
    assert [data["tool_call_id"] for event, data in seen if event == "tool:pre"] == ["corrected"]
    assert [data["tool_call_id"] for event, data in seen if event == "tool:post"] == (
        ["corrected"] if decision == "allow" else []
    )
    if decision == "deny":
        assert "denied" in corrected[2].lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("remove", [False, True])
async def test_known_name_keeps_registry_lookup_after_awaited_hook(parallel, remove):
    loop = StreamingOrchestrator(config={})
    hooks = HookRegistry()
    owner = coordinator()
    original = SimpleNamespace(execute=AsyncMock())
    replacement = SimpleNamespace(
        execute=AsyncMock(return_value=ToolResult(success=True, output="replacement"))
    )
    tools = {"known": original}

    async def policy(event, data):
        if remove:
            tools.pop("known")
        else:
            tools["known"] = replacement
        return HookResult()

    hooks.register("tool:pre", policy)
    result = await dispatch(
        loop, parallel, ToolCall(id="known-call", name="known", arguments={}),
        tools, hooks, owner,
    )
    original.execute.assert_not_awaited()
    assert replacement.execute.await_count == (0 if remove else 1)
    assert ("not found" if remove else "replacement") in result[2]
    assert result[:2] == ("known-call", "known")
