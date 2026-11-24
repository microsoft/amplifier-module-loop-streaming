"""Use the fully assembled uncounted request; settle only the reply that actually completed."""
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ContextLengthError
from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockCoordinator, NRoundToolProvider, OneShotTool, ScriptedHooks, ScriptedHookResult,
)
from tests.test_request_retention import RetainingContext


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["tail", "persist"])
async def test_uncounted_settles_only_the_final_tool_round_after_answer(mode):
    context, coord = RetainingContext(), MockCoordinator()
    await context.add_message({"role": "system", "content": "Keep Unicode 月 in the answer"})
    hooks = ScriptedHooks({"provider:request": ScriptedHookResult(
        action="inject_context", ephemeral=True, context_injection="Include current instructions")})
    prepared, settled = [], []
    async def prepare(*, provider, retain_contents, project_view, request_options):
        rows = await context.get_messages()
        request = project_view(rows)
        prepared.append(request)
        assert request_options == {"extended_thinking": True}
        assert "Keep Unicode 月" in str(request.messages)
        assert sum("Include current instructions" in str(m.content) for m in request.messages) == 1
        assert request.tools[0].name == "mock_tool"
        assert request.tools[0].parameters == OneShotTool.input_schema
        if mode == "persist":
            assert any("Include current instructions" in content for content in retain_contents)
        def settled_after_reply():
            assert context.add_message_calls[-1]["role"] == "assistant"
            assert hooks.emitted[-1][0] == "execution:end"
            settled.append(request)
        return {"dispatch": request, "base_view": rows, "on_settled": settled_after_reply}
    coord.register_capability("context.estimated_request_view", prepare)
    coord.register_capability("context.request_retention", context.retaining_view)
    class Provider(NRoundToolProvider):
        async def complete(self, request, **kwargs):
            assert request is prepared[-1] and not settled
            assert kwargs == {"extended_thinking": True}
            result = await super().complete(request, **kwargs)
            result.metadata = {"provider_response_completed": True}
            result.finish_reason = "stop"
            return result
    provider = Provider(n_tool_rounds=1)
    answer = await StreamingOrchestrator({
        "ephemeral_injection_mode": mode, "extended_thinking": True, "min_delay_between_calls_ms": 0,
    }).execute("What was decided?", context, {"main": provider}, {"mock_tool": OneShotTool()}, hooks, coord)
    assert answer and len(prepared) == len(provider.requests) == 2
    assert settled == [provider.requests[-1]] and context.requirements == []


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["absent", "partial", "length", "stop", "overflow", "fallback"])
async def test_uncounted_partial_stop_fallback_and_overflow_never_schedule(disposition):
    context, coord = RetainingContext(), MockCoordinator()
    calls, projected, settled = [], [], []
    async def prepare(*, provider, retain_contents, project_view, request_options):
        rows = await context.get_messages()
        request = project_view(rows)
        projected.append(request)
        return {"dispatch": None if disposition == "fallback" else request,
                "base_view": rows, "on_settled": lambda: settled.append(True)}
    coord.register_capability("context.estimated_request_view", prepare)
    class Provider(NRoundToolProvider):
        async def complete(self, request, **kwargs):
            calls.append(request)
            if disposition == "overflow":
                raise ContextLengthError("synthetic endpoint overflow")
            response = await super().complete(request, **kwargs)
            response.metadata = {} if disposition == "absent" else {
                "provider_response_completed": disposition != "partial"}
            response.finish_reason = "length" if disposition == "length" else "stop"
            if disposition == "stop":
                coord.cancellation.is_cancelled = True
            return response
    provider = Provider(n_tool_rounds=0)
    run = StreamingOrchestrator({}).execute("one", context, {"main": provider}, {}, ScriptedHooks({}), coord)
    if disposition == "overflow":
        with pytest.raises(ContextLengthError, match="synthetic endpoint overflow"):
            await run
    else:
        await run
    assert len(calls) == len(projected) == 1 and not settled
    assert calls[0] == projected[0]


@pytest.mark.asyncio
async def test_budget_capable_provider_never_enters_estimated_extension():
    coord = MockCoordinator()
    getter = AsyncMock(side_effect=AssertionError("uncounted extension used for counted route"))
    coord.register_capability("context.estimated_request_view", getter)
    class BudgetProvider(NRoundToolProvider):
        def request_budget(self, request, **kwargs):
            return None
    provider = BudgetProvider(n_tool_rounds=0)
    answer = await StreamingOrchestrator({}).execute("one", RetainingContext(),
        {"main": provider}, {}, ScriptedHooks({}), coord)
    assert answer and provider.call_count == 1
    getter.assert_not_called()
