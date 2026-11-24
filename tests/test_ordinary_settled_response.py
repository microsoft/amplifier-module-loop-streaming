"""A confirmed ordinary response offers only background work for that same completed turn."""
from types import SimpleNamespace

import pytest
from amplifier_core import ContextLengthError
from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockCoordinator, NRoundToolProvider, OneShotTool, ScriptedHooks, ScriptedHookResult,
)
from tests.test_request_retention import RetainingContext


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "complete", "refit_wrapper", "tools", "no_observer", "absent_confirmation", "partial", "length", "missing_usage",
    "zero_input", "invalid_output", "overflow", "stop_after_reply", "switch_during_call", "switch_after_reply",
    "steer_after_reply", "model_change", "goal", "end_failure", "counted",
])
async def test_only_same_completed_natural_uncounted_turn_permits_background_work(case):
    events = []
    context, coord = RetainingContext(), MockCoordinator()
    providers = {}
    loop = StreamingOrchestrator({"min_delay_between_calls_ms": 0})
    class Hooks(ScriptedHooks):
        async def emit(self, event, data):
            if event == "execution:end":
                events.append("end")
                if case == "stop_after_reply":
                    coord.cancellation.is_cancelled = True
                elif case == "switch_after_reply":
                    providers["main"] = object()
                elif case == "model_change":
                    raw.model = "changed"
                elif case == "steer_after_reply":
                    loop._steering_queue.steer("new input")
                elif case == "goal":
                    coord.session_state["goal"] = {"condition": "existing goal", "turns_used": 0, "last_reason": None, "cap": 2}
                elif case == "end_failure":
                    raise RuntimeError("incomplete turn")
            return await super().emit(event, data)
    hooks = Hooks({"provider:request": ScriptedHookResult(
        action="inject_context", ephemeral=True, context_injection="This request's reminder")})
    def observe(*, provider, retain_contents, input_tokens, output_tokens, prompt_characters):
        assert context.add_message_calls[-1]["role"] == "assistant"
        assert provider is raw
        assert (input_tokens, output_tokens, prompt_characters) == (131600, 9, len(prompt))
        assert any("This request's reminder" in item for item in retain_contents)
        events.append("observe")
        return lambda: events.append("schedule")
    if case != "no_observer":
        coord.register_capability("context.ordinary_settled_response", observe)
    coord.register_capability("context.request_retention", context.retaining_view)

    class Provider(NRoundToolProvider):
        model = "a"

        def get_info(self):
            return SimpleNamespace(defaults={"model": self.model}, id="main")

        async def complete(self, request, **kwargs):
            events.append("request")
            if case == "overflow":
                raise ContextLengthError("rejected at provider")
            if case == "switch_during_call":
                providers["main"] = object()
            result = await super().complete(request, **kwargs)
            result.metadata = {} if case == "absent_confirmation" else {
                "provider_response_completed": case != "partial"}
            result.finish_reason = "length" if case == "length" else "stop"
            result.usage = None if case == "missing_usage" else SimpleNamespace(
                input_tokens=0 if case == "zero_input" else 131600,
                output_tokens=-1 if case == "invalid_output" else 9,
                model_dump=lambda: {"input_tokens":131600, "output_tokens":9})
            return result
    raw = Provider(n_tool_rounds=1 if case == "tools" else 0)
    if case == "refit_wrapper":
        raw.refit_request = lambda *args, **kwargs: None
    if case == "counted":
        raw.request_budget = lambda *args, **kwargs: None
    providers["main"] = raw
    prompt = "A complete ordinary input 月"
    run = loop.execute(prompt, context, providers, {"mock_tool": OneShotTool()}, hooks, coord)
    if case == "overflow":
        with pytest.raises(ContextLengthError, match="rejected at provider"):
            await run
        assert events == ["request"]
        return
    if case == "end_failure":
        with pytest.raises(RuntimeError, match="incomplete turn"):
            await run
    else:
        assert await run
    if case in ("complete", "refit_wrapper", "tools"):
        assert events == (["request", "request"] if case == "tools" else ["request"]) + [
            "observe", "end", "schedule"]
    else:
        assert "schedule" not in events
    assert len(context.requirements) == (2 if case == "tools" else 1)
