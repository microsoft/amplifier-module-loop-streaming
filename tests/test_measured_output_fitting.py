"""Joint regressions for measured output-reserve fitting.

These run the real StreamingOrchestrator with the real context-simple measured
view.  Only the provider transport/count endpoint and hooks are stubbed: the
assertions therefore cover the complete Context -> Loop contract rather than a
fake measured-view result.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from amplifier_core import ContextLengthError, ToolResult

pytest.importorskip("amplifier_module_context_simple")

from amplifier_module_context_simple import SimpleContextManager
from amplifier_module_loop_streaming import StreamingOrchestrator


class _HookResult:
    action = "continue"
    reason = None
    ephemeral = False
    context_injection = None
    context_injection_role = "system"
    append_to_last_tool_result = False
    data = None


class _Hooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, name: str, payload: dict[str, Any] | None = None):
        self.events.append((name, payload or {}))
        if name == "provider:request":
            result = _HookResult()
            result.action = "inject_context"
            result.ephemeral = True
            result.context_injection = "<system-reminder>ACTIVE-OVERLAY</system-reminder>"
            return result
        return _HookResult()

    def payloads(self, name: str) -> list[dict[str, Any]]:
        return [payload for event, payload in self.events if event == name]


class _Cancellation:
    is_cancelled = False
    is_immediate = False
    state = "running"

    def register_tool_start(self, *_args) -> None:
        pass

    def register_tool_complete(self, *_args) -> None:
        pass

    async def trigger_callbacks(self) -> None:
        pass


class _Coordinator:
    def __init__(self) -> None:
        self.cancellation = _Cancellation()
        self.session_state: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}

    def register_capability(self, name: str, capability: Any) -> None:
        self.capabilities[name] = capability

    def get_capability(self, name: str) -> Any:
        return self.capabilities.get(name)

    async def process_hook_result(self, result, *_args):
        return result


class _Tool:
    name = "preserved_tool"
    description = "records a real tool declaration in the counted request"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, _arguments) -> ToolResult:
        return ToolResult(success=True, output="tool result")


class _MeasuredTransport:
    """Native-count provider whose answer is a deterministic function of cap."""

    def __init__(self, *, fit_at: int | None) -> None:
        self.fit_at = fit_at
        self.counted: list[Any] = []
        self.counted_options: list[dict[str, Any] | None] = []
        self.requests: list[Any] = []
        self.complete_options: list[dict[str, Any]] = []

    def get_info(self):
        return SimpleNamespace(
            capabilities=["request_budget:provider_count"],
            defaults={"context_window": 100_000, "max_output_tokens": 64_000},
        )

    @staticmethod
    def _warning_present(request) -> bool:
        return any(
            "orchestrator-context-degraded" in str(message.content)
            for message in request.messages
        )

    def request_budget(self, request, *, context_estimate: int, request_options=None):
        self.counted.append(request)
        self.counted_options.append(request_options)
        cap = request.max_output_tokens or 64_000
        warning = self._warning_present(request)
        # The synthetic native count describes input, not reserved output.
        # Only the warning increases it; the model's allowance changes by cap.
        estimated = 89_503 if self.fit_at == 1_000 else 45_000
        if warning:
            estimated += 2  # 6,400 output allows 89,504, so the warning matters.
        allowance = 100_000 - cap - 4_096
        if self.fit_at is None:
            allowance = min(allowance, 30_000 - 4_096)
        return {
            "estimated_input_tokens": estimated,
            "input_limit_tokens": allowance,
            "context_token_budget": 30_000,
            "max_output_tokens": cap,
            "measurement": {
                "kind": "provider_count",
                "source": "test.measured-output",
                "input_tokens": estimated,
            },
        }

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        self.complete_options.append(kwargs)
        return SimpleNamespace(text="accepted", content=None, usage=None)

    def parse_tool_calls(self, _response):
        return []


async def _context_with_protected_markers(*, hooks=None) -> tuple[SimpleContextManager, list[int]]:
    """Create a real actual-meter Context with no legal input reductions."""
    context = SimpleContextManager(
        max_tokens=200_000,
        compact_threshold=0.99,
        target_usage=0.50,
        protected_recent=1.0,
        protected_tool_results=1,
        compaction_notice_enabled=False,
        token_meter="actual",
        hooks=hooks,
    )
    factory_calls = [0]

    async def factory() -> str:
        factory_calls[0] += 1
        return "SYSTEM-FIRST-MARKER"

    await context.set_system_prompt_factory(factory)
    await context.add_message({"role": "user", "content": "FIRST-USER-MARKER"})
    await context.add_message(
        {"role": "developer", "content": "DEVELOPER-REQUIRED-MARKER"}
    )
    await context.add_message(
        {
            "role": "assistant",
            "content": "TOOL-PAIR-ASSISTANT-MARKER",
            "tool_calls": [
                {"id": "kept-tool-call", "name": "preserved_tool", "arguments": {}}
            ],
        }
    )
    await context.add_message(
        {
            "role": "tool",
            "name": "preserved_tool",
            "tool_call_id": "kept-tool-call",
            "content": "TOOL-PAIR-RESULT-MARKER",
        }
    )
    await context.add_message(
        {
            "role": "user",
            "content": "<system-reminder>REQUIRED-REMINDER-MARKER</system-reminder>",
            "metadata": {"ephemeral": True, "source": "hook"},
        }
    )
    return context, factory_calls


def _wire_without_cap(request) -> dict[str, Any]:
    """A deep snapshot of every request field except the intended cap change."""
    payload = request.model_dump(mode="python")
    payload.pop("max_output_tokens", None)
    return deepcopy(payload)


async def _run(provider: _MeasuredTransport, *, stream: bool = False):
    context, factory_calls = await _context_with_protected_markers()
    hooks = _Hooks()
    coordinator = _Coordinator()
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    loop = StreamingOrchestrator(
        {
            "extended_thinking": True,
            "ephemeral_injection_mode": "tail",
            "reminder_placement": "tail",
        }
    )
    if stream:
        async def stream_method(request, *, tools):
            provider.requests.append(request)
            yield {"content": "streamed"}

        provider.stream = stream_method  # type: ignore[attr-defined]
    result = await loop.execute(
        "LATEST-USER-MARKER",
        context,
        {"main": provider},
        {"preserved_tool": _Tool()},
        hooks,
        coordinator,
    )
    return result, context, hooks, coordinator, factory_calls


@pytest.mark.asyncio
async def test_measured_fit_dispatches_the_exact_32k_counted_request_without_input_loss() -> None:
    """64k hard-oversize can retain all input by lowering only to the 32k rung."""
    provider = _MeasuredTransport(fit_at=32_000)
    result, context, hooks, _coordinator, factory_calls = await _run(provider)

    assert result == "accepted"
    assert factory_calls == [1]
    assert len(provider.counted) == 2  # 64k initial count, then legal 50% rung.
    assert [request.max_output_tokens for request in provider.counted] == [None, 32_000]
    assert len(provider.requests) == 1
    counted_final = provider.counted[-1]
    assert provider.requests[0] is counted_final
    assert _wire_without_cap(provider.counted[0]) == _wire_without_cap(counted_final)
    assert provider.complete_options == [{"extended_thinking": True}]
    assert provider.counted_options[-1] == {"extended_thinking": True}
    body = "\n".join(str(message.content) for message in provider.requests[0].messages)
    for marker in (
        "SYSTEM-FIRST-MARKER",
        "DEVELOPER-REQUIRED-MARKER",
        "TOOL-PAIR-ASSISTANT-MARKER",
        "TOOL-PAIR-RESULT-MARKER",
        "REQUIRED-REMINDER-MARKER",
        "FIRST-USER-MARKER",
        "ACTIVE-OVERLAY",
        "LATEST-USER-MARKER",
    ):
        assert marker in body
    assert provider.requests[0].tools is not None
    assert context._last_compaction_stats is None  # Output relief alone is non-sticky.
    assert hooks.payloads("orchestrator:context_degradation") == []
    assert len(hooks.payloads("provider:request")) == 1


@pytest.mark.asyncio
async def test_measured_fit_streams_once_with_the_fitted_counted_request() -> None:
    provider = _MeasuredTransport(fit_at=32_000)
    result, _context, _hooks, _coordinator, _factory_calls = await _run(provider, stream=True)

    assert result == "streamed"
    assert len(provider.requests) == 1
    assert provider.requests[0] is provider.counted[-1]
    assert provider.requests[0].max_output_tokens == 32_000
    assert provider.complete_options == []


@pytest.mark.asyncio
async def test_measured_fit_fails_after_exactly_six_legal_rungs_without_sdk_dispatch() -> None:
    """An immutable protected input gets one initial count plus all six rungs."""
    provider = _MeasuredTransport(fit_at=None)
    context, _factory_calls = await _context_with_protected_markers()
    hooks = _Hooks()
    coordinator = _Coordinator()
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    loop = StreamingOrchestrator(
        {"ephemeral_injection_mode": "tail", "reminder_placement": "tail"}
    )
    with pytest.raises(ContextLengthError, match="cannot fit protected content"):
        await loop.execute(
            "LATEST-USER-MARKER", context, {"main": provider}, {}, hooks, coordinator
        )

    assert len(provider.counted) == 7
    assert [request.max_output_tokens for request in provider.counted] == [
        None,
        32_000,
        25_600,
        19_200,
        12_800,
        6_400,
        1_000,
    ]
    assert provider.requests == []
    assert context._last_compaction_stats is None


@pytest.mark.asyncio
async def test_severe_measured_fit_counts_one_view_warning_and_emits_one_degradation_event() -> None:
    provider = _MeasuredTransport(fit_at=1_000)
    result, context, hooks, _coordinator, _factory_calls = await _run(provider)

    assert result == "accepted"
    assert len(provider.counted) == 7
    severe = [
        request for request in provider.counted
        if request.max_output_tokens is not None and request.max_output_tokens < 10_000
    ]
    assert [request.max_output_tokens for request in severe] == [6_400, 1_000]
    assert all(_MeasuredTransport._warning_present(request) for request in severe)
    assert provider.requests[0].max_output_tokens == 1_000
    assert sum(
        "orchestrator-context-degraded" in str(message.content)
        for message in provider.requests[0].messages
    ) == 1
    assert hooks.payloads("orchestrator:context_degradation") == [
        {"mode": "reduced_output", "max_output_tokens": 1_000}
    ]
    canonical = await context.get_messages()
    assert not any(
        "orchestrator-context-degraded" in str(message.get("content"))
        for message in canonical
    )


@pytest.mark.asyncio
async def test_measured_finalization_recounts_to_low_cap_keeps_tools_and_disables_new_calls() -> None:
    class ToolCall:
        id = "only-tool"
        name = "preserved_tool"
        arguments: dict[str, Any] = {}

    class FinalizingTransport(_MeasuredTransport):
        def request_budget(self, request, *, context_estimate: int, request_options=None):
            # The conversational request fits at 32k, but the complete tool
            # transcript plus finalization overlay needs the 1k floor.
            old_fit = self.fit_at
            self.fit_at = 1_000 if request.tool_choice == "none" else 32_000
            try:
                return super().request_budget(
                    request,
                    context_estimate=context_estimate,
                    request_options=request_options,
                )
            finally:
                self.fit_at = old_fit

        async def complete(self, request, **kwargs):
            self.requests.append(request)
            self.complete_options.append(kwargs)
            if len(self.requests) == 1:
                return SimpleNamespace(
                    text="calling tool",
                    content=None,
                    usage=None,
                    _calls=[ToolCall()],
                )
            return SimpleNamespace(text="final answer", content=None, usage=None, _calls=[])

        def parse_tool_calls(self, response):
            return response._calls

    provider = FinalizingTransport(fit_at=32_000)
    context, _factory_calls = await _context_with_protected_markers()
    hooks, coordinator = _Hooks(), _Coordinator()
    coordinator.register_capability("context.measured_request_view", context.get_measured_request_view)
    loop = StreamingOrchestrator(
        {
            "max_iterations": 1,
            "extended_thinking": True,
            "ephemeral_injection_mode": "tail",
            "reminder_placement": "tail",
        }
    )

    assert await loop.execute(
        "LATEST-USER-MARKER",
        context,
        {"main": provider},
        {"preserved_tool": _Tool()},
        hooks,
        coordinator,
    ) == "final answer"

    assert len(provider.requests) == 2
    final = provider.requests[-1]
    assert final.max_output_tokens == 1_000
    assert final.tool_choice == "none"
    assert final.tools is not None
    assert provider.complete_options == [{"extended_thinking": True}] * 2
    assert [request.max_output_tokens for request in provider.counted[:2]] == [None, 32_000]
    # Finalization may first apply legal history reductions at the original cap.
    final_counts = [request for request in provider.counted if request.tool_choice == "none"]
    assert sum(request.max_output_tokens is None for request in final_counts) == 2
    assert [request.max_output_tokens for request in final_counts[-6:]] == [
        32_000,
        25_600,
        19_200,
        12_800,
        6_400,
        1_000,
    ]
    assert final is final_counts[-1]
    assert all(request.tools == final.tools for request in final_counts)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "absent", "malformed"])
async def test_measured_recount_loss_fails_closed_and_leaves_no_sticky_context(failure: str) -> None:
    class RecountFailureTransport(_MeasuredTransport):
        def request_budget(self, request, *, context_estimate: int, request_options=None):
            if request.max_output_tokens is not None:
                self.counted.append(request)
                self.counted_options.append(request_options)
                if failure == "none":
                    return None
                if failure == "absent":
                    return {
                        "estimated_input_tokens": 100_000,
                        "input_limit_tokens": 90_000,
                        "context_token_budget": 30_000,
                        "max_output_tokens": request.max_output_tokens,
                    }
                return {
                    "estimated_input_tokens": 100_000,
                    "input_limit_tokens": 90_000,
                    "context_token_budget": 30_000,
                    "max_output_tokens": 0,
                    "measurement": {"kind": "provider_count", "source": "bad", "input_tokens": 45_000},
                }
            return super().request_budget(
                request, context_estimate=context_estimate, request_options=request_options
            )

    provider = RecountFailureTransport(fit_at=None)
    context, _factory_calls = await _context_with_protected_markers()
    hooks = _Hooks()
    # Build explicitly here so the failed call still lets us inspect the real Context.
    coordinator = _Coordinator()
    coordinator.register_capability("context.measured_request_view", context.get_measured_request_view)
    loop = StreamingOrchestrator({"ephemeral_injection_mode": "tail", "reminder_placement": "tail"})
    with pytest.raises(ContextLengthError):
        await loop.execute("CURRENT", context, {"main": provider}, {}, hooks, coordinator)

    assert provider.requests == []
    assert context._last_compaction_stats is None


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["counter", "budget_event", "delay"])
async def test_measured_output_fit_cancellation_rolls_back_before_dispatch(phase) -> None:
    context, _ = await _context_with_protected_markers()
    context.protected_recent = 0
    await context.add_message({"role": "assistant", "content": "discardable old detail"})
    coordinator = _Coordinator()
    coordinator.register_capability("context.measured_request_view", context.get_measured_request_view)

    class CancellingTransport(_MeasuredTransport):
        async def request_budget(self, request, **kwargs):
            result = super().request_budget(request, **kwargs)
            if phase == "counter" and request.max_output_tokens is not None:
                raise asyncio.CancelledError()
            return result

    class CancellingHooks(_Hooks):
        async def emit(self, event, payload=None):
            result = await super().emit(event, payload)
            if phase == "budget_event" and event == "orchestrator:provider_budget":
                coordinator.cancellation.is_cancelled = True
            return result

    provider = CancellingTransport(fit_at=32_000)
    loop = StreamingOrchestrator({})

    async def cancelling_delay(*_args):
        raise asyncio.CancelledError()

    if phase == "delay":
        loop._apply_rate_limit_delay = cancelling_delay
    call = loop.execute("CURRENT", context, {"main": provider}, {}, CancellingHooks(), coordinator)
    if phase == "budget_event":
        await call
    else:
        with pytest.raises(asyncio.CancelledError):
            await call
    assert provider.requests == []
    assert len(provider.counted) >= 3  # A reduction rung was staged before the fit.
    assert not context._removed_seqs
    assert not context._truncated_seqs
    assert context._last_compaction_stats is None


@pytest.mark.asyncio
async def test_measured_output_fit_commits_staged_compaction_before_dispatch() -> None:
    hooks, coordinator = _Hooks(), _Coordinator()
    context, _ = await _context_with_protected_markers(hooks=hooks)
    context.protected_recent = 0
    await context.add_message({"role": "assistant", "content": "discardable old detail"})
    coordinator.register_capability("context.measured_request_view", context.get_measured_request_view)

    class CommitObservingProvider(_MeasuredTransport):
        async def complete(self, request, **kwargs):
            assert context._removed_seqs
            assert context._last_compaction_stats["outcome"] == "reduced_output"
            assert len(hooks.payloads("context:compaction")) == 1
            return await super().complete(request, **kwargs)

    provider = CommitObservingProvider(fit_at=32_000)
    await StreamingOrchestrator({}).execute("CURRENT", context, {"main": provider}, {}, hooks, coordinator)
    assert len(provider.counted) >= 3
    assert provider.requests == [provider.counted[-1]]
    assert context._last_compaction_stats["count_calls"] == len(provider.counted)
    assert any(m.get("content") == "discardable old detail" for m in await context.get_messages())


@pytest.mark.asyncio
@pytest.mark.parametrize("finalization", [False, True])
@pytest.mark.parametrize("when", ["before_fit", "during_fit"])
async def test_cooperative_cancel_during_output_fit_uses_normal_cancel_lifecycle(
    finalization, when,
) -> None:
    from types import SimpleNamespace

    context, _ = await _context_with_protected_markers()
    context.protected_recent = 0
    await context.add_message({"role": "assistant", "content": "discardable old detail"})
    coordinator = _Coordinator()
    coordinator.register_capability("context.measured_request_view", context.get_measured_request_view)
    hooks = _Hooks()

    class CancellingProvider(_MeasuredTransport):
        def request_budget(self, request, **kwargs):
            # For finalization, complete a normal tool turn before the failure.
            if finalization and request.tool_choice != "none":
                self.counted.append(request)
                self.counted_options.append(kwargs.get("request_options"))
                return {
                    "estimated_input_tokens": 1, "input_limit_tokens": 100_000,
                    "context_token_budget": 1, "max_output_tokens": 64_000,
                    "measurement": {"kind": "provider_count", "source": "test", "input_tokens": 1},
                }
            result = super().request_budget(request, **kwargs)
            if when == "during_fit" and request.max_output_tokens is not None:
                coordinator.cancellation.is_cancelled = True
            elif when == "before_fit" and context._removed_seqs:
                coordinator.cancellation.is_cancelled = True
            return result

        async def complete(self, request, **kwargs):
            self.requests.append(request)
            return SimpleNamespace(text="", content=None, usage=None)

        def parse_tool_calls(self, response):
            return [SimpleNamespace(id="new-tool", name="preserved_tool", arguments={})]

    provider = CancellingProvider(fit_at=None)
    loop = StreamingOrchestrator({"max_iterations": 1})
    await loop.execute(
        "CURRENT", context, {"main": provider}, {"preserved_tool": _Tool()}, hooks, coordinator
    )
    assert len(provider.requests) == (1 if finalization else 0)
    assert len(hooks.payloads("cancel:requested")) == 1
    assert len(hooks.payloads("cancel:completed")) == 1
    assert hooks.payloads("orchestrator:complete")[-1]["status"] == "cancelled"
    assert not context._removed_seqs
    assert context._last_compaction_stats is None
    caps = [r.max_output_tokens for r in provider.counted if r.max_output_tokens is not None]
    assert caps == ([32_000] if when == "during_fit" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("original_cap", [None, 1_000])
async def test_measured_fit_requires_a_reducible_reported_cap(original_cap) -> None:
    class NoLadderTransport(_MeasuredTransport):
        def request_budget(self, request, **kwargs):
            result = super().request_budget(request, **kwargs)
            if original_cap is None:
                result.pop("max_output_tokens")
            else:
                result["max_output_tokens"] = original_cap
            return result

    provider = NoLadderTransport(fit_at=None)
    with pytest.raises(ContextLengthError, match="cannot fit protected content"):
        await _run(provider)
    assert all(request.max_output_tokens is None for request in provider.counted)
    assert provider.requests == []


@pytest.mark.asyncio
async def test_provider_ignoring_reduced_output_cap_is_not_dispatched() -> None:
    class IgnoringCapTransport(_MeasuredTransport):
        def request_budget(self, request, **kwargs):
            result = super().request_budget(request, **kwargs)
            result["max_output_tokens"] = 64_000
            return result

    provider = IgnoringCapTransport(fit_at=32_000)
    with pytest.raises(ContextLengthError, match="did not honor"):
        await _run(provider)
    assert provider.requests == []


@pytest.mark.asyncio
async def test_old_measured_getter_is_called_without_the_new_keyword() -> None:
    context, _ = await _context_with_protected_markers()
    coordinator = _Coordinator()
    calls = []

    async def old_getter(*, provider, retain_contents, count_view):
        calls.append(True)
        return await context.get_measured_request_view(
            provider=provider, retain_contents=retain_contents, count_view=count_view
        )

    coordinator.register_capability("context.measured_request_view", old_getter)
    provider = _MeasuredTransport(fit_at=32_000)
    with pytest.raises(ContextLengthError, match="cannot fit protected content"):
        await StreamingOrchestrator({}).execute("CURRENT", context, {"main": provider}, {}, _Hooks(), coordinator)
    assert calls == [True]
    assert all(request.max_output_tokens is None for request in provider.counted)
    assert provider.requests == []


@pytest.mark.asyncio
async def test_negotiated_callback_typeerror_propagates_without_legacy_retry() -> None:
    context, _ = await _context_with_protected_markers()
    coordinator = _Coordinator()
    calls = []
    failure = TypeError("callback implementation failed")

    async def broken_getter(*, provider, retain_contents, count_view, fit_output=None):
        calls.append(fit_output)
        raise failure

    coordinator.register_capability("context.measured_request_view", broken_getter)
    provider = _MeasuredTransport(fit_at=32_000)
    with pytest.raises(TypeError) as raised:
        await StreamingOrchestrator({}).execute("CURRENT", context, {"main": provider}, {}, _Hooks(), coordinator)
    assert raised.value is failure
    assert len(calls) == 1 and callable(calls[0])
    assert provider.requests == []


@pytest.mark.asyncio
async def test_measured_protected_floor_error_finalizes_goal_without_generation() -> None:
    context, _ = await _context_with_protected_markers()
    coordinator = _Coordinator()
    coordinator.register_capability("context.measured_request_view", context.get_measured_request_view)
    coordinator.session_state["goal"] = {"condition": "finish", "turns_used": 0, "cap": None}
    hooks = _Hooks()
    provider = _MeasuredTransport(fit_at=None)
    loop = StreamingOrchestrator({})
    with pytest.raises(ContextLengthError):
        await loop.execute("CURRENT", context, {"main": provider}, {}, hooks, coordinator)
    assert provider.requests == []  # Includes evaluators and summaries, not just main calls.
    assert coordinator.session_state["goal"] is None
    assert loop._pending_orchestrator_complete is None
    complete = hooks.payloads("orchestrator:complete")
    assert len(complete) == 1
    assert complete[0]["status"] == "error" and complete[0]["goal_final"] is True
    assert [p["state"] for p in hooks.payloads("orchestrator:goal_progress")] == ["error"]
