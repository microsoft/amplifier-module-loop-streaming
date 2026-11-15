"""Focused generic Loop coverage for bounded provider overflow recovery."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from amplifier_core import ContextLengthError

from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockResponse,
    OneShotTool,
    RequestCapturingProvider,
    ScriptedHookResult,
    ScriptedHooks,
    ToolCallStub,
)
from tests.test_measured_compaction_runtime import (
    _RepairMeasuredContext as OverflowContext,
    _CountTolerantContext,
)
from tests.test_provider_budget_guard import (
    HardFitBudgetContext,
    _retaining_coordinator as _legacy_coordinator,
)


def _retaining_coordinator(context):
    coordinator = _legacy_coordinator(context)
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    return coordinator


def _fit(cap: int = 128) -> dict[str, object]:
    return {
        "estimated_input_tokens": 1,
        "input_limit_tokens": 10,
        "context_token_budget": 0,
        "max_output_tokens": cap,
        "measurement": {
            "kind": "provider_count",
            "source": "test.provider.count",
            "input_tokens": 1,
        },
    }


def _overflow(target: int = 1, cap: int = 128) -> dict[str, int]:
    return {
        "estimated_input_tokens": 100,
        "input_limit_tokens": 10,
        "context_token_budget": target,
        "max_output_tokens": cap,
    }


class RecoveringProvider(RequestCapturingProvider):
    def __init__(self, *, recovery: object = None, fail_call: int = 1) -> None:
        super().__init__()
        self.recovery = _overflow() if recovery is None else recovery
        self.fail_call = fail_call
        self.complete_calls = 0
        self.budget_options: list[object] = []
        self.recovery_options: list[object] = []
        self.counted_requests: list[object] = []

    def get_info(self):
        return SimpleNamespace(capabilities=["request_budget:provider_count"])

    def request_budget(self, request, *, context_estimate: int, request_options=None):
        self.budget_options.append(request_options)
        self.counted_requests.append(request)
        return _fit(cap=request.max_output_tokens or 128)

    def recover_context_overflow(
        self, request, error, *, context_estimate: int, request_options=None
    ):
        self.recovery_options.append(request_options)
        return self.recovery

    async def complete(self, request, **kwargs):
        self.complete_calls += 1
        self.requests.append(request)
        if self.complete_calls == self.fail_call:
            raise ContextLengthError("provider rejected input")
        return MockResponse(text="recovered")


@pytest.mark.asyncio
async def test_normal_overflow_recovers_once_with_named_options_and_preserved_overlay() -> None:
    context = OverflowContext()
    provider = RecoveringProvider()
    hooks = ScriptedHooks({
        "provider:request": ScriptedHookResult(
            action="inject_context", ephemeral=True, context_injection="CURRENT-OVERLAY"
        )
    })

    result = await StreamingOrchestrator(
        {
            "extended_thinking": True,
            "ephemeral_injection_mode": "tail",
            "reminder_placement": "tail",
        }
    ).execute(
        "work",
        context,
        {"main": provider},
        {},
        hooks,
        _retaining_coordinator(context),
    )

    assert result == "recovered"
    assert provider.complete_calls == 2
    assert provider.budget_options == [
        {"extended_thinking": True},
        {"extended_thinking": True},
    ]
    assert provider.recovery_options == [{"extended_thinking": True}]
    assert context.hard_fit_calls == []
    assert context.repair_reasons == ["provider_overflow"]
    assert provider.requests[1] is provider.counted_requests[1]
    assert "CURRENT-OVERLAY" in "\n".join(m.content for m in provider.requests[1].messages)
    assert "CURRENT-OVERLAY" not in str(context._messages)
    assert provider.requests[1].max_output_tokens == 128
    assert [name for name, _ in hooks.emitted].count("provider:request") == 1


@pytest.mark.asyncio
async def test_kwargs_only_budget_and_recovery_keep_legacy_call_shape() -> None:
    class KwargsOnlyProvider(RecoveringProvider):
        def request_budget(self, request, **kwargs):
            self.budget_options.append(dict(kwargs))
            return _fit()

        def recover_context_overflow(self, request, error, **kwargs):
            self.recovery_options.append(dict(kwargs))
            return self.recovery

    context = OverflowContext()
    provider = KwargsOnlyProvider()
    await StreamingOrchestrator({"extended_thinking": True}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert provider.budget_options == [
        {"context_estimate": provider.budget_options[0]["context_estimate"]},
        {"context_estimate": provider.budget_options[1]["context_estimate"]},
    ]
    assert provider.recovery_options == [
        {"context_estimate": provider.recovery_options[0]["context_estimate"]}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feedback",
    [
        None,
        {"estimated_input_tokens": True, "input_limit_tokens": 10, "context_token_budget": 1},
        {"estimated_input_tokens": 10, "input_limit_tokens": 10, "context_token_budget": 1},
        {"estimated_input_tokens": 100, "input_limit_tokens": 0, "context_token_budget": 1},
        _overflow(target=0),
        _overflow(target=999),
        _overflow(cap=129),
    ],
)
async def test_unavailable_invalid_or_unsafe_recovery_feedback_never_retries(feedback) -> None:
    context = OverflowContext()
    provider = RecoveringProvider()
    provider.recovery = feedback
    hooks = ScriptedHooks({})

    with pytest.raises(ContextLengthError, match="provider rejected input"):
        await StreamingOrchestrator({}).execute(
            "work",
            context,
            {"main": provider},
            {},
            hooks,
            _retaining_coordinator(context),
        )

    assert provider.complete_calls == 1
    assert context.hard_fit_calls == []
    assert context.repair_reasons == []
    assert [
        data["result"]
        for name, data in hooks.emitted
        if name == "orchestrator:provider_overflow_recovery"
    ] == ["unavailable" if feedback is None else "invalid"]


@pytest.mark.asyncio
async def test_recovery_honors_a_lower_provider_output_cap() -> None:
    context = OverflowContext()
    provider = RecoveringProvider(recovery=_overflow(cap=64))

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert provider.requests[-1].max_output_tokens == 64


@pytest.mark.asyncio
async def test_cancellation_during_recovery_never_sends_a_retry() -> None:
    context = OverflowContext()
    coordinator = _retaining_coordinator(context)

    class CancellingProvider(RecoveringProvider):
        def recover_context_overflow(self, request, error, *, context_estimate: int, request_options=None):
            coordinator.cancellation.is_cancelled = True
            return _overflow()

    provider = CancellingProvider()
    with pytest.raises(ContextLengthError, match="provider rejected input"):
        await StreamingOrchestrator({}).execute(
            "work", context, {"main": provider}, {}, ScriptedHooks({}), coordinator
        )

    assert provider.complete_calls == 1


@pytest.mark.asyncio
async def test_recovery_applies_rate_limit_delay_before_retry(monkeypatch) -> None:
    context = OverflowContext()
    provider = RecoveringProvider()
    rate_limit_calls: list[tuple[int, bool]] = []

    async def record_rate_limit_delay(self, _hooks, iteration: int) -> None:
        rate_limit_calls.append((iteration, self._last_provider_call_end is not None))

    monkeypatch.setattr(
        StreamingOrchestrator, "_apply_rate_limit_delay", record_rate_limit_delay
    )

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert rate_limit_calls == [(1, False), (1, True)]


@pytest.mark.asyncio
async def test_cancellation_during_recovery_rate_limit_never_sends_a_retry(
    monkeypatch,
) -> None:
    context = OverflowContext()
    coordinator = _retaining_coordinator(context)
    provider = RecoveringProvider()

    async def cancel_recovery_retry(self, _hooks, _iteration: int) -> None:
        if self._last_provider_call_end is not None:
            coordinator.cancellation.is_cancelled = True

    monkeypatch.setattr(
        StreamingOrchestrator, "_apply_rate_limit_delay", cancel_recovery_retry
    )

    with pytest.raises(ContextLengthError, match="provider rejected input"):
        await StreamingOrchestrator({}).execute(
            "work",
            context,
            {"main": provider},
            {},
            ScriptedHooks({}),
            coordinator,
        )

    assert provider.complete_calls == 1


@pytest.mark.asyncio
async def test_unrecoverable_normal_overflow_keeps_provider_error_event() -> None:
    class NoRecoveryProvider(RequestCapturingProvider):
        async def complete(self, request, **kwargs):
            self.requests.append(request)
            raise ContextLengthError("unrecoverable")

    context = OverflowContext()
    hooks = ScriptedHooks({})
    provider = NoRecoveryProvider()
    with pytest.raises(ContextLengthError, match="unrecoverable"):
        await StreamingOrchestrator({}).execute(
            "work", context, {"main": provider}, {}, hooks, _retaining_coordinator(context)
        )

    assert [name for name, _ in hooks.emitted].count("provider:error") == 1


class RecoveringStreamProvider(RecoveringProvider):
    def __init__(self, *, after_chunk: bool = False) -> None:
        super().__init__()
        self.after_chunk = after_chunk
        self.stream_calls = 0

    async def stream(self, request, *, tools):
        self.stream_calls += 1
        self.requests.append(request)
        if self.stream_calls == 1:
            if self.after_chunk:
                yield {"block_type": "thinking", "content": "hidden"}
            raise ContextLengthError("stream rejected input")
        yield {"content": "stream recovered"}


@pytest.mark.asyncio
async def test_stream_recovers_only_before_its_first_provider_chunk() -> None:
    context = OverflowContext()
    provider = RecoveringStreamProvider()

    result = await StreamingOrchestrator({"extended_thinking": True}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert result == "stream recovered"
    assert provider.stream_calls == 2
    assert provider.budget_options == [{}, {}]
    assert provider.recovery_options == [{}]


@pytest.mark.asyncio
async def test_stream_without_recovery_uses_the_direct_stream_path(monkeypatch) -> None:
    class DirectStreamProvider(RequestCapturingProvider):
        async def stream(self, request, *, tools):
            self.requests.append(request)
            yield {"content": "direct"}

    orchestrator = StreamingOrchestrator({})
    monkeypatch.setattr(
        orchestrator,
        "_stream_with_overflow_recovery",
        lambda *_args, **_kwargs: pytest.fail("unavailable recovery wrapped a stream"),
    )
    context = OverflowContext()

    result = await orchestrator.execute(
        "work",
        context,
        {"main": DirectStreamProvider()},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert result == "direct"


@pytest.mark.asyncio
async def test_closing_a_recovery_stream_closes_its_provider_iterator() -> None:
    from amplifier_core.message_models import ChatRequest, Message

    class ClosableStream:
        def __init__(self) -> None:
            self.closed = False
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return {"content": "first"}

        async def aclose(self) -> None:
            self.closed = True

    class ClosableStreamProvider:
        def __init__(self) -> None:
            self.iterator = ClosableStream()

        def stream(self, _request, *, tools):
            return self.iterator

    orchestrator = StreamingOrchestrator({})
    provider = ClosableStreamProvider()
    response_stream = orchestrator._stream_with_overflow_recovery(
        provider,
        ChatRequest(messages=[Message(role="user", content="work")]),
        OverflowContext(),
        {},
        ScriptedHooks({}),
        recover_overflow=None,
    )

    assert await anext(response_stream) == "first"
    assert not provider.iterator.closed
    await response_stream.aclose()
    assert provider.iterator.closed


@pytest.mark.asyncio
async def test_stream_overflow_after_nontext_chunk_does_not_recover() -> None:
    context = OverflowContext()
    provider = RecoveringStreamProvider(after_chunk=True)

    with pytest.raises(ContextLengthError, match="stream rejected input"):
        await StreamingOrchestrator({}).execute(
            "work",
            context,
            {"main": provider},
            {},
            ScriptedHooks({}),
            _retaining_coordinator(context),
        )

    assert provider.stream_calls == 1
    assert provider.recovery_options == []


@pytest.mark.asyncio
async def test_finalization_overflow_recovers_once_and_keeps_tool_choice_none() -> None:
    class FinalizationProvider(RecoveringProvider):
        def parse_tool_calls(self, response):
            return [ToolCallStub()] if self.complete_calls == 1 else []

        def request_budget(self, request, *, context_estimate: int, request_options=None):
            self.budget_options.append(request_options)
            self.counted_requests.append(request)
            return _fit(cap=request.max_output_tokens or 128)

    context = OverflowContext()
    provider = FinalizationProvider(recovery=_overflow(cap=64), fail_call=2)
    tool_calls = []

    class CountingTool(OneShotTool):
        async def execute(self, arguments):
            tool_calls.append(arguments)
            return await super().execute(arguments)

    hooks = ScriptedHooks({})
    await StreamingOrchestrator({"max_iterations": 1}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": CountingTool()},
        hooks,
        _retaining_coordinator(context),
    )

    assert provider.complete_calls == 3
    assert provider.requests[-2].tool_choice == "none"
    assert provider.requests[-1].tool_choice == "none"
    assert provider.requests[-1].max_output_tokens == 64
    assert provider.requests[-1] is provider.counted_requests[-1]
    assert provider.requests[-1].tools == provider.requests[-2].tools
    assert provider.requests[-1].messages[-1] == provider.requests[-2].messages[-1]
    assert len(tool_calls) == 1
    assert [name for name, _ in hooks.emitted].count("provider:request") == 2


@pytest.mark.asyncio
async def test_finalization_invalid_recovery_feedback_closes_tool_turn() -> None:
    class FinalizationProvider(RecoveringProvider):
        def parse_tool_calls(self, response):
            return [ToolCallStub()] if self.complete_calls == 1 else []

    context = OverflowContext()
    provider = FinalizationProvider(recovery=None, fail_call=2)
    # Explicitly make recovery unavailable rather than treating None as the
    # constructor's valid default.
    provider.recovery = None
    with pytest.raises(ContextLengthError, match="provider rejected input"):
        await StreamingOrchestrator({"max_iterations": 1}).execute(
            "work",
            context,
            {"main": provider},
            {"mock_tool": OneShotTool()},
            ScriptedHooks({}),
            _retaining_coordinator(context),
        )

    assert provider.complete_calls == 2
    assert context._messages[-1] == {
        "role": "assistant",
        "content": "The final response could not be generated because the context is too long.",
    }


async def _execute_recovery(context, provider, *, coordinator=None, hooks=None):
    return await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        hooks or ScriptedHooks({}),
        coordinator or _retaining_coordinator(context),
    )


@pytest.mark.asyncio
async def test_repair_freezes_selected_model_tools_options_and_current_overlays(monkeypatch) -> None:
    import amplifier_module_loop_streaming as loop_module

    spec_calls = []
    original_spec = loop_module._build_tool_spec

    def record_spec(tool):
        spec_calls.append(tool.description)
        return original_spec(tool)

    monkeypatch.setattr(loop_module, "_build_tool_spec", record_spec)
    tool = OneShotTool()
    failed_options = []

    class SelectedProvider(RecoveringProvider):
        def request_budget(self, request, *, context_estimate, request_options=None):
            if not self.complete_calls:
                # A request-scoped provider wrapper's selected model.
                request.model = "selected-model"
            else:
                assert request.model == "selected-model"
                assert request.reasoning_effort == "high"
                assert request_options == failed_options[0] == {"extended_thinking": True}
            return super().request_budget(
                request, context_estimate=context_estimate, request_options=request_options
            )

        async def complete(self, request, **kwargs):
            if not self.complete_calls:
                failed_options.append(deepcopy(kwargs))
            return await super().complete(request, **kwargs)

    class ChangingToolContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            if reason:
                tool.description = "must not rebuild this declaration"
            return await super().get_measured_request_view(
                provider=provider, retain_contents=retain_contents,
                count_view=count_view, reason=reason,
            )

    context = ChangingToolContext()
    provider = SelectedProvider()
    hooks = ScriptedHooks({
        "prompt:submit": ScriptedHookResult(
            action="inject_context", ephemeral=True, context_injection="SUBMIT-OVERLAY"
        ),
        "provider:request": ScriptedHookResult(
            action="inject_context", ephemeral=True, context_injection="CURRENT-OVERLAY"
        ),
    })
    result = await StreamingOrchestrator({
        "reasoning_effort": "high", "extended_thinking": True,
        "ephemeral_injection_mode": "tail", "reminder_placement": "tail",
    }).execute(
        "work", context, {"main": provider}, {"mock_tool": tool}, hooks,
        _retaining_coordinator(context),
    )
    assert result == "recovered"
    assert spec_calls == ["test tool"]
    assert provider.requests[-1] is provider.counted_requests[-1]
    assert provider.requests[-1].tools == provider.requests[0].tools
    content = "\n".join(message.content for message in provider.requests[-1].messages)
    assert content.count("CURRENT-OVERLAY") == 1
    assert content.count("SUBMIT-OVERLAY") == 1
    assert [name for name, _ in hooks.emitted].count("provider:request") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget",
    [
        None,
        {key: value for key, value in _fit().items() if key != "measurement"},
        {**_fit(), "measurement": {"kind": "estimate", "source": "test", "input_tokens": 1}},
        {**_fit(), "measurement": {"kind": "provider_count", "source": "", "input_tokens": 1}},
        {**_fit(), "measurement": {"kind": "provider_count", "source": "test", "input_tokens": True}},
        {**_fit(), "measurement": {"kind": "provider_count", "source": "test", "input_tokens": -1}},
        {**_fit(), "estimated_input_tokens": True},
        {**_fit(), "max_output_tokens": 0},
        {**_fit(), "estimated_input_tokens": 11},  # count fits, guarded envelope does not
        {
            **_fit(), "estimated_input_tokens": 101, "input_limit_tokens": 200,
            "measurement": {"kind": "provider_count", "source": "test", "input_tokens": 100},
        },  # fitting but no reduction against observed rejection
        {**_fit(), "max_output_tokens": 129},
    ],
)
async def test_repair_requires_a_fitting_reduced_provider_count(budget) -> None:
    class PostRejectBudgetProvider(RecoveringProvider):
        def request_budget(self, request, *, context_estimate, request_options=None):
            self.budget_options.append(request_options)
            return _fit() if not self.complete_calls else deepcopy(budget)

    context = OverflowContext()
    provider = PostRejectBudgetProvider()
    with pytest.raises(ContextLengthError):
        await _execute_recovery(context, provider)
    assert provider.complete_calls == 1
    assert context.transactions[-1].rolled_back == 1
    assert context.transactions[-1].committed == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "estimated,count,fresh_limit,fits",
    [
        (16, 15, 20, False),  # native count fits fresh allowance, not server's 10
        (11, 9, 20, False),  # count fits both, safety-reserved estimate does not
        (10, 9, 20, True),  # exact server-limit fit
        (8, 7, 8, True),  # fresh count can narrow the server's allowance
        (9, 7, 8, False),  # narrower fresh allowance remains binding
    ],
)
async def test_repair_fits_both_server_and_fresh_counted_allowances(
    estimated, count, fresh_limit, fits
) -> None:
    class FreshAllowanceProvider(RecoveringProvider):
        def request_budget(self, request, *, context_estimate, request_options=None):
            result = super().request_budget(
                request, context_estimate=context_estimate, request_options=request_options
            )
            if self.complete_calls:
                result.update(
                    estimated_input_tokens=estimated,
                    input_limit_tokens=fresh_limit,
                    measurement={
                        "kind": "provider_count",
                        "source": "test.provider.count",
                        "input_tokens": count,
                    },
                )
            return result

    context = OverflowContext()
    provider = FreshAllowanceProvider()
    if fits:
        assert await _execute_recovery(context, provider) == "recovered"
        assert provider.complete_calls == 2
        assert provider.requests[-1] is provider.counted_requests[-1]
        assert context.transactions[-1].committed == 1
        assert context.transactions[-1].rolled_back == 0
    else:
        with pytest.raises(ContextLengthError, match="input budget"):
            await _execute_recovery(context, provider)
        assert provider.complete_calls == 1
        assert context.transactions[-1].committed == 0
        assert context.transactions[-1].rolled_back == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pressure", [None, {key: value for key, value in _fit().items() if key != "measurement"}])
async def test_unavailable_opaque_pressure_can_be_compacted_before_final_provider_count(pressure) -> None:
    pressure_attempts = []

    class TwoStageContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            if reason is None:
                return await _CountTolerantContext.get_measured_request_view(
                    self, provider=provider, retain_contents=retain_contents,
                    count_view=count_view,
                )
            # Match Context's pressure-then-prepare sequence: the opaque view
            # cannot be counted, but that must not prevent semantic compaction.
            pressure_attempts.append(await count_view(list(self._messages)))
            assert pressure_attempts[-1]["budget_decision"] == pressure
            return await super().get_measured_request_view(
                provider=provider, retain_contents=retain_contents,
                count_view=count_view, reason=reason,
            )

    class OpaqueCountProvider(RecoveringProvider):
        def request_budget(self, request, *, context_estimate, request_options=None):
            self.budget_options.append(request_options)
            self.counted_requests.append(request)
            opaque = any(message.content == "old history" * 100 for message in request.messages)
            return deepcopy(pressure) if opaque else _fit(cap=request.max_output_tokens or 128)

    context = TwoStageContext()
    provider = OpaqueCountProvider()
    result = await _execute_recovery(context, provider)

    assert result == "recovered"
    assert provider.complete_calls == 2
    assert len(provider.counted_requests) == 3
    assert len(pressure_attempts) == 1
    assert pressure_attempts[0]["dispatch"] is provider.counted_requests[1]
    assert provider.requests[-1] is provider.counted_requests[2]
    assert context.repair_reasons == ["provider_overflow"]
    assert [transaction.committed for transaction in context.transactions] == [1, 1]
    assert [transaction.rolled_back for transaction in context.transactions] == [0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["normal", "streaming", "finalization"])
async def test_permanently_unavailable_count_allows_initial_send_but_no_retry(phase) -> None:
    class TolerantContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            if reason is None:
                return await _CountTolerantContext.get_measured_request_view(
                    self, provider=provider, retain_contents=retain_contents, count_view=count_view
                )
            return await super().get_measured_request_view(
                provider=provider, retain_contents=retain_contents,
                count_view=count_view, reason=reason,
            )

    class UnavailableProvider(RecoveringProvider):
        def request_budget(self, request, *, context_estimate, request_options=None):
            self.budget_options.append(request_options)
            return None

    class UnavailableStreamProvider(UnavailableProvider):
        async def stream(self, request, *, tools):
            self.requests.append(request)
            raise ContextLengthError("stream rejected input")
            yield

    class UnavailableFinalProvider(UnavailableProvider):
        def parse_tool_calls(self, response):
            return [ToolCallStub()] if self.complete_calls == 1 else []

    context = TolerantContext()
    provider = (
        UnavailableFinalProvider(fail_call=2) if phase == "finalization" else
        UnavailableStreamProvider() if phase == "streaming" else UnavailableProvider()
    )
    with pytest.raises(ContextLengthError, match="requires a provider count"):
        await StreamingOrchestrator(
            {"max_iterations": 1} if phase == "finalization" else {}
        ).execute(
            "work", context, {"main": provider},
            {"mock_tool": OneShotTool()} if phase == "finalization" else {},
            ScriptedHooks({}), _retaining_coordinator(context),
        )
    assert len(provider.requests) == (2 if phase == "finalization" else 1)
    assert len(provider.budget_options) == (3 if phase == "finalization" else 2)
    assert context.transactions[-1].rolled_back == 1


@pytest.mark.asyncio
async def test_lost_count_capability_never_permits_an_unchecked_retry() -> None:
    class LosingProvider(RecoveringProvider):
        async def complete(self, request, **kwargs):
            self.request_budget = None
            return await super().complete(request, **kwargs)

    context = OverflowContext()
    provider = LosingProvider()
    with pytest.raises(ContextLengthError, match="requires a provider count"):
        await _execute_recovery(context, provider)
    assert provider.complete_calls == 1
    assert context.transactions[-1].rolled_back == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["same", "deepcopy", "cap_only", "metadata_only"])
async def test_unchanged_message_content_is_not_repair(mode) -> None:
    class UnchangedContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            if reason is None:
                return await super().get_measured_request_view(
                    provider=provider, retain_contents=retain_contents, count_view=count_view
                )
            base_view = list(self._messages)
            if mode == "deepcopy":
                base_view = deepcopy(base_view)
            if mode == "metadata_only":
                base_view = deepcopy(base_view)
                base_view[0]["metadata"] = {"different": True}
            attempt = await count_view(base_view)
            return {"base_view": base_view, "final_attempt": attempt}

    provider = RecoveringProvider(
        recovery=_overflow(cap=64) if mode == "cap_only" else _overflow()
    )
    with pytest.raises(ContextLengthError, match="content unchanged"):
        await _execute_recovery(UnchangedContext(), provider)
    assert provider.complete_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper", ["copy_dispatch", "mutate_dispatch", "swap_decision", "uncounted_dispatch"]
)
async def test_repair_must_dispatch_callback_owned_unchanged_counted_object(tamper) -> None:
    class TamperingContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            result = await super().get_measured_request_view(
                provider=provider, retain_contents=retain_contents,
                count_view=count_view, reason=reason,
            )
            if reason is not None:
                attempt = result["final_attempt"]
                if tamper == "copy_dispatch":
                    attempt["dispatch"] = attempt["dispatch"].model_copy(deep=True)
                elif tamper == "mutate_dispatch":
                    attempt["dispatch"].messages[-1].content = "not counted"
                elif tamper == "swap_decision":
                    attempt["budget_decision"] = {**_fit(), "estimated_input_tokens": 2}
                elif tamper == "uncounted_dispatch":
                    result["final_attempt"] = {
                        "dispatch": object(), "budget_decision": _fit()
                    }
            return result

    context = TamperingContext()
    provider = RecoveringProvider()
    with pytest.raises(ContextLengthError, match="count"):
        await _execute_recovery(context, provider)
    assert provider.complete_calls == 1
    assert context.transactions[-1].rolled_back == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("getter_kind", ["missing", "old", "kwargs", "positional_only", "uninspectable"])
async def test_old_measured_getter_declines_without_a_reactive_invocation(getter_kind) -> None:
    context = OverflowContext()
    coordinator = _retaining_coordinator(context)
    invocations = []

    async def legacy(*, provider, retain_contents, count_view):
        invocations.append("ordinary")
        return await context.get_measured_request_view(
            provider=provider, retain_contents=retain_contents, count_view=count_view
        )

    async def kwargs_only(**kwargs):
        return await legacy(**kwargs)

    async def positional_reason(reason=None, /, **kwargs):
        return await legacy(**kwargs)

    class Uninspectable:
        __signature__ = "not a signature"

        async def __call__(self, **kwargs):
            return await legacy(**kwargs)

    getter = {
        "missing": None, "old": legacy, "kwargs": kwargs_only,
        "positional_only": positional_reason, "uninspectable": Uninspectable(),
    }[getter_kind]
    coordinator.register_capability("context.measured_request_view", getter)
    # No explicit hard_fit fallback in this fixture: old measured signatures
    # alone must not authorize a reactive invocation.
    async def old_retention(*, provider, retain_contents, token_budget=None):
        return await context.retaining_view(
            provider=provider, retain_contents=retain_contents, token_budget=token_budget
        )
    coordinator.register_capability("context.request_retention", old_retention)
    provider = RecoveringProvider()
    with pytest.raises(ContextLengthError, match="provider rejected input"):
        await _execute_recovery(context, provider, coordinator=coordinator)
    assert invocations == ([] if getter_kind == "missing" else ["ordinary"])
    assert len(provider.recovery_options) == 1
    assert provider.complete_calls == 1
    assert context.repair_reasons == []


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["normal", "streaming", "finalization"])
@pytest.mark.parametrize("measured_getter", ["missing", "old"])
async def test_explicit_legacy_hard_fit_retries_only_the_changed_counted_request(
    phase, measured_getter
) -> None:
    context = HardFitBudgetContext()
    old = "old history" * 100
    context._messages.append({"role": "assistant", "content": old})
    tool_calls = []

    class CountingTool(OneShotTool):
        async def execute(self, arguments):
            tool_calls.append(arguments)
            return await super().execute(arguments)

    class FinalProvider(RecoveringProvider):
        def parse_tool_calls(self, response):
            return [ToolCallStub()] if self.complete_calls == 1 else []

    provider = (
        FinalProvider(fail_call=2) if phase == "finalization" else
        RecoveringStreamProvider() if phase == "streaming" else RecoveringProvider()
    )
    provider.recovery = _overflow(cap=64)
    hooks = ScriptedHooks({
        "provider:request": ScriptedHookResult(
            action="inject_context", ephemeral=True, context_injection="LEGACY-CURRENT-OVERLAY"
        )
    })
    coordinator = _legacy_coordinator(context)
    measured_calls = []

    async def old_measured(*, provider, retain_contents, count_view):
        measured_calls.append(list(retain_contents))
        rows = list(context._messages)
        return {
            "base_view": rows,
            "final_attempt": await count_view(rows),
            "outcome": "not_needed",
            "count_calls": 1,
        }

    if measured_getter == "old":
        coordinator.register_capability("context.measured_request_view", old_measured)
    result = await StreamingOrchestrator({
        "extended_thinking": True,
        "max_iterations": 1 if phase == "finalization" else -1,
        "ephemeral_injection_mode": "tail", "reminder_placement": "tail",
    }).execute(
        "work", context, {"main": provider},
        {"mock_tool": CountingTool()} if phase == "finalization" else {},
        hooks, coordinator,
    )

    assert result == ("stream recovered" if phase == "streaming" else "recovered")
    expected_ordinary = 2 if phase == "finalization" else 1
    assert context.hard_fit_calls == (
        [False] * expected_ordinary + [True] if measured_getter == "missing" else [True]
    )
    assert len(measured_calls) == (expected_ordinary if measured_getter == "old" else 0)
    assert context.request_calls[-1][1] == 1
    assert len(provider.requests) == (3 if phase == "finalization" else 2)
    assert provider.requests[-1] is provider.counted_requests[-1]
    assert provider.requests[-1].max_output_tokens == 64
    assert provider.requests[-1].tools == provider.requests[-2].tools
    assert provider.requests[-1].tool_choice == provider.requests[-2].tool_choice
    if phase == "finalization":
        assert provider.requests[-1].tool_choice == "none"
    contents = [message.content for message in provider.requests[-1].messages]
    assert old not in contents
    assert "\n".join(contents).count("LEGACY-CURRENT-OVERLAY") == 1
    assert any(message.get("content") == old for message in context._messages)
    assert len(tool_calls) == (1 if phase == "finalization" else 0)
    assert [name for name, _ in hooks.emitted].count("provider:request") == (2 if phase == "finalization" else 1)
    assert len(provider.recovery_options) == 1
    expected_options = {} if phase == "streaming" else {"extended_thinking": True}
    assert provider.budget_options[-1] == provider.recovery_options[-1] == expected_options


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["unchanged", "none", "estimate", "malformed", "server_limit", "reserved_limit"]
)
async def test_legacy_hard_fit_still_requires_changed_content_and_strict_final_fit(failure) -> None:
    class LegacyContext(HardFitBudgetContext):
        async def retaining_view(
            self, *, provider=None, retain_contents, token_budget=None, hard_fit=False
        ):
            if failure == "unchanged" and hard_fit:
                self.hard_fit_calls.append(hard_fit)
                return list(self._messages)
            return await super().retaining_view(
                provider=provider, retain_contents=retain_contents,
                token_budget=token_budget, hard_fit=hard_fit,
            )

    class PostRejectProvider(RecoveringProvider):
        def request_budget(self, request, *, context_estimate, request_options=None):
            result = super().request_budget(
                request, context_estimate=context_estimate, request_options=request_options
            )
            if self.complete_calls:
                if failure == "none":
                    return None
                if failure == "estimate":
                    result.pop("measurement")
                elif failure == "malformed":
                    result["max_output_tokens"] = 0
                elif failure in ("server_limit", "reserved_limit"):
                    result.update(estimated_input_tokens=11, input_limit_tokens=20)
                    result["measurement"]["input_tokens"] = 11 if failure == "server_limit" else 9
            return result

    context = LegacyContext()
    context._messages.append({"role": "assistant", "content": "old history" * 100})
    provider = PostRejectProvider()
    hooks = ScriptedHooks({})
    with pytest.raises(ContextLengthError):
        await _execute_recovery(
            context, provider, coordinator=_legacy_coordinator(context), hooks=hooks
        )
    assert provider.complete_calls == 1
    assert context.hard_fit_calls == [False, True]
    assert [name for name, _ in hooks.emitted].count("provider:request") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("during", ["retention", "count", "spacing"])
async def test_stop_during_legacy_repair_prevents_retry(during, monkeypatch) -> None:
    class StoppingContext(HardFitBudgetContext):
        async def retaining_view(
            self, *, provider=None, retain_contents, token_budget=None, hard_fit=False
        ):
            result = await super().retaining_view(
                provider=provider, retain_contents=retain_contents,
                token_budget=token_budget, hard_fit=hard_fit,
            )
            if hard_fit and during == "retention":
                coordinator.cancellation.is_cancelled = True
            return result

    class StoppingProvider(RecoveringProvider):
        def request_budget(self, request, *, context_estimate, request_options=None):
            result = super().request_budget(
                request, context_estimate=context_estimate, request_options=request_options
            )
            if self.complete_calls and during == "count":
                coordinator.cancellation.is_cancelled = True
            return result

    async def stop_spacing(self, _hooks, _iteration):
        if self._last_provider_call_end is not None and during == "spacing":
            coordinator.cancellation.is_cancelled = True

    monkeypatch.setattr(StreamingOrchestrator, "_apply_rate_limit_delay", stop_spacing)
    context = StoppingContext()
    context._messages.append({"role": "assistant", "content": "old history" * 100})
    coordinator = _legacy_coordinator(context)
    provider = StoppingProvider()
    hooks = ScriptedHooks({})
    with pytest.raises(ContextLengthError, match="provider rejected input"):
        await _execute_recovery(context, provider, coordinator=coordinator, hooks=hooks)
    assert provider.complete_calls == 1
    assert [name for name, _ in hooks.emitted].count("provider:request") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("getter_kind", ["kwargs", "positional_only", "uninspectable"])
async def test_legacy_hard_fit_requires_explicit_inspectable_keyword(getter_kind) -> None:
    context = HardFitBudgetContext()
    context._messages.append({"role": "assistant", "content": "old history" * 100})
    coordinator = _legacy_coordinator(context)
    invocations = []

    async def record_ordinary(kwargs):
        invocations.append(kwargs)
        return list(context._messages)

    async def kwargs_only(**kwargs):
        return await record_ordinary(kwargs)

    async def positional_only(hard_fit=False, /, **kwargs):
        return await record_ordinary(kwargs)

    class Uninspectable:
        __signature__ = "not a signature"

        async def __call__(self, **kwargs):
            return await record_ordinary(kwargs)

    coordinator.register_capability("context.request_retention", {
        "kwargs": kwargs_only, "positional_only": positional_only,
        "uninspectable": Uninspectable(),
    }[getter_kind])
    provider = RecoveringProvider()
    with pytest.raises(ContextLengthError, match="provider rejected input"):
        await _execute_recovery(context, provider, coordinator=coordinator)
    assert len(invocations) == 1
    assert "hard_fit" not in invocations[0]
    assert provider.complete_calls == 1


@pytest.mark.asyncio
async def test_explicit_legacy_hard_fit_typeerror_is_not_reinvoked() -> None:
    class BrokenContext(HardFitBudgetContext):
        async def retaining_view(
            self, *, provider=None, retain_contents, token_budget=None, hard_fit=False
        ):
            self.hard_fit_calls.append(hard_fit)
            if hard_fit:
                raise TypeError("legacy implementation failure")
            return list(self._messages)

    context = BrokenContext()
    context._messages.append({"role": "assistant", "content": "old history" * 100})
    provider = RecoveringProvider()
    with pytest.raises(TypeError, match="legacy implementation failure"):
        await _execute_recovery(context, provider, coordinator=_legacy_coordinator(context))
    assert context.hard_fit_calls == [False, True]
    assert provider.complete_calls == 1


@pytest.mark.asyncio
async def test_reason_getter_typeerror_is_not_retried_without_the_keyword() -> None:
    invocations = []

    class BrokenContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            invocations.append(reason)
            if reason:
                self.repair_reasons.append(reason)
                raise TypeError("implementation failure")
            return await super().get_measured_request_view(
                provider=provider, retain_contents=retain_contents, count_view=count_view
            )

    context = BrokenContext()
    provider = RecoveringProvider()
    with pytest.raises(TypeError, match="implementation failure"):
        await _execute_recovery(context, provider)
    assert context.repair_reasons == ["provider_overflow"]
    assert invocations == [None, "provider_overflow"]
    assert provider.complete_calls == 1
    assert context.hard_fit_calls == [], "a reason-aware getter failure must not enter legacy fallback"


@pytest.mark.asyncio
async def test_reason_aware_measured_native_error_never_enters_legacy_fallback() -> None:
    class RefusingContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            if reason:
                self.repair_reasons.append(reason)
                raise ContextLengthError("protected native history cannot be reduced")
            return await super().get_measured_request_view(
                provider=provider, retain_contents=retain_contents, count_view=count_view
            )

    context = RefusingContext()
    provider = RecoveringProvider()
    with pytest.raises(ContextLengthError, match="protected native history"):
        await _execute_recovery(context, provider)
    assert context.repair_reasons == ["provider_overflow"]
    assert context.hard_fit_calls == []
    assert provider.complete_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("during", ["compaction", "count", "commit"])
async def test_stop_during_measured_repair_never_sends_a_retry(during) -> None:
    class StoppingContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            result = await super().get_measured_request_view(
                provider=provider, retain_contents=retain_contents,
                count_view=count_view, reason=reason,
            )
            if reason:
                if during == "compaction":
                    coordinator.cancellation.is_cancelled = True
                elif during == "commit":
                    original_commit = result["transaction"].commit

                    async def stop_before_commit(**kwargs):
                        coordinator.cancellation.is_cancelled = True
                        return await original_commit(**kwargs)

                    result["transaction"].commit = stop_before_commit
            return result

    class StoppingProvider(RecoveringProvider):
        def request_budget(self, request, *, context_estimate, request_options=None):
            if self.complete_calls and during == "count":
                coordinator.cancellation.is_cancelled = True
            return super().request_budget(
                request, context_estimate=context_estimate, request_options=request_options
            )

    context = StoppingContext()
    coordinator = _retaining_coordinator(context)
    provider = StoppingProvider()
    with pytest.raises(ContextLengthError, match="provider rejected input"):
        await _execute_recovery(context, provider, coordinator=coordinator)
    assert provider.complete_calls == 1
    assert context.transactions[-1].rolled_back == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["request", "budget"])
async def test_commit_await_revalidates_counted_snapshot_and_rolls_back(change) -> None:
    class CommitMutationContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            result = await super().get_measured_request_view(
                provider=provider, retain_contents=retain_contents,
                count_view=count_view, reason=reason,
            )
            if reason:
                attempt = result["final_attempt"]
                original_commit = result["transaction"].commit

                async def mutate_during_commit(**kwargs):
                    await asyncio.sleep(0)
                    if change == "request":
                        attempt["dispatch"].messages[-1].content = "not counted"
                    else:
                        attempt["budget_decision"]["estimated_input_tokens"] += 1
                    return await original_commit(**kwargs)

                result["transaction"].commit = mutate_during_commit
            return result

    context = CommitMutationContext()
    provider = RecoveringProvider()
    with pytest.raises(ContextLengthError, match="changed after counting"):
        await _execute_recovery(context, provider)
    assert provider.complete_calls == 1
    assert context.transactions[-1].committed == 0
    assert context.transactions[-1].rolled_back == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["veto", "exception", "cancellation"])
async def test_open_transaction_rolls_back_when_commit_does_not_select(failure) -> None:
    class CommitFailureContext(OverflowContext):
        async def get_measured_request_view(
            self, *, provider, retain_contents, count_view, reason=None
        ):
            result = await super().get_measured_request_view(
                provider=provider, retain_contents=retain_contents,
                count_view=count_view, reason=reason,
            )
            if reason:
                async def fail_commit(**_kwargs):
                    await asyncio.sleep(0)
                    if failure == "exception":
                        raise RuntimeError("commit failed")
                    if failure == "cancellation":
                        raise asyncio.CancelledError()
                    return False

                result["transaction"].commit = fail_commit
            return result

    context = CommitFailureContext()
    provider = RecoveringProvider()
    expected = (
        RuntimeError if failure == "exception" else
        asyncio.CancelledError if failure == "cancellation" else ContextLengthError
    )
    with pytest.raises(expected):
        await _execute_recovery(context, provider)
    assert provider.complete_calls == 1
    assert context.transactions[-1].committed == 0
    assert context.transactions[-1].rolled_back == 1


@pytest.mark.asyncio
async def test_task_cancellation_during_repair_count_rolls_back() -> None:
    started = asyncio.Event()
    context = OverflowContext()

    class BlockingProvider(RecoveringProvider):
        async def request_budget(self, request, *, context_estimate, request_options=None):
            if self.complete_calls:
                started.set()
                await asyncio.Event().wait()
            return super().request_budget(
                request, context_estimate=context_estimate, request_options=request_options
            )

    provider = BlockingProvider()
    task = asyncio.create_task(_execute_recovery(context, provider))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.complete_calls == 1
    assert context.transactions[-1].rolled_back == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_second_overflow_never_recompacts_or_sends_a_third_request(streaming) -> None:
    class TwiceRejectedProvider(RecoveringProvider):
        async def complete(self, request, **kwargs):
            self.complete_calls += 1
            self.requests.append(request)
            raise ContextLengthError("always rejected")

    class TwiceRejectedStreamProvider(RecoveringStreamProvider):
        async def stream(self, request, *, tools):
            self.stream_calls += 1
            self.requests.append(request)
            raise ContextLengthError("always rejected")
            yield  # make this an async iterator without a provider event

    context = OverflowContext()
    provider = TwiceRejectedStreamProvider() if streaming else TwiceRejectedProvider()
    with pytest.raises(ContextLengthError, match="always rejected"):
        await _execute_recovery(context, provider)
    assert len(provider.requests) == 2
    assert len(provider.recovery_options) == 1
    assert context.repair_reasons == ["provider_overflow"]


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk", [{"content": "visible"}, {"usage": {}}, {"block_type": "thinking", "content": "hidden"}])
async def test_any_prior_stream_event_prevents_repair(chunk) -> None:
    class EventBeforeRejectProvider(RecoveringStreamProvider):
        async def stream(self, request, *, tools):
            self.stream_calls += 1
            self.requests.append(request)
            yield chunk
            raise ContextLengthError("after event")

    context = OverflowContext()
    provider = EventBeforeRejectProvider()
    with pytest.raises(ContextLengthError, match="after event"):
        await _execute_recovery(context, provider)
    assert provider.stream_calls == 1
    assert provider.recovery_options == []
    assert context.repair_reasons == []