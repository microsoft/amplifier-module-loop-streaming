"""Focused generic Loop coverage for bounded provider overflow recovery."""

from __future__ import annotations

import pytest
from amplifier_core import ContextLengthError

from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockResponse,
    OneShotTool,
    RequestCapturingProvider,
    ScriptedHooks,
    ToolCallStub,
)
from tests.test_provider_budget_guard import (
    HardFitBudgetContext,
    _retaining_coordinator,
)


def _fit(cap: int = 128) -> dict[str, int]:
    return {
        "estimated_input_tokens": 1,
        "input_limit_tokens": 10,
        "context_token_budget": 0,
        "max_output_tokens": cap,
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

    def request_budget(self, request, *, context_estimate: int, request_options=None):
        self.budget_options.append(request_options)
        # The post-rebuild None is authorized only by a valid server overflow.
        return _fit() if len(self.budget_options) == 1 else None

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
    context = HardFitBudgetContext()
    context._messages.append({"role": "assistant", "content": "old history" * 100})
    provider = RecoveringProvider()
    hooks = ScriptedHooks({})

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
    assert context.hard_fit_calls == [False, True]
    assert provider.requests[1].max_output_tokens == 128
    assert [name for name, _ in hooks.emitted].count("provider:request") == 1


@pytest.mark.asyncio
async def test_kwargs_only_budget_and_recovery_keep_legacy_call_shape() -> None:
    class KwargsOnlyProvider(RecoveringProvider):
        def request_budget(self, request, **kwargs):
            self.budget_options.append(dict(kwargs))
            return _fit() if len(self.budget_options) == 1 else None

        def recover_context_overflow(self, request, error, **kwargs):
            self.recovery_options.append(dict(kwargs))
            return self.recovery

    context = HardFitBudgetContext()
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
    context = HardFitBudgetContext()
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
    assert context.hard_fit_calls == [False]
    assert [
        data["result"]
        for name, data in hooks.emitted
        if name == "orchestrator:provider_overflow_recovery"
    ] == ["unavailable" if feedback is None else "invalid"]


@pytest.mark.asyncio
async def test_recovery_honors_a_lower_provider_output_cap() -> None:
    context = HardFitBudgetContext()
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
    context = HardFitBudgetContext()
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
    context = HardFitBudgetContext()
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
    context = HardFitBudgetContext()
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

    context = HardFitBudgetContext()
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
    context = HardFitBudgetContext()
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
    context = HardFitBudgetContext()

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
        HardFitBudgetContext(),
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
    context = HardFitBudgetContext()
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
            return _fit() if len(self.budget_options) <= 2 else None

    context = HardFitBudgetContext()
    provider = FinalizationProvider(recovery=_overflow(cap=64), fail_call=2)
    await StreamingOrchestrator({"max_iterations": 1}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert provider.complete_calls == 3
    assert provider.requests[-2].tool_choice == "none"
    assert provider.requests[-1].tool_choice == "none"
    assert provider.requests[-1].max_output_tokens == 64


@pytest.mark.asyncio
async def test_finalization_invalid_recovery_feedback_closes_tool_turn() -> None:
    class FinalizationProvider(RecoveringProvider):
        def parse_tool_calls(self, response):
            return [ToolCallStub()] if self.complete_calls == 1 else []

    context = HardFitBudgetContext()
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