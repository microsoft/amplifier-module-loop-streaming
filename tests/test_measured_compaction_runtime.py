"""Contract coverage for Context's optional measured request-view capability."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from amplifier_core import ContextLengthError

from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockCoordinator,
    MockResponse,
    NRoundToolProvider,
    OneShotTool,
    RequestCapturingProvider,
    ScriptedHookResult,
    ScriptedHooks,
)
from tests.test_provider_budget_guard import (
    HardFitBudgetContext,
    _retaining_coordinator,
)


def _provider_count(count: int = 5) -> dict[str, object]:
    return {
        "estimated_input_tokens": count + 4,
        "input_limit_tokens": 100,
        "context_token_budget": 0,
        "measurement": {
            "kind": "provider_count",
            "source": "test.provider.count",
            "input_tokens": count,
        },
    }


class _Transaction:
    def __init__(self) -> None:
        self.committed = 0
        self.rolled_back = 0
        self.terminal = False

    async def commit(self, *, is_cancelled=None) -> bool:
        if self.terminal:
            return self.committed == 1
        if is_cancelled is not None and is_cancelled():
            self.rollback()
            return False
        self.committed += 1
        self.terminal = True
        return True

    def rollback(self) -> None:
        if self.terminal:
            return
        self.rolled_back += 1
        self.terminal = True


class _MeasuredContext(HardFitBudgetContext):
    def __init__(self) -> None:
        super().__init__()
        self.measured_calls: list[tuple[object, list[str]]] = []
        self.transactions: list[_Transaction] = []

    async def get_measured_request_view(self, *, provider, retain_contents, count_view):
        base_view = list(self._messages)
        attempt = await count_view(base_view)
        transaction = _Transaction()
        self.transactions.append(transaction)
        self.measured_calls.append((provider, list(retain_contents)))
        decision = attempt["budget_decision"]
        count = decision["measurement"]["input_tokens"]
        return {
            "base_view": base_view,
            "final_attempt": attempt,
            "outcome": "not_needed",
            "measured_before": count,
            "measured_after": count,
            "policy_budget": 100,
            "trigger": 80.0,
            "target": 50,
            "count_calls": 1,
            "transaction": transaction,
        }


class _MeasuredProvider(RequestCapturingProvider):
    def __init__(self, *, fail_first: bool = False) -> None:
        super().__init__()
        self.budget_calls: list[object] = []
        self.complete_calls = 0
        self.fail_first = fail_first

    def get_info(self):
        return SimpleNamespace(capabilities=["request_budget:provider_count"])

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.budget_calls.append(request)
        return _provider_count()

    def recover_context_overflow(
        self, request, error, *, context_estimate, request_options=None
    ):
        return {
            "estimated_input_tokens": 100,
            "input_limit_tokens": 10,
            "context_token_budget": 1,
        }

    async def complete(self, request, **kwargs):
        self.complete_calls += 1
        self.requests.append(request)
        if self.fail_first and self.complete_calls == 1:
            raise ContextLengthError("provider rejected input")
        response = MockResponse(text="ok")
        response.usage = SimpleNamespace(input_tokens=17, cache_write_tokens=3)
        return response


@pytest.mark.asyncio
async def test_measured_view_counts_and_dispatches_the_identical_request_once() -> None:
    context = _MeasuredContext()
    provider = _MeasuredProvider()
    coordinator = MockCoordinator()
    recorded: list[tuple[int, int]] = []
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    coordinator.register_capability(
        "context.foreground_usage",
        lambda: lambda *, input_tokens, cache_write_tokens=0: recorded.append(
            (input_tokens, cache_write_tokens)
        ),
    )
    hooks = ScriptedHooks(
        {
            "provider:request": ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="COUNTED-OVERLAY",
            )
        }
    )

    await StreamingOrchestrator(
        {"ephemeral_injection_mode": "tail", "reminder_placement": "tail"}
    ).execute("current request", context, {"main": provider}, {}, hooks, coordinator)

    assert len(context.measured_calls) == 1
    assert context.request_calls == []
    assert provider.budget_calls == [provider.requests[0]]
    assert "COUNTED-OVERLAY" in "\n".join(
        message.content for message in provider.requests[0].messages
    )
    assert [transaction.committed for transaction in context.transactions] == [1]
    assert [transaction.rolled_back for transaction in context.transactions] == [0]
    assert recorded == [(17, 3)]


@pytest.mark.asyncio
async def test_measured_overflow_recovery_uses_context_base_view_and_retries_once() -> None:
    context = _MeasuredContext()
    provider = _MeasuredProvider(fail_first=True)
    coordinator = _retaining_coordinator(context)
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )

    await StreamingOrchestrator({}).execute(
        "long enough request",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        coordinator,
    )

    assert provider.complete_calls == 2
    assert len(provider.budget_calls) == 2
    assert len(provider.requests) == 2
    assert context.request_calls[-1][1] == 1


@pytest.mark.asyncio
async def test_measured_transaction_rolls_back_when_budget_event_sets_cancellation() -> None:
    context = _MeasuredContext()
    provider = _MeasuredProvider()
    coordinator = _retaining_coordinator(context)
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )

    class CancellingHooks(ScriptedHooks):
        async def emit(self, event, payload=None):
            result = await super().emit(event, payload)
            if event == "orchestrator:provider_budget":
                coordinator.cancellation.is_cancelled = True
            return result

    await StreamingOrchestrator({}).execute(
        "current request",
        context,
        {"main": provider},
        {},
        CancellingHooks({}),
        coordinator,
    )

    assert provider.requests == []
    assert context.transactions[0].committed == 0
    assert context.transactions[0].rolled_back == 1


@pytest.mark.asyncio
async def test_task_cancellation_during_measured_budget_event_rolls_back() -> None:
    context = _MeasuredContext()
    provider = _MeasuredProvider()
    coordinator = _retaining_coordinator(context)
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    started = asyncio.Event()

    class BlockingHooks(ScriptedHooks):
        async def emit(self, event, payload=None):
            if event == "orchestrator:provider_budget":
                started.set()
                await asyncio.Event().wait()
            return await super().emit(event, payload)

    task = asyncio.create_task(
        StreamingOrchestrator({}).execute(
            "current request",
            context,
            {"main": provider},
            {},
            BlockingHooks({}),
            coordinator,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.requests == []
    assert context.transactions[0].rolled_back == 1



@pytest.mark.asyncio
async def test_measured_rate_delay_cancellation_rolls_back_without_sdk_call(
    monkeypatch,
) -> None:
    context = _MeasuredContext()
    provider = _MeasuredProvider()
    coordinator = _retaining_coordinator(context)
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )

    async def cancel_during_delay(_self, _hooks, _iteration) -> None:
        coordinator.cancellation.is_cancelled = True

    monkeypatch.setattr(
        StreamingOrchestrator, "_apply_rate_limit_delay", cancel_during_delay
    )
    await StreamingOrchestrator({}).execute(
        "current request",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        coordinator,
    )

    assert provider.requests == []
    assert context.transactions[0].rolled_back == 1


class _MeasuredFinalizingProvider(NRoundToolProvider):
    def __init__(self) -> None:
        super().__init__(n_tool_rounds=1)
        self.budget_calls: list[object] = []

    def get_info(self):
        return SimpleNamespace(capabilities=["request_budget:provider_count"])

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.budget_calls.append(request)
        return _provider_count()

    def recover_context_overflow(
        self, request, error, *, context_estimate, request_options=None
    ):
        return {
            "estimated_input_tokens": 100,
            "input_limit_tokens": 10,
            "context_token_budget": 1,
        }

    async def complete(self, request, **kwargs):
        response = await super().complete(request, **kwargs)
        if self.call_count == 2:
            raise ContextLengthError("finalization rejected input")
        return response


@pytest.mark.asyncio
async def test_measured_finalization_commit_veto_closes_tool_turn_without_send() -> None:
    coordinator = _retaining_coordinator(_MeasuredContext())

    class CancellingFinalContext(_MeasuredContext):
        async def get_measured_request_view(self, **kwargs):
            result = await super().get_measured_request_view(**kwargs)
            if len(self.transactions) == 2:
                class CancellingTransaction(_Transaction):
                    async def commit(self, *, is_cancelled=None) -> bool:
                        coordinator.cancellation.is_cancelled = True
                        return await super().commit(is_cancelled=is_cancelled)

                transaction = CancellingTransaction()
                self.transactions[-1] = transaction
                result["transaction"] = transaction
            return result

    context = CancellingFinalContext()
    coordinator.register_capability(
        "context.request_retention", context.retaining_view
    )
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    provider = _MeasuredFinalizingProvider()

    await StreamingOrchestrator({"max_iterations": 1}).execute(
        "current request",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({}),
        coordinator,
    )

    assert provider.call_count == 1
    assert [transaction.committed for transaction in context.transactions] == [1, 0]
    assert [transaction.rolled_back for transaction in context.transactions] == [0, 1]


@pytest.mark.asyncio
async def test_measured_finalization_recovery_uses_context_base_view_once() -> None:
    context = _MeasuredContext()
    provider = _MeasuredFinalizingProvider()
    coordinator = _retaining_coordinator(context)
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )

    await StreamingOrchestrator({"max_iterations": 1}).execute(
        "current request",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({}),
        coordinator,
    )

    assert provider.call_count == 3
    assert len(provider.budget_calls) == 3
    assert context.request_calls[-1][1] == 1
    assert [transaction.committed for transaction in context.transactions] == [1, 1]


@pytest.mark.asyncio
async def test_none_cache_write_usage_is_normalized_to_zero() -> None:
    class CoreUsageShape:
        def model_dump(self):
            return {"input_tokens": 17, "cache_write_tokens": None}

    class NoneCacheWriteProvider(_MeasuredProvider):
        async def complete(self, request, **kwargs):
            response = await super().complete(request, **kwargs)
            response.usage = CoreUsageShape()
            return response

    context = _MeasuredContext()
    provider = NoneCacheWriteProvider()
    coordinator = MockCoordinator()
    recorded: list[tuple[int, int]] = []
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    coordinator.register_capability(
        "context.foreground_usage",
        lambda: lambda *, input_tokens, cache_write_tokens=0: recorded.append(
            (input_tokens, cache_write_tokens)
        ),
    )

    await StreamingOrchestrator({}).execute(
        "current request", context, {"main": provider}, {}, ScriptedHooks({}), coordinator
    )

    assert recorded == [(17, 0)]


@pytest.mark.asyncio
async def test_previously_claimed_no_count_stream_marks_usage_stale() -> None:
    class UncountedStreamProvider:
        def __init__(self) -> None:
            self.requests: list[object] = []

        async def stream(self, request, *, tools):
            self.requests.append(request)
            yield {"content": "streamed"}

    context = _MeasuredContext()
    coordinator = MockCoordinator()
    recorded: list[tuple[object, object]] = []
    claims = 0

    def claim_usage():
        nonlocal claims
        claims += 1

        def recorder(*, input_tokens, cache_write_tokens=0):
            recorded.append((input_tokens, cache_write_tokens))

        return recorder

    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    coordinator.register_capability(
        "context.foreground_usage",
        claim_usage,
    )
    loop = StreamingOrchestrator({})

    await loop.execute(
        "counted complete",
        context,
        {"main": _MeasuredProvider()},
        {},
        ScriptedHooks({}),
        coordinator,
    )
    stream_provider = UncountedStreamProvider()
    await loop.execute(
        "uncounted stream",
        context,
        {"stream": stream_provider},
        {},
        ScriptedHooks({}),
        coordinator,
    )

    assert len(stream_provider.requests) == 1
    assert claims == 1
    assert recorded == [(17, 3), (None, 0)]


@pytest.mark.asyncio
async def test_never_claimed_no_count_stream_does_not_advertise_ownership() -> None:
    class UncountedStreamProvider:
        async def stream(self, request, *, tools):
            yield {"content": "streamed"}

    context = _MeasuredContext()
    coordinator = MockCoordinator()
    recorded: list[tuple[object, object]] = []
    coordinator.register_capability(
        "context.foreground_usage",
        lambda: lambda *, input_tokens, cache_write_tokens=0: recorded.append(
            (input_tokens, cache_write_tokens)
        ),
    )

    await StreamingOrchestrator({}).execute(
        "uncounted stream",
        context,
        {"stream": UncountedStreamProvider()},
        {},
        ScriptedHooks({}),
        coordinator,
    )

    assert recorded == []


@pytest.mark.asyncio
async def test_rejected_measured_stream_marks_its_count_stale_before_retry() -> None:
    class RecoveringMeasuredStreamProvider(_MeasuredProvider):
        def __init__(self) -> None:
            super().__init__()
            self.stream_calls = 0

        async def stream(self, request, *, tools):
            self.stream_calls += 1
            self.requests.append(request)
            if self.stream_calls == 1:
                raise ContextLengthError("stream rejected input")
            yield {"content": "recovered"}

    context = _MeasuredContext()
    provider = RecoveringMeasuredStreamProvider()
    coordinator = _retaining_coordinator(context)
    recorded: list[tuple[object, object]] = []
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    coordinator.register_capability(
        "context.foreground_usage",
        lambda: lambda *, input_tokens, cache_write_tokens=0: recorded.append(
            (input_tokens, cache_write_tokens)
        ),
    )

    await StreamingOrchestrator({}).execute(
        "current request",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        coordinator,
    )

    assert provider.stream_calls == 2
    assert recorded == [(5, 0), (None, 0)]


@pytest.mark.asyncio
async def test_synchronously_rejected_measured_stream_marks_count_stale_before_retry() -> None:
    class SynchronousRejectingStreamProvider(_MeasuredProvider):
        def __init__(self) -> None:
            super().__init__()
            self.stream_calls = 0

        def stream(self, request, *, tools):
            self.stream_calls += 1
            self.requests.append(request)
            if self.stream_calls == 1:
                raise ContextLengthError("stream rejected input")

            async def recovered():
                yield {"content": "recovered"}

            return recovered()

    context = _MeasuredContext()
    provider = SynchronousRejectingStreamProvider()
    coordinator = _retaining_coordinator(context)
    recorded: list[tuple[object, object]] = []
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    coordinator.register_capability(
        "context.foreground_usage",
        lambda: lambda *, input_tokens, cache_write_tokens=0: recorded.append(
            (input_tokens, cache_write_tokens)
        ),
    )

    await StreamingOrchestrator({}).execute(
        "current request",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        coordinator,
    )

    assert provider.stream_calls == 2
    assert recorded == [(5, 0), (None, 0)]


@pytest.mark.asyncio
async def test_measured_final_dispatch_rejects_hard_oversize_before_sdk_call() -> None:
    class HardOversizeContext(_MeasuredContext):
        async def get_measured_request_view(self, **kwargs):
            result = await super().get_measured_request_view(**kwargs)
            result["final_attempt"] = dict(result["final_attempt"])
            result["final_attempt"]["budget_decision"] = {
                **_provider_count(5),
                "estimated_input_tokens": 101,
                "input_limit_tokens": 100,
            }
            return result

    context = HardOversizeContext()
    provider = _MeasuredProvider()
    coordinator = MockCoordinator()
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )

    with pytest.raises(ContextLengthError, match="input budget at measured dispatch"):
        await StreamingOrchestrator({}).execute(
            "current request",
            context,
            {"main": provider},
            {},
            ScriptedHooks({}),
            coordinator,
        )

    assert provider.requests == []
    assert context.transactions[0].rolled_back == 1


@pytest.mark.asyncio
async def test_invalid_measured_dispatch_rolls_back_its_staged_transaction() -> None:
    class InvalidDispatchContext(_MeasuredContext):
        async def get_measured_request_view(self, **kwargs):
            result = await super().get_measured_request_view(**kwargs)
            result["final_attempt"] = dict(result["final_attempt"])
            result["final_attempt"]["dispatch"] = object()
            return result

    context = InvalidDispatchContext()
    provider = _MeasuredProvider()
    coordinator = MockCoordinator()
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )

    with pytest.raises(TypeError, match="did not return a ChatRequest"):
        await StreamingOrchestrator({}).execute(
            "current request",
            context,
            {"main": provider},
            {},
            ScriptedHooks({}),
            coordinator,
        )

    assert provider.requests == []
    assert context.transactions[0].rolled_back == 1


# ---------------------------------------------------------------------------
# Native-count unavailability on the measured path. Every assertion below
# reads the payloads the REAL orchestrator emitted through the hooks object
# it was handed -- never a logger-only or helper-level probe.
# ---------------------------------------------------------------------------


def _budget_payloads(hooks) -> list[dict]:
    return [
        payload
        for event, payload in hooks.emitted
        if event == "orchestrator:provider_budget"
    ]


class _CountTolerantContext(_MeasuredContext):
    """Measured context double that tolerates a decision with no usable count."""

    async def get_measured_request_view(self, *, provider, retain_contents, count_view):
        base_view = list(self._messages)
        attempt = await count_view(base_view)
        transaction = _Transaction()
        self.transactions.append(transaction)
        self.measured_calls.append((provider, list(retain_contents)))
        decision = attempt["budget_decision"]
        measurement = (decision or {}).get("measurement") or {}
        count = measurement.get("input_tokens")
        return {
            "base_view": base_view,
            "final_attempt": attempt,
            "outcome": "not_needed",
            "measured_before": count,
            "measured_after": count,
            "policy_budget": 100,
            "trigger": 80.0,
            "target": 50,
            "count_calls": 1,
            "transaction": transaction,
        }


class _NoMeasurementProvider(_MeasuredProvider):
    """Advertises provider_count but returns a valid decision without one."""

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.budget_calls.append(request)
        return {
            "estimated_input_tokens": 9,
            "input_limit_tokens": 100,
            "context_token_budget": 0,
        }


class _NoDecisionProvider(_MeasuredProvider):
    def request_budget(self, request, *, context_estimate, request_options=None):
        self.budget_calls.append(request)
        return None


class _MalformedMeasurementProvider(_MeasuredProvider):
    """Structurally invalid envelope: must stay fail-closed, not fall back."""

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.budget_calls.append(request)
        return {
            **_provider_count(),
            "max_output_tokens": 0,
        }


def _measured_coordinator(context) -> MockCoordinator:
    coordinator = MockCoordinator()
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    return coordinator


@pytest.mark.asyncio
async def test_measured_path_reports_a_decision_without_a_usable_count() -> None:
    context = _CountTolerantContext()
    provider = _NoMeasurementProvider()
    hooks = ScriptedHooks({})

    await StreamingOrchestrator({}).execute(
        "current request",
        context,
        {"main": provider},
        {},
        hooks,
        _measured_coordinator(context),
    )

    assert len(provider.requests) == 1
    assert _budget_payloads(hooks) == [
        {
            "result": "unavailable",
            "mode": "measured",
            "reason": "measurement_absent",
        }
    ]


@pytest.mark.asyncio
async def test_measured_path_reports_a_none_budget_decision() -> None:
    context = _CountTolerantContext()
    provider = _NoDecisionProvider()
    hooks = ScriptedHooks({})

    await StreamingOrchestrator({}).execute(
        "current request",
        context,
        {"main": provider},
        {},
        hooks,
        _measured_coordinator(context),
    )

    assert len(provider.requests) == 1
    assert _budget_payloads(hooks) == [
        {
            "result": "unavailable",
            "mode": "measured",
            "reason": "no_decision",
        }
    ]


@pytest.mark.asyncio
async def test_measured_valid_count_emits_no_unavailable_event() -> None:
    context = _MeasuredContext()
    provider = _MeasuredProvider()
    hooks = ScriptedHooks({})

    await StreamingOrchestrator({}).execute(
        "current request",
        context,
        {"main": provider},
        {},
        hooks,
        _measured_coordinator(context),
    )

    payloads = _budget_payloads(hooks)
    assert [payload["result"] for payload in payloads] == ["fits"]
    assert all(payload["result"] != "unavailable" for payload in payloads)
    assert payloads[0]["measurement_kind"] == "provider_count"


@pytest.mark.asyncio
async def test_malformed_measured_envelope_stays_fail_closed_without_fallback() -> None:
    context = _CountTolerantContext()
    provider = _MalformedMeasurementProvider()
    hooks = ScriptedHooks({})

    with pytest.raises(ContextLengthError, match="invalid max_output_tokens"):
        await StreamingOrchestrator({}).execute(
            "current request",
            context,
            {"main": provider},
            {},
            hooks,
            _measured_coordinator(context),
        )

    assert provider.requests == []
    assert _budget_payloads(hooks) == []


class _NoMeasurementFinalizingProvider(NRoundToolProvider):
    """One tool round then finalization; never rejects, so no recovery probe.

    Deliberately NOT derived from ``_MeasuredFinalizingProvider``: that double
    raises ``ContextLengthError`` on its second completion to exercise overflow
    recovery, which would add a third budget invocation and make the
    one-event-per-invocation assertion below ambiguous.
    """

    def __init__(self) -> None:
        super().__init__(n_tool_rounds=1)
        self.budget_calls: list[object] = []

    def get_info(self):
        return SimpleNamespace(capabilities=["request_budget:provider_count"])

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.budget_calls.append(request)
        return {
            "estimated_input_tokens": 9,
            "input_limit_tokens": 100,
            "context_token_budget": 0,
        }


@pytest.mark.asyncio
async def test_measured_finalization_reports_its_absent_count_once_per_call() -> None:
    context = _CountTolerantContext()
    provider = _NoMeasurementFinalizingProvider()
    hooks = ScriptedHooks({})
    coordinator = _retaining_coordinator(context)
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )

    await StreamingOrchestrator({"max_iterations": 1}).execute(
        "current request",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        hooks,
        coordinator,
    )

    # One event per budget invocation: ordinary turn, then finalization.
    assert _budget_payloads(hooks) == [
        {
            "result": "unavailable",
            "mode": "measured",
            "reason": "measurement_absent",
        },
        {
            "result": "unavailable",
            "mode": "measured",
            "reason": "measurement_absent",
        },
    ]
    assert len(provider.budget_calls) == len(_budget_payloads(hooks))
