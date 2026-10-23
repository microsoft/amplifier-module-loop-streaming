"""Contract coverage for the paired Context final-dispatch handoff."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from amplifier_core import ContextLengthError

from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockContext,
    MockCoordinator,
    MockResponse,
    NRoundToolProvider,
    OneShotTool,
    RequestCapturingProvider,
    ScriptedHookResult,
    ScriptedHooks,
)
from tests.test_measured_compaction_runtime import _MeasuredContext, _MeasuredProvider
from tests.test_provider_budget_guard import (
    HardFitBudgetContext,
    _retaining_coordinator,
)
from tests.test_provider_overflow_recovery import RecoveringProvider


@dataclass(frozen=True)
class OpaqueTicket:
    sequence: int


class DispatchHandoff:
    """Small synchronous Context-capability fake; tickets are deliberately opaque."""

    def __init__(self) -> None:
        self.requests: list[object] = []
        self.bound: list[OpaqueTicket] = []

    def record_final_request(self, request) -> OpaqueTicket:
        self.requests.append(request)
        return OpaqueTicket(len(self.requests))

    def bind_signed_response(self, ticket: OpaqueTicket) -> None:
        self.bound.append(ticket)


def _register_handoff(coordinator: MockCoordinator, handoff: DispatchHandoff) -> None:
    coordinator.register_capability(
        "context.final_request_record", handoff.record_final_request
    )
    coordinator.register_capability(
        "context.signed_replay", handoff.bind_signed_response
    )


@pytest.mark.asyncio
async def test_records_final_tail_overlay_and_binds_after_assistant_admission() -> None:
    context = MockContext()
    provider = RequestCapturingProvider()
    coordinator = MockCoordinator()
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)
    hooks = ScriptedHooks(
        {
            "provider:request": ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="FINAL-REQUEST-OVERLAY",
            )
        }
    )

    await StreamingOrchestrator(
        {"ephemeral_injection_mode": "tail", "reminder_placement": "tail"}
    ).execute("work", context, {"main": provider}, {}, hooks, coordinator)

    assert handoff.requests == provider.requests
    assert "FINAL-REQUEST-OVERLAY" in "\n".join(
        message.content for message in handoff.requests[0].messages
    )
    assert handoff.bound == [OpaqueTicket(1)]


@pytest.mark.asyncio
async def test_binds_tool_use_assistant_response_after_admission() -> None:
    context = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=1)
    coordinator = MockCoordinator()
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({}),
        coordinator,
    )

    assert handoff.requests == provider.requests
    assert handoff.bound == [OpaqueTicket(1), OpaqueTicket(2)]


@pytest.mark.asyncio
async def test_binds_thinking_for_completed_and_tool_use_assistant_turns() -> None:
    class ThinkingBlock:
        type = "thinking"

        def model_dump(self):
            return {"type": "thinking", "thinking": "private", "signature": "sig"}

    class ThinkingToolProvider(NRoundToolProvider):
        async def complete(self, chat_request, **kwargs):
            self.call_count += 1
            self.requests.append(chat_request)
            response = MockResponse(text=f"round {self.call_count}")
            response.content = [ThinkingBlock()]
            return response

    context = MockContext()
    provider = ThinkingToolProvider(n_tool_rounds=1)
    coordinator = MockCoordinator()
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({}),
        coordinator,
    )

    assistant_messages = [
        message
        for message in context.add_message_calls
        if message.get("role") == "assistant"
    ]
    assert all("thinking_block" in message for message in assistant_messages)
    assert handoff.bound == [OpaqueTicket(1), OpaqueTicket(2)]


@pytest.mark.asyncio
async def test_finalization_dispatch_records_and_binds_its_accepted_response() -> None:
    context = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=1)
    coordinator = MockCoordinator()
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)

    await StreamingOrchestrator({"max_iterations": 1}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({}),
        coordinator,
    )

    assert handoff.requests == provider.requests
    assert handoff.requests[-1].tool_choice == "none"
    assert handoff.bound == [OpaqueTicket(1), OpaqueTicket(2)]


@pytest.mark.asyncio
async def test_measured_dispatch_records_only_after_transaction_commit() -> None:
    context = _MeasuredContext()
    provider = _MeasuredProvider()
    coordinator = _retaining_coordinator(context)
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)
    committed_at_record: list[int] = []

    def record_after_commit(request) -> OpaqueTicket:
        committed_at_record.append(context.transactions[-1].committed)
        return handoff.record_final_request(request)

    coordinator.register_capability("context.final_request_record", record_after_commit)

    await StreamingOrchestrator({}).execute(
        "work", context, {"main": provider}, {}, ScriptedHooks({}), coordinator
    )

    assert committed_at_record == [1]
    assert handoff.requests == provider.requests
    assert handoff.bound == [OpaqueTicket(1)]


@pytest.mark.asyncio
async def test_measured_cancellation_before_dispatch_records_nothing() -> None:
    context = _MeasuredContext()
    provider = _MeasuredProvider()
    coordinator = _retaining_coordinator(context)
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)

    class CancellingHooks(ScriptedHooks):
        async def emit(self, event, payload=None):
            result = await super().emit(event, payload)
            if event == "orchestrator:provider_budget":
                coordinator.cancellation.is_cancelled = True
            return result

    await StreamingOrchestrator({}).execute(
        "work", context, {"main": provider}, {}, CancellingHooks({}), coordinator
    )

    assert provider.requests == []
    assert context.transactions[0].rolled_back == 1
    assert handoff.requests == []
    assert handoff.bound == []


@pytest.mark.asyncio
async def test_hard_limit_preflight_records_nothing() -> None:
    class OverBudgetProvider(RequestCapturingProvider):
        def request_budget(self, request, *, context_estimate, request_options=None):
            return {
                "estimated_input_tokens": 100,
                "input_limit_tokens": 10,
                "context_token_budget": 0,
            }

    context = MockContext()
    provider = OverBudgetProvider()
    coordinator = MockCoordinator()
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)

    with pytest.raises(ContextLengthError, match="cannot retain a smaller context"):
        await StreamingOrchestrator({}).execute(
            "work", context, {"main": provider}, {}, ScriptedHooks({}), coordinator
        )

    assert provider.requests == []
    assert handoff.requests == []
    assert handoff.bound == []


@pytest.mark.asyncio
async def test_overflow_retry_records_and_binds_only_corrected_request() -> None:
    context = HardFitBudgetContext()
    provider = RecoveringProvider()
    coordinator = _retaining_coordinator(context)
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)

    await StreamingOrchestrator({}).execute(
        "work", context, {"main": provider}, {}, ScriptedHooks({}), coordinator
    )

    assert handoff.requests == provider.requests
    assert len(handoff.requests) == 2
    assert handoff.bound == [OpaqueTicket(2)]


@pytest.mark.asyncio
async def test_missing_either_handoff_capability_makes_no_handoff_calls() -> None:
    for registered in ("record", "bind"):
        context = MockContext()
        provider = RequestCapturingProvider()
        coordinator = MockCoordinator()
        handoff = DispatchHandoff()
        if registered == "record":
            coordinator.register_capability(
                "context.final_request_record", handoff.record_final_request
            )
        else:
            coordinator.register_capability(
                "context.signed_replay", handoff.bind_signed_response
            )

        await StreamingOrchestrator({}).execute(
            "work", context, {"main": provider}, {}, ScriptedHooks({}), coordinator
        )

        assert provider.requests
        assert handoff.requests == []
        assert handoff.bound == []


@pytest.mark.asyncio
async def test_failed_assistant_admission_does_not_bind_ticket() -> None:
    class FailingAssistantContext(MockContext):
        async def add_message(self, message):
            if message.get("role") == "assistant":
                raise RuntimeError("assistant admission failed")
            await super().add_message(message)

    context = FailingAssistantContext()
    provider = RequestCapturingProvider()
    coordinator = MockCoordinator()
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)

    with pytest.raises(RuntimeError, match="assistant admission failed"):
        await StreamingOrchestrator({}).execute(
            "work", context, {"main": provider}, {}, ScriptedHooks({}), coordinator
        )

    assert handoff.requests == provider.requests
    assert handoff.bound == []


@pytest.mark.asyncio
async def test_stream_overflow_retry_binds_only_retry_ticket() -> None:
    class StreamRecoveringProvider(RecoveringProvider):
        def stream(self, request, *, tools):
            self.complete_calls += 1
            self.requests.append(request)

            async def chunks():
                if self.complete_calls == 1:
                    raise ContextLengthError("provider rejected input")
                yield {"content": "recovered"}

            return chunks()

    context = HardFitBudgetContext()
    provider = StreamRecoveringProvider()
    coordinator = _retaining_coordinator(context)
    handoff = DispatchHandoff()
    _register_handoff(coordinator, handoff)

    result = await StreamingOrchestrator({}).execute(
        "work", context, {"main": provider}, {}, ScriptedHooks({}), coordinator
    )

    assert result == "recovered"
    assert handoff.requests == provider.requests
    assert len(handoff.requests) == 2
    assert handoff.bound == [OpaqueTicket(2)]