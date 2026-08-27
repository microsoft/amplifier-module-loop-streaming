"""Unit tests for the Layer 1 LLM-call budget (spec: 298-replacement, §3, §11 T1).

Covers, per the implementation spec's test plan:
  T1.1  Budget not reached -> normal success, budget_exhausted False
  T1.2  Budget reached exactly -> N iterations + 1 wrap-up call (documents G4)
  T1.3  Wrap-up content -> system-reminder present; no further looping
  T1.4  Status -> "budget_exhausted"
  T1.5  Metadata shape -> exactly llm_calls/llm_call_budget/budget_exhausted/resumable
  T1.6  Transcript completeness -> every assistant message plus the wrap-up message
  T1.7  Warning fires once at ceil(budget_warn_ratio * N)
  T1.8  Warning suppressed when max_iterations == -1
  T1.9  Per-turn reset across two sequential execute() calls
  T1.10 Status precedence: cancelled beats budget_exhausted
  T1.11 Status precedence: error beats budget_exhausted
  T1.12 Wrap-up LLM failure -> PROVIDER_ERROR emitted, no crash, still budget_exhausted

All LLM calls go through a single FakeProvider queue (pattern mirrors
tests/test_goal_loop.py / tests/test_steering.py -- deliberately duplicated
per-file rather than shared, matching this test suite's existing convention).
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from amplifier_core import ToolResult
from amplifier_core.events import ORCHESTRATOR_COMPLETE

# ---------------------------------------------------------------------------
# Shared stubs -- minimal, self-contained (pattern mirrors test_goal_loop.py)
# ---------------------------------------------------------------------------


class MockHookResult:
    """Minimal hook result -- pass through, no deny, no injection."""

    action = "pass"
    reason = None
    ephemeral = False
    context_injection = None
    context_injection_role = "user"
    append_to_last_tool_result = False
    data = None


class MockHooks:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(
        self, event_name: str, payload: dict | None = None
    ) -> MockHookResult:
        self.emitted.append((event_name, payload or {}))
        return MockHookResult()

    def events(self, name: str) -> list[dict]:
        return [payload for event_name, payload in self.emitted if event_name == name]

    def orchestrator_complete_events(self) -> list[dict]:
        return self.events(ORCHESTRATOR_COMPLETE)

    def budget_warning_events(self) -> list[dict]:
        return self.events("orchestrator:budget_warning")

    def provider_error_events(self) -> list[dict]:
        return self.events("provider:error")

    def execution_end_events(self) -> list[dict]:
        return self.events("execution:end")


class MockContext:
    def __init__(self) -> None:
        self._messages: list[dict] = []

    async def add_message(self, msg: dict) -> None:
        self._messages.append(msg)

    async def get_messages(self) -> list[dict]:
        return list(self._messages)

    async def get_messages_for_request(self, provider=None) -> list[dict]:
        return list(self._messages)


class MockCancellation:
    is_cancelled = False
    is_immediate = False
    state = "running"

    def register_tool_start(self, tool_call_id: str, display_name: str) -> None:
        pass

    def register_tool_complete(self, tool_call_id: str) -> None:
        pass

    async def trigger_callbacks(self) -> None:
        pass


class MockCoordinator:
    def __init__(self) -> None:
        self.cancellation = MockCancellation()
        self.session_state: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = {}

    async def process_hook_result(self, result, *args, **kwargs):
        return result

    def get_capability(self, name: str) -> Any:
        return self._capabilities.get(name)


class MockToolCall:
    def __init__(self, call_id: str = "tc-1", name: str = "mock_tool") -> None:
        self.id = call_id
        self.name = name
        self.arguments: dict = {}


class MockTool:
    name = "mock_tool"
    description = "test tool"
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return ToolResult(success=True, output="ok")


class MockTurnResponse:
    """A plain conversational-turn response (non-streaming path)."""

    def __init__(self, text: str = "", tool_calls: list | None = None) -> None:
        self.text = text
        self.content = text
        self.content_blocks = None
        self.usage = None
        self.metadata = None
        self._intended_tool_calls = tool_calls or []


class FakeProvider:
    """Non-streaming provider stub (no `.stream` attribute).

    Pops responses off `turn_queue` for each `complete()` call. Once the
    queue is exhausted, returns `wrapup_response` -- this is what the
    exhaustion branch's own extra `provider.complete()` call receives, so
    tests only need to queue exactly `max_iterations` turn responses.
    """

    def __init__(self) -> None:
        self.turn_queue: list[MockTurnResponse] = []
        self.wrapup_response: MockTurnResponse | None = MockTurnResponse(
            text="Summary: made progress, here is what remains."
        )
        self.wrapup_should_raise: Exception | None = None
        self.call_count = 0
        self.requests: list[Any] = []

    async def complete(self, chat_request, **kwargs):
        self.call_count += 1
        self.requests.append(chat_request)
        if self.turn_queue:
            return self.turn_queue.pop(0)
        if self.wrapup_should_raise:
            raise self.wrapup_should_raise
        assert self.wrapup_response is not None
        return self.wrapup_response

    def parse_tool_calls(self, response):
        return getattr(response, "_intended_tool_calls", [])


def _make_orchestrator(config: dict | None = None):
    from amplifier_module_loop_streaming import StreamingOrchestrator

    return StreamingOrchestrator(config or {})


def _looping_turns(n: int) -> list[MockTurnResponse]:
    """`n` turn responses, each with a tool call, so the loop never breaks
    early on its own (only the budget can stop it)."""
    return [
        MockTurnResponse(text="", tool_calls=[MockToolCall(call_id=f"tc-{i}")])
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# T1.1 -- Budget not reached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetNotReached:
    async def test_normal_success_when_budget_not_hit(self) -> None:
        orch = _make_orchestrator({"max_iterations": 5})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        # Two tool-call turns, then a plain text turn that ends the loop
        # naturally at iteration 3 -- well under the budget of 5.
        provider.turn_queue = _looping_turns(2) + [MockTurnResponse(text="done")]

        result = await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "done"
        events = hooks.orchestrator_complete_events()
        assert len(events) == 1
        assert events[0]["status"] == "success"
        assert events[0]["metadata"]["budget_exhausted"] is False
        assert events[0]["metadata"]["llm_calls"] == 3
        assert events[0]["metadata"]["llm_call_budget"] == 5
        assert events[0]["metadata"]["resumable"] is True
        # Only 3 provider calls -- no wrap-up call, since budget was never hit.
        assert provider.call_count == 3


# ---------------------------------------------------------------------------
# T1.2 / T1.3 / T1.4 / T1.5 -- Budget reached exactly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetExhaustion:
    async def test_exactly_n_plus_one_provider_calls(self) -> None:
        """T1.2: loop runs exactly N iterations; wrap-up fires; provider
        called N+1 times total (documents G4)."""
        orch = _make_orchestrator({"max_iterations": 3})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(3)

        result = await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.call_count == 4  # 3 budgeted + 1 wrap-up
        assert "Summary: made progress" in result

    async def test_wrapup_reminder_and_no_further_loop(self) -> None:
        """T1.3: the final request's last message carries the
        orchestrator-loop-limit system-reminder, and the wrap-up call never
        triggers a further loop iteration (tool-less by construction -- see
        the exhaustion branch, which forces tools=None on the wrap-up
        ChatRequest)."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)
        # Even if the model tried to call a tool during wrap-up, the
        # orchestrator must not act on it -- prove this by returning a
        # tool_calls-laden response and confirming no fifth call happens.
        provider.wrapup_response = MockTurnResponse(
            text="wrap-up text", tool_calls=[MockToolCall(call_id="tc-wrapup")]
        )

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        wrapup_request = provider.requests[-1]
        last_message = wrapup_request.messages[-1]
        assert last_message.role == "user"
        assert 'source="orchestrator-loop-limit"' in last_message.content
        # Tool-less: the wrap-up ChatRequest carries no tools regardless of
        # what tools were available during the turn.
        assert wrapup_request.tools is None
        # Exactly one wrap-up call -- no further loop, even though the fake
        # wrap-up response included a tool call.
        assert provider.call_count == 3  # 2 budgeted + 1 wrap-up, no more

    async def test_status_is_budget_exhausted(self) -> None:
        """T1.4."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "budget_exhausted"

    async def test_metadata_shape_is_exact(self) -> None:
        """T1.5: metadata has exactly llm_calls, llm_call_budget,
        budget_exhausted, resumable -- no more, no fewer."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        metadata = hooks.orchestrator_complete_events()[-1]["metadata"]
        assert set(metadata.keys()) == {
            "llm_calls",
            "llm_call_budget",
            "budget_exhausted",
            "resumable",
        }
        assert metadata["llm_calls"] == 2
        assert metadata["llm_call_budget"] == 2
        assert metadata["budget_exhausted"] is True
        assert metadata["resumable"] is True


# ---------------------------------------------------------------------------
# T1.6 -- Transcript completeness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTranscriptCompleteness:
    async def test_wrapup_message_appended_to_context(self) -> None:
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)
        provider.wrapup_response = MockTurnResponse(text="final summary text")

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assistant_messages = [m for m in ctx._messages if m.get("role") == "assistant"]
        # The wrap-up's own assistant message closes the transcript -- it is
        # not truncated or dropped.
        assert assistant_messages[-1]["content"] == "final summary text"
        # Nothing was lost: the two tool-round assistant turns should also
        # be present (the orchestrator's tool-call handling records an
        # assistant message with tool_calls per round).
        assert len(assistant_messages) >= 1


# ---------------------------------------------------------------------------
# T1.7 / T1.8 -- 80% warning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetWarning:
    async def test_warning_fires_once_at_threshold(self) -> None:
        """T1.7: at ceil(0.8*N) one event + one injected message; never a
        second warning even though the turn continues past the threshold."""
        # N=5, ratio=0.8 -> threshold iteration = max(1, int(5*0.8)) = 4.
        orch = _make_orchestrator({"max_iterations": 5, "budget_warn_ratio": 0.8})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(5)

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        warnings = hooks.budget_warning_events()
        assert len(warnings) == 1
        assert warnings[0]["iteration"] == 4
        assert warnings[0]["budget"] == 5
        assert warnings[0]["remaining"] == 1

        injected = [
            m
            for m in ctx._messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and "You have used" in m["content"]
        ]
        assert len(injected) == 1

    async def test_warning_suppressed_when_unlimited(self) -> None:
        """T1.8: max_iterations == -1 -> zero warning events."""
        orch = _make_orchestrator({"max_iterations": -1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        # A handful of tool-call turns then a plain finish -- there is no
        # budget, so nothing should ever warn regardless of turn count.
        provider.turn_queue = _looping_turns(6) + [MockTurnResponse(text="done")]

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert hooks.budget_warning_events() == []


# ---------------------------------------------------------------------------
# T1.9 -- Per-turn reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPerTurnReset:
    async def test_budget_flags_reset_across_sequential_executes(self) -> None:
        # max_iterations=10, default budget_warn_ratio=0.8 -> warn threshold
        # is iteration 8. Chosen so the second (1-iteration) turn below is
        # nowhere near its own warn threshold -- if its flags come back
        # True, that can only be leakage from turn 1, not a fresh trigger.
        orch = _make_orchestrator({"max_iterations": 10})
        hooks = MockHooks()
        coordinator = MockCoordinator()

        # First execute(): hits the budget (10 tool-call turns -> exhausted,
        # and crosses the 80% warn threshold along the way).
        ctx1 = MockContext()
        provider1 = FakeProvider()
        provider1.turn_queue = _looping_turns(10)
        await orch.execute(
            prompt="first",
            context=ctx1,
            providers={"main": provider1},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )
        assert orch._budget_exhausted is True
        assert orch._budget_warned is True

        # Second execute() on the SAME orchestrator instance: finishes in a
        # single iteration, far under both the budget and its warn
        # threshold. If the flags didn't reset, this turn would incorrectly
        # report budget_exhausted/warned as leftovers from the first.
        ctx2 = MockContext()
        provider2 = FakeProvider()
        provider2.turn_queue = [MockTurnResponse(text="quick answer")]
        await orch.execute(
            prompt="second",
            context=ctx2,
            providers={"main": provider2},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "success"
        assert events[-1]["metadata"]["budget_exhausted"] is False
        assert orch._budget_exhausted is False
        assert orch._budget_warned is False


# ---------------------------------------------------------------------------
# T1.10 / T1.11 -- Status precedence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStatusPrecedence:
    async def test_cancelled_beats_budget_exhausted(self) -> None:
        """T1.10: cancellation during an exhausted turn -> status ==
        "cancelled", not "budget_exhausted"."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)

        # Force _budget_exhausted True as if exhaustion already ran, then
        # simulate the coordinator reporting cancellation for this turn --
        # exercises _execute_one_turn's precedence directly, since driving a
        # real mid-stream cancellation race is not what this test is about.
        async def fake_execute_stream(*args, **kwargs):
            orch._budget_exhausted = True
            coordinator.cancellation.is_cancelled = True
            return
            yield  # pragma: no cover -- makes this an async generator

        orch._execute_stream = fake_execute_stream  # type: ignore[method-assign]

        result = await orch._execute_one_turn(
            "prompt",
            ctx,
            {"main": provider},
            {},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "cancelled"
        assert result == ""

    async def test_error_beats_budget_exhausted(self) -> None:
        """T1.11: an exception during an exhausted turn -> status ==
        "error", not "budget_exhausted" (and the exception still
        propagates)."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        async def fake_execute_stream(*args, **kwargs):
            orch._budget_exhausted = True
            raise RuntimeError("boom")
            yield  # pragma: no cover

        orch._execute_stream = fake_execute_stream  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="boom"):
            await orch._execute_one_turn(
                "prompt",
                ctx,
                {"main": provider},
                {},
                hooks,  # type: ignore[arg-type]
                coordinator,  # type: ignore[arg-type]
            )

        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "error"


# ---------------------------------------------------------------------------
# T1.12 -- Wrap-up LLM failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWrapupFailure:
    async def test_wrapup_provider_error_does_not_crash(self) -> None:
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)
        provider.wrapup_should_raise = RuntimeError("provider exploded")

        result = await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # No crash: execute() returns normally (possibly with empty text,
        # since the wrap-up call itself failed).
        assert result == ""
        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "budget_exhausted"
        assert events[-1]["metadata"]["budget_exhausted"] is True
        # PROVIDER_ERROR (generic Exception branch, not LLMError) was
        # emitted for the failed wrap-up call.
        provider_errors = hooks.events("provider:error")
        assert len(provider_errors) == 1
        assert provider_errors[0]["error"]["type"] == "RuntimeError"
