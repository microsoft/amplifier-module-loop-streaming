"""Unit tests for the /goal auto-continue loop's stall-detection work
(docs/designs/goal-command.md).

Covers, per the implementation task, at minimum:
  1. Continuation counting (+ version tolerance: an old-style goal dict with
     only the original 4 keys still works via _ensure_goal_defaults).
  2. no_tool_turns increment/reset across continuation turns.
  3. Dual-condition stall trip: mechanical condition (a) alone is NOT enough
     -- the judge (condition b) must also confirm a static blocker.
  4. Escalation-then-stall: first trip escalates (one more turn); a second
     trip (post-escalation) hard-stops with state "stalled".
  5. No-goal passthrough: zero behavior change when no goal is active.

All LLM calls (evaluator, stall judge, summary) are mocked via a single
FakeProvider that discriminates calls by their system-prompt text, since all
three share the same `provider.complete()` entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from amplifier_core import ToolResult

# ---------------------------------------------------------------------------
# Shared stubs -- minimal, self-contained (pattern mirrors tests/test_steering.py)
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

    def goal_progress_events(self) -> list[dict]:
        return [
            payload
            for name, payload in self.emitted
            if name == "orchestrator:goal_progress"
        ]

    def orchestrator_complete_events(self) -> list[dict]:
        from amplifier_core.events import ORCHESTRATOR_COMPLETE

        return [
            payload for name, payload in self.emitted if name == ORCHESTRATOR_COMPLETE
        ]


class MockContext:
    def __init__(self) -> None:
        self._messages: list[dict] = []

    async def add_message(self, msg: dict) -> None:
        self._messages.append(msg)

    async def get_messages(self) -> list[dict]:
        # Used by _evaluate_goal / stall-judge transcript reading.
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
    """Duck-typed coordinator stub carrying `session_state` (the real
    RustCoordinator has this; the test_steering.py stub predates goal
    support and lacks it -- see that file's pre-existing unrelated
    failures)."""

    def __init__(self) -> None:
        self.cancellation = MockCancellation()
        self.session_state: dict[str, Any] = {}

    async def process_hook_result(self, result, *args, **kwargs):
        return result


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
        self.content = None
        self.content_blocks = None
        self.usage = None
        self.metadata = None
        self._intended_tool_calls = tool_calls or []


class _TextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


def _llm_text_response(text: str) -> SimpleNamespace:
    """Response shape expected by _evaluate_goal/_judge_stall/_summarize_goal_run,
    which read `response.content` blocks (not `.text`)."""
    return SimpleNamespace(content=[_TextBlock(text)])


class FakeProvider:
    """Single provider stub shared by turn conversation, evaluator, stall
    judge, and summary calls -- discriminated by system-prompt text, exactly
    as the real orchestrator issues them (each uses a distinct system
    prompt).
    """

    def __init__(self) -> None:
        self.turn_queue: list[MockTurnResponse] = []
        # Each item: (satisfied: bool, reason: str)
        self.eval_queue: list[tuple[bool, str]] = []
        # Each item: (is_stall: bool, detail: str)
        self.judge_queue: list[tuple[bool, str]] = []
        self.summary_text = "Recap: goal pursued across several turns, then resolved."
        # Optional test hook: called right before an evaluator call is
        # answered (used to snapshot orchestrator state, e.g.
        # goal_dict["no_tool_turns"], at each evaluation point).
        self.on_eval_call: Callable[[], None] | None = None
        # DEFECT 2 regression tracking: the `model` kwarg (if any) actually
        # received by `complete()`, per call kind. Populated regardless of
        # whether the call came from the evaluator, stall judge, or
        # summary -- lets tests assert the cheap-model override actually
        # reaches the provider call, not just `ChatRequest.model`.
        self.eval_call_models: list[str | None] = []
        self.judge_call_models: list[str | None] = []
        self.summary_call_models: list[str | None] = []
        self.summary_call_count = 0

    async def complete(self, chat_request, **kwargs):
        system_msg = next(
            (m for m in chat_request.messages if m.role == "system"), None
        )
        system_text = (
            system_msg.content
            if system_msg and isinstance(system_msg.content, str)
            else ""
        )

        if "tool-less evaluator" in system_text:
            if self.on_eval_call:
                self.on_eval_call()
            self.eval_call_models.append(kwargs.get("model"))
            satisfied, reason = self.eval_queue.pop(0)
            verdict = "YES" if satisfied else "NO"
            return _llm_text_response(f"{verdict}\n{reason}")

        if "tool-less judge" in system_text:
            self.judge_call_models.append(kwargs.get("model"))
            is_stall, detail = self.judge_queue.pop(0)
            verdict = "YES" if is_stall else "NO"
            return _llm_text_response(f"{verdict}\n{detail}")

        if "factual recaps" in system_text:
            self.summary_call_count += 1
            self.summary_call_models.append(kwargs.get("model"))
            return _llm_text_response(self.summary_text)

        # Plain turn-conversation call.
        return self.turn_queue.pop(0)

    def parse_tool_calls(self, response):
        return getattr(response, "_intended_tool_calls", [])


def _make_orchestrator(config: dict | None = None):
    from amplifier_module_loop_streaming import StreamingOrchestrator

    return StreamingOrchestrator(config or {})


# ---------------------------------------------------------------------------
# 1. Continuation counting (+ version tolerance)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestContinuationCounting:
    async def test_continuations_increment_and_reasons_recorded(self) -> None:
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        # Old-style goal dict: ONLY the original 4 keys. Proves
        # _ensure_goal_defaults backfills reasons/continuations/no_tool_turns/
        # escalated via setdefault -- an older caller keeps working.
        goal_dict = {
            "condition": "the file exists",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        # Initial turn + 2 continuation turns, each with a tool call so
        # stall bookkeeping never kicks in and doesn't interfere.
        for _ in range(3):
            provider.turn_queue.append(
                MockTurnResponse(text="", tool_calls=[MockToolCall()])
            )
            provider.turn_queue.append(MockTurnResponse(text="done this round"))

        provider.eval_queue.extend(
            [
                (False, "file not created yet"),
                (False, "still missing"),
                (True, "file now exists"),
            ]
        )

        result = await orch.execute(
            prompt="create the file",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert isinstance(result, str)
        # Backfilled defaults present and correctly evolved.
        assert goal_dict["continuations"] == 2
        assert goal_dict["reasons"] == [
            "file not created yet",
            "still missing",
            "file now exists",
        ]
        assert goal_dict["escalated"] is False
        assert goal_dict["no_tool_turns"] == 0

        # Goal is cleared on completion.
        assert coordinator.session_state["goal"] is None

        states = [e["state"] for e in hooks.goal_progress_events()]
        assert states == ["continuing", "continuing", "achieved"]

        # Every emitted payload matches the exact contract shape.
        for event in hooks.goal_progress_events():
            assert set(event.keys()) == {
                "orchestrator",
                "state",
                "turn",
                "continuations",
                "cap",
                "reason",
                "reasons",
                "stall_detail",
                "summary",
                "metadata",
            }

        # Terminal event carries the fast-model summary; non-terminal doesn't
        # need one (it's simply None until the run ends).
        achieved_event = hooks.goal_progress_events()[-1]
        assert achieved_event["summary"] == provider.summary_text

        # ORCHESTRATOR_COMPLETE payloads carry the continuations field too.
        complete_events = hooks.orchestrator_complete_events()
        assert all("continuations" in e for e in complete_events)
        assert complete_events[-1]["continuations"] == 2


# ---------------------------------------------------------------------------
# 2. no_tool_turns increment / reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNoToolTurnsBookkeeping:
    async def test_increment_on_zero_tools_reset_on_tools(self) -> None:
        orch = _make_orchestrator({"goal_stall_threshold": 3})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "task complete",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        # Initial turn: no tools (never counted -- not a continuation).
        provider.turn_queue.append(MockTurnResponse(text="thinking"))
        # Turn A (continuation 1): no tools.
        provider.turn_queue.append(MockTurnResponse(text="still thinking"))
        # Turn B (continuation 2): tools run (2 provider calls: tool-call
        # round, then the follow-up plain response that ends the turn).
        provider.turn_queue.append(
            MockTurnResponse(text="", tool_calls=[MockToolCall()])
        )
        provider.turn_queue.append(MockTurnResponse(text="ran the tool"))
        # Turn C (continuation 3): no tools again.
        provider.turn_queue.append(MockTurnResponse(text="thinking again"))
        # Turn D (continuation 4): no tools; goal is satisfied right after.
        provider.turn_queue.append(MockTurnResponse(text="final answer"))

        no_tool_turns_trace: list[int] = []
        provider.on_eval_call = lambda: no_tool_turns_trace.append(
            goal_dict.get("no_tool_turns", 0)
        )

        provider.eval_queue.extend(
            [
                (False, "not yet (initial)"),  # eval after initial turn
                (False, "not yet (after A)"),  # eval after turn A
                (False, "not yet (after B)"),  # eval after turn B
                (False, "not yet (after C)"),  # eval after turn C
                (True, "done (after D)"),  # eval after turn D
            ]
        )

        await orch.execute(
            prompt="go",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # Trace of no_tool_turns *as observed at the start of each evaluator
        # call* -- i.e. reflecting whatever bookkeeping the PRIOR iteration
        # did (bookkeeping for iteration N only happens after iteration N's
        # own eval call resolves, so evalN's snapshot always reflects
        # iteration N-1's outcome):
        #   eval1 (after initial turn): 0 -- nothing has run bookkeeping yet.
        #   eval2 (after turn A): still 0 -- iteration 1 (initial turn) was
        #     never a continuation, so it never ran bookkeeping.
        #   eval3 (after turn B): 1 -- reflects turn A's bookkeeping (0
        #     tools -> incremented).
        #   eval4 (after turn C): 0 -- reflects turn B's bookkeeping (tools
        #     ran -> reset).
        #   eval5 (after turn D, satisfied): 1 -- reflects turn C's
        #     bookkeeping (0 tools -> incremented). Turn D's own bookkeeping
        #     never runs because eval5 resolves satisfied=True first.
        assert no_tool_turns_trace == [0, 0, 1, 0, 1]

        # Final state check: D had 0 tool calls; since eval5 was satisfied,
        # bookkeeping for D never ran, so no_tool_turns stays at C's value.
        assert goal_dict["no_tool_turns"] == 1
        assert goal_dict["escalated"] is False


# ---------------------------------------------------------------------------
# 3. Dual-condition stall trip: mechanical alone is not enough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDualConditionStallTrip:
    async def test_mechanical_condition_alone_does_not_trip_without_judge_confirmation(
        self,
    ) -> None:
        """no_tool_turns reaches the threshold, but the judge says this is
        genuine incremental progress (not a stall) -- the loop must keep
        going normally: no escalation, no stalled state."""
        orch = _make_orchestrator({"goal_stall_threshold": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        # Initial turn (not counted) + 2 no-tool continuation turns to reach
        # the threshold, then a satisfying turn.
        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.turn_queue.append(MockTurnResponse(text="t1"))
        provider.turn_queue.append(MockTurnResponse(text="t2"))
        provider.turn_queue.append(MockTurnResponse(text="t3"))

        provider.eval_queue.extend(
            [
                (False, "narrowing down candidate A"),  # eval1, after initial turn
                (False, "ruled out candidate A, trying B"),  # eval2, after turn A
                (
                    False,
                    "still narrowing on candidate B",
                ),  # eval3, after turn B -> no_tool_turns hits threshold
                (True, "found it"),  # eval4, after turn C
            ]
        )
        # Judge is only consulted once no_tool_turns hits the threshold (2),
        # i.e. once, right after the second continuation turn's evaluation.
        provider.judge_queue.append(
            (False, "genuine incremental narrowing, not a stall")
        )

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.judge_queue == []  # consumed exactly once
        states = [e["state"] for e in hooks.goal_progress_events()]
        assert states == ["continuing", "continuing", "continuing", "achieved"]
        assert goal_dict["escalated"] is False
        assert "stalled" not in states

    async def test_judge_confirms_static_blocker_triggers_escalation(self) -> None:
        """Same mechanical setup, but the judge confirms a static blocker --
        this must escalate (state 'continuing', escalated flips True), not
        hard-stop immediately (that's the one-shot retry, tested next)."""
        orch = _make_orchestrator({"goal_stall_threshold": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.turn_queue.append(MockTurnResponse(text="t1"))
        provider.turn_queue.append(MockTurnResponse(text="t2"))
        # Escalation turn (stall prompt) resolves the goal to keep the test bounded.
        provider.turn_queue.append(MockTurnResponse(text="t3-escalation"))

        provider.eval_queue.extend(
            [
                (False, "blocked: missing credentials"),  # eval1, after initial turn
                (False, "blocked: missing credentials"),  # eval2, after turn A
                (
                    False,
                    "blocked: missing credentials",
                ),  # eval3, after turn B -> threshold hit -> escalate
                (True, "credentials provided, solved"),  # eval4, after escalation turn
            ]
        )
        provider.judge_queue.append((True, "same blocker recurring, no new progress"))

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        states = [e["state"] for e in hooks.goal_progress_events()]
        # continuing (turn A), continuing (turn B -- threshold not yet hit),
        # continuing (escalation fires here, still "continuing"), achieved.
        assert states == ["continuing", "continuing", "continuing", "achieved"]
        assert goal_dict["escalated"] is True
        assert goal_dict["continuations"] == 3  # 3 turns were sent back total


# ---------------------------------------------------------------------------
# 4. Escalation-then-stall: second trip after escalation hard-stops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEscalationThenStall:
    async def test_second_stall_after_escalation_hard_stops(self) -> None:
        orch = _make_orchestrator({"goal_stall_threshold": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        # Initial turn + 2 no-tool continuations (trip #1, escalates) +
        # 1 more no-tool escalation turn (trip #2, hard stop). Never
        # satisfied -- the run must end via "stalled", not "achieved".
        for i in range(4):
            provider.turn_queue.append(MockTurnResponse(text=f"t{i}"))

        provider.eval_queue.extend(
            [
                (False, "blocked: missing API key"),  # after initial turn
                (False, "blocked: missing API key"),  # after turn A
                (
                    False,
                    "blocked: missing API key",
                ),  # after turn B (trip #1 -> escalate)
                (
                    False,
                    "blocked: missing API key",
                ),  # after turn C/escalation (trip #2 -> stall)
            ]
        )
        # First judge call: confirms stall -> escalate.
        # Second judge call (after the escalation turn, still blocked):
        # confirms stall again -> hard stop.
        provider.judge_queue.extend(
            [
                (True, "same blocker, no progress"),
                (True, "escalation did not help, still the same blocker"),
            ]
        )

        result = await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert isinstance(result, str)
        assert provider.judge_queue == []  # both judge calls consumed
        assert provider.eval_queue == []  # all evals consumed, none left over

        states = [e["state"] for e in hooks.goal_progress_events()]
        assert states[-1] == "stalled"
        assert states.count("stalled") == 1
        # Never "achieved" -- this run genuinely never satisfied the goal.
        assert "achieved" not in states

        stalled_event = hooks.goal_progress_events()[-1]
        assert (
            stalled_event["stall_detail"]
            == "escalation did not help, still the same blocker"
        )
        assert stalled_event["summary"] == provider.summary_text

        # Loud, unambiguous terminal state: goal is cleared.
        assert coordinator.session_state["goal"] is None
        assert goal_dict["escalated"] is True


# ---------------------------------------------------------------------------
# 5. No-goal passthrough: zero behavior change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNoGoalPassthrough:
    async def test_no_goal_active_is_single_pass_through(self) -> None:
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue.append(MockTurnResponse(text="just answer normally"))

        # No "goal" key in session_state at all.
        assert "goal" not in coordinator.session_state

        result = await orch.execute(
            prompt="hello",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert isinstance(result, str)
        # No goal_progress events at all -- the goal machinery never engages.
        assert hooks.goal_progress_events() == []
        # Exactly one ORCHESTRATOR_COMPLETE, emitted immediately (not
        # deferred), with goal_turn/goal_final/continuations reflecting the
        # non-goal defaults.
        complete_events = hooks.orchestrator_complete_events()
        assert len(complete_events) == 1
        assert complete_events[0]["goal_turn"] is None
        assert complete_events[0]["goal_final"] is True
        assert complete_events[0]["continuations"] is None
        # No provider calls beyond the single turn conversation call --
        # no evaluator/judge/summary calls were made.
        assert provider.eval_queue == []
        assert provider.judge_queue == []

    async def test_no_provider_calls_are_evaluator_shaped_without_goal(self) -> None:
        """Sanity check that the FakeProvider's discrimination never
        accidentally routes a plain turn call to the evaluator/judge/summary
        branches when no goal is active."""
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue.append(
            MockTurnResponse(text="", tool_calls=[MockToolCall()])
        )
        provider.turn_queue.append(MockTurnResponse(text="done"))

        await orch.execute(
            prompt="do something",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.turn_queue == []
        assert hooks.goal_progress_events() == []


# ---------------------------------------------------------------------------
# 6. DEFECT 1 regression: stall detector must re-arm after escalation, even
#    when the turn completing the second zero-tool streak also hits the cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStallRearmsAfterEscalation:
    """Regression test for DEFECT 1.

    Real session 48adf75a (cap=8, unsatisfiable goal): turns 1-4 zero tool
    calls -> stall judge trips -> escalates for turn 5. Turn 5 (the
    escalation turn) makes a tool call, correctly resetting the zero-tool
    streak. Turns 6, 7, 8 spin again with zero tool calls -- turn 8 should
    have re-tripped the stall judge, but turn 8 was ALSO the configured
    cap, and the cap check ran (and returned) before the stall
    bookkeeping/judge-consultation block ever executed for that turn. The
    judge was never re-consulted and the run silently reported "cap_hit"
    instead of "stalled", burning the full turn budget for nothing.

    This test reproduces the same shape -- spin -> escalate -> tool call
    -> spin again -- with the cap chosen to land exactly on the turn that
    completes the second zero-tool streak, and asserts both that the judge
    IS consulted a second time and that the terminal state is "stalled",
    not "cap_hit". Must FAIL against pre-fix code (only 1 of 2 queued
    judge answers gets consumed, and the terminal state is "cap_hit").
    """

    async def test_second_trip_at_cap_boundary_reports_stalled_not_cap_hit(
        self,
    ) -> None:
        orch = _make_orchestrator({"goal_stall_threshold": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            # Cap lands exactly on the turn that completes the SECOND
            # zero-tool streak: initial(t0) + A + B (trips judge #1 ->
            # escalate) + escalation (1 tool call, resets streak) + C + D
            # (should trip judge #2). That's turn 6.
            "cap": 6,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.turn_queue.append(MockTurnResponse(text="A"))
        provider.turn_queue.append(MockTurnResponse(text="B"))
        # Escalation turn: a tool-call round (empty text + tool_calls) then
        # the follow-up final response after the tool executes -- mirrors
        # the two-provider-call shape used elsewhere in this file (see
        # TestNoToolTurnsBookkeeping) for any turn that actually runs a
        # tool. `self._tool_calls_this_turn` is only incremented when a
        # tool is *executed*, so a single response object that sets both
        # non-empty text AND tool_calls does not exercise the real code
        # path.
        provider.turn_queue.append(
            MockTurnResponse(text="", tool_calls=[MockToolCall()])
        )
        provider.turn_queue.append(MockTurnResponse(text="escalation-followup"))
        provider.turn_queue.append(MockTurnResponse(text="C"))
        provider.turn_queue.append(MockTurnResponse(text="D"))

        provider.eval_queue.extend(
            [
                (False, "blocked: missing credentials"),  # after t0
                (False, "blocked: missing credentials"),  # after A
                (False, "blocked: missing credentials"),  # after B -> trip #1
                (False, "blocked: missing credentials"),  # after escalation (reset)
                (False, "blocked: missing credentials"),  # after C
                (False, "blocked: missing credentials"),  # after D -> trip #2, at cap
            ]
        )
        provider.judge_queue.extend(
            [
                (True, "same blocker recurring, no new progress"),
                (True, "escalation did not help, still the same blocker"),
            ]
        )

        result = await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            # The escalation turn's tool call must actually resolve to a
            # real tool (so `_tool_calls_this_turn` increments and the
            # zero-tool streak genuinely resets) -- an empty tools dict
            # would make the call fail "tool not found" *before* the
            # increment, silently defeating the reset this test exists to
            # exercise.
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert isinstance(result, str)
        # Both judge calls consumed -- the second consultation must happen
        # even though this turn also hits the cap (DEFECT 1).
        assert provider.judge_queue == [], (
            "stall judge was never re-consulted after escalation -- "
            "DEFECT 1 (cap check preempting the stall check)"
        )
        assert provider.eval_queue == []

        states = [e["state"] for e in hooks.goal_progress_events()]
        assert states[-1] == "stalled", (
            f"expected terminal state 'stalled', got {states[-1]!r} -- the "
            "run must not silently report 'cap_hit' when the judge confirms "
            "a second stall at the cap boundary"
        )
        assert "cap_hit" not in states
        assert goal_dict["escalated"] is True

        stalled_event = hooks.goal_progress_events()[-1]
        assert (
            stalled_event["stall_detail"]
            == "escalation did not help, still the same blocker"
        )

    async def test_rescue_still_works_when_cap_leaves_room(self) -> None:
        """Sanity companion: when the cap does NOT coincide with the second
        streak, an escalation that succeeds (tool call resets the streak,
        goal is then achieved) still reaches "achieved", not "stalled" or
        "cap_hit" -- confirms the fix doesn't disturb the rescue path
        evidenced by sessions d34d7b31/c8b810b0.
        """
        orch = _make_orchestrator({"goal_stall_threshold": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.turn_queue.append(MockTurnResponse(text="A"))
        provider.turn_queue.append(MockTurnResponse(text="B"))
        # Escalation turn: tool-call round + follow-up (see comment in the
        # sibling test above for why this needs two queued responses).
        provider.turn_queue.append(
            MockTurnResponse(text="", tool_calls=[MockToolCall()])
        )
        provider.turn_queue.append(MockTurnResponse(text="escalation-followup"))

        provider.eval_queue.extend(
            [
                (False, "blocked: missing credentials"),
                (False, "blocked: missing credentials"),
                (False, "blocked: missing credentials"),
                (True, "credentials provided by the rescue attempt, solved"),
            ]
        )
        provider.judge_queue.append((True, "same blocker recurring, no new progress"))

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.judge_queue == []
        states = [e["state"] for e in hooks.goal_progress_events()]
        assert states[-1] == "achieved"
        assert goal_dict["escalated"] is True


# ---------------------------------------------------------------------------
# 7. DEFECT 2 regression: cheap-model override must reach provider.complete()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheapModelKwargPropagation:
    async def test_evaluator_judge_and_summary_pass_model_kwarg(self) -> None:
        """DEFECT 2 root cause: `_select_cheap_model`'s result was set on
        `ChatRequest.model` but never passed as a `model=` kwarg to
        `provider.complete()`. The actually-installed Anthropic provider's
        `complete()` / `_complete_chat_request()` reads the effective model
        from `kwargs.get("model", self.default_model)` -- it never reads
        `request.model` -- so the override was silently dropped and every
        evaluator/stall-judge/summary call ran on the session's default
        (expensive) model. Asserts the kwarg is now present for all three
        call kinds.
        """
        orch = _make_orchestrator({"goal_stall_threshold": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.turn_queue.append(MockTurnResponse(text="A"))
        provider.turn_queue.append(MockTurnResponse(text="escalation-turn"))

        provider.eval_queue.extend(
            [
                (False, "blocked"),  # after t0 -- not a continuation, no bookkeeping
                (False, "blocked"),  # after A -> threshold(1) trips -> judge #1
                (True, "done"),  # after escalation turn -> achieved
            ]
        )
        provider.judge_queue.append((True, "same blocker, escalate"))

        # Provider name contains "anthropic" so _select_cheap_model resolves
        # to the hint-table entry "claude-haiku-4-5".
        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main-anthropic": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.eval_call_models == [
            "claude-haiku-4-5",
            "claude-haiku-4-5",
            "claude-haiku-4-5",
        ]
        assert provider.judge_call_models == ["claude-haiku-4-5"]
        # continuations=2 (A, escalation turn) -- achieved-with-continuations
        # still gets a summary (DEFECT 3 skip rule only applies at zero).
        assert provider.summary_call_models == ["claude-haiku-4-5"]


# ---------------------------------------------------------------------------
# 8. DEFECT 3 regression: skip the summary call when there's nothing to say
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSkipSummaryWhenNothingToSummarize:
    async def test_achieved_on_first_attempt_skips_summary_call(self) -> None:
        """A goal achieved with zero continuations (satisfied on the very
        first evaluation) has nothing worth recapping -- the summary LLM
        call must be skipped entirely (measured cost: 5-12s of dead wait on
        this exact happy path). Assert both that the payload's `summary`
        is None AND that the FakeProvider's summary branch was never
        invoked (proves the call was skipped, not merely swallowed).
        """
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.eval_queue.append((True, "solved on the first attempt"))

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        achieved_event = hooks.goal_progress_events()[-1]
        assert achieved_event["state"] == "achieved"
        assert goal_dict["continuations"] == 0
        assert achieved_event["summary"] is None
        assert provider.summary_call_count == 0

    async def test_stalled_run_still_gets_a_summary(self) -> None:
        """Companion sanity check: a run that genuinely needed the loop
        (here, ending in "stalled") is NOT affected by the skip rule --
        the summary still runs, same as before.
        """
        orch = _make_orchestrator({"goal_stall_threshold": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.turn_queue.append(MockTurnResponse(text="A"))
        # Escalation turn (also zero tools -- immediately re-trips).
        provider.turn_queue.append(MockTurnResponse(text="B-escalation"))

        provider.eval_queue.extend(
            [
                (False, "blocked"),  # after t0
                (False, "blocked"),  # after A -> trip #1 -> escalate
                (False, "blocked"),  # after escalation turn -> trip #2
            ]
        )
        provider.judge_queue.extend(
            [
                (True, "stall #1 -> escalate"),
                (True, "stall #2 -> hard stop"),
            ]
        )

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        states = [e["state"] for e in hooks.goal_progress_events()]
        assert states[-1] == "stalled"
        assert provider.summary_call_count == 1


# ---------------------------------------------------------------------------
# 9. Reason-history dedupe: rendered payload collapses repeated reasons
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReasonDedupe:
    async def test_consecutive_identical_reasons_collapsed_in_payload(self) -> None:
        """`reasons[]` on an unsatisfiable-goal run can be a string of
        near-identical evaluator sentences -- consumers render this as a
        list, so N verbatim copies is noise, not signal. The rendered
        payload must collapse consecutive identical reasons into one entry
        annotated with a repeat count; the raw per-turn history remains
        untouched on the goal dict itself.
        """
        orch = _make_orchestrator({"goal_stall_threshold": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        for i in range(4):
            provider.turn_queue.append(MockTurnResponse(text=f"t{i}"))

        provider.eval_queue.extend(
            [
                (False, "blocked: missing API key"),
                (False, "blocked: missing API key"),
                (False, "blocked: missing API key"),
                (True, "credentials provided, solved"),
            ]
        )
        provider.judge_queue.append(
            (False, "genuinely still working the same blocker, not a stall")
        )

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # Raw history is untouched: 4 distinct entries recorded verbatim.
        assert goal_dict["reasons"] == [
            "blocked: missing API key",
            "blocked: missing API key",
            "blocked: missing API key",
            "credentials provided, solved",
        ]

        achieved_event = hooks.goal_progress_events()[-1]
        # Rendered payload collapses the 3 identical consecutive reasons
        # into one entry annotated with a repeat count.
        assert achieved_event["reasons"] == [
            "blocked: missing API key (repeated 3x)",
            "credentials provided, solved",
        ]

    async def test_no_repeats_passes_through_unchanged(self) -> None:
        """Sanity check: when nothing repeats, the dedupe is a no-op."""
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "the file exists",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.turn_queue.append(MockTurnResponse(text="t1"))
        provider.eval_queue.extend(
            [
                (False, "file not created yet"),
                (True, "file now exists"),
            ]
        )

        await orch.execute(
            prompt="create the file",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        achieved_event = hooks.goal_progress_events()[-1]
        assert achieved_event["reasons"] == [
            "file not created yet",
            "file now exists",
        ]
