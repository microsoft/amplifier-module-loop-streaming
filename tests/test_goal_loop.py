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
from amplifier_foundation import ProviderPreference

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
    failures).

    Also carries a `get_capability` lookup (real coordinators expose this;
    see `_resolve_goal_model`'s `model_role_resolver` lookup) backed by a
    plain dict -- defaults to no capabilities registered (i.e. every
    `get_capability` call returns None), matching a session with no
    routing bundle installed.
    """

    def __init__(self, capabilities: dict[str, Any] | None = None) -> None:
        self.cancellation = MockCancellation()
        self.session_state: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = capabilities or {}

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
        # First element is either a plain bool (legacy shape -- see
        # `complete()`'s "tool-less judge" branch) or an explicit taxonomy
        # verdict string.
        self.judge_queue: list[tuple[bool | str, str]] = []
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
        # DEFECT 4 regression tracking: the full ChatRequest and kwargs seen
        # by each call kind, so tests can assert `metadata == {"stream":
        # False}`, `extended_thinking is False`, and `max_output_tokens` is
        # set -- not just that the call happened.
        self.eval_call_requests: list[Any] = []
        self.eval_call_kwargs: list[dict] = []
        self.judge_call_requests: list[Any] = []
        self.judge_call_kwargs: list[dict] = []
        self.summary_call_requests: list[Any] = []
        self.summary_call_kwargs: list[dict] = []
        # When set, the summary branch raises this instead of returning a
        # response -- used to exercise _goal_summary_fallback's
        # generation-failed path.
        self.summary_should_raise: Exception | None = None
        # `goal_provider_preferences` glob resolution support (see
        # `_resolve_goal_pref_glob`): the model names this provider reports
        # as available, and/or an exception to raise instead.
        self.list_models_result: list[str] = []
        self.list_models_error: Exception | None = None
        self.list_models_call_count = 0

    async def list_models(self) -> list[str]:
        self.list_models_call_count += 1
        if self.list_models_error:
            raise self.list_models_error
        return list(self.list_models_result)

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
            self.eval_call_requests.append(chat_request)
            self.eval_call_kwargs.append(kwargs)
            satisfied, reason = self.eval_queue.pop(0)
            verdict = "YES" if satisfied else "NO"
            return _llm_text_response(f"{verdict}\n{reason}")

        if "tool-less judge" in system_text:
            self.judge_call_models.append(kwargs.get("model"))
            self.judge_call_requests.append(chat_request)
            self.judge_call_kwargs.append(kwargs)
            flag, detail = self.judge_queue.pop(0)
            # `flag` is either a plain bool (legacy shape: True/False --
            # mapped to a default locked/resolvable verdict word below, for
            # tests that only care about the stalled/not-stalled outcome)
            # or an explicit taxonomy verdict string ("time-locked",
            # "structure-locked", "history-locked", "resolvable" --
            # case-insensitive) for tests that assert on the specific
            # verdict. See _judge_stall / _GOAL_STALL_VERDICT_WORDS.
            if isinstance(flag, str):
                verdict_word = flag.upper()
            else:
                verdict_word = "HISTORY-LOCKED" if flag else "RESOLVABLE"
            return _llm_text_response(f"{verdict_word}\n{detail}")

        if "single, short line for a developer" in system_text:
            # All three per-state summary prompts (stalled/cap_hit/error)
            # share this phrase -- see _GOAL_SUMMARY_SYSTEM_PROMPTS.
            self.summary_call_count += 1
            self.summary_call_models.append(kwargs.get("model"))
            self.summary_call_requests.append(chat_request)
            self.summary_call_kwargs.append(kwargs)
            if self.summary_should_raise:
                raise self.summary_should_raise
            return _llm_text_response(self.summary_text)

        # Plain turn-conversation call.
        return self.turn_queue.pop(0)

    def parse_tool_calls(self, response):
        return getattr(response, "_intended_tool_calls", [])


class FakeModelRoleResolver:
    """Test double for the ``model_role_resolver`` coordinator capability
    (contract: ``async def resolve(model_role: str | list[str]) ->
    list[ProviderPreference]``, per amplifier_module_hooks_routing's
    ``MatrixModelRoleResolver`` and tool-delegate's consumption of the same
    capability). Registered on a ``MockCoordinator`` via its
    ``capabilities={"model_role_resolver": ...}`` constructor arg.

    ``resolve_calls`` records every ``model_role`` argument passed in, so
    tests can assert the resolver is only consulted once per run (caching)
    and that it's asked for the configured ``goal_model_role``.
    """

    def __init__(
        self,
        preferences: list[ProviderPreference] | None = None,
        name: str = "fake-matrix",
    ) -> None:
        self.preferences = preferences if preferences is not None else []
        self.name = name
        self.resolve_calls: list[str | list[str]] = []

    async def resolve(self, model_role: str | list[str]) -> list[ProviderPreference]:
        self.resolve_calls.append(model_role)
        return list(self.preferences)


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

        # Every emitted payload matches the exact contract shape (plus the
        # additive distinct_blockers/condition/schema_version fields -- see
        # _distinct_blocker_count and _GOAL_PROGRESS_SCHEMA_VERSION).
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
                "distinct_blockers",
                "stall_verdict",
                "condition",
                "schema_version",
            }

        # `achieved` never gets a summary, regardless of how many
        # continuations the run took (extended DEFECT 3 fix, see
        # _goal_run_needs_summary) -- the CLI renders no prose on success
        # at all, so the summary call is skipped unconditionally, not just
        # in the zero-continuations case.
        achieved_event = hooks.goal_progress_events()[-1]
        assert achieved_event["summary"] is None
        assert provider.summary_call_count == 0

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
# 4b. Busy-stall trigger (b): tool-activity-INDEPENDENT pre-filter. This is
#     the real defect the task fixes -- trigger (a) above can only ever
#     observe a stall during zero-tool-call turns, so it never fires for
#     the dominant real-world failure mode: the agent stays busy (tool
#     calls every turn) while the goal has already become unsatisfiable
#     (real sessions a3126f2f r8, 6e64b3db r1/r2 -- see
#     GOAL-HARDENING-DESIGN.md sec 1.2).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBusyStallTrigger:
    async def test_busy_trigger_fires_and_hard_stops_despite_tool_calls_every_turn(
        self,
    ) -> None:
        # goal_stall_threshold impossibly high -- proves trigger (a) (idle)
        # genuinely never reaches threshold, isolating trigger (b) (busy).
        orch = _make_orchestrator(
            {
                "goal_stall_threshold": 100,
                "goal_busy_stall_window": 3,
                "goal_busy_stall_min_overlap": 0.5,
            }
        )
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

        # 4 turns total (initial + 3 continuations); EVERY turn makes a
        # tool call, so no_tool_turns stays 0 for the whole run.
        for _ in range(4):
            provider.turn_queue.append(
                MockTurnResponse(text="", tool_calls=[MockToolCall()])
            )
            provider.turn_queue.append(MockTurnResponse(text="ran a tool"))

        provider.eval_queue.extend(
            [
                (False, "blocked: missing config file for module x"),
                (False, "blocked: missing config file for module y"),
                (False, "blocked: missing config file for module z"),
                (False, "blocked: missing config file for module w"),
            ]
        )
        # Window (3) is first reached after the 3rd eval (-> escalate) and
        # again after the 4th (-> hard stop).
        provider.judge_queue.extend(
            [
                ("history-locked", "same blocker recurring despite activity"),
                ("history-locked", "still the same blocker after escalation"),
            ]
        )

        result = await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert isinstance(result, str)
        assert provider.judge_queue == []  # both judge calls consumed
        assert provider.eval_queue == []

        states = [e["state"] for e in hooks.goal_progress_events()]
        assert states[-1] == "stalled"
        assert "achieved" not in states

        stalled_event = hooks.goal_progress_events()[-1]
        assert stalled_event["stall_verdict"] == "history-locked"
        assert (
            stalled_event["stall_detail"]
            == "still the same blocker after escalation"
        )

        # The mechanical IDLE trigger genuinely never reached threshold --
        # proves this run was caught SOLELY by the busy pre-filter.
        assert goal_dict["no_tool_turns"] == 0
        assert goal_dict["escalated"] is True

    async def test_busy_pretrip_does_not_fire_when_reasons_genuinely_differ(
        self,
    ) -> None:
        """Dissimilar reasons across the window must never trip the
        pre-filter, regardless of tool activity -- this is what keeps
        trigger (b) from manufacturing false stalls on a genuinely
        exploratory run.
        """
        orch = _make_orchestrator(
            {
                "goal_stall_threshold": 100,
                "goal_busy_stall_window": 3,
                "goal_busy_stall_min_overlap": 0.5,
            }
        )
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

        for _ in range(4):
            provider.turn_queue.append(
                MockTurnResponse(text="", tool_calls=[MockToolCall()])
            )
            provider.turn_queue.append(MockTurnResponse(text="ran a tool"))

        provider.eval_queue.extend(
            [
                (False, "checking the database connection pool settings"),
                (False, "verifying the network firewall rules configuration"),
                (False, "narrowing down the deployment pipeline permissions"),
                (True, "root cause found and fixed, condition satisfied"),
            ]
        )

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # Pre-filter never tripped -- judge was never consulted at all.
        assert provider.judge_call_requests == []
        states = [e["state"] for e in hooks.goal_progress_events()]
        assert states == ["continuing", "continuing", "continuing", "achieved"]

    async def test_idle_trigger_wins_when_both_conditions_hold_same_turn(
        self,
    ) -> None:
        """When idle_trip and busy_trip could both hold on the same turn,
        idle framing wins (and only one judge call is made) -- a turn
        never pays for two judge calls (see execute()'s stall-detection
        block comment).
        """
        # window(3) - 1 == threshold(2): the busy pre-filter would reach
        # its window on the SAME continuation turn idle_trip first reaches
        # threshold (reasons accumulate 1/turn overall; no_tool_turns only
        # accumulates 1/zero-tool continuation -- see execute()'s
        # stall-detection block comment for why idle wins the race).
        orch = _make_orchestrator(
            {
                "goal_stall_threshold": 2,
                "goal_busy_stall_window": 3,
                "goal_busy_stall_min_overlap": 0.5,
            }
        )
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

        # Initial turn (not counted) + 2 no-tool continuations (idle_trip
        # threshold reached) + 1 escalation turn.
        for i in range(4):
            provider.turn_queue.append(MockTurnResponse(text=f"t{i}"))

        provider.eval_queue.extend(
            [
                (False, "blocked: missing credentials for the deploy step"),
                (False, "blocked: missing credentials for the deploy step"),
                (
                    False,
                    "blocked: missing credentials for the deploy step",
                ),  # idle_trip reaches threshold here; busy_trip would
                # independently be true too (identical reasons, window
                # reached) but is short-circuited -- see the assertion
                # below that the judge got idle framing.
                (True, "credentials provided, solved"),
            ]
        )
        provider.judge_queue.append((True, "same blocker, no progress"))

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.judge_queue == []  # exactly one judge call consumed
        assert len(provider.judge_call_requests) == 1
        # Idle framing used: the judge's user prompt names "no tool
        # actions", not the busy framing's "regardless of ... activity".
        judge_request = provider.judge_call_requests[0]
        user_msg = next(m for m in judge_request.messages if m.role == "user")
        assert "no tool actions at all" in user_msg.content


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
        # goal_busy_stall_window disabled (set impossibly high): this test
        # exercises the IDLE trigger's DEFECT-1 cap-boundary interaction in
        # isolation. Every reason in this test is identical across all 6
        # turns, so the (separate, additive) busy pre-filter would
        # otherwise trip too -- and, correctly, EARLIER than this test's
        # cap boundary (see TestBusyStallTrigger for that behavior
        # exercised on its own terms) -- which would change this test's
        # judge-call count and defeat the specific DEFECT-1 timing it
        # pins.
        orch = _make_orchestrator(
            {"goal_stall_threshold": 2, "goal_busy_stall_window": 100}
        )
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
        """DEFECT 2 root cause: the resolved model was set on
        `ChatRequest.model` but never passed as a `model=` kwarg to
        `provider.complete()`. The actually-installed Anthropic provider's
        `complete()` / `_complete_chat_request()` reads the effective model
        from `kwargs.get("model", self.default_model)` -- it never reads
        `request.model` -- so the override was silently dropped and every
        evaluator/stall-judge/summary call ran on the session's default
        (expensive) model. Asserts the kwarg is now present for all three
        call kinds.

        Model selection now goes through the `model_role_resolver`
        coordinator capability (see `_resolve_goal_model`) rather than the
        deleted hardcoded hint table -- this test registers a
        `FakeModelRoleResolver` that resolves the configured `fast` role to
        a concrete model, exactly as a real routing bundle would.

        The run ends "stalled" rather than "achieved" so that a summary
        call actually happens: `achieved` never generates a summary at all
        (see _goal_run_needs_summary), so it can't be used to prove the
        override reaches the summary call -- only stalled/cap_hit/error
        can.
        """
        orch = _make_orchestrator({"goal_stall_threshold": 1})
        ctx = MockContext()
        hooks = MockHooks()
        resolver = FakeModelRoleResolver(
            preferences=[
                ProviderPreference(provider="main-anthropic", model="claude-haiku-4-5")
            ]
        )
        coordinator = MockCoordinator(capabilities={"model_role_resolver": resolver})
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
        # Escalation turn: still blocked, so the second no-tool streak
        # re-trips the judge immediately (threshold=1) and, since already
        # escalated, hard-stops as "stalled".
        provider.turn_queue.append(MockTurnResponse(text="escalation-turn"))

        provider.eval_queue.extend(
            [
                (False, "blocked"),  # after t0 -- not a continuation, no bookkeeping
                (False, "blocked"),  # after A -> threshold(1) trips -> judge #1
                (False, "blocked"),  # after escalation turn -> judge #2 -> stalled
            ]
        )
        provider.judge_queue.extend(
            [
                (True, "same blocker, escalate"),
                (True, "escalation did not help, still blocked"),
            ]
        )

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main-anthropic": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        states = [e["state"] for e in hooks.goal_progress_events()]
        assert states[-1] == "stalled"

        assert provider.eval_call_models == [
            "claude-haiku-4-5",
            "claude-haiku-4-5",
            "claude-haiku-4-5",
        ]
        assert provider.judge_call_models == [
            "claude-haiku-4-5",
            "claude-haiku-4-5",
        ]
        assert provider.summary_call_models == ["claude-haiku-4-5"]


# ---------------------------------------------------------------------------
# 7b. Model-role routing: goal-loop LLM calls must go through the
#     `model_role_resolver` coordinator capability, not a hardcoded
#     substring hint table.
#
# Root cause: `_select_cheap_model` picked a model from a hardcoded
# provider-name substring table, ignoring the user's configured routing
# matrix entirely -- even though `hooks-routing` registers a
# `model_role_resolver` capability (contract: `async def
# resolve(model_role) -> list[ProviderPreference]`) that IS reachable from
# here. `_resolve_goal_model` now looks that capability up lazily (hooks
# mount after the orchestrator), resolves `self.goal_model_role` (default
# "fast"), and calls whichever provider INSTANCE the resolved preference
# names -- which may differ from the session's default provider.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestModelRoleResolution:
    async def test_resolver_present_all_three_call_kinds_use_resolved_model(
        self,
    ) -> None:
        """When a `model_role_resolver` is registered, the evaluator, stall
        judge, and summary calls all use its resolved model -- never a
        hardcoded name (there is no longer a hint table to fall back to)."""
        orch = _make_orchestrator({"goal_stall_threshold": 1})
        ctx = MockContext()
        hooks = MockHooks()
        resolver = FakeModelRoleResolver(
            preferences=[ProviderPreference(provider="main", model="routed-fast-model")]
        )
        coordinator = MockCoordinator(capabilities={"model_role_resolver": resolver})
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
            [(False, "blocked"), (False, "blocked"), (False, "blocked")]
        )
        provider.judge_queue.extend(
            [(True, "same blocker, escalate"), (True, "still blocked")]
        )

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.eval_call_models == ["routed-fast-model"] * 3
        assert provider.judge_call_models == ["routed-fast-model"] * 2
        assert provider.summary_call_models == ["routed-fast-model"]

    async def test_resolver_names_a_different_provider_that_provider_is_called(
        self,
    ) -> None:
        """The matrix's resolved role may point at an installed provider
        other than the session default -- e.g. `fast` routes to an OpenAI
        provider while the main conversation runs on Anthropic. Calling the
        WRONG provider instance with a model name from a DIFFERENT provider
        is a real failure mode this must avoid: only the resolved
        provider's `complete()` must be invoked.
        """
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        resolver = FakeModelRoleResolver(
            preferences=[
                ProviderPreference(provider="openai-secondary", model="gpt-5-mini")
            ]
        )
        coordinator = MockCoordinator(capabilities={"model_role_resolver": resolver})

        anthropic_provider = FakeProvider()
        openai_provider = FakeProvider()
        anthropic_provider.eval_queue.append((True, "n/a"))  # must never be consulted
        openai_provider.eval_queue.append((True, "resolved via the other provider"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        satisfied, reason = await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"anthropic-main": anthropic_provider, "openai-secondary": openai_provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert satisfied is True
        assert reason == "resolved via the other provider"
        # The resolved (non-default) provider was called with its own model.
        assert openai_provider.eval_call_models == ["gpt-5-mini"]
        # The session-default provider was never touched.
        assert anthropic_provider.eval_call_requests == []
        assert anthropic_provider.eval_queue == [(True, "n/a")]  # untouched, unconsumed

    async def test_no_resolver_registered_warns_and_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No routing bundle installed at all: `coordinator.get_capability`
        returns None. Must log a WARNING naming the specific cause and fall
        back to the session's default provider/model (no model override) --
        the run must still complete rather than erroring out.
        """
        import logging

        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()  # no capabilities registered at all
        provider = FakeProvider()
        provider.eval_queue.append((True, "solved anyway"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        with caplog.at_level(logging.WARNING):
            satisfied, reason = await orch._evaluate_goal(
                "the thing is done",
                ctx,
                {"main": provider},
                hooks,  # type: ignore[arg-type]
                coordinator,  # type: ignore[arg-type]
            )

        assert satisfied is True
        assert reason == "solved anyway"
        # Falls back to the session default provider, no model override.
        assert provider.eval_call_models == [None]
        assert any(
            "no model_role_resolver capability is registered" in r.message
            for r in caplog.records
        )

    async def test_resolver_returns_empty_warns_distinctly_and_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A resolver IS registered but resolves the configured role to no
        candidates (e.g. no installed provider serves it). This must log a
        WARNING with a message DISTINCT from the no-resolver-registered
        case, and still fall back to the session default so the run
        completes.
        """
        import logging

        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        resolver = FakeModelRoleResolver(preferences=[])  # resolves to []
        coordinator = MockCoordinator(capabilities={"model_role_resolver": resolver})
        provider = FakeProvider()
        provider.eval_queue.append((True, "solved anyway"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        with caplog.at_level(logging.WARNING):
            satisfied, reason = await orch._evaluate_goal(
                "the thing is done",
                ctx,
                {"main": provider},
                hooks,  # type: ignore[arg-type]
                coordinator,  # type: ignore[arg-type]
            )

        assert satisfied is True
        assert reason == "solved anyway"
        assert provider.eval_call_models == [None]
        no_candidates_messages = [
            r.message
            for r in caplog.records
            if "resolved to no candidates" in r.message
        ]
        assert no_candidates_messages
        # Distinct from the no-resolver-registered message.
        assert not any(
            "no model_role_resolver capability is registered" in m
            for m in no_candidates_messages
        )

    async def test_resolution_is_cached_across_a_multi_turn_run(self) -> None:
        """`resolver.resolve()` performs a live `provider.list_models()`
        round-trip whenever the matrix entry is a glob, and the evaluator
        runs every turn -- resolution must be cached for the lifetime of
        one `execute()` call, not repeated per turn.
        """
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        resolver = FakeModelRoleResolver(
            preferences=[ProviderPreference(provider="main", model="routed-fast-model")]
        )
        coordinator = MockCoordinator(capabilities={"model_role_resolver": resolver})
        provider = FakeProvider()

        goal_dict = {
            "condition": "the file exists",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        for _ in range(3):
            provider.turn_queue.append(MockTurnResponse(text="working"))
        provider.eval_queue.extend(
            [
                (False, "not yet"),
                (False, "still not yet"),
                (True, "done now"),
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

        # Three evaluator calls happened (one per turn)...
        assert provider.eval_call_models == ["routed-fast-model"] * 3
        # ...but the resolver itself was only ever consulted once.
        assert resolver.resolve_calls == ["fast"]

    async def test_pref_config_forwarded_as_kwargs_extended_thinking_wins(
        self,
    ) -> None:
        """A resolved role's `ProviderPreference.config` (e.g.
        `{"reasoning_effort": "high"}`) must be forwarded as `complete()`
        kwargs -- this is how the delegate path applies per-role config.
        `extended_thinking=False` must still win even when the role's own
        config tries to turn it on (DEFECT 4 invariant).
        """
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        resolver = FakeModelRoleResolver(
            preferences=[
                ProviderPreference(
                    provider="main",
                    model="routed-fast-model",
                    config={"reasoning_effort": "high", "extended_thinking": True},
                )
            ]
        )
        coordinator = MockCoordinator(capabilities={"model_role_resolver": resolver})
        provider = FakeProvider()
        provider.eval_queue.append((True, "looks satisfied"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"main": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        kwargs = provider.eval_call_kwargs[0]
        assert kwargs.get("reasoning_effort") == "high"
        # extended_thinking=False always wins, even though the role's own
        # config tried to set it True.
        assert kwargs.get("extended_thinking") is False
        assert kwargs.get("model") == "routed-fast-model"

    async def test_goal_model_role_config_knob_changes_requested_role(self) -> None:
        """The `goal_model_role` config knob (default "fast") controls
        which role name is requested from the resolver."""
        orch = _make_orchestrator({"goal_model_role": "quality"})
        ctx = MockContext()
        hooks = MockHooks()
        resolver = FakeModelRoleResolver(
            preferences=[ProviderPreference(provider="main", model="quality-model")]
        )
        coordinator = MockCoordinator(capabilities={"model_role_resolver": resolver})
        provider = FakeProvider()
        provider.eval_queue.append((True, "looks satisfied"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"main": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert resolver.resolve_calls == ["quality"]
        assert provider.eval_call_models == ["quality-model"]


# ---------------------------------------------------------------------------
# 7c. goal_provider_preferences: cost-regression fix. When model_role
#     routing doesn't yield a usable provider (no routing bundle installed,
#     empty resolution, or unmounted provider), the previous behavior fell
#     straight through to the session's DEFAULT (expensive, conversational)
#     provider/model -- a cost regression relative to the hardcoded
#     cheap-model table this replaced. `goal_provider_preferences` is an
#     ordered fallback list, consulted only in that gap, resolved with the
#     same natural-sort glob ranking the routing matrix itself uses.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGoalProviderPreferencesFallback:
    async def test_resolver_present_and_resolving_role_wins_preferences_not_consulted(
        self,
    ) -> None:
        """When the role resolver yields a usable provider, the
        `goal_provider_preferences` list is never even consulted -- the
        preference-matching provider's `list_models()` is never called."""
        orch = _make_orchestrator(
            {
                "goal_provider_preferences": [
                    {"provider": "anthropic", "model": "claude-haiku-*"}
                ]
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        resolver = FakeModelRoleResolver(
            preferences=[ProviderPreference(provider="main", model="routed-model")]
        )
        coordinator = MockCoordinator(capabilities={"model_role_resolver": resolver})
        main_provider = FakeProvider()
        main_provider.eval_queue.append((True, "role resolved this"))
        # Also mount a provider that WOULD match the preference entry, to
        # prove it is never touched when the role resolver already won.
        anthropic_provider = FakeProvider()
        anthropic_provider.list_models_result = ["claude-haiku-4-5"]

        await ctx.add_message({"role": "user", "content": "do the thing"})

        satisfied, reason = await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"main": main_provider, "anthropic": anthropic_provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert satisfied is True
        assert reason == "role resolved this"
        assert main_provider.eval_call_models == ["routed-model"]
        # Preference-matching provider was never even queried for its
        # model list -- proves the preference list was not consulted.
        assert anthropic_provider.list_models_call_count == 0
        assert anthropic_provider.eval_call_requests == []

    async def test_no_resolver_registered_preference_list_used_not_session_default(
        self,
    ) -> None:
        """No routing bundle installed at all (no resolver registered):
        the evaluator call must use the configured preference's resolved
        model, NOT the session default provider/model."""
        orch = _make_orchestrator(
            {
                "goal_provider_preferences": [
                    {"provider": "anthropic", "model": "claude-haiku-*"}
                ]
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()  # no capabilities registered at all
        provider = FakeProvider()
        provider.list_models_result = ["claude-haiku-4-5"]
        provider.eval_queue.append((True, "solved via preference"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        satisfied, reason = await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"anthropic": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert satisfied is True
        assert reason == "solved via preference"
        assert provider.eval_call_models == ["claude-haiku-4-5"]

    async def test_resolver_returns_empty_preference_list_used(self) -> None:
        """A resolver IS registered but resolves to no candidates -- same
        as the no-resolver case, the preference list must be used."""
        orch = _make_orchestrator(
            {
                "goal_provider_preferences": [
                    {"provider": "anthropic", "model": "claude-haiku-*"}
                ]
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        resolver = FakeModelRoleResolver(preferences=[])
        coordinator = MockCoordinator(capabilities={"model_role_resolver": resolver})
        provider = FakeProvider()
        provider.list_models_result = ["claude-haiku-4-5"]
        provider.eval_queue.append((True, "solved via preference"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        satisfied, _reason = await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"anthropic": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert satisfied is True
        assert provider.eval_call_models == ["claude-haiku-4-5"]

    async def test_first_preference_provider_not_mounted_advances_to_next(
        self,
    ) -> None:
        """A preference entry whose provider isn't mounted for this session
        is skipped -- resolution advances to the next entry in the list."""
        orch = _make_orchestrator(
            {
                "goal_provider_preferences": [
                    {"provider": "openai", "model": "gpt-5-mini"},
                    {"provider": "anthropic", "model": "claude-haiku-*"},
                ]
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()  # no resolver
        provider = FakeProvider()
        provider.list_models_result = ["claude-haiku-4-5"]
        provider.eval_queue.append((True, "solved via second preference"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        # Only "anthropic" is mounted -- "openai" (first preference) is not.
        satisfied, _reason = await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"anthropic": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert satisfied is True
        assert provider.eval_call_models == ["claude-haiku-4-5"]

    async def test_glob_matching_nothing_advances_to_next_preference(self) -> None:
        """A preference glob that matches nothing in the provider's
        `list_models()` result is skipped -- resolution advances to the
        next entry, and the raw glob is never sent to the provider."""
        orch = _make_orchestrator(
            {
                "goal_provider_preferences": [
                    {"provider": "anthropic", "model": "claude-opus-*"},
                    {"provider": "anthropic", "model": "claude-haiku-*"},
                ]
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()  # no resolver
        provider = FakeProvider()
        # No opus models available -- first preference's glob matches
        # nothing; second preference's glob matches.
        provider.list_models_result = ["claude-haiku-4-5", "claude-sonnet-4-5"]
        provider.eval_queue.append((True, "solved via second glob"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        satisfied, _reason = await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"anthropic": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert satisfied is True
        # Never the raw, unresolved "claude-opus-*" glob.
        assert provider.eval_call_models == ["claude-haiku-4-5"]

    async def test_ranking_parity_clean_alias_beats_dated_snapshot(self) -> None:
        """Pins the deliberate divergence from
        `amplifier_foundation.spawn_utils.resolve_model_pattern`'s bare
        lexicographic `sort(reverse=True)`: given a clean alias alongside
        date-stamped snapshots, the clean alias must win -- lexicographic
        sort would instead pick a snapshot (a fixed string date sorts
        higher than the bare alias)."""
        orch = _make_orchestrator(
            {
                "goal_provider_preferences": [
                    {"provider": "anthropic", "model": "claude-haiku-*"}
                ]
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()  # no resolver
        provider = FakeProvider()
        provider.list_models_result = [
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5-20250101",
        ]
        provider.eval_queue.append((True, "resolved to clean alias"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"anthropic": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert provider.eval_call_models == ["claude-haiku-4-5"]

    async def test_ranking_parity_multi_digit_version_sorts_numerically(
        self,
    ) -> None:
        """Pins the second documented lexicographic-sort defect: multi-digit
        version segments must compare as integers, so
        `claude-opus-4-10` outranks `claude-opus-4-7` (a plain string sort
        would rank `4-7` above `4-10` because `'7' > '1'`)."""
        orch = _make_orchestrator(
            {
                "goal_provider_preferences": [
                    {"provider": "anthropic", "model": "claude-opus-4-*"}
                ]
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()  # no resolver
        provider = FakeProvider()
        provider.list_models_result = ["claude-opus-4-7", "claude-opus-4-10"]
        provider.eval_queue.append((True, "resolved to highest version"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"anthropic": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert provider.eval_call_models == ["claude-opus-4-10"]

    async def test_nothing_resolves_anywhere_warns_and_uses_session_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No resolver, and no `goal_provider_preferences` entry resolves
        either (provider not mounted): falls all the way back to the
        session default with a WARNING, and the run still completes."""
        import logging

        orch = _make_orchestrator(
            {
                "goal_provider_preferences": [
                    {"provider": "openai", "model": "gpt-5-mini"}
                ]
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()  # no resolver
        provider = FakeProvider()  # mounted as "main" -- never matches "openai"
        provider.eval_queue.append((True, "session default used"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        with caplog.at_level(logging.WARNING):
            satisfied, reason = await orch._evaluate_goal(
                "the thing is done",
                ctx,
                {"main": provider},
                hooks,  # type: ignore[arg-type]
                coordinator,  # type: ignore[arg-type]
            )

        assert satisfied is True
        assert reason == "session default used"
        # Falls back to the session default provider, no model override.
        assert provider.eval_call_models == [None]
        assert any(
            "no goal_provider_preferences entry resolved either" in r.message
            for r in caplog.records
        )

    async def test_caching_holds_for_preference_resolution_across_multi_turn_run(
        self,
    ) -> None:
        """`_resolve_goal_pref_glob` performs a live `provider.list_models()`
        round-trip -- resolution via the preference-list fallback must
        still be cached for the lifetime of one `execute()` call, exactly
        like role-resolver resolution, not repeated on every turn."""
        orch = _make_orchestrator(
            {
                "goal_provider_preferences": [
                    {"provider": "anthropic", "model": "claude-haiku-*"}
                ]
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()  # no resolver
        provider = FakeProvider()
        provider.list_models_result = ["claude-haiku-4-5"]

        goal_dict = {
            "condition": "the file exists",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        for _ in range(3):
            provider.turn_queue.append(MockTurnResponse(text="working"))
        provider.eval_queue.extend(
            [
                (False, "not yet"),
                (False, "still not yet"),
                (True, "done now"),
            ]
        )

        await orch.execute(
            prompt="create the file",
            context=ctx,
            providers={"anthropic": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # Three evaluator calls happened (one per turn)...
        assert provider.eval_call_models == ["claude-haiku-4-5"] * 3
        # ...but `list_models()` (the expensive glob-resolution round-trip)
        # was only ever called once.
        assert provider.list_models_call_count == 1


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


# ---------------------------------------------------------------------------
# 10. Run summary stored in full: no length cap in the orchestrator --
# truncation (if any) is a display-time concern for the CLI renderer, not
# something the orchestrator does before storing/emitting the summary.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSummaryStoredInFull:
    async def test_overlong_model_summary_passes_through_unclipped(
        self,
    ) -> None:
        """Wiring check: a model response over ~120 chars (the length the
        per-state prompts merely *ask* the model for) must come out of the
        emitted payload byte-for-byte identical to what the model returned
        -- no code-level truncation. Storage and display are different
        concerns; display-time clipping (if any) lives in amplifier-app-cli's
        goal_progress_hook.py, not here.
        """
        orch = _make_orchestrator({"goal_stall_threshold": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.summary_text = (
            "the evaluator keeps reporting the exact same blocker every "
            "single turn and no new progress has been made toward the "
            "condition at all whatsoever, across every turn of this run"
        )
        assert len(provider.summary_text) > 120

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
                (False, "blocked: missing API key"),
            ]
        )
        provider.judge_queue.extend(
            [
                (True, "same blocker, no progress"),
                (True, "escalation did not help, still the same blocker"),
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

        stalled_event = hooks.goal_progress_events()[-1]
        assert stalled_event["state"] == "stalled"
        # Full text, verbatim -- not merely "under some cap".
        assert stalled_event["summary"] == provider.summary_text


# ---------------------------------------------------------------------------
# 10b. `condition` (fully-expanded) and `schema_version` on every emitted
# goal_progress payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConditionAndSchemaVersionInPayload:
    async def test_condition_is_the_fully_expanded_text_on_every_event(
        self,
    ) -> None:
        """`goal["condition"]` is already the fully-expanded form by the
        time it reaches the orchestrator (app-cli expands @mentions once,
        at /goal set-time) -- the payload must carry that exact text
        verbatim, on every state, not just the terminal one.
        """
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = _make_orchestrator({})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        expanded_condition = (
            "the file expanded-from-mention.txt exists and contains 'done'"
        )
        goal_dict = {
            "condition": expanded_condition,
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.eval_queue.append((True, "done"))

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        events = hooks.goal_progress_events()
        assert events  # sanity: at least the achieved event fired
        for event in events:
            assert event["condition"] == expanded_condition
            assert event["schema_version"] == (
                StreamingOrchestrator._GOAL_PROGRESS_SCHEMA_VERSION
            )
            assert isinstance(event["schema_version"], int)
            # The dead, always-null field is gone -- not merely null.
            assert "metadata" not in event


# ---------------------------------------------------------------------------
# 11. Per-state deterministic fallback strings when generation fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPerStateFallbacks:
    async def test_stalled_fallback_names_the_no_tool_turn_count(self) -> None:
        orch = _make_orchestrator({"goal_stall_threshold": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.summary_should_raise = RuntimeError("model unavailable")

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
                (False, "blocked: missing API key"),
            ]
        )
        provider.judge_queue.extend(
            [
                (True, "same blocker, no progress"),
                (True, "escalation did not help, still the same blocker"),
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

        stalled_event = hooks.goal_progress_events()[-1]
        assert stalled_event["state"] == "stalled"
        assert stalled_event["summary"] == (
            f"no progress across the last {goal_dict['no_tool_turns']} turns"
        )

    async def test_cap_hit_fallback_is_deterministic(self) -> None:
        """Minimal cap_hit scenario: cap=1 means the very first evaluation
        (of the initial turn) already hits the cap, with zero continuations
        -- the shortest possible path to "cap_hit".
        """
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.summary_should_raise = RuntimeError("model unavailable")

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": 1,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.eval_queue.append((False, "not there yet"))

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        cap_hit_event = hooks.goal_progress_events()[-1]
        assert cap_hit_event["state"] == "cap_hit"
        assert cap_hit_event["summary"] == "completion not confirmed before the cap"

    async def test_error_fallback_is_deterministic(self) -> None:
        """No provider mounted at all -- `_evaluate_goal` raises immediately,
        producing an "error" terminal state; the summary call then also has
        no provider to call, so it falls back rather than crashing.
        """
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": None,
        }
        coordinator.session_state["goal"] = goal_dict

        provider = FakeProvider()
        provider.turn_queue.append(MockTurnResponse(text="t0"))

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={},  # no provider mounted -> _evaluate_goal raises
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        error_event = hooks.goal_progress_events()[-1]
        assert error_event["state"] == "error"
        assert error_event["summary"] == "evaluator failed"


@pytest.mark.asyncio
class TestPerStateSummaryGeneration:
    """Confirms cap_hit and error select the correct per-state prompt and
    successfully call through to the provider (companion to the stalled
    case already covered by TestEscalationThenStall and
    TestCheapModelKwargPropagation).
    """

    async def test_cap_hit_gets_a_real_generated_summary(self) -> None:
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 0,
            "last_reason": None,
            "cap": 1,
        }
        coordinator.session_state["goal"] = goal_dict

        provider.turn_queue.append(MockTurnResponse(text="t0"))
        provider.eval_queue.append((False, "not there yet"))

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        cap_hit_event = hooks.goal_progress_events()[-1]
        assert cap_hit_event["state"] == "cap_hit"
        assert provider.summary_call_count == 1
        assert cap_hit_event["summary"] == provider.summary_text

    async def test_error_gets_a_real_generated_summary(self) -> None:
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

        class BrokenContext(MockContext):
            async def get_messages(self) -> list[dict]:
                raise RuntimeError("transcript read failed")

        ctx = BrokenContext()
        provider.turn_queue.append(MockTurnResponse(text="t0"))

        await orch.execute(
            prompt="solve it",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        error_event = hooks.goal_progress_events()[-1]
        assert error_event["state"] == "error"
        assert "transcript read failed" in error_event["reason"]
        assert provider.summary_call_count == 1
        assert error_event["summary"] == provider.summary_text


# ---------------------------------------------------------------------------
# 12. Distinct-blocker-signature count: "hit a wall" vs "flailing"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDistinctBlockerCount:
    async def test_same_blocker_every_turn_counts_as_one(self) -> None:
        """ "Hit a wall": every evaluator reason is the same blocker
        (mod whitespace/case) -- distinct_blockers must be 1, regardless of
        how many turns repeated it.
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
                (False, "  Blocked:   missing api key  "),  # same, mod case/ws
                (False, "blocked: missing API key"),
                (False, "blocked: missing API key"),
            ]
        )
        provider.judge_queue.extend(
            [
                (True, "same blocker, no progress"),
                (True, "escalation did not help, still the same blocker"),
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

        stalled_event = hooks.goal_progress_events()[-1]
        assert stalled_event["state"] == "stalled"
        assert stalled_event["distinct_blockers"] == 1

    async def test_different_blocker_each_turn_counts_all_of_them(self) -> None:
        """ "Flailing": a different blocker every turn -- distinct_blockers
        must equal the number of distinct reasons recorded, not collapse
        them via similarity.
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
                (False, "missing credentials"),
                (False, "wrong output format"),
                (False, "network timeout"),
                (False, "still not matching"),
            ]
        )
        provider.judge_queue.extend(
            [
                (True, "different blocker every turn, judged a stall anyway"),
                (True, "still flailing, hard stop"),
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

        stalled_event = hooks.goal_progress_events()[-1]
        assert stalled_event["state"] == "stalled"
        assert stalled_event["distinct_blockers"] == 4

    async def test_zero_reasons_is_zero(self) -> None:
        """Edge case sanity: an event built before any evaluator reason
        exists (shouldn't happen in practice, but the helper must not
        crash) reports 0, not 1 or an exception.
        """
        from amplifier_module_loop_streaming import StreamingOrchestrator

        assert StreamingOrchestrator._distinct_blocker_count([]) == 0


# ---------------------------------------------------------------------------
# 6. DEFECT 4 regression: internal /goal LLM calls must not leak into the
#    user-facing streaming overlay (real session e97e192b).
#
# Root cause: _evaluate_goal / _judge_stall / _summarize_goal_run each build
# a ChatRequest and call provider.complete() without metadata={"stream":
# False}, without an explicit extended_thinking=False opt-out, and without a
# capped max_output_tokens. Per docs/provider-streaming-contract.md, a
# provider streams by default (config.get("use_streaming", True)) and only
# takes the non-streaming path when request.metadata["stream"] is False --
# so these background utility calls took the same streaming branch as a
# real user turn, emitting llm:stream_block_start/delta/end events that
# hooks-streaming-ui cannot distinguish from foreground output. Precedent
# fix: amplifier-foundation's hooks-session-naming module
# (__init__.py:~519-549) already sets all three of these on its own
# background call.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInternalCallsDoNotStream:
    async def test_evaluate_goal_sets_stream_false_and_disables_thinking(
        self,
    ) -> None:
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.eval_queue.append((True, "looks satisfied"))

        await ctx.add_message({"role": "user", "content": "do the thing"})

        satisfied, reason = await orch._evaluate_goal(
            "the thing is done",
            ctx,
            {"main": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert satisfied is True
        assert reason == "looks satisfied"
        assert len(provider.eval_call_requests) == 1
        request = provider.eval_call_requests[0]
        kwargs = provider.eval_call_kwargs[0]

        # metadata={"stream": False} -- keeps this call off the streaming
        # branch entirely (identity check per the provider contract, not
        # truthiness).
        assert request.metadata == {"stream": False}
        # extended_thinking=False passed explicitly as a kwarg -- NOT left
        # to "just don't set reasoning_effort", since a session-level
        # provider config (reasoning_effort/effort) would otherwise force
        # thinking on regardless of what this orchestrator does.
        assert kwargs.get("extended_thinking") is False
        # max_output_tokens capped -- not left to inherit the provider's
        # session default (64000 in the real session e97e192b that
        # triggered this fix).
        assert request.max_output_tokens == orch._GOAL_INTERNAL_CALL_MAX_TOKENS
        assert request.max_output_tokens is not None
        assert request.max_output_tokens < 64000

    async def test_judge_stall_sets_stream_false_and_disables_thinking(
        self,
    ) -> None:
        orch = _make_orchestrator()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.judge_queue.append((True, "same blocker again"))

        goal_dict = {
            "condition": "solved",
            "turns_used": 2,
            "last_reason": None,
            "cap": None,
            "reasons": ["blocked: x", "blocked: x"],
            "no_tool_turns": 2,
        }

        is_stalled, detail, verdict = await orch._judge_stall(
            goal_dict,
            {"main": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert is_stalled is True
        assert detail == "same blocker again"
        assert verdict == "history-locked"
        assert len(provider.judge_call_requests) == 1
        request = provider.judge_call_requests[0]
        kwargs = provider.judge_call_kwargs[0]

        assert request.metadata == {"stream": False}
        assert kwargs.get("extended_thinking") is False
        assert request.max_output_tokens == orch._GOAL_INTERNAL_CALL_MAX_TOKENS
        assert request.max_output_tokens is not None
        assert request.max_output_tokens < 64000

    async def test_summarize_goal_run_sets_stream_false_and_disables_thinking(
        self,
    ) -> None:
        orch = _make_orchestrator()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        goal_dict = {
            "condition": "solved",
            "turns_used": 3,
            "last_reason": "still blocked",
            "cap": None,
            "reasons": ["blocked: x", "blocked: x", "blocked: x"],
            "no_tool_turns": 3,
            "continuations": 2,
        }

        summary = await orch._summarize_goal_run(
            goal_dict,
            {"main": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
            final_state="stalled",
        )

        assert summary  # non-empty (falls back deterministically otherwise)
        assert len(provider.summary_call_requests) == 1
        request = provider.summary_call_requests[0]
        kwargs = provider.summary_call_kwargs[0]

        assert request.metadata == {"stream": False}
        assert kwargs.get("extended_thinking") is False
        assert request.max_output_tokens == orch._GOAL_INTERNAL_CALL_MAX_TOKENS
        assert request.max_output_tokens is not None
        assert request.max_output_tokens < 64000

    async def test_no_goal_path_never_touches_internal_call_sites(self) -> None:
        """Zero behavior change on the no-goal path: none of the three
        internal-call fixes matter if the goal loop is never entered, and
        this must remain true after the DEFECT 4 fix. No goal set ->
        _evaluate_goal / _judge_stall / _summarize_goal_run are never
        called at all.
        """
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue.append(MockTurnResponse(text="just an answer"))
        # No goal set in coordinator.session_state.

        result = await orch.execute(
            prompt="hello",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "just an answer"
        assert provider.eval_call_requests == []
        assert provider.judge_call_requests == []
        assert provider.summary_call_requests == []


# ---------------------------------------------------------------------------
# 13. _flatten_message_for_evaluator: characterization of current behavior
#     (__init__.py:94-122). Zero coverage existed for this function before
#     these tests -- pinned here so a future change can't silently alter
#     what the /goal evaluator actually sees.
# ---------------------------------------------------------------------------


class TestFlattenMessageForEvaluator:
    """Pure-function characterization tests -- no orchestration, no mocks.
    Each test pins the CURRENT input -> output mapping exactly as
    implemented, including the one behavior (below) that contradicts the
    function's own docstring/inline comment.
    """

    def test_plain_string_content_renders_as_role_colon_content(self) -> None:
        """Case 1: `content` is a plain string -> `"{role}: {content}"`
        verbatim, hitting the `isinstance(content, str)` branch (line 103).
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        msg = {"role": "user", "content": "please fix the bug"}
        assert _flatten_message_for_evaluator(msg) == "user: please fix the bug"

    def test_tool_role_message_as_this_orchestrator_writes_it_is_fully_visible(
        self,
    ) -> None:
        """CRUX TEST (UPDATED for evaluator input hygiene) -- pins that tool
        results DO reach the evaluator, refuting the docstring/`[tool
        result omitted]`-branch implication that they don't.

        This orchestrator's own tool-result write sites (__init__.py
        :2416-2424 normal path, :2366-2374 graceful cancel, :2313-2321
        immediate cancel) all write `role="tool"` messages with a PLAIN
        STRING `content` -- never a `{"type": "tool_result", ...}` content
        block. A plain string hits `isinstance(content, str)`, not the
        `type == "tool_result"` block branch, so the `[tool result
        omitted]` placeholder is never substituted.

        POST-HYGIENE: tool results remain VISIBLE, but are now BOUNDED --
        a short result (well under `tool_content_clip_chars`, as here)
        still renders VERBATIM AND IN FULL, unclipped. See the sibling
        test `test_overlong_tool_result_is_head_tail_clipped_with_marker`
        below for the overlong case, which is clipped rather than dropped.

        The `type == "tool_result"` block-handling branch remains dead code
        on this orchestrator's own message-writing path (see the sibling
        test `test_tool_result_content_block_form_renders_omitted_placeholder`
        below, which is the only way to reach it).
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        tool_result_content = (
            '{"error": "Tool execution was cancelled by user", '
            '"cancelled": true, "tool": "write_file", '
            '"detail": "verbatim marker 8f3c2a should survive intact"}'
        )
        msg = {
            "role": "tool",
            "name": "write_file",
            "tool_call_id": "call_abc123",
            "content": tool_result_content,
        }

        # Default clip (2000 chars) -- well above this short result's
        # length, so it is untouched.
        result = _flatten_message_for_evaluator(msg)

        assert result == f"tool: {tool_result_content}"
        # Explicit, not-omitted, not-clipped assertions on the marker text.
        assert "verbatim marker 8f3c2a should survive intact" in result
        assert "[tool result omitted]" not in result
        assert "chars truncated" not in result

    def test_overlong_tool_result_is_head_tail_clipped_with_marker(self) -> None:
        """Evaluator input hygiene: a tool result LONGER than
        `tool_content_clip_chars` is clipped, not dropped -- it remains
        VISIBLE (bounded, not omitted), with BOTH the head (e.g. a command
        echo) and the tail (e.g. an exit-status line) retained, and an
        explicit marker naming how many characters were cut in between.

        Head+tail (not head-only) because verdict-bearing detail commonly
        sits at both ends of tool output.
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        head_marker = "HEAD_MARKER_running_command_abc123"
        tail_marker = "TAIL_MARKER_exit_status_0_success"
        middle_filler = "x" * 5000
        tool_result_content = f"{head_marker}{middle_filler}{tail_marker}"
        msg = {"role": "tool", "content": tool_result_content}

        result = _flatten_message_for_evaluator(msg, tool_content_clip_chars=200)

        assert result.startswith("tool: ")
        assert head_marker in result
        assert tail_marker in result
        assert "chars truncated" in result
        # The clip must actually have shrunk the content -- not merely
        # decorated the full text with a marker.
        assert len(result) < len(f"tool: {tool_result_content}")
        assert "[tool result omitted]" not in result

    def test_text_content_block_extracts_text_field(self) -> None:
        """Case 3: a `type: "text"` block -> its `text` field is extracted."""
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "here is my answer"}],
        }
        assert _flatten_message_for_evaluator(msg) == "assistant: here is my answer"

    def test_tool_call_block_includes_clipped_arguments(
        self,
    ) -> None:
        """GAP CLOSED (evaluator input hygiene): a `type: "tool_call"`
        block used to render only `[called tool: NAME]`, dropping the
        call's arguments entirely. The evaluator could see WHICH tool ran
        but never WHAT it was asked to do -- a plausible cause of observed
        confabulation.

        Now `block["input"]` (the tool-call-argument dict per the
        ToolCallBlock content-block schema) IS included, JSON-rendered and
        clipped via the same `tool_content_clip_chars` bound as tool
        results. This test uses a short argument dict (well under the
        default clip) so it should appear verbatim, unclipped.
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        argument_marker = "/tmp/argument-marker-should-now-be-visible.txt"
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "name": "write_file",
                    "input": {"path": argument_marker, "content": "..."},
                }
            ],
        }

        result = _flatten_message_for_evaluator(msg)

        assert result.startswith("assistant: [called tool: write_file args:")
        assert argument_marker in result
        assert "chars truncated" not in result

    def test_tool_call_block_overlong_arguments_are_clipped(self) -> None:
        """Evaluator input hygiene: tool-call arguments longer than
        `tool_content_clip_chars` are clipped (head+tail, with a marker),
        not shipped in full and not dropped back to the pre-hygiene
        name-only rendering.
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        head_marker = "HEAD_ARG_MARKER"
        tail_marker = "TAIL_ARG_MARKER"
        msg = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "name": "write_file",
                    "input": {
                        "path": head_marker,
                        "content": "y" * 5000,
                        "trailer": tail_marker,
                    },
                }
            ],
        }

        result = _flatten_message_for_evaluator(msg, tool_content_clip_chars=200)

        assert "[called tool: write_file args:" in result
        assert "chars truncated" in result
        # The tool name itself is always visible even when args are clipped.
        assert "write_file" in result

    def test_tool_call_block_missing_name_falls_back_to_question_mark(
        self,
    ) -> None:
        """`block.get('name', '?')` fallback when a tool_call block has no
        `name` key at all.
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        msg = {"role": "assistant", "content": [{"type": "tool_call"}]}
        assert _flatten_message_for_evaluator(msg) == "assistant: [called tool: ?]"

    def test_tool_result_content_block_form_renders_omitted_placeholder(
        self,
    ) -> None:
        """Case 5: a `{"type": "tool_result", ...}` content BLOCK (as
        opposed to a plain-string `role="tool"` message) renders the
        `[tool result omitted]` placeholder regardless of what the block
        actually contains.

        NOTE: this orchestrator's own tool-result writes never take this
        shape (see the CRUX test above) -- this branch is only reachable
        via a message constructed directly in this block form, which is not
        how __init__.py:2313-2424 write tool results. It is effectively
        dead code on this orchestrator's own path.
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        msg = {
            "role": "tool",
            "content": [
                {"type": "tool_result", "content": "this text must not appear"}
            ],
        }

        result = _flatten_message_for_evaluator(msg)

        assert result == "tool: [tool result omitted]"
        assert "this text must not appear" not in result

    def test_thinking_redacted_thinking_and_reasoning_blocks_are_dropped(
        self,
    ) -> None:
        """Case 6: `thinking` / `redacted_thinking` / `reasoning` blocks are
        intentionally skipped entirely -- none of their content, nor any
        placeholder for them, appears in the output.
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "internal chain of thought A"},
                {"type": "redacted_thinking", "data": "opaque redacted blob B"},
                {"type": "reasoning", "text": "internal reasoning trace C"},
                {"type": "text", "text": "the only visible answer"},
            ],
        }

        result = _flatten_message_for_evaluator(msg)

        assert result == "assistant: the only visible answer"
        assert "internal chain of thought A" not in result
        assert "opaque redacted blob B" not in result
        assert "internal reasoning trace C" not in result

    def test_mixed_content_blocks_joined_with_newline_empty_parts_skipped(
        self,
    ) -> None:
        """Case 7: multiple blocks of different types are joined with `\\n`,
        and any block that contributes an empty string (e.g. a `text` block
        with empty text) is filtered out of the join -- not preserved as a
        blank line. Non-dict list entries are silently skipped too (line
        108's `continue`).
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "dropped"},
                {"type": "text", "text": "first visible line"},
                {"type": "text", "text": ""},  # empty -> filtered, no blank line
                "not-a-dict-block",  # non-dict entry -> skipped via `continue`
                {"type": "tool_call", "name": "search"},
            ],
        }

        result = _flatten_message_for_evaluator(msg)

        assert result == "assistant: first visible line\n[called tool: search]"

    def test_missing_content_key_and_missing_role_key_fallbacks(self) -> None:
        """Case 8a: no `content` key -> defaults to `""` -> the function's
        final `if text else ""` guard means the WHOLE render is `""` (not
        `"role: "`).
        Case 8b: no `role` key -> defaults to `"unknown"`.
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        assert _flatten_message_for_evaluator({"role": "user"}) == ""
        assert _flatten_message_for_evaluator({"content": "hi"}) == "unknown: hi"

    def test_empty_string_content_renders_as_empty_string(self) -> None:
        """Case 8c: `content` explicitly `""` also hits the falsy-text
        short-circuit and returns `""`, same as a missing key.
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        assert _flatten_message_for_evaluator({"role": "user", "content": ""}) == ""

    def test_non_str_non_list_content_falls_back_to_str_of_content(self) -> None:
        """Case 9: a `content` value that is neither `str` nor `list` (e.g.
        a `dict` or an `int`) falls through to the `else: text = str(content)`
        branch (line 121) -- the raw Python `str()` rendering, not an error.
        """
        from amplifier_module_loop_streaming import _flatten_message_for_evaluator

        dict_msg = {"role": "assistant", "content": {"unexpected": "shape"}}
        assert _flatten_message_for_evaluator(dict_msg) == (
            "assistant: " + str({"unexpected": "shape"})
        )

        int_msg = {"role": "assistant", "content": 42}
        assert _flatten_message_for_evaluator(int_msg) == "assistant: 42"


# ---------------------------------------------------------------------------
# 14. Assembled evaluator transcript: truncation to the last 40 messages
#     (_GOAL_MAX_TRANSCRIPT_MESSAGES, __init__.py:266) and the "(truncated,
#     most recent messages shown)" marker (__init__.py:1665-1671, :1685).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEvaluatorTranscriptTruncation:
    async def test_over_40_messages_only_last_40_appear_with_truncated_marker(
        self,
    ) -> None:
        """With more than `_GOAL_MAX_TRANSCRIPT_MESSAGES` (40) messages in
        context, `_evaluate_goal` must truncate the transcript it sends to
        the evaluator to only the last 40, and must set the
        "(truncated, most recent messages shown)" marker in the prompt. The
        earliest messages must not appear anywhere in the sent prompt; the
        most recent 40 must all appear.
        """
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.eval_queue.append((True, "n/a"))

        # Delimited markers (not bare decimal indices) -- a bare index like
        # "message number 1" is a SUBSTRING of "message number 10"/"11"/...,
        # which would make an early naive version of this test pass for the
        # wrong reason. "MSG_<i>_END" cannot falsely match another index.
        total_messages = 45
        for i in range(total_messages):
            await ctx.add_message({"role": "user", "content": f"MSG_{i}_END"})

        await orch._evaluate_goal(
            "the condition",
            ctx,
            {"main": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        assert len(provider.eval_call_requests) == 1
        request = provider.eval_call_requests[0]
        user_message = next(m for m in request.messages if m.role == "user")
        prompt_text = user_message.content

        assert "(truncated, most recent messages shown)" in prompt_text

        # The oldest 5 messages (indices 0-4) were dropped by truncation to
        # the last 40 (indices 5-44).
        for dropped_index in range(total_messages - 40):
            assert f"MSG_{dropped_index}_END" not in prompt_text

        # The most recent 40 messages (indices 5-44) all survive.
        for kept_index in range(total_messages - 40, total_messages):
            assert f"MSG_{kept_index}_END" in prompt_text

    async def test_40_or_fewer_messages_no_truncation_marker(self) -> None:
        """Sanity counterpart: at or below the 40-message cap, no
        truncation occurs and the marker is absent -- confirms the marker
        is conditional on actual truncation, not always present.
        """
        orch = _make_orchestrator()
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.eval_queue.append((True, "n/a"))

        for i in range(40):
            await ctx.add_message({"role": "user", "content": f"message number {i}"})

        await orch._evaluate_goal(
            "the condition",
            ctx,
            {"main": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        request = provider.eval_call_requests[0]
        user_message = next(m for m in request.messages if m.role == "user")
        prompt_text = user_message.content

        assert "(truncated, most recent messages shown)" not in prompt_text
        assert "message number 0" in prompt_text
        assert "message number 39" in prompt_text

    async def test_transcript_char_budget_drops_oldest_messages_first(
        self,
    ) -> None:
        """Evaluator input hygiene: `goal_transcript_char_budget` is a
        backstop ABOVE the message-count window, applied NEWEST-FIRST --
        when it binds, the OLDEST messages within the window are dropped
        and the most recent (most verdict-relevant) ones survive.
        """
        orch = _make_orchestrator(
            {
                # Budget only large enough for the last couple of messages.
                "goal_transcript_char_budget": 60,
                "goal_tool_content_clip_chars": 0,  # disabled -- not exercised here
            }
        )
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.eval_queue.append((True, "n/a"))

        for i in range(10):
            await ctx.add_message({"role": "user", "content": f"MSG_{i}_END"})

        await orch._evaluate_goal(
            "the condition",
            ctx,
            {"main": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        request = provider.eval_call_requests[0]
        user_message = next(m for m in request.messages if m.role == "user")
        prompt_text = user_message.content

        assert "(truncated, most recent messages shown)" in prompt_text
        # The newest message must survive.
        assert "MSG_9_END" in prompt_text
        # The oldest message must have been dropped by the char budget.
        assert "MSG_0_END" not in prompt_text

    async def test_transcript_char_budget_disabled_by_non_positive_value(
        self,
    ) -> None:
        """`goal_transcript_char_budget <= 0` disables the budget entirely
        -- sanity counterpart proving the knob is opt-in, not a silent
        always-on truncation.
        """
        orch = _make_orchestrator({"goal_transcript_char_budget": 0})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.eval_queue.append((True, "n/a"))

        for i in range(10):
            await ctx.add_message({"role": "user", "content": f"MSG_{i}_END"})

        await orch._evaluate_goal(
            "the condition",
            ctx,
            {"main": provider},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        request = provider.eval_call_requests[0]
        user_message = next(m for m in request.messages if m.role == "user")
        prompt_text = user_message.content

        assert "(truncated, most recent messages shown)" not in prompt_text
        for i in range(10):
            assert f"MSG_{i}_END" in prompt_text


@pytest.mark.asyncio
class TestEvaluatorInputHygieneEndToEnd:
    """Empirical verification (see goal-build.md item 4): a real large tool
    result run through the ACTUAL assembly path (`_evaluate_goal`) produces
    a bounded evaluator prompt -- not merely a unit-level clip. Compares
    the hygiene knobs DISABLED (reproducing the pre-hygiene unbounded
    behavior exactly, since the only prior bound was message count) against
    the hygiene knobs at their defaults.
    """

    async def test_large_tool_result_bounded_vs_unbounded_prompt_size(
        self,
    ) -> None:
        large_tool_output = (
            "COMMAND_ECHO_START read_file /var/log/huge.log\n"
            + ("line of log output filler content here\n" * 6000)
            + "EXIT_STATUS_END code=0"
        )

        async def _run(config: dict) -> str:
            orch = _make_orchestrator(config)
            ctx = MockContext()
            hooks = MockHooks()
            coordinator = MockCoordinator()
            provider = FakeProvider()
            provider.eval_queue.append((True, "n/a"))

            await ctx.add_message({"role": "user", "content": "please read the log"})
            await ctx.add_message(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "name": "read_file",
                            "input": {"path": "/var/log/huge.log"},
                        }
                    ],
                }
            )
            await ctx.add_message(
                {
                    "role": "tool",
                    "name": "read_file",
                    "tool_call_id": "call_1",
                    "content": large_tool_output,
                }
            )

            await orch._evaluate_goal(
                "the log has been read",
                ctx,
                {"main": provider},
                hooks,  # type: ignore[arg-type]
                coordinator,  # type: ignore[arg-type]
            )

            request = provider.eval_call_requests[0]
            user_message = next(m for m in request.messages if m.role == "user")
            return user_message.content

        # BEFORE (knobs disabled -- reproduces the pre-hygiene behavior,
        # since a 40-message window was the ONLY bound that existed):
        before_prompt = await _run(
            {"goal_tool_content_clip_chars": 0, "goal_transcript_char_budget": 0}
        )
        # AFTER (default hygiene knobs):
        after_prompt = await _run({})

        before_size = len(before_prompt)
        after_size = len(after_prompt)

        print(
            f"\n[evaluator input hygiene] tool-result char count: "
            f"{len(large_tool_output)}\n"
            f"[evaluator input hygiene] BEFORE (unbounded) prompt size: "
            f"{before_size} chars\n"
            f"[evaluator input hygiene] AFTER (bounded) prompt size: "
            f"{after_size} chars\n"
        )

        # The unbounded run must actually contain the full tool output --
        # otherwise this isn't a fair "before" baseline.
        assert large_tool_output in before_prompt
        # The bounded run must be dramatically smaller.
        assert after_size < before_size
        assert after_size < 5000
        # But NOT empty / NOT silently dropped -- the clip marker and both
        # ends of the tool output must still be visible (visibility, not
        # omission).
        assert "chars truncated" in after_prompt
        assert "COMMAND_ECHO_START read_file /var/log/huge.log" in after_prompt
        assert "EXIT_STATUS_END code=0" in after_prompt
        # Tool-call arguments are now visible too.
        assert "/var/log/huge.log" in after_prompt
