"""Live hosts may change a goal while an ordinary turn or utility call awaits.

Real execute() flow, deterministic providers, and hooks; no network/model calls.
"""
import asyncio

import pytest

from .test_goal_loop import (
    FakeProvider, MockContext, MockCoordinator, MockHooks, MockTurnResponse,
    MockTool, MockToolCall, _make_orchestrator,
)


def goal(condition="original", **extra):
    return {"condition": condition, "turns_used": 0, "last_reason": None,
            "cap": None, **extra}


class ChangingProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.callbacks = {}
        self.calls = []

    async def complete(self, request, **kwargs):
        text = next((m.content for m in request.messages if m.role == "system"), "")
        text = text if isinstance(text, str) else ""
        kind = ("eval" if "tool-less evaluator" in text else
                "judge" if "tool-less judge" in text else
                "summary" if "single, short line for a developer" in text else "turn")
        self.calls.append(kind)
        # Yield at the same boundary at which real network calls permit host
        # control changes, then apply one-shot deterministic host actions.
        await asyncio.sleep(0)
        action = self.callbacks.pop(kind, None)
        if action:
            action()
        return await super().complete(request, **kwargs)


async def run(provider, coordinator, hooks=None, config=None, tools=None):
    hooks = hooks or MockHooks()
    engine = _make_orchestrator(config)
    result = await engine.execute("work", MockContext(), {"main": provider},
                                  tools or {}, hooks, coordinator)
    return engine, hooks, result


@pytest.mark.asyncio
@pytest.mark.parametrize("already_present", [False, True])
async def test_partial_goal_at_entry_or_created_mid_turn_gets_full_defaults(already_present):
    coordinator, provider, hooks = MockCoordinator(), ChangingProvider(), MockHooks()
    active = goal(task_id="task", task_revision=1)
    if already_present:
        coordinator.session_state["goal"] = active
    else:
        provider.callbacks["turn"] = lambda: coordinator.session_state.update(goal=active)
    # Verify the answer's complete event is deferred until its new goal has
    # been evaluated, not prematurely published as the final completion.
    provider.callbacks["eval"] = lambda: assert_no_completions(hooks)
    provider.turn_queue.append(MockTurnResponse(text="answer"))
    provider.eval_queue.append((True, "done"))
    engine, hooks, result = await run(provider, coordinator, hooks)
    assert result == "answer"
    assert active["progress_evidence"]
    assert active["turns_used"] == 1
    assert coordinator.session_state["goal"] is None
    assert engine._pending_orchestrator_complete is None
    assert [e["state"] for e in hooks.goal_progress_events()] == ["achieved"]
    assert [(e["goal_turn"], e["goal_final"]) for e in hooks.orchestrator_complete_events()] == [(1, True)]


def assert_no_completions(hooks):
    assert hooks.orchestrator_complete_events() == []


@pytest.mark.asyncio
async def test_goal_added_during_tool_turn_keeps_evidence_and_continues():
    coordinator, provider = MockCoordinator(), ChangingProvider()
    active = goal()
    provider.callbacks["turn"] = lambda: coordinator.session_state.update(goal=active)
    provider.turn_queue.extend([MockTurnResponse(tool_calls=[MockToolCall()]),
                               MockTurnResponse(text="first"), MockTurnResponse(text="second")])
    provider.eval_queue.extend([(False, "needs another step"), (True, "finished")])
    _, hooks, result = await run(provider, coordinator, tools={"mock_tool": MockTool()})
    assert result == "second"
    assert active["continuations"] == 1
    assert active["turns_used"] == 2
    assert active["progress_evidence"][0]["tools"]
    assert [e["goal_final"] for e in hooks.orchestrator_complete_events()] == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [True, False, "error"])
@pytest.mark.parametrize("change", ["replace", "revise", "clear", "pause"])
async def test_obsolete_evaluation_cannot_complete_clear_or_continue_new_goal(outcome, change):
    coordinator, provider = MockCoordinator(), ChangingProvider()
    original = goal(task_id="task", task_revision=1)
    replacement = goal("revised", task_id="task", task_revision=2)
    coordinator.session_state["goal"] = original
    provider.turn_queue.append(MockTurnResponse(text="already saved answer"))
    provider.eval_queue.extend([(bool(outcome), "obsolete"), (True, "current result")])

    def update():
        if change in {"clear", "pause"}:
            coordinator.session_state["goal"] = None
        elif change == "replace":
            coordinator.session_state["goal"] = replacement
        else:
            original.update(condition="revised", task_revision=2)
        if outcome == "error":
            provider.eval_queue.pop(0)
            raise RuntimeError("obsolete evaluator failed")

    provider.callbacks["eval"] = update
    _, hooks, result = await run(provider, coordinator)
    assert result == "already saved answer"
    assert "obsolete" not in original["reasons"]
    assert provider.calls.count("turn") == 1
    assert hooks.goal_progress_events() == []
    assert provider.calls.count("eval") == 1
    if change in {"clear", "pause"}:
        assert coordinator.session_state["goal"] is None
    else:
        expected = replacement if change == "replace" else original
        assert coordinator.session_state["goal"] is expected
        assert expected.get("reasons", []) == []
    assert [e["goal_final"] for e in hooks.orchestrator_complete_events()] == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["replace", "clear"])
@pytest.mark.parametrize("boundary", ["judge", "summary"])
async def test_stale_judge_and_terminal_summary_do_not_end_successor_goal(change, boundary):
    coordinator, provider = MockCoordinator(), ChangingProvider()
    original = goal(cap=1 if boundary == "summary" else None)
    successor = goal("successor")
    coordinator.session_state["goal"] = original
    provider.turn_queue.extend([MockTurnResponse(text="first"), MockTurnResponse(text="second")])
    provider.eval_queue.extend([(False, "blocked"), (False, "blocked"), (True, "successor done")]
                               if boundary == "judge" else [(False, "blocked"), (True, "successor done")])
    provider.judge_queue.append((True, "stalled"))
    provider.callbacks[boundary] = lambda: coordinator.session_state.update(
        goal=successor if change == "replace" else None)
    _, hooks, _ = await run(provider, coordinator, config={"goal_stall_threshold": 1})
    events = hooks.goal_progress_events()
    assert not any(e["state"] in {"stalled", "cap_hit", "error"} for e in events)
    if change == "replace":
        assert coordinator.session_state["goal"] is successor
        assert successor.get("reasons", []) == []
    else:
        assert coordinator.session_state["goal"] is None
    assert hooks.orchestrator_complete_events()[-1]["goal_final"] is True


@pytest.mark.asyncio
async def test_pause_in_continuing_progress_hook_stops_before_next_turn():
    coordinator, provider = MockCoordinator(), ChangingProvider()
    coordinator.session_state["goal"] = goal()
    provider.turn_queue.append(MockTurnResponse(text="first"))
    provider.eval_queue.append((False, "needs work"))

    class PausingHooks(MockHooks):
        async def emit(self, event_name, payload=None):
            result = await super().emit(event_name, payload)
            if event_name == "orchestrator:goal_progress":
                coordinator.session_state["goal"] = None
            return result

    _, hooks, _ = await run(provider, coordinator, PausingHooks())
    assert provider.calls == ["turn", "eval"]
    assert coordinator.session_state["goal"] is None
    assert [e["goal_final"] for e in hooks.orchestrator_complete_events()] == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["eval", "judge", "summary"])
@pytest.mark.parametrize("mode", ["flag", "raise"])
async def test_cancellation_during_utility_call_stops_and_clears_current_goal(boundary, mode):
    coordinator, provider, hooks = MockCoordinator(), ChangingProvider(), MockHooks()
    active = goal(cap=1 if boundary == "summary" else None)
    coordinator.session_state["goal"] = active
    provider.turn_queue.extend([MockTurnResponse(text="first"), MockTurnResponse(text="second")])
    provider.eval_queue.extend([(False, "blocked"), (False, "blocked")])
    provider.judge_queue.append((True, "stalled"))

    def cancel():
        if mode == "raise":
            raise asyncio.CancelledError()
        coordinator.cancellation.is_cancelled = True

    provider.callbacks[boundary] = cancel
    if mode == "raise":
        with pytest.raises(asyncio.CancelledError):
            await run(provider, coordinator, hooks, config={"goal_stall_threshold": 1})
    else:
        await run(provider, coordinator, hooks, config={"goal_stall_threshold": 1})
    assert coordinator.session_state["goal"] is None
    assert hooks.goal_progress_events()[-1]["state"] == "cancelled"
    assert hooks.orchestrator_complete_events()[-1]["goal_final"] is True
    assert provider.calls.count("turn") == (2 if boundary == "judge" else 1)
    if boundary == "eval":
        assert active["reasons"] == []


@pytest.mark.asyncio
async def test_cancellation_of_obsolete_evaluator_does_not_clear_successor():
    coordinator, provider, hooks = MockCoordinator(), ChangingProvider(), MockHooks()
    coordinator.session_state["goal"] = goal()
    successor = goal("replacement")
    provider.turn_queue.append(MockTurnResponse(text="saved answer"))

    def replace_and_cancel():
        coordinator.session_state["goal"] = successor
        raise asyncio.CancelledError()

    provider.callbacks["eval"] = replace_and_cancel
    with pytest.raises(asyncio.CancelledError):
        await run(provider, coordinator, hooks)
    assert coordinator.session_state["goal"] is successor
    assert hooks.goal_progress_events() == []
    assert hooks.orchestrator_complete_events()[-1]["goal_final"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["replace", "revise"])
async def test_goal_revised_during_conversation_is_evaluated_with_current_condition(change):
    coordinator, provider = MockCoordinator(), ChangingProvider()
    original = goal(task_revision=1)
    replacement = goal("revised", task_revision=2)
    coordinator.session_state["goal"] = original

    def revise():
        if change == "replace":
            coordinator.session_state["goal"] = replacement
        else:
            original.update(condition="revised", task_revision=2)

    provider.callbacks["turn"] = revise
    provider.turn_queue.append(MockTurnResponse(text="answer"))
    provider.eval_queue.append((True, "revised condition met"))
    _, hooks, result = await run(provider, coordinator)
    assert result == "answer"
    active = replacement if change == "replace" else original
    assert active["progress_evidence"]
    assert active["reasons"] == ["revised condition met"]
    assert hooks.goal_progress_events()[-1]["condition"] == "revised"
    assert len(hooks.orchestrator_complete_events()) == 1
    assert coordinator.session_state["goal"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["orchestrator:complete", "orchestrator:goal_progress"])
async def test_utility_cancellation_preserves_cancellation_when_diagnostics_fail(event):
    coordinator, provider = MockCoordinator(), ChangingProvider()
    coordinator.session_state["goal"] = goal()
    provider.turn_queue.append(MockTurnResponse(text="saved answer"))
    cancellation = asyncio.CancelledError()

    def cancel():
        raise cancellation

    class FailingHooks(MockHooks):
        async def emit(self, event_name, payload=None):
            result = await super().emit(event_name, payload)
            if event_name == event:
                raise RuntimeError("diagnostic hook failed")
            return result

    hooks = FailingHooks()
    provider.callbacks["eval"] = cancel
    with pytest.raises(asyncio.CancelledError) as raised:
        await run(provider, coordinator, hooks)
    assert raised.value is cancellation
    assert coordinator.session_state["goal"] is None
    assert hooks.goal_progress_events()[-1]["state"] == "cancelled"
