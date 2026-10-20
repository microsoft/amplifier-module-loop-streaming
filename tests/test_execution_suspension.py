"""Settled boundaries preserve effects and native identity through remount."""

import asyncio
import json

import pytest

from amplifier_core import ToolResult
from amplifier_module_loop_streaming import ExecutionSuspended, StreamingOrchestrator
from amplifier_module_loop_streaming.suspension import ToolExecutionSuspended
from tests.test_durable_checkpoint import BatchProvider, MockContext, OrderedTool, _receipts
from tests.test_ephemeral_cache_persist_mode import MockCoordinator, ScriptedHooks


class Owner:
    def __init__(self, path, context):
        self.path, self.context = path, context
        self.hold = False
        self.claimed = set()

    def requested(self):
        return self.hold

    async def checkpoint(self, frame):
        self.path.write_text(json.dumps({
            "continuation": frame, "messages": await self.context.get_messages()
        }))
        return True

    async def claim(self, frame):
        if self.hold or frame["id"] in self.claimed:
            return False
        self.claimed.add(frame["id"])
        return True


def setup_owner(tmp_path):
    context, coordinator = MockContext(), MockCoordinator()
    owner = Owner(tmp_path / "checkpoint.json", context)
    coordinator.register_capability("execution.suspension", owner)
    coordinator.register_capability("execution.resume", owner.claim)
    return context, coordinator, owner


async def run(loop, provider, tool, context, coordinator, frame=None):
    args = (context, {"main": provider}, {"mock_tool": tool}, ScriptedHooks({}), coordinator)
    return await (loop.resume(frame, *args) if frame else loop.execute("work", *args))


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_at", ["hook", "context", "rate_limit"])
async def test_pause_during_preparation_preserves_exact_request_without_replaying_hooks(tmp_path, wait_at):
    context, coordinator, owner = setup_owner(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    provider, tool = BatchProvider(count=0), OrderedTool()
    hooks = ScriptedHooks({})
    config = {"max_iterations": 1, "reminder_placement": "tail"}
    loop = StreamingOrchestrator(config)
    original_hook = hooks.emit
    original_messages = context.get_messages_for_request
    original_delay = loop._apply_rate_limit_delay
    counts = {"hook": 0, "context": 0, "rate_limit": 0}

    async def hold(kind):
        counts[kind] += 1
        if wait_at == kind:
            entered.set()
            await release.wait()

    async def held_hook(name, payload=None):
        if name == "provider:request":
            await hold("hook")
        return await original_hook(name, payload)

    async def held_messages(**kwargs):
        await hold("context")
        return await original_messages(**kwargs)

    async def held_delay(*args):
        await hold("rate_limit")
        return await original_delay(*args)

    hooks.emit, context.get_messages_for_request = held_hook, held_messages
    loop._apply_rate_limit_delay = held_delay
    task = asyncio.create_task(loop.execute("original request", context, {"main": provider},
                                          {"mock_tool": tool}, hooks, coordinator))
    await asyncio.wait_for(entered.wait(), 2)
    owner.hold = True
    release.set()
    with pytest.raises(ExecutionSuspended) as stopped:
        await task
    frame = stopped.value.continuation
    assert frame["phase"] == "prepared_request" and frame["iteration"] == 1
    assert provider.calls == 0
    before = dict(counts)
    owner.hold = False
    assert await StreamingOrchestrator(config).resume(
        frame, context, {"main": provider}, {"mock_tool": tool}, hooks, coordinator) == "done"
    assert counts == before and provider.calls == 1
    assert len([row for row in context._messages if row["role"] == "user" and
                row["content"] == "original request"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at", [
    "turn-return", "eval-return", "judge-return",
    "eval-hook", "judge-hook", "summary-hook", "continuation-hook",
])
async def test_goal_safe_step_resume_preserves_assessment_and_continuation_accounting(tmp_path, pause_at):
    from tests.test_goal_loop import FakeProvider, MockHooks, MockTurnResponse

    context, coordinator, owner = setup_owner(tmp_path)
    goal = {"condition": "Finish work", "turns_used": 0, "last_reason": None, "cap": 2}
    coordinator.session_state["goal"] = goal
    config = {"max_iterations": 3, "goal_stall_threshold": 1}
    calls, fired = [], False

    def pause_once(where):
        nonlocal fired
        if pause_at == where and not fired:
            owner.hold, fired = True, True

    class Provider(FakeProvider):
        async def complete(self, request, **kwargs):
            system = next((m.content for m in request.messages if m.role == "system"), "")
            kind = ("eval" if "tool-less evaluator" in system else
                    "judge" if "tool-less judge" in system else
                    "summary" if "single, short line for a developer" in system else "turn")
            calls.append(kind)
            result = await super().complete(request, **kwargs)
            pause_once(kind + "-return")
            return result

    class Hooks(MockHooks):
        async def emit(self, name, payload=None):
            if name == "provider:request" and payload.get("iteration") == 0:
                kind = ("eval" if len(calls) == 1 else "judge" if calls[-1] == "eval" else "summary")
                pause_once(kind + "-hook")
            if name == "orchestrator:goal_progress" and payload.get("state") == "continuing":
                pause_once("continuation-hook")
            return await super().emit(name, payload)

    provider, hooks = Provider(), Hooks()
    provider.turn_queue = [MockTurnResponse("first answer"), MockTurnResponse("second answer")]
    provider.eval_queue = [(False, "next step"), (False, "remaining work")]
    provider.judge_queue = [(False, "new material")]
    with pytest.raises(ExecutionSuspended) as paused:
        await StreamingOrchestrator(config).execute(
            "original work", context, {"main": provider}, {}, hooks, coordinator)
    assert fired and paused.value.continuation["phase"] == "goal"
    count_before = len(calls)
    await asyncio.sleep(0)
    assert len(calls) == count_before
    owner.hold = False
    result = await StreamingOrchestrator(config).resume(
        paused.value.continuation, context, {"main": provider}, {}, hooks, coordinator)
    assert result == "second answer"
    assert calls == ["turn", "eval", "turn", "eval", "judge", "summary"]
    assert goal["turns_used"] == 2 and goal["continuations"] == 1
    assert goal["reasons"] == ["next step", "remaining work"]
    assert coordinator.session_state["goal"] is None
    assert [row["goal_final"] for row in hooks.orchestrator_complete_events()] == [False, True]
    assert len([row for row in context._messages if row["role"] == "user" and
                row["content"] == "original work"]) == 1


@pytest.mark.asyncio
async def test_model_finishes_then_unstarted_tools_suspend_and_remount_once(tmp_path):
    context, coordinator, owner = setup_owner(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldProvider(BatchProvider):
        async def complete(self, request, **kwargs):
            if self.calls == 0:
                entered.set()
                await release.wait()
            return await super().complete(request, **kwargs)

    provider, tool = HeldProvider(), OrderedTool()
    config = {"max_iterations": 3}
    task = asyncio.create_task(run(StreamingOrchestrator(config), provider, tool, context, coordinator))
    await entered.wait()
    owner.hold = True
    assert not task.done()
    assert tool.completed == []
    release.set()
    with pytest.raises(ExecutionSuspended) as stopped:
        await task
    assert stopped.value.continuation["phase"] == "tools"
    assert provider.calls == 1 and tool.completed == []
    # The resumed engine and context are new objects loaded only from disk.
    saved = json.loads(owner.path.read_text())
    restored = MockContext()
    restored._messages = saved["messages"]
    owner.context, owner.hold = restored, False
    assert await run(StreamingOrchestrator(config), provider, tool, restored, coordinator,
                     saved["continuation"]) == "done"
    assert provider.calls == 2
    assert tool.completed == [1, 0]
    assert len([m for m in restored._messages if m["role"] == "user" and m["content"] == "work"]) == 1
    assert _receipts(restored) == [("call-0", "result-0"), ("call-1", "result-1")]
    # Lost reply/duplicate request cannot admit the same continuation again.
    with pytest.raises(ValueError, match="history changed"):
        await run(StreamingOrchestrator(config), provider, tool, restored, coordinator, saved["continuation"])
    assert tool.completed == [1, 0] and provider.calls == 2


@pytest.mark.asyncio
async def test_tool_batch_settles_in_original_order_before_pause(tmp_path):
    context, coordinator, owner = setup_owner(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldTool(OrderedTool):
        async def execute(self, arguments):
            if arguments["index"] == 0:
                entered.set()
                await release.wait()
            return await super().execute(arguments)

    provider, tool = BatchProvider(), HeldTool()
    config = {"max_iterations": 3}
    task = asyncio.create_task(run(StreamingOrchestrator(config), provider, tool, context, coordinator))
    await entered.wait()
    owner.hold = True
    assert not task.done() and provider.calls == 1
    # A hung operation remains waiting; requesting pause does not cancel it.
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(ExecutionSuspended) as stopped:
        await task
    frame = stopped.value.continuation
    assert frame["phase"] == "provider" and frame["iteration"] == 1
    assert _receipts(context) == [("call-0", "result-0"), ("call-1", "result-1")]
    assert frame["turn"]["_llm_calls_this_turn"] == 1
    owner.hold = False
    assert await run(StreamingOrchestrator(config), provider, tool, context, coordinator, frame) == "done"
    assert tool.completed == [1, 0] and provider.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["false", "disk"])
async def test_checkpoint_failure_never_starts_unstarted_effect(tmp_path, failure):
    context, coordinator, owner = setup_owner(tmp_path)
    provider, tool = BatchProvider(), OrderedTool()
    owner.hold = True

    async def fail(frame):
        if failure == "disk":
            raise OSError("disk full")
        return False

    owner.checkpoint = fail
    with pytest.raises((OSError, RuntimeError)):
        await run(StreamingOrchestrator({}), provider, tool, context, coordinator)
    assert provider.calls == 0 and tool.completed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["history", "configuration", "goal", "admission", "duplicate", "stop"])
async def test_current_authority_and_snapshot_win_over_resume(tmp_path, change):
    context, coordinator, owner = setup_owner(tmp_path)
    provider, tool = BatchProvider(), OrderedTool()
    config = {"max_iterations": 3}
    owner.hold = True
    with pytest.raises(ExecutionSuspended) as stopped:
        await run(StreamingOrchestrator(config), provider, tool, context, coordinator)
    frame = stopped.value.continuation
    owner.hold = False
    if change == "history":
        await context.add_message({"role": "user", "content": "changed"})
    elif change == "configuration":
        config = {"max_iterations": 1}
    elif change == "goal":
        coordinator.session_state["goal"] = {"condition": "replacement"}
    elif change == "admission":
        owner.hold = True
    elif change == "stop":
        coordinator.cancellation.is_cancelled = True
    else:
        owner.claimed.add(frame["id"])
    with pytest.raises((ValueError, RuntimeError)):
        await run(StreamingOrchestrator(config), provider, tool, context, coordinator, frame)
    assert provider.calls == 0 and tool.completed == []


@pytest.mark.asyncio
async def test_parent_and_child_resume_owned_continuations_without_replaying_sibling(tmp_path):
    parent_dir, child_dir = tmp_path / "parent", tmp_path / "child"
    parent_dir.mkdir()
    child_dir.mkdir()
    context, coordinator, owner = setup_owner(parent_dir)
    child_context, child_coordinator, child_owner = setup_owner(child_dir)
    entered, release = asyncio.Event(), asyncio.Event()
    child_provider = BatchProvider(count=1)

    class ChildEffect(OrderedTool):
        async def execute(self, arguments):
            entered.set()
            await release.wait()
            return await super().execute(arguments)

    child_tool = ChildEffect(count=1)

    class DelegateAndSibling(OrderedTool):
        def __init__(self):
            super().__init__(count=1)
            self.spawned, self.resumed, self.sibling_effects = 0, 0, 0

        async def execute(self, arguments):
            if arguments["index"] == 1:
                self.sibling_effects += 1
                return ToolResult(success=True, output="sibling result")
            self.spawned += 1
            try:
                result = await run(StreamingOrchestrator({"max_iterations": 3}),
                                   child_provider, child_tool, child_context, child_coordinator)
            except ExecutionSuspended as suspended:
                raise ToolExecutionSuspended({
                    "version": 1, "kind": "fixture-delegate", "child": suspended.continuation
                }) from None
            return ToolResult(success=True, output=result)

        async def resume_execution(self, continuation):
            self.resumed += 1
            saved = json.loads(child_owner.path.read_text())
            restored_child = MockContext()
            restored_child._messages = saved["messages"]
            result = await run(StreamingOrchestrator({"max_iterations": 3}),
                               child_provider, child_tool, restored_child, child_coordinator,
                               continuation["child"])
            return ToolResult(success=True, output=result)

    provider, tool = BatchProvider(), DelegateAndSibling()
    config = {"max_iterations": 3}
    task = asyncio.create_task(run(StreamingOrchestrator(config), provider, tool, context, coordinator))
    await entered.wait()
    owner.hold = child_owner.hold = True
    release.set()
    with pytest.raises(ExecutionSuspended) as stopped:
        await task
    frame = stopped.value.continuation
    assert frame["phase"] == "settling_tools"
    assert provider.calls == child_provider.calls == 1
    assert child_tool.completed == [0] and tool.sibling_effects == 1
    assert _receipts(context) == []  # A partial batch never masquerades as complete.
    saved = json.loads(owner.path.read_text())
    restored_parent = MockContext()
    restored_parent._messages = saved["messages"]
    owner.hold = child_owner.hold = False
    result = await run(StreamingOrchestrator(config), provider, tool, restored_parent,
                       coordinator, saved["continuation"])
    assert result == "done"
    assert tool.spawned == tool.resumed == tool.sibling_effects == 1
    assert child_tool.completed == [0]
    assert provider.calls == child_provider.calls == 2
    assert [identity for identity, _ in _receipts(restored_parent)] == ["call-0", "call-1"]
