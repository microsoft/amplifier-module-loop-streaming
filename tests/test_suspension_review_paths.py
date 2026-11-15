"""Public-entry regressions for the suspension review on PR67 / 1c2c283."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from amplifier_core import ContextLengthError, HookResult, ToolResult
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_module_loop_streaming import ExecutionSuspended, StreamingOrchestrator
from amplifier_module_loop_streaming.suspension import ToolExecutionSuspended
from tests.test_approved_dispatch import environment  # noqa: F401
from tests.test_measured_compaction_runtime import _Transaction
from tests.test_provider_overflow_recovery import RecoveringStreamProvider, _overflow

pytestmark = pytest.mark.asyncio


class DurableOwner:
    def __init__(self, path, context):
        self.path, self.context = path, context
        self.hold = False
        self.claims = []
        self.consumed = set()

    def requested(self):
        return self.hold

    async def checkpoint(self, frame):
        self.path.write_text(json.dumps({"frame": frame, "messages": await self.context.get_messages()}))
        return True

    async def claim(self, frame):
        self.claims.append(frame["id"])
        if self.hold or frame["id"] in self.consumed:
            return False
        self.consumed.add(frame["id"])
        return True


class ContinuingTool:
    description = "Offline public-resume fixture"
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, name):
        self.name = name
        self.execute_calls = self.resume_calls = 0
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.finished = False

    async def execute(self, arguments):
        self.execute_calls += 1
        raise ToolExecutionSuspended({"kind": "fixture", "name": self.name})

    async def resume_execution(self, continuation):
        assert continuation == {"kind": "fixture", "name": self.name}
        self.resume_calls += 1
        self.entered.set()
        await self.release.wait()
        self.finished = True
        return ToolResult(success=True, output={"known_result": self.name})


async def suspended_batch(environment, tmp_path):
    _, coordinator, loop, _ = environment
    context = coordinator.get("context")
    loop.config["max_iterations"] = 3
    owner = DurableOwner(tmp_path / "batch.json", context)
    coordinator.register_capability("execution.suspension", owner)
    coordinator.register_capability("execution.resume", owner.claim)
    first, second = ContinuingTool("first"), ContinuingTool("second")
    await coordinator.mount("tools", first, name="first")
    await coordinator.mount("tools", second, name="second")

    class Provider:
        name, config = "fixture", {}
        calls = 0
        def get_info(self):
            return SimpleNamespace(defaults={"model": "fixture"})
        def parse_tool_calls(self, response):
            return response.tool_calls or []
        async def complete(self, request, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ChatResponse(content=[], tool_calls=[
                    ToolCall(id="original-first", name="first", arguments={}),
                    ToolCall(id="original-second", name="second", arguments={}),
                ])
            return ChatResponse(content=[TextBlock(text="done")])

    provider = Provider()
    args = (context, {"fixture": provider}, coordinator.get("tools"), coordinator.hooks, coordinator)
    with pytest.raises(ExecutionSuspended):
        await loop.execute("original input", *args)
    frame = json.loads(owner.path.read_text())["frame"]
    assert frame["phase"] == "settling_tools"
    assert first.execute_calls == second.execute_calls == 1
    return coordinator, loop.config, owner, first, second, provider, args, frame


async def test_public_resume_preflights_all_completion_owners(environment, tmp_path):
    coordinator, config, owner, first, second, provider, args, frame = await suspended_batch(environment, tmp_path)
    second.resume_execution = None
    restored = StreamingOrchestrator(dict(config))
    scheduled = []
    original = restored._execute_tool_only
    async def observe_dispatch(*pos, **kw):
        scheduled.append(asyncio.current_task())
        return await original(*pos, **kw)
    restored._execute_tool_only = observe_dispatch
    try:
        with pytest.raises(ValueError, match="explicit execution resumption"):
            await restored.resume(frame, *args)
        assert scheduled == [], "invalid owner must be rejected before scheduling any sibling"
        assert not first.entered.is_set() and first.resume_calls == 0
        assert first.execute_calls == second.execute_calls == provider.calls == 1
        # Refusal does not silently release or reuse the consumed host claim.
        assert owner.claims == [frame["id"]]
        second.resume_execution = ContinuingTool.resume_execution.__get__(second)
        with pytest.raises(RuntimeError, match="admission refused"):
            await StreamingOrchestrator(dict(config)).resume(frame, *args)
        assert owner.claims == [frame["id"], frame["id"]]
        assert first.resume_calls == second.resume_calls == 0
    finally:
        # The red implementation can leave a live sibling; always observe it.
        first.release.set()
        second.release.set()
        if scheduled:
            await asyncio.wait_for(asyncio.gather(*scheduled, return_exceptions=True), 2)


async def test_public_resume_observes_sibling_after_unexpected_dispatch_error(environment, tmp_path):
    coordinator, config, owner, first, second, provider, args, frame = await suspended_batch(environment, tmp_path)
    failed = asyncio.Event()
    dispatch_tasks = []
    class UnexpectedDispatch(StreamingOrchestrator):
        async def _execute_tool_only(self, call, *pos, **kw):
            dispatch_tasks.append(asyncio.current_task())
            if call.name == "second":
                await first.entered.wait()
                failed.set()
                raise RuntimeError("unexpected dispatch setup failure")
            return await super()._execute_tool_only(call, *pos, **kw)
    restored = UnexpectedDispatch(dict(config))
    root = asyncio.create_task(restored.resume(frame, *args))
    try:
        await asyncio.wait_for(failed.wait(), 2)
        # Allow gather's exception wakeup to run without waiting for completion.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not root.done(), "a started sibling must settle before the batch exits"
        first.release.set()
        assert await asyncio.wait_for(root, 2) == "done"
        rows = [row for row in await args[0].get_messages() if row.get("role") == "tool"]
        assert [row["tool_call_id"] for row in rows] == ["original-first", "original-second"]
        known, unknown = [json.loads(row["content"]) for row in rows]
        assert known["success"] is True and known["output"] == {"known_result": "first"}
        assert unknown["success"] is False and unknown["output"]["status"] == "unknown"
        assert unknown["output"]["retryable"] is False
        assert first.finished and first.resume_calls == 1 and second.resume_calls == 0
        assert first.execute_calls == second.execute_calls == 1
        assert owner.claims == [frame["id"]] and provider.calls == 2
    finally:
        first.release.set()
        second.release.set()
        await asyncio.wait_for(asyncio.gather(root, *dispatch_tasks, return_exceptions=True), 2)


@pytest.mark.parametrize("wait_at", ["recovery", "preflight", "rate_limit"])
@pytest.mark.parametrize("retry_fails", [False, True])
async def test_public_stream_recovery_hold_retains_prepared_retry(environment, tmp_path, wait_at, retry_fails):
    _, coordinator, loop, _ = environment
    context = coordinator.get("context")
    owner = DurableOwner(tmp_path / "stream.json", context)
    coordinator.register_capability("execution.suspension", owner)
    coordinator.register_capability("execution.resume", owner.claim)
    entered, release = asyncio.Event(), asyncio.Event()
    views, delays, requests, hold_at_send, closed, transactions = [], [], [], [], [], []
    old = "old history " * 100
    await context.add_message({"role": "assistant", "content": old})

    async def gate(kind):
        if wait_at == kind:
            entered.set()
            await release.wait()

    async def measured_view(*, provider, retain_contents, count_view, reason=None):
        # Deliberately opt into reactive measured repair. A legacy hard_fit-only
        # fixture must now decline, so it cannot exercise a prepared retry hold.
        views.append((reason, list(retain_contents)))
        rows = await context.get_messages()
        if reason is not None:
            assert reason == "provider_overflow"
            await gate("recovery")
            rows = [row for row in rows if row.get("content") != old]
        transaction = _Transaction()
        transactions.append(transaction)
        try:
            attempt = await count_view(rows)
            return {
                "base_view": rows,
                "final_attempt": attempt,
                "outcome": "compacted" if reason else "not_needed",
                "count_calls": 1,
                "transaction": transaction,
            }
        except BaseException:
            transaction.rollback()
            raise
    coordinator.register_capability("context.measured_request_view", measured_view)

    class Provider(RecoveringStreamProvider):
        async def request_budget(self, request, *, context_estimate, request_options=None):
            result = super().request_budget(request, context_estimate=context_estimate, request_options=request_options)
            if len(self.budget_options) == 2:
                await gate("preflight")
            return result
        async def stream(self, request, *, tools):
            self.stream_calls += 1
            self.requests.append(request)
            hold_at_send.append(owner.hold)
            try:
                if self.stream_calls == 1 or (retry_fails and self.stream_calls == 2):
                    raise ContextLengthError("stream rejected input")
                yield {"content": "stream recovered"}
            finally:
                closed.append(self.stream_calls)
    provider = Provider()
    provider.recovery = _overflow(cap=64)
    await coordinator.mount("providers", provider, name="main")
    async def count_request(event, data):
        requests.append(event)
        return HookResult()
    coordinator.hooks.register("provider:request", count_request)
    async def delay(hooks, iteration):
        delays.append(iteration)
        if len(delays) == 2:
            await gate("rate_limit")
    loop._apply_rate_limit_delay = delay
    args = (context, {"main": provider}, {}, coordinator.hooks, coordinator)
    task = asyncio.create_task(loop.execute("work", *args))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        owner.hold = True
    finally:
        release.set()
    with pytest.raises(ExecutionSuspended) as stopped:
        await asyncio.wait_for(task, 2)
    frame = stopped.value.continuation
    assert json.loads(owner.path.read_text())["frame"] == frame
    assert frame["phase"] == "prepared_request"
    prepared = frame["prepared_request"]
    assert prepared["stream"] is True and prepared["overflow_retried"] is True
    assert prepared["request"]["max_output_tokens"] == 64
    assert [reason for reason, _ in views] == [None, "provider_overflow"]
    assert provider.counted_requests[-1].model_dump(mode="json") == prepared["request"]
    assert len(provider.counted_requests) == 2
    assert old not in [row["content"] for row in prepared["request"]["messages"]]
    assert [transaction.committed for transaction in transactions] == [1, 1]
    assert [transaction.rolled_back for transaction in transactions] == [0, 0]
    assert frame["turn"]["_llm_calls_this_turn"] == provider.stream_calls == 1
    assert hold_at_send == [False] and closed == [1]
    before = (list(views), list(delays), list(requests),
              list(provider.budget_options), list(provider.recovery_options),
              len(provider.counted_requests),
              [(transaction.committed, transaction.rolled_back) for transaction in transactions])

    # Hold still active: explicit host refusal is not implicit authorization.
    with pytest.raises(RuntimeError, match="admission refused"):
        await StreamingOrchestrator(dict(loop.config)).resume(frame, *args)
    assert provider.stream_calls == 1 and owner.consumed == set()
    owner.hold = False
    restored = StreamingOrchestrator(dict(loop.config))
    restored._apply_rate_limit_delay = delay
    if retry_fails:
        with pytest.raises(ContextLengthError, match="stream rejected input"):
            await restored.resume(frame, *args)
    else:
        assert await restored.resume(frame, *args) == "stream recovered"
    assert provider.requests[1].model_dump(mode="json") == prepared["request"]
    assert restored._llm_calls_this_turn == provider.stream_calls == 2
    assert hold_at_send == [False, False] and closed == [1, 2]
    assert (views, delays, requests, provider.budget_options, provider.recovery_options,
            len(provider.counted_requests),
            [(transaction.committed, transaction.rolled_back) for transaction in transactions]) == before
    assert owner.claims == [frame["id"], frame["id"]] and owner.consumed == {frame["id"]}
    assert len([row for row in await context.get_messages()
                if row["role"] == "user" and row["content"] == "work"]) == 1
    assert any(row.get("content") == old for row in await context.get_messages())
    assert loop._tool_calls_this_turn == restored._tool_calls_this_turn == 0
