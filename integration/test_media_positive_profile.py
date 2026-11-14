"""Public cooperative bridge + real Core/F1; synthetic transport, not vendor proof."""
import base64
import asyncio
import copy
import hashlib
import importlib.util
import io
import json
import inspect
import os
import sys
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import pytest
from PIL import Image
from amplifier_core import AmplifierSession, HookResult, ToolResult
from amplifier_core.message_models import ChatRequest, Message, ToolCall
from amplifier_foundation.session.checkpoint import acknowledge_legacy_checkpoint, verified_checkpoint_refs
from amplifier_foundation.session.media import MediaPolicy, SessionMediaStore, bind_session_media
from amplifier_module_loop_streaming.media import NAMESPACE, checkpoint_settled_results
from amplifier_module_loop_streaming import ExecutionSuspended

pytestmark = pytest.mark.asyncio


def bridge_module():
    # Keep the frozen negative profile independent: no imports of its fixtures.
    source = Path(os.environ["MEDIA_HOOK_BRIDGE_SOURCE"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == os.environ["MEDIA_HOOK_BRIDGE_SHA256"]
    spec = importlib.util.spec_from_file_location("positive_profile_bridge", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def acknowledge(store, directory, messages):
    directory.mkdir(exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in messages), encoding="utf-8")
    receipt = acknowledge_legacy_checkpoint(
        directory, session_id=store.session_id, expected_messages=messages)
    return store.acknowledge_checkpoint(receipt)


class Owner:
    """Offline host checkpoint/one-time claim owner, not a mock Core session."""

    def __init__(self, path, context):
        self.path, self.context, self.hold, self.claimed = path, context, False, set()

    def requested(self):
        return self.hold

    async def checkpoint(self, frame):
        self.path.write_text(json.dumps({
            "continuation": frame, "messages": await self.context.get_messages()}))
        return True

    async def claim(self, frame):
        if self.hold or frame["id"] in self.claimed:
            return False
        self.claimed.add(frame["id"])
        return True


@contextmanager
def native_post_calls(coordinator):
    """Observe the actual C API invocations without replacing either endpoint.

    Callback counts alone cannot prove process-once. CPython's C-call profiler
    observes the exact bound Rust registry/coordinator APIs captured through
    Core's public attributes. Caller-local event data only labels the trace;
    it never supplies a replacement implementation or acceptance witness.
    """
    emit, process = coordinator.hooks.emit, coordinator.process_hook_result
    assert inspect.isbuiltin(emit) and inspect.isbuiltin(process)
    previous, calls = sys.getprofile(), []

    def capture(frame, event, function):
        if event in {"c_call", "c_return", "c_exception"}:
            name, owner = getattr(function, "__name__", None), getattr(function, "__self__", None)
            if ((name == "emit" and owner is emit.__self__) or
                    (name == "process_hook_result" and owner is process.__self__)):
                local = frame.f_locals
                # Core 2.0.2 bridge calls its native emit from the attributed
                # emitter (name/data) and processor (event/name/result).
                hook_event = local.get("event", local.get("name"))
                if (frame.f_code.co_name == "post_result" and
                        frame.f_globals.get("__name__") == "amplifier_module_loop_streaming.media"):
                    hook_event = "tool:post"
                if hook_event == "tool:post":
                    data = local.get("enriched", local.get("data", {}))
                    result = local.get("result")
                    if name == "process_hook_result":
                        data = getattr(result, "data", None) or {}
                    tool_name = (getattr(local.get("call"), "name", None)
                                 if frame.f_code.co_name == "post_result" else local.get("name"))
                    calls.append((event, name, data.get("tool_call_id"), tool_name))
        if previous is not None:
            previous(frame, event, function)

    sys.setprofile(capture)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def assert_native_once(trace, count, *, processed=None):
    processed = count if processed is None else processed
    assert Counter((event, name) for event, name, *_ in trace) == Counter({
        ("c_call", "emit"): count, ("c_return", "emit"): count,
        ("c_call", "process_hook_result"): processed,
        ("c_return", "process_hook_result"): processed,
    }), trace


class ProfileImages:
    name, description, input_schema = "image", "offline PNG/JPEG fixture", {"type": "object"}

    def __init__(self, mime):
        buffer = io.BytesIO()
        Image.new("RGB", (3, 2), "purple").save(buffer, format="PNG" if mime == "image/png" else "JPEG")
        self.raw, self.mime, self.calls = buffer.getvalue(), mime, 0

    async def execute(self, arguments):
        self.calls += 1
        return ToolResult(success=True, output={"text": "useful caption"}, content=[
            {"type": "image", "source": {"type": "base64", "media_type": self.mime,
                                         "data": base64.b64encode(self.raw).decode()}},
        ])


@asynccontextmanager
async def profile(tmp_path, *, policy=None, outer=None, mime="image/png", config=None, setup=None):
    # Public direct filesystem discovery uses the conventional package name.
    # Keep the fixture directory scoped to this session; no installed entrypoint
    # or user's runtime configuration is changed.
    patch = pytest.MonkeyPatch()
    patch.syspath_prepend(str(Path(__file__).parent))
    session = AmplifierSession({"session": {
        "orchestrator": {"module": "loop-streaming", "config": {"programmatic_dispatch": True, **(config or {})}},
        "context": {"module": "context-simple"},
    }, "providers": [{"module": "provider-media-profile", "instance_id": "main",
                      "config": {"mime": mime}}]}, session_id="stream-positive-profile")
    coordinator = session.coordinator
    selected = bridge_module()
    bridge = selected.install_media_hook_bridge(coordinator)
    graph = bridge.begin_participating_graph(
        session, owner="streaming.integration.public-profile.v1",
        participants=(*selected.participating_module_ids(session), "host.profile"))
    store = SessionMediaStore(tmp_path / "media", coordinator.session_id,
                              policy=MediaPolicy(1024 * 1024, 10000, 8),
                              checkpoint_verifier=verified_checkpoint_refs)
    cap = bind_session_media(coordinator, store=store, enabled=True)
    await session.initialize()
    host = graph.participant("host.profile")
    loop, context, tool = coordinator.get("orchestrator"), coordinator.get("context"), ProfileImages(mime)
    await host.mount("tools", tool, name="image")
    async def checkpoint():
        return acknowledge(store, tmp_path / "session", await context.get_messages())
    coordinator.register_capability("session.durable_checkpoint", checkpoint)
    calls, staged = [], []
    async def capture(event, data):
        calls.append((data["tool_name"], data["tool_call_id"]))
        refs = ((data.get("result") or {}).get("output") or {}).get(NAMESPACE, {}).get("refs", [])
        staged.extend(copy.deepcopy(refs))
        return await policy(event, data) if policy else HookResult()
    host.hooks.register("tool:post", capture, name="profile.policy", priority=10)
    if outer:
        host.hooks.register("tool:post", outer, name="profile.outer", priority=20)
    if setup:
        await setup(host, loop, context, cap)
    receipt = graph.seal()
    assert bridge.validate_receipt(receipt)
    evidence = graph.participants_evidence()
    assert [item["module"] for item in evidence["loads"]] == [
        "loop-streaming", "context-simple", "provider-media-profile"]
    assert all(item["mounted"] for item in evidence["loads"])
    assert coordinator.get("providers", "main") is not None
    try:
        with native_post_calls(coordinator) as native:
            # A public host capability exposes test evidence without changing
            # the native endpoints or using bridge-private observation helpers.
            coordinator.register_capability("profile.native_calls", native)
            yield session, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged
    finally:
        store.close()
        await session.cleanup()
        patch.undo()


def private_image_blocks(view):
    return [block for message in view["messages"] if isinstance(message["content"], list)
            for block in message["content"] if block.get("type") == "image"]


async def assert_private_pixels(coordinator, canonical, refs, raw, mime):
    adapter = coordinator.get("providers", "main")
    sink = coordinator.get_capability("profile.offline_transport")
    counted, sent = len(sink.counted), len(sink.sent)
    before = canonical.model_dump_json()
    options = {"extended_thinking": True}
    await adapter.request_budget(canonical, context_estimate=1, request_options=options)
    assert adapter.media_budget_witness(canonical, request_options=options)["validated"]
    response = await adapter.complete(canonical, **options)
    assert response.content[0].text == "done"
    assert len(sink.counted) == counted + 1 and len(sink.sent) == sent + 1
    assert sink.sent[-1] is sink.counted[-1] and sink.sent[-1].messages == []
    assert sink.sent_views[-1] == sink.counted_views[-1]
    blocks = private_image_blocks(sink.sent_views[-1])
    assert [block["source"]["media_type"] for block in blocks] == [mime] * len(refs)
    assert [hashlib.sha256(base64.b64decode(block["source"]["data"])).hexdigest()
            for block in blocks] == [ref["sha256"] for ref in refs]
    assert canonical.model_dump_json() == before and base64.b64encode(raw).decode() not in before


@pytest.mark.parametrize("site", ["parallel", "sequential"])
@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
@pytest.mark.parametrize("action", ["retain", "null", "string", "inplace", "error", "deny", "null-inject", "outer-redact"])
async def test_public_positive_settlement_final_effective_refs_checkpoint_projection(tmp_path, site, mime, action):
    async def policy(event, data):
        if action in {"null", "null-inject"}:
            return HookResult(action="modify", data={**data, "result": None})
        if action == "string":
            return HookResult(action="modify", data={**data, "result": "redacted"})
        if action == "inplace":
            data["result"]["output"].pop(NAMESPACE)
        if action == "error":
            raise ValueError("participating privacy callback failed")
        if action == "deny":
            return HookResult(action="deny", reason="profile deny")
        return HookResult()
    async def outer(event, data):
        if action == "outer-redact":
            return HookResult(action="modify", data={**data, "result": None})
        if action == "null-inject":
            assert data["result"] is None
            return HookResult(action="inject_context", ephemeral=True, context_injection="one reminder")
        return HookResult()
    async with profile(tmp_path, policy=policy, outer=outer, mime=mime) as state:
        _, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged = state
        call = ToolCall(id="actual-image-call", name="image", arguments={})
        if site == "parallel":
            settled = await loop._execute_tool_unscoped(call, coordinator.get("tools"), coordinator.hooks, "group", coordinator)
            await context.add_message(settled.message())
            await checkpoint_settled_results(coordinator, has_media=bool(settled.metadata))
        else:
            await loop._execute_tool_with_result_unscoped(call, coordinator.get("tools"), context, coordinator.hooks, coordinator)
        rows = await context.get_messages()
        row = next(row for row in rows if row.get("tool_call_id") == call.id)
        refs = row.get("metadata", {}).get(NAMESPACE, {}).get("refs", [])
        assert calls == [("image", call.id)] and tool.calls == 1 and staged
        assert_native_once(coordinator.get_capability("profile.native_calls"), 1)
        assert bool(refs) is (action == "retain")
        if action == "retain":
            assert refs == staged
            await assert_private_pixels(coordinator, ChatRequest(messages=[Message(**row)]), refs, tool.raw, mime)
        else:
            assert not list((tmp_path / "media" / "assets").iterdir())
        if action in {"null", "null-inject", "outer-redact"}:
            assert row["content"] == "null"
        if action == "string":
            assert row["content"] == "redacted"
        assert base64.b64encode(tool.raw).decode() not in json.dumps(rows)
        consumers = cap.report("loop", event="readiness", outcome="inspect")["consumers"]
        assert bool(consumers) is (action != "error")
        if action == "null-inject":
            assert sum("one reminder" in str(row.get("content"))
                       for row in rows + loop._pending_ephemeral_injections) == 1


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
@pytest.mark.parametrize("action", ["retain", "outer-redact", "inner-null", "deny", "error"])
async def test_public_positive_actual_nested_lease_origin_and_redaction(tmp_path, mime, action):
    returned = []
    async def setup(host, loop, context, cap):
        class Client:
            name, description, input_schema = "tool_exec", "actual nested lease fixture", {"type": "object"}
            async def execute(self, arguments):
                lease = host.get_capability("tools.dispatch").bind()
                result = await lease.call("image", {}, request_id="inner-request")
                returned.append(result)
                return ToolResult(success=True, output=result["output"])
        await host.mount("tools", Client(), name="tool_exec")
    async def policy(event, data):
        if data["tool_name"] == "image":
            assert data["parent_tool_call_id"] == "actual-parent"
            if action == "inner-null":
                return HookResult(action="modify", data={**data, "result": None})
            if action == "deny":
                return HookResult(action="deny", reason="nested denied")
            if action == "error":
                raise ValueError("inner observer failed")
        return HookResult()
    async def outer(event, data):
        if data["tool_name"] == "tool_exec" and action == "outer-redact":
            return HookResult(action="modify", data={**data, "result": None})
        return HookResult()
    async with profile(tmp_path, policy=policy, outer=outer, mime=mime, setup=setup) as state:
        _, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged = state
        with loop.approved_dispatch.run_scope(coordinator):
            result = await loop._execute_tool_only(
                ToolCall(id="actual-parent", name="tool_exec", arguments={}),
                coordinator.get("tools"), coordinator.hooks, "group", coordinator)
            # Returned nested evidence expires with the original parent scope;
            # outer publication must already have occurred within that scope.
            await context.add_message(result.message())
            await checkpoint_settled_results(coordinator, has_media=bool(result.metadata))
        refs = result.metadata.get(NAMESPACE, {}).get("refs", [])
        assert tool.calls == 1 and len(returned) == 1
        assert [name for name, _ in calls] == ["image", "tool_exec"]
        native = coordinator.get_capability("profile.native_calls")
        assert_native_once(native, 2)
        assert Counter(name for event, api, _, name in native
                       if event == "c_call" and api == "process_hook_result") == {
                           "image": 1, "tool_exec": 1}
        assert bool(refs) is (action == "retain")
        if action == "retain":
            assert returned[0]["status"] == "completed" and returned[0]["success"] is True
            assert refs == returned[0]["output"][NAMESPACE]["refs"]
            assert all(ref["provenance"]["id"] != "actual-parent" for ref in refs)
            assert all(ref["provenance"]["id"] == calls[0][1] for ref in refs)
            await assert_private_pixels(
                coordinator, ChatRequest(messages=[Message(**result.message())]), refs, tool.raw, mime)
        elif action == "deny":
            assert returned[0]["status"] == "denied"
        elif action == "inner-null":
            assert returned[0]["status"] == "unknown" and returned[0]["success"] is None
        if action == "outer-redact":
            assert result.result is None
        assert base64.b64encode(tool.raw).decode() not in json.dumps(await context.get_messages())


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
@pytest.mark.parametrize("finalization", [False, True])
@pytest.mark.parametrize("pause", [None, "prepared", "settled"])
async def test_public_positive_whole_engine_final_request_and_refs_only_resume(
    tmp_path, mime, finalization, pause,
):
    owned = {}
    async def setup(host, loop, context, cap):
        owner = owned["suspension"] = Owner(tmp_path / "continuation.json", context)
        host.register_capability("execution.suspension", owner)
        host.register_capability("execution.resume", owner.claim)
        requests = 0
        async def stop(event, data):
            nonlocal requests
            requests += 1
            owned["provider_requests"] = requests
            if pause == "prepared" and requests == 2:
                owner.hold = True
            return HookResult()
        host.hooks.register("provider:request", stop, name="profile.suspension")
    async def policy(event, data):
        if pause == "settled":
            owned["suspension"].hold = True
        return HookResult()
    config = {"max_iterations": 1 if finalization else 3, "extended_thinking": True,
              "reasoning_effort": "high", "ephemeral_injection_mode": "tail",
              "reminder_placement": "tail"}
    async with profile(tmp_path, mime=mime, config=config, policy=policy, setup=setup) as state:
        _, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged = state
        sink = coordinator.get_capability("profile.offline_transport")
        args = (context, coordinator.get("providers"), coordinator.get("tools"), coordinator.hooks, coordinator)
        if pause:
            with pytest.raises(ExecutionSuspended) as stopped:
                await loop.execute("inspect fixture", *args)
            frame = stopped.value.continuation
            assert frame["phase"] == (
                "provider" if pause == "settled" else
                "prepared_finalization" if finalization else "prepared_request")
            assert base64.b64encode(tool.raw).decode() not in json.dumps(frame)
            before_native = list(coordinator.get_capability("profile.native_calls"))
            before_requests = owned["provider_requests"]
            owned["suspension"].hold = False
            answer = await loop.resume(frame, *args)
            assert coordinator.get_capability("profile.native_calls") == before_native
            assert owned["provider_requests"] == before_requests + (pause == "settled")
            # A claimed frame is never executed twice, including tool effects.
            with pytest.raises(Exception, match="claim|resum|history changed"):
                await loop.resume(frame, *args)
        else:
            answer = await state[0].execute("inspect fixture")
        assert answer == "done" and tool.calls == 1
        assert len(sink.sent) == 2 and len(sink.counted) == (3 if pause == "prepared" else 2)
        assert calls == [("image", "engine-image")]
        assert_native_once(coordinator.get_capability("profile.native_calls"), 1)
        assert sink.count_options == [{"extended_thinking": True}] * len(sink.counted)
        assert sink.send_options == [{"extended_thinking": True}] * 2
        for view in sink.counted_views:
            assert view["tools"][0]["name"] == "image" and len(view["tools"]) == 1
            assert view["reasoning_effort"] == "high"
        assert sink.sent_views[-1] == sink.counted_views[-1]
        blocks = private_image_blocks(sink.sent_views[-1])
        assert len(blocks) == 1 and base64.b64decode(blocks[0]["source"]["data"]) == tool.raw
        assert sink.sent_views[-1]["tool_choice"] == ("none" if finalization else None)
        if finalization:
            assert "wrap up naturally" in str(sink.sent_views[-1]["messages"][-1]["content"])
        rows = await context.get_messages()
        assert sum(row["role"] == "user" and row["content"] == "inspect fixture" for row in rows) == 1
        assert base64.b64encode(tool.raw).decode() not in json.dumps(rows)
        row = next(row for row in rows if row.get("tool_call_id") == "engine-image")
        assert row["metadata"][NAMESPACE]["refs"] == staged
        assert cap.report("loop", event="readiness", outcome="inspect")["consumers"]["loop-streaming"]


@pytest.mark.parametrize("site", ["parallel", "sequential"])
@pytest.mark.parametrize("interruption", ["cancel", "unknown-registration"])
@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
async def test_public_positive_interrupted_chain_never_promotes_or_replays(tmp_path, site, interruption, mime):
    entered, release = asyncio.Event(), asyncio.Event()
    owned = {}
    async def policy(event, data):
        if interruption == "unknown-registration":
            # Deliberately outside a declared participant: receipt must fail.
            owned["coordinator"].hooks.register("tool:post", lambda e, d: HookResult(), name="unknown")
        else:
            entered.set()
            await release.wait()
        return HookResult()
    async with profile(tmp_path, policy=policy, mime=mime) as state:
        _, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged = state
        owned["coordinator"] = coordinator
        call = ToolCall(id="interrupted-call", name="image", arguments={})
        operation = (loop._execute_tool_unscoped(
            call, coordinator.get("tools"), coordinator.hooks, "group", coordinator)
            if site == "parallel" else loop._execute_tool_with_result_unscoped(
                call, coordinator.get("tools"), context, coordinator.hooks, coordinator))
        if interruption == "cancel":
            task = asyncio.create_task(operation)
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await operation
            if site == "parallel":
                assert not result.metadata
            else:
                assert not any(row.get("metadata", {}).get(NAMESPACE) for row in await context.get_messages())
        assert calls == [("image", call.id)] and tool.calls == 1 and staged
        assert_native_once(coordinator.get_capability("profile.native_calls"), 1,
                           processed=0 if interruption == "cancel" else 1)
        await asyncio.sleep(0)
        assert calls == [("image", call.id)] and tool.calls == 1
        assert not list((tmp_path / "media" / "assets").iterdir())
        assert not cap.report("loop", event="readiness", outcome="inspect")["consumers"]


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
@pytest.mark.parametrize("finalization", [False, True])
@pytest.mark.parametrize("outcome", ["suspend", "cancel"])
async def test_public_positive_exact_abandonment_after_private_count(
    tmp_path, mime, finalization, outcome,
):
    entered, hold = asyncio.Event(), asyncio.Event()
    owned = {}

    async def setup(host, loop, context, cap):
        owner = owned["owner"] = Owner(tmp_path / "continuation.json", context)
        host.register_capability("execution.suspension", owner)
        host.register_capability("execution.resume", owner.claim)

        async def after_count(event, data):
            sink = host.get_capability("profile.offline_transport")
            if len(sink.counted) != 2:
                return HookResult()
            provider = host.get("providers", "main")
            canonical, options = sink.canonical_counts[-1]
            # Reject equal-but-distinct canonical objects and wrong options
            # without stealing the prepared association from its actual owner.
            owned["negative_release"] = (
                provider.release_media_request(canonical.model_copy(deep=True), request_options=options),
                provider.release_media_request(canonical, request_options={"wrong": True}),
            )
            owned["witness"] = provider.media_budget_witness(canonical, request_options=options)
            if outcome == "suspend":
                owner.hold = True
            else:
                # The adapter already froze the accepted count options. A
                # caller-side mutation while the observer awaits must not
                # change the loop's independently recorded cleanup identity.
                sink.live_count_options[-1]["mutated-after-count"] = True
                entered.set()
                await hold.wait()
            return HookResult()

        host.hooks.register("orchestrator:provider_budget", after_count, name="profile.after-private-count")

    config = {"max_iterations": 1 if finalization else 3, "extended_thinking": True,
              "reminder_placement": "tail"}
    async with profile(tmp_path, mime=mime, config=config, setup=setup) as state:
        session, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged = state
        sink = coordinator.get_capability("profile.offline_transport")
        args = (context, coordinator.get("providers"), coordinator.get("tools"),
                coordinator.hooks, coordinator)
        if outcome == "suspend":
            with pytest.raises(ExecutionSuspended) as stopped:
                await loop.execute("inspect fixture", *args)
            assert stopped.value.continuation["phase"] == (
                "prepared_finalization" if finalization else "prepared_request")
            assert base64.b64encode(tool.raw).decode() not in json.dumps(stopped.value.continuation)
        else:
            task = asyncio.create_task(loop.execute("inspect fixture", *args))
            await asyncio.wait_for(entered.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert owned["negative_release"] == (False, False)
        assert owned["witness"]["validated"] is True
        assert len(sink.counted) == 2 and len(sink.sent) == 1
        assert len(private_image_blocks(sink.counted_views[-1])) == 1
        assert sink.counted[-1].messages == []  # real private buffer revoked
        canonical, options = sink.canonical_counts[-1]
        released = [item for item in sink.releases if item[2] is True]
        assert len(released) == 1
        assert released[0][0] is canonical and released[0][1] == options == {"extended_thinking": True}
        # Cleanup did not close the shared provider; another MEDIA request can
        # count and complete with the same privately hydrated pixels.
        provider = coordinator.get("providers", "main")
        assert provider.release_media_request(canonical, request_options=options) is False
        rows = await context.get_messages()
        media_row = next(row for row in rows if row.get("tool_call_id") == "engine-image")
        fresh = ChatRequest(messages=[Message(**media_row)])
        await provider.request_budget(fresh, context_estimate=1, request_options=options)
        assert provider.media_budget_witness(fresh, request_options=options)["validated"] is True
        response = await provider.complete(fresh, **options)
        assert response.content[0].text == "done"
        assert sink.sent[-1] is sink.counted[-1]
        assert sink.sent_views[-1] == sink.counted_views[-1]
        assert base64.b64decode(private_image_blocks(sink.sent_views[-1])[0]["source"]["data"]) == tool.raw
        assert sink.sent[-1].messages == []
        assert provider.release_media_request(fresh, request_options=options) is False
        assert tool.calls == 1 and calls == [("image", "engine-image")]
        assert_native_once(coordinator.get_capability("profile.native_calls"), 1)
        assert base64.b64encode(tool.raw).decode() not in json.dumps(await context.get_messages())


@pytest.mark.parametrize("outcome", ["revoke", "hold", "cancel"])
async def test_public_positive_nested_count_refusal_exact_cleanup(tmp_path, outcome):
    from amplifier_core import ToolResult
    from amplifier_module_loop_streaming.suspension import NestedExecutionHeld
    owned = {}

    async def setup(host, loop, context, cap):
        owner = Owner(tmp_path / "nested-continuation.json", context)
        host.register_capability("execution.suspension", owner)
        sink = host.get_capability("profile.offline_transport")
        provider = host.get("providers", "main")

        async def after_count(canonical, options):
            owned["witness"] = provider.media_budget_witness(canonical, request_options=options)
            if outcome == "revoke":
                loop.approved_dispatch.validate_context = lambda coordinator: False
            elif outcome == "hold":
                owner.hold = True
            else:
                host.cancellation.request_immediate()

        sink.after_count = after_count

        class Client:
            name, description, input_schema = "tool_exec", "nested cleanup fixture", {"type": "object"}

            async def execute(self, arguments):
                lease = host.get_capability("tools.dispatch").bind()
                try:
                    complete = host.get_capability("model.complete")
                    expected = NestedExecutionHeld if outcome == "hold" else RuntimeError
                    with pytest.raises(expected):
                        await complete(ChatRequest(messages=[Message(role="user", content=[
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                       "data": "aW1hZ2U="}},
                        ])]))
                    owned["refused"] = True
                finally:
                    lease.close()
                    if outcome == "revoke":
                        loop.approved_dispatch.validate_context = None
                return ToolResult(success=True, output={"refused": True})

        await host.mount("tools", Client(), name="tool_exec")

    async with profile(tmp_path, setup=setup) as state:
        _, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged = state
        sink = coordinator.get_capability("profile.offline_transport")
        with loop.approved_dispatch.run_scope(coordinator):
            await loop._execute_tool_only(
                ToolCall(id="nested-count-parent", name="tool_exec", arguments={}),
                coordinator.get("tools"), coordinator.hooks, "group", coordinator)
        assert owned["refused"] and owned["witness"]["validated"] is True
        assert len(sink.counted) == 1 and not sink.sent
        assert not private_image_blocks(sink.counted_views[0])
        assert "no pixels supplied" in json.dumps(sink.counted_views[0])
        canonical, options = sink.canonical_counts[0]
        assert options == {"extended_thinking": False, "model": "fixture"}
        assert sink.releases == [(canonical, options, True)]
        assert sink.releases[0][0] is canonical
        assert sink.counted[0].messages == []
        assert coordinator.get("providers", "main").release_media_request(
            canonical, request_options=options) is False


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
async def test_public_positive_full_goal_evaluator_stall_judge_and_summary(tmp_path, mime):
    config = {"goal_provider_preferences": [{"provider": "main", "model": "fixture"}]}
    async with profile(tmp_path, mime=mime, config=config) as state:
        _, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged = state
        settled = await loop._execute_tool_unscoped(
            ToolCall(id="utility-image", name="image", arguments={}),
            coordinator.get("tools"), coordinator.hooks, "group", coordinator)
        assert settled.metadata[NAMESPACE]["refs"] == staged and staged
        # An ordinary text tool row with real refs in the recognized metadata
        # carrier, not refs quoted in arbitrary text. Utilities are not generic
        # secret redactors for JSON-looking text.
        row = settled.message()
        row["content"] = "useful caption"
        await context.add_message(row)
        await checkpoint_settled_results(coordinator, has_media=True)
        original = copy.deepcopy(await context.get_messages())
        sink = coordinator.get_capability("profile.offline_transport")
        sink.image_call = False
        sink.responses[:] = [
            "NO\nUTILITY_EVAL_PENDING",
            "HISTORY-LOCKED\nNOT_DEMONSTRATED\nUTILITY_JUDGE_BLOCKED",
            "UTILITY_SUMMARY: awaiting prerequisite.",
        ]
        providers = coordinator.get("providers")
        assert coordinator.get_capability("model_role_resolver") is None

        def check_transport(index):
            # These real utilities call complete directly, not request_budget.
            # Assert that fact instead of inventing a provider-count claim.
            assert not sink.counted and len(sink.sent) == index + 1
            assert sink.send_options[-1] == {
                "model": "fixture", "extended_thinking": False}
            view = sink.sent_views[-1]
            assert view["tools"] is None
            assert [message["role"] for message in view["messages"]] == ["system", "user"]
            assert view["model"] == "fixture" and view["metadata"]["stream"] is False
            assert view["max_output_tokens"] == 512
            encoded = json.dumps(view)
            assert base64.b64encode(tool.raw).decode() not in encoded
            assert NAMESPACE not in encoded
            assert all(ref["sha256"] not in encoded and
                       ref["provenance"]["id"] not in encoded for ref in staged)
            assert not private_image_blocks(view)
            assert "Finish offline fixture" in encoded
            return encoded

        evaluation = await loop._evaluate_goal(
            "Finish offline fixture", context, providers, coordinator.hooks, coordinator)
        assert evaluation == (False, "UTILITY_EVAL_PENDING")
        assert "useful caption" in check_transport(0)
        assert "Image unavailable" in json.dumps(sink.sent_views[-1])
        assert await context.get_messages() == original

        goal = {"condition": "Finish offline fixture", "reasons": [evaluation[1]],
                "no_tool_turns": 1, "continuations": 1}
        judged = await loop._judge_stall(goal, providers, coordinator.hooks, coordinator, trigger="idle")
        assert judged == (True, "UTILITY_JUDGE_BLOCKED", "history-locked", "not_demonstrated")
        assert "UTILITY_EVAL_PENDING" in check_transport(1)
        assert await context.get_messages() == original

        summary = await loop._summarize_goal_run(
            goal, providers, coordinator.hooks, coordinator, "stalled")
        assert summary == "UTILITY_SUMMARY: awaiting prerequisite."
        assert "UTILITY_EVAL_PENDING" in check_transport(2)
        assert not sink.responses and await context.get_messages() == original
        assert_native_once(coordinator.get_capability("profile.native_calls"), 1)
