"""Public cooperative bridge + real Core/F1; synthetic transport, not vendor proof."""
import base64
import asyncio
import copy
import hashlib
import io
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from PIL import Image
from amplifier_core import AmplifierSession, HookResult, ToolResult
from amplifier_core.message_models import ChatRequest, Message, ToolCall
from amplifier_foundation.session.checkpoint import verified_checkpoint_refs
from amplifier_foundation.session.media import MediaPolicy, SessionMediaStore, bind_session_media
from amplifier_foundation.session.media_provider import MediaPreparedProvider
from amplifier_module_loop_streaming.media import NAMESPACE, checkpoint_settled_results
from integration.test_media_hook_bridge_pairing import bridge_module
from integration.test_media_store_pairing import acknowledge
from tests.test_ephemeral_cache_persist_mode import MockResponse
from tests.test_execution_suspension import Owner
from amplifier_module_loop_streaming import ExecutionSuspended

pytestmark = pytest.mark.asyncio


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
    session = AmplifierSession({"session": {
        "orchestrator": {"module": "loop-streaming", "config": {"programmatic_dispatch": True, **(config or {})}},
        "context": {"module": "context-simple"},
    }, "providers": []}, session_id="stream-positive-profile")
    coordinator = session.coordinator
    bridge = bridge_module().install_media_hook_bridge(coordinator)
    graph = bridge.begin_participating_graph(
        session, owner="streaming.integration.public-profile.v1",
        participants=("loop-streaming", "context-simple", "host.profile"))
    await session.initialize()
    host = graph.participant("host.profile")
    loop, context, tool = coordinator.get("orchestrator"), coordinator.get("context"), ProfileImages(mime)
    await host.mount("tools", tool, name="image")
    store = SessionMediaStore(tmp_path / "media", coordinator.session_id,
                              policy=MediaPolicy(1024 * 1024, 10000, 8),
                              checkpoint_verifier=verified_checkpoint_refs)
    cap = bind_session_media(coordinator, store=store, enabled=True)
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
    try:
        yield session, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged
    finally:
        store.close()
        await session.cleanup()


async def assert_private_pixels(cap, canonical, refs, raw, mime):
    counted, sent = [], []
    class SyntheticTransport:
        name, config = "profile-synthetic-transport", {}
        def get_info(self):
            return SimpleNamespace(defaults={"model": "fixture"}, capabilities=[])
        def get_media_request_support(self, *, model, request_options=None):
            return {"v": 1, "positions": ["tool"], "mime_types": [mime]}
        def request_budget(self, request, *, context_estimate, request_options=None):
            assert request_options == {"extended_thinking": True}
            counted.append(request)
            blocks = [block for message in request.messages if isinstance(message.content, list)
                      for block in message.content if getattr(block, "type", None) == "image"]
            assert [hashlib.sha256(base64.b64decode(block.source["data"])).hexdigest() for block in blocks] == [r["sha256"] for r in refs]
            return {"estimated_input_tokens": 20, "input_limit_tokens": 10000, "context_token_budget": 10000,
                    "measurement": {"kind": "provider_count", "source": "profile", "input_tokens": 20}}
        async def complete(self, request, **options):
            assert options == {"extended_thinking": True}
            assert counted[-1] is request
            sent.append(request)
            return "done"
    adapter = MediaPreparedProvider(SyntheticTransport(), cap, provider_identity={
        "provider": "profile-synthetic-transport", "account": "offline", "generation": 1}, model="fixture")
    before = canonical.model_dump_json()
    try:
        options = {"extended_thinking": True}
        await adapter.request_budget(canonical, context_estimate=1, request_options=options)
        assert adapter.media_budget_witness(canonical, request_options=options)["validated"]
        assert await adapter.complete(canonical, **options) == "done"
        assert len(counted) == len(sent) == 1 and sent[0].messages == []
        assert canonical.model_dump_json() == before and base64.b64encode(raw).decode() not in before
    finally:
        adapter.close()


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
        assert bool(refs) is (action == "retain")
        if action == "retain":
            assert refs == staged
            await assert_private_pixels(cap, ChatRequest(messages=[Message(**row)]), refs, tool.raw, mime)
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
        assert bool(refs) is (action == "retain")
        if action == "retain":
            assert returned[0]["status"] == "completed" and returned[0]["success"] is True
            assert refs == returned[0]["output"][NAMESPACE]["refs"]
            assert all(ref["provenance"]["id"] != "actual-parent" for ref in refs)
            assert all(ref["provenance"]["id"] == calls[0][1] for ref in refs)
            await assert_private_pixels(
                cap, ChatRequest(messages=[Message(**result.message())]), refs, tool.raw, mime)
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
        counted, sent = [], []
        class SyntheticCompleteOwner:
            name, config = "profile-complete-owner", {}
            calls = 0
            def get_info(self):
                return SimpleNamespace(defaults={"model": "fixture"}, capabilities=["request_budget:provider_count"])
            def get_media_request_support(self, *, model, request_options=None):
                return {"v": 1, "positions": ["tool"], "mime_types": [mime]}
            def request_budget(self, request, *, context_estimate, request_options=None):
                assert request_options == {"extended_thinking": True}
                assert len(request.tools) == 1 and request.tools[0].name == "image"
                assert request.reasoning_effort == "high"
                if self.calls:
                    blocks = [block for message in request.messages if isinstance(message.content, list)
                              for block in message.content if getattr(block, "type", None) == "image"]
                    assert len(blocks) == 1 and base64.b64decode(blocks[0].source["data"]) == tool.raw
                    assert request.tool_choice == ("none" if finalization else None)
                    if finalization:
                        assert "wrap up naturally" in str(request.messages[-1].content)
                counted.append(request)
                return {"estimated_input_tokens": 20, "input_limit_tokens": 10000,
                        "context_token_budget": 10000,
                        "measurement": {"kind": "provider_count", "source": "profile", "input_tokens": 20}}
            async def complete(self, request, **options):
                assert options == {"extended_thinking": True} and counted[-1] is request
                self.calls += 1
                sent.append(request)
                return MockResponse("done")
            def parse_tool_calls(self, response):
                return [ToolCall(id="engine-image", name="image", arguments={})] if self.calls == 1 else []
        adapter = MediaPreparedProvider(SyntheticCompleteOwner(), cap, provider_identity={
            "provider": "profile-complete-owner", "account": "offline", "generation": 1}, model="fixture")
        try:
            args = (context, {"main": adapter}, coordinator.get("tools"), coordinator.hooks, coordinator)
            if pause:
                with pytest.raises(ExecutionSuspended) as stopped:
                    await loop.execute("inspect fixture", *args)
                frame = stopped.value.continuation
                assert frame["phase"] == (
                    "provider" if pause == "settled" else
                    "prepared_finalization" if finalization else "prepared_request")
                assert base64.b64encode(tool.raw).decode() not in json.dumps(frame)
                owned["suspension"].hold = False
                answer = await loop.resume(frame, *args)
            else:
                answer = await loop.execute("inspect fixture", *args)
            assert answer == "done" and tool.calls == 1
            assert len(sent) == 2 and len(counted) == (3 if pause == "prepared" else 2)
            assert calls == [("image", "engine-image")]
            rows = await context.get_messages()
            assert base64.b64encode(tool.raw).decode() not in json.dumps(rows)
            row = next(row for row in rows if row.get("tool_call_id") == "engine-image")
            assert row["metadata"][NAMESPACE]["refs"] == staged
            assert cap.report("loop", event="readiness", outcome="inspect")["consumers"]["loop-streaming"]
        finally:
            adapter.close()


@pytest.mark.parametrize("site", ["parallel", "sequential"])
@pytest.mark.parametrize("interruption", ["cancel", "unknown-registration"])
async def test_public_positive_interrupted_chain_never_promotes_or_replays(tmp_path, site, interruption):
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
    async with profile(tmp_path, policy=policy) as state:
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
        assert not list((tmp_path / "media" / "assets").iterdir())
        assert not cap.report("loop", event="readiness", outcome="inspect")["consumers"]


@pytest.mark.parametrize("kind", ["evaluator", "judge", "summary"])
async def test_public_positive_goal_utility_drops_pixels_before_actual_adapter_count(tmp_path, kind):
    async with profile(tmp_path) as state:
        _, coordinator, loop, context, tool, store, cap, bridge, graph, calls, staged = state
        settled = await loop._execute_tool_unscoped(
            ToolCall(id="utility-image", name="image", arguments={}),
            coordinator.get("tools"), coordinator.hooks, "group", coordinator)
        await context.add_message(settled.message())
        await checkpoint_settled_results(coordinator, has_media=True)
        original = copy.deepcopy(await context.get_messages())
        utility = ChatRequest(messages=[Message(**row) for row in original], stream=False)
        await loop._goal_utility_boundary(kind, "fixture", utility, {}, coordinator)
        assert "Image unavailable in text-only utility" in utility.model_dump_json()
        # Tool text can quote refs; only the recognized metadata carrier is media.
        assert all(NAMESPACE not in (message.metadata or {}) for message in utility.messages)
        await assert_private_pixels(cap, utility, [], tool.raw, tool.mime)
        assert await context.get_messages() == original
