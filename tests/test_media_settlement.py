"""Media contract fixtures: offline structural capability, no host activation."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from amplifier_core import HookRegistry, HookResult, ToolResult
from amplifier_core.message_models import ChatRequest, Message, ToolCall

from amplifier_module_loop_streaming import ExecutionSuspended, SettledToolResult, StreamingOrchestrator
from amplifier_module_loop_streaming.media import NAMESPACE, UNAVAILABLE
from amplifier_module_loop_streaming.media import effective_result, report_consumer_ready
from amplifier_module_loop_streaming.suspension import ToolExecutionSuspended
from tests.test_durable_checkpoint import MockContext
from tests.test_ephemeral_cache_persist_mode import MockCoordinator, ScriptedHooks
from tests.test_execution_suspension import Owner
from tests.test_approved_dispatch import environment, call_client  # noqa: F401

pytestmark = pytest.mark.asyncio

PIXELS = "aW1hZ2UtcHJpdmFjeS1zZW50aW5lbA=="


def ref(call_id="call"):
    return {
        "v": 1, "session_id": "session", "sha256": "a" * 64,
        "mime": "image/png", "bytes": 22, "width": 1, "height": 1,
        "provenance": {"kind": "tool", "session_id": "session", "id": call_id,
                       "created_at": "2026-01-01T00:00:00+00:00", "expires_at": None},
    }


class Media:
    """Owner check spy; Foundation independently owns byte/ref validation."""
    version = 1

    def __init__(self):
        self.staged, self.publications, self.reports = {}, [], []

    async def ingest(self, value, *, source_format, origin, policy):
        assert source_format == "core-tool-result" and policy is None
        assert origin["kind"] == "tool" and origin["session_id"] == "session"
        refs = [ref(origin["id"])] if value.content else []
        self.staged[origin["id"]] = refs
        if refs:
            value = ToolResult(success=value.success, output={
                "text": value.output, NAMESPACE: {"refs": refs},
            }, content=[{"type": "text", "text": "saved image"}])
        return {"value": value, "staged_refs": refs}

    async def publish(self, refs, *, origin, disposition):
        assert all(r in self.staged[origin["id"]] for r in refs), "forged origin/ref"
        self.publications.append((copy.deepcopy(refs), dict(origin), disposition))
        return {"refs": copy.deepcopy(refs), "publication": {
            "state": "published" if disposition == "retain" else "dropped",
            "durability": "assets-only",
        }}

    def validate(self, *args, **kwargs):
        return {"valid": True}

    def resolve(self, *args, **kwargs):
        pytest.fail("Loop may not hydrate")

    def prepare_request(self, *args, **kwargs):
        pytest.fail("Loop may not prepare/hydrate")

    def report(self, *args, **kwargs):
        assert isinstance(args[0], str)
        self.reports.append((args, kwargs))


class ImageTool:
    name, description, input_schema = "image", "fixture image", {"type": "object"}

    async def execute(self, arguments):
        return ToolResult(success=True, output="useful text", content=[
            {"type": "text", "text": "caption"},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": PIXELS,
            }},
        ])


class MutatingHooks(ScriptedHooks):
    def __init__(self, mutation):
        super().__init__({})
        self.mutation = mutation

    async def emit(self, event, payload=None):
        if event == "tool:post":
            assert PIXELS not in json.dumps(payload)
            self.emitted.append((event, copy.deepcopy(payload)))
            return self.mutation(payload)
        return await super().emit(event, payload)


async def fixture_boundary(event_data, *, hooks=None, coordinator=None):
    """Synthetic trustworthy hook boundary, NOT a qualified host adapter."""
    hook_result = await hooks.emit("tool:post", event_data)
    hook_result = await coordinator.process_hook_result(hook_result, "tool:post", event_data["tool_name"])
    return {"hook_result": hook_result, "result": effective_result(hook_result, event_data["result"]),
            "complete": True}


def setup(cap=True, boundary=True):
    coordinator = MockCoordinator()
    coordinator.session_id = "session"
    media = Media()
    if cap:
        coordinator.register_capability(NAMESPACE, media)
    if boundary:
        coordinator.register_capability("session.media.tool_post", fixture_boundary)
    return coordinator, media, StreamingOrchestrator({})


async def execute(mutation=lambda data: HookResult(), *, cap=True, sequential=False):
    coordinator, media, loop = setup(cap)
    hooks = MutatingHooks(mutation)
    call = ToolCall(id="call", name="image", arguments={})
    if sequential:
        context = MockContext()
        coordinator.register_capability("session.durable_checkpoint", lambda: None)
        info = await loop._execute_tool_with_result_unscoped(
            call, {"image": ImageTool()}, context, hooks, coordinator)
        return context._messages[0], media, info
    result = await loop._execute_tool_unscoped(
        call, {"image": ImageTool()}, hooks, "group", coordinator)
    return result, media, hooks


@pytest.mark.parametrize("sequential", [False, True])
async def test_stages_before_observers_publishes_final_refs_at_actual_call(sequential):
    result, media, _ = await execute(sequential=sequential)
    message = result if sequential else result.message()
    assert message["metadata"] == {NAMESPACE: {"refs": [ref()]}}
    assert message["tool_call_id"] == "call"
    assert media.publications == [([ref()], {
        "kind": "tool", "session_id": "session", "id": "call",
    }, "retain")]
    assert PIXELS not in json.dumps(message)
    if not sequential:
        assert SettledToolResult.from_legacy(result.model_dump()) == result
        assert result.result["output"][NAMESPACE]["refs"] == [ref()]


@pytest.mark.parametrize("case", [
    "null", "missing", "remove-output", "inplace", "string", "list", "deny", "exception",
    "forged", "empty",
])
async def test_effective_hook_result_never_resurrects_original_refs(case):
    def mutation(data):
        if case == "exception":
            raise ValueError("hook failure")
        if case == "inplace":
            data["result"]["output"].pop(NAMESPACE)
            return HookResult()
        if case == "null":
            value = None
        elif case == "missing":
            return HookResult(action="modify", data={"tool_name": "image"})
        elif case == "remove-output":
            value = {"success": True}
        elif case == "string":
            value = json.dumps(data["result"])  # quoted data is NOT media
        elif case == "list":
            value = [data["result"]]
        elif case == "deny":
            return HookResult(action="deny", reason="redacted", data=data)
        elif case == "forged":
            value = copy.deepcopy(data["result"])
            value["output"][NAMESPACE]["refs"] = [ref("another-call")]
        else:
            value = ""
        return HookResult(action="modify", data={"result": value})

    result, media, _ = await execute(mutation)
    assert result.metadata == {}
    assert not any(disposition == "retain" for _, _, disposition in media.publications)
    if case not in {"forged"}:
        assert media.publications[-1][2] == "drop"
    if case in {"null", "missing", "deny"}:
        assert result.content == "null" and result.result is None
    if case == "string":
        assert isinstance(result.result, str)


async def test_unsupported_capability_has_truthful_text_and_no_pixels():
    result, _, hooks = await execute(cap=False)
    assert result.metadata == {}
    assert UNAVAILABLE in result.content
    assert PIXELS not in json.dumps(hooks.emitted)
    assert NAMESPACE not in result.content


@pytest.mark.parametrize("failure", ["exception", "pixels-left", "mapping-not-toolresult"])
async def test_ingest_failure_never_exposes_raw_result_to_hooks(failure):
    coordinator, media, loop = setup()
    async def invalid(value, **kwargs):
        if failure == "exception":
            raise ValueError(PIXELS)
        return {"value": value if failure == "pixels-left" else value.model_dump(), "staged_refs": []}
    media.ingest = invalid
    hooks = MutatingHooks(lambda data: HookResult())
    result = await loop._execute_tool_unscoped(
        ToolCall(id="call", name="image", arguments={}),
        {"image": ImageTool()}, hooks, "group", coordinator)
    assert result.metadata == {}
    assert not any(event == "tool:post" for event, _ in hooks.emitted)
    assert PIXELS not in json.dumps(hooks.emitted)
    assert PIXELS not in result.content and not media.publications


async def test_absent_capability_keeps_rich_caption_and_ordinary_output_semantics():
    coordinator, _, loop = setup(cap=False)
    class Caption(ImageTool):
        async def execute(self, arguments):
            value = await super().execute(arguments)
            return ToolResult(content=value.content)
    result = await loop._execute_tool_unscoped(
        ToolCall(id="call", name="image", arguments={}),
        {"image": Caption()}, MutatingHooks(lambda data: HookResult()), "group", coordinator)
    assert "caption" in result.content and UNAVAILABLE in result.content
    assert PIXELS not in result.content


async def test_registered_but_disabled_media_preserves_text_and_warns_for_images():
    coordinator, media, loop = setup()
    media.enabled = False
    async def forbidden(*args, **kwargs):
        pytest.fail("Disabled producer must not ingest")
    media.ingest = forbidden
    hooks = MutatingHooks(lambda data: HookResult())
    call = ToolCall(id="call", name="image", arguments={})
    result = await loop._execute_tool_unscoped(
        call, {"image": ImageTool()}, hooks, "group", coordinator)
    assert not result.metadata and UNAVAILABLE in result.content
    assert not media.publications and PIXELS not in json.dumps(hooks.emitted)
    class Text:
        async def execute(self, arguments):
            return ToolResult(success=True, output="ordinary text")
    result = await loop._execute_tool_unscoped(
        call, {"image": Text()}, hooks, "group", coordinator)
    assert result.content == "ordinary text" and result.success is True


async def test_coordinator_processed_result_is_authoritative():
    coordinator, media, loop = setup()
    async def process(result, event, name):
        return HookResult(action="modify", data={"result": None}) if event == "tool:post" else result
    coordinator.process_hook_result = process
    result = await loop._execute_tool_unscoped(
        ToolCall(id="call", name="image", arguments={}),
        {"image": ImageTool()}, MutatingHooks(lambda data: HookResult()),
        "group", coordinator)
    assert result.metadata == {} and result.result is None
    assert media.publications[-1][2] == "drop"


@pytest.mark.parametrize("case", ["null", "missing", "remove-output", "inplace-modify", "string", "deny"])
async def test_real_core_registry_final_chained_data_is_authoritative(case):
    coordinator, media, loop = setup()
    hooks = HookRegistry()
    async def mutation(event, data):
        assert PIXELS not in json.dumps(data)
        if case == "null":
            data["result"] = None
        elif case == "missing":
            data.pop("result")
        elif case == "remove-output":
            data["result"].pop("output")
        elif case == "inplace-modify":
            data["result"]["output"].pop(NAMESPACE)
        elif case == "string":
            data["result"] = json.dumps(data["result"])
        elif case == "deny":
            return HookResult(action="deny", reason="redacted")
        return HookResult(action="modify", data=data)
    hooks.register("tool:post", mutation)
    result = await loop._execute_tool_unscoped(
        ToolCall(id="call", name="image", arguments={}),
        {"image": ImageTool()}, hooks, "group", coordinator)
    assert result.metadata == {}
    assert media.publications[-1][2] == "drop"


@pytest.mark.parametrize("site", ["parallel", "sequential", "nested"])
@pytest.mark.parametrize("case", ["none", "null-inject", "missing-inject", "exception", "inplace"])
async def test_lossy_real_registry_without_witness_never_promotes_media(site, case):
    from amplifier_module_loop_streaming.dispatch import AttributedHooks
    coordinator, media, loop = setup(boundary=False)
    hooks = HookRegistry()
    posts = []
    async def mutate(event, data):
        posts.append(copy.deepcopy(data))
        if case == "exception":
            raise ValueError("privacy policy failed")
        if case == "inplace":
            data["result"]["output"].pop(NAMESPACE)
            return HookResult()
        if case == "null-inject":
            data["result"] = None
        elif case == "missing-inject":
            data.pop("result")
        return HookResult(action="modify", data=data) if case != "none" else HookResult()
    async def inject(event, data):
        return HookResult(action="inject_context", context_injection="ordinary reminder", ephemeral=True)
    hooks.register("tool:post", mutate, priority=10)
    if case.endswith("inject"):
        hooks.register("tool:post", inject, priority=20)
    if site == "nested":
        hooks = AttributedHooks(hooks, {"parent_tool_call_id": "outer"}, {})
    call = ToolCall(id="call", name="image", arguments={})
    if site == "sequential":
        context = MockContext()
        await loop._execute_tool_with_result_unscoped(
            call, {"image": ImageTool()}, context, hooks, coordinator)
        result = context._messages[0]
        content, metadata = result["content"], result.get("metadata", {})
    else:
        result = await loop._execute_tool_unscoped(
            call, {"image": ImageTool()}, hooks, "group", coordinator)
        content, metadata = result.content, result.metadata
    assert len(posts) == 1 and PIXELS not in json.dumps(posts)
    assert not metadata
    assert not any(disposition == "retain" for _, _, disposition in media.publications)
    assert media.publications[-1][2] == "drop"
    assert UNAVAILABLE in content or content == "null"
    await report_consumer_ready(coordinator)
    assert media.reports == []


@pytest.mark.parametrize("failure", ["incomplete", "missing-result", "wrong-hook", "exception"])
async def test_boundary_ambiguous_or_failed_witness_drops_staging_without_duplicate_hooks(failure):
    coordinator, media, loop = setup()
    hooks = MutatingHooks(lambda data: HookResult())
    async def boundary(data, *, hooks=None, coordinator=None):
        assert data["tool_name"] == "image"
        post = await hooks.emit("tool:post", data)
        post = await coordinator.process_hook_result(post, "tool:post", data["tool_name"])
        if failure == "exception":
            raise RuntimeError("hook boundary failed")
        receipt = {"hook_result": post, "result": data["result"], "complete": True}
        if failure == "incomplete":
            receipt["complete"] = 1
        elif failure == "missing-result":
            receipt.pop("result")
        else:
            receipt["hook_result"] = {"action": "continue"}
        return receipt
    coordinator.register_capability("session.media.tool_post", boundary)
    result = await loop._execute_tool_unscoped(
        ToolCall(id="call", name="image", arguments={}),
        {"image": ImageTool()}, hooks, "group", coordinator)
    assert not result.metadata
    assert len([event for event, _ in hooks.emitted if event == "tool:post"]) == 1
    assert media.publications[-1][2] == "drop"


async def test_boundary_witness_preserves_final_null_and_nested_attribution():
    from amplifier_module_loop_streaming.dispatch import AttributedHooks, AttributedCoordinator
    coordinator, media, loop = setup()
    hooks = HookRegistry()
    post_calls, process_calls, seen = [], [], []
    async def redact(event, data):
        post_calls.append(data)
        assert data["parent_tool_call_id"] == "outer"
        data["result"] = None
        return HookResult(action="modify", data=data)
    async def inject(event, data):
        seen.append(data["result"])
        return HookResult(action="inject_context", context_injection="keep reminder", ephemeral=True)
    hooks.register("tool:post", redact, priority=10)
    hooks.register("tool:post", inject, priority=20)
    observation = {}
    attributed = AttributedHooks(hooks, {"parent_tool_call_id": "outer"}, observation)
    attributed_coordinator = AttributedCoordinator(coordinator, observation)
    original_process = coordinator.process_hook_result
    async def process(result, *args):
        process_calls.append(args)
        return await original_process(result, *args)
    coordinator.process_hook_result = process
    async def boundary(data, *, hooks=None, coordinator=None):
        assert hooks is attributed and coordinator is attributed_coordinator
        post = await hooks.emit("tool:post", data)
        post = await coordinator.process_hook_result(post, "tool:post", data["tool_name"])
        # This fixture's trusted witness observes final result from the last
        # handler; production must install its facade before registration.
        return {"hook_result": post, "result": seen[-1], "complete": True}
    coordinator.register_capability("session.media.tool_post", boundary)
    result = await loop._execute_tool_unscoped(
        ToolCall(id="call", name="image", arguments={}), {"image": ImageTool()},
        attributed, "group", attributed_coordinator)
    assert result.result is None and result.content == "null" and not result.metadata
    assert len(post_calls) == 1
    assert len([row for row in process_calls if row[0] == "tool:post"]) == 1
    assert loop._pending_ephemeral_injections[0]["content"] == "keep reminder"
    assert media.publications[-1][2] == "drop"


@pytest.mark.parametrize("cancellation", ["none", "graceful", "immediate"])
async def test_batch_order_completed_media_survives_cancellation(cancellation):
    coordinator, media, loop = setup()
    context = MockContext()
    second = asyncio.Event()
    class OrderedImages(ImageTool):
        async def execute(self, arguments):
            if arguments["index"] == 0:
                await second.wait()
            else:
                second.set()
                if cancellation == "immediate":
                    # Let the other sibling fully settle before cancelling.
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    raise asyncio.CancelledError()
            if cancellation == "graceful":
                coordinator.cancellation.is_cancelled = True
            return await super().execute(arguments)
    checkpoint_rows = []
    coordinator.register_capability(
        "session.durable_checkpoint", lambda: checkpoint_rows.extend(context._messages))
    calls = [ToolCall(id=f"call-{i}", name="image", arguments={"index": i}) for i in range(2)]
    operation = loop._settle_tool_batch(
        calls, context, {"image": OrderedImages()}, MutatingHooks(lambda data: HookResult()),
        coordinator, 1)
    if cancellation == "immediate":
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        assert await operation is (cancellation == "none")
    rows = [r for r in context._messages if r["role"] == "tool"]
    assert [r["tool_call_id"] for r in rows] == ["call-0", "call-1"]
    assert rows[0]["metadata"][NAMESPACE]["refs"] == [ref("call-0")]
    if cancellation != "immediate":
        assert rows[1]["metadata"][NAMESPACE]["refs"] == [ref("call-1")]
    assert PIXELS not in json.dumps(context._messages)
    assert bool(checkpoint_rows) is (cancellation == "none")


async def test_settled_continuation_roundtrip_retains_refs_without_tool_replay(tmp_path):
    coordinator, media, loop = setup()
    context = MockContext()
    owner = Owner(tmp_path / "saved.json", context)
    coordinator.register_capability("execution.suspension", owner)
    coordinator.register_capability("session.durable_checkpoint", lambda: None)
    calls = [ToolCall(id="call", name="image", arguments={}),
             ToolCall(id="later", name="later", arguments={})]
    class Later:
        async def execute(self, arguments):
            raise ToolExecutionSuspended({"version": 1, "kind": "not_started"})
    with loop.approved_dispatch.run_scope(coordinator), pytest.raises(ExecutionSuspended) as caught:
        await loop._settle_tool_batch(
            calls, context, {"image": ImageTool(), "later": Later()},
            MutatingHooks(lambda data: HookResult()), coordinator, 1)
    frame = caught.value.continuation
    record = frame["settled_tools"][0]["result"]
    assert isinstance(record, dict) and record["metadata"] == {NAMESPACE: {"refs": [ref()]}}
    assert PIXELS not in owner.path.read_text()
    publications = copy.deepcopy(media.publications)
    class Text:
        async def execute(self, arguments):
            return ToolResult(success=True, output="continued")
    assert await loop._settle_tool_batch(
        calls, context, {"later": Text()}, MutatingHooks(lambda data: HookResult()),
        coordinator, 1, frame["settled_tools"])
    assert context._messages[0]["metadata"] == record["metadata"]
    assert media.publications == publications


async def test_legacy_settled_triple_stays_readable_without_parsing_text():
    legacy = ["old", "read", '{"experimental.media.v1":{"refs":[]}}']
    result = SettledToolResult.from_legacy(legacy)
    assert result.content == legacy[2] and result.metadata == {} and result.result is None
    assert result.model_dump()["tool_call_id"] == "old"


async def test_estimator_and_evaluator_recognize_refs_without_payload_or_locator_leak():
    from amplifier_module_loop_streaming import _flatten_message_for_evaluator
    from amplifier_module_loop_streaming._text_estimate import estimate_messages
    message = {"role": "tool", "content": "", "metadata": {NAMESPACE: {"refs": [ref()]}}}
    before = copy.deepcopy(message)
    assert estimate_messages([message]).has_unmeasured_images
    flattened = _flatten_message_for_evaluator(message)
    assert "Image unavailable" in flattened and "no pixels supplied" in flattened
    assert "sha256" not in flattened and message == before
    assert not estimate_messages([{"role": "user", "content": json.dumps(message)}]).has_unmeasured_images


@pytest.mark.parametrize("finalization", [False, True])
@pytest.mark.parametrize("checkpoint", ["ok", "absent", "failure"])
async def test_settled_media_checkpoint_precedes_next_count_send_and_finalization(finalization, checkpoint):
    from tests.test_durable_checkpoint import BatchProvider
    coordinator, media, loop = setup()
    context = MockContext()
    committed = []
    class Provider(BatchProvider):
        def __init__(self):
            super().__init__(count=1)
            self.requests, self.counted = [], []

        def request_budget(self, request, *, context_estimate, request_options=None):
            rows = [m for m in request.messages if (m.metadata or {}).get(NAMESPACE)]
            if rows:
                assert committed
                assert rows[0].metadata[NAMESPACE]["refs"] == [ref("call-0")]
                assert PIXELS not in request.model_dump_json()
            self.counted.append(request)
            return {"estimated_input_tokens": 100, "input_limit_tokens": 10000,
                    "context_token_budget": 10000}

        async def complete(self, request, **kwargs):
            assert self.counted[-1] is request
            self.requests.append(request)
            return await super().complete(request, **kwargs)

        def parse_tool_calls(self, response):
            return [ToolCall(id="call-0", name="image", arguments={})] if self.calls == 1 else []

    provider = Provider()
    def save():
        rows = [r for r in context._messages if r["role"] == "tool"]
        assert rows[0]["metadata"] == {NAMESPACE: {"refs": [ref("call-0")]}}
        assert media.publications[-1][2] == "retain"
        committed.extend(rows)
        return False if checkpoint == "failure" else None
    if checkpoint != "absent":
        coordinator.register_capability("session.durable_checkpoint", save)
    loop = StreamingOrchestrator({"max_iterations": 1 if finalization else 3})
    operation = loop.execute(
        "work", context, {"main": provider}, {"image": ImageTool()},
        MutatingHooks(lambda data: HookResult()), coordinator)
    if checkpoint != "ok":
        with pytest.raises(RuntimeError, match="checkpoint"):
            await operation
        assert provider.calls == len(provider.counted) == 1
    else:
        assert await operation == "done"
        assert provider.calls == 2 and len(provider.counted) == 2
        assert provider.requests[1].tool_choice == ("none" if finalization else None)


@pytest.mark.parametrize("finalization", [False, True])
@pytest.mark.parametrize("lost_capability", [False, True])
async def test_prepared_request_suspension_is_refs_only_and_resume_does_not_reexecute(
    tmp_path, finalization, lost_capability
):
    from tests.test_durable_checkpoint import BatchProvider
    coordinator, media, loop = setup()
    context = MockContext()
    owner = Owner(tmp_path / "prepared.json", context)
    coordinator.register_capability("execution.suspension", owner)
    coordinator.register_capability("execution.resume", owner.claim)
    coordinator.register_capability("session.durable_checkpoint", lambda: None)
    class Provider(BatchProvider):
        def __init__(self):
            super().__init__(count=1)
            self.requests = []
        async def complete(self, request, **kwargs):
            self.requests.append(request)
            return await super().complete(request, **kwargs)
        def parse_tool_calls(self, response):
            return [ToolCall(id="call", name="image", arguments={})] if self.calls == 1 else []
    class Hooks(MutatingHooks):
        def __init__(self):
            super().__init__(lambda data: HookResult())
            self.provider_hooks = 0
        async def emit(self, event, payload=None):
            if event == "provider:request":
                self.provider_hooks += 1
                if self.provider_hooks == 2:
                    owner.hold = True
            return await super().emit(event, payload)
    provider, hooks = Provider(), Hooks()
    loop = StreamingOrchestrator({"max_iterations": 1 if finalization else 3})
    args = (context, {"main": provider}, {"image": ImageTool()}, hooks, coordinator)
    with pytest.raises(ExecutionSuspended) as caught:
        await loop.execute("work", *args)
    frame = caught.value.continuation
    assert frame["phase"] == ("prepared_finalization" if finalization else "prepared_request")
    serialized = owner.path.read_text()
    assert PIXELS not in serialized and NAMESPACE in serialized
    canonical_request = frame["prepared_request"]["request"]
    saved_rows = [row for row in canonical_request["messages"] if row["role"] == "tool"]
    assert saved_rows[0]["tool_call_id"] == "call"
    assert saved_rows[0]["metadata"] == {NAMESPACE: {"refs": [ref()]}}
    before = copy.deepcopy(media.publications)
    owner.hold = False
    if lost_capability:
        coordinator.register_capability(NAMESPACE, None)
        with pytest.raises(RuntimeError, match="Prepared media request lost compatible handling"):
            await StreamingOrchestrator(loop.config).resume(frame, *args)
        assert media.publications == before and provider.calls == 1
        return
    assert await StreamingOrchestrator(loop.config).resume(frame, *args) == "done"
    assert media.publications == before and hooks.provider_hooks == 2
    media_rows = [m for m in provider.requests[-1].messages if (m.metadata or {}).get(NAMESPACE)]
    assert media_rows[0].metadata[NAMESPACE]["refs"] == [ref()]
    assert PIXELS not in provider.requests[-1].model_dump_json()


async def test_restored_refs_on_unsupported_path_are_unavailable_at_same_position():
    from tests.test_ephemeral_cache_persist_mode import RequestCapturingProvider
    coordinator, _, loop = setup(cap=False)
    context, provider = MockContext(), RequestCapturingProvider()
    original = {"role": "user", "content": "original", "metadata": {
        NAMESPACE: {"refs": [ref()]}, "owner": {"opaque": "preserved"},
    }}
    await context.add_message(copy.deepcopy(original))
    await loop.execute("continue", context, {"main": provider}, {}, ScriptedHooks({}), coordinator)
    request_row = provider.requests[0].messages[0]
    assert request_row.role == "user" and "original" in request_row.content
    assert UNAVAILABLE in request_row.content
    assert request_row.metadata == {"owner": {"opaque": "preserved"}}
    assert context._messages[0] == original


async def test_utility_boundary_keeps_requests_text_only_before_saving_or_adapter():
    coordinator, _, loop = setup()
    request = ChatRequest(messages=[Message(role="tool", tool_call_id="call", content="text",
        metadata={NAMESPACE: {"refs": [ref()]}, "other": 3})])
    await loop._goal_utility_boundary("evaluate", "main", request, {}, coordinator)
    assert "Image unavailable in text-only utility" in request.messages[0].content
    assert request.messages[0].metadata == {"other": 3}
    request = ChatRequest(messages=[Message(role="user", content=[
        {"type": "text", "text": "caption"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PIXELS}},
    ])])
    await loop._goal_utility_boundary("judge", "main", request, {}, coordinator)
    assert PIXELS not in request.model_dump_json()
    assert "caption" in request.model_dump_json()
    assert "no pixels supplied" in request.model_dump_json()


@pytest.mark.parametrize("case", ["null", "missing", "string", "list"])
async def test_real_nested_dispatch_never_parses_or_resurrects_effective_result(environment, case):
    _, coordinator, _, _ = environment
    quoted = '{"success":true,"output":{"experimental.media.v1":{"refs":[]}}}'
    async def replacement(event, data):
        if data["tool_name"] == "read":
            data = dict(data)
            if case == "missing":
                data.pop("result")
            else:
                data["result"] = None if case == "null" else quoted if case == "string" else [quoted]
            return HookResult(action="modify", data=data)
        return HookResult()
    coordinator.hooks.register("tool:post", replacement)
    result, _ = await call_client(environment)
    assert result["status"] == "unknown" and result["success"] is None
    assert result["output"] == (None if case in {"null", "missing"} else quoted if case == "string" else [quoted])


@pytest.mark.parametrize("outcome", ["success", "failed", "unknown", "string", "redacted", "revoked"])
async def test_nested_verifier_binds_only_returned_effective_refs_to_active_parent(environment, outcome):
    from tests.test_approved_dispatch import Client
    from amplifier_module_loop_streaming import mount
    _, coordinator, loop, _ = environment
    # Publish this exact loop's verifier as mount does, without replacing its
    # configured dispatcher or execution owner.
    coordinator.register_capability("tools.dispatch.media_refs", loop.approved_dispatch.media_refs)
    sid = str(coordinator.session_id)
    actual_refs, origins, checks = [], [], []
    class NestedMedia(Media):
        async def ingest(self, value, *, source_format, origin, policy):
            if value.content:
                reference = ref(origin["id"])
                reference["session_id"] = reference["provenance"]["session_id"] = sid
                actual_refs.append(reference)
                self.staged[origin["id"]] = [reference]
                return {"value": ToolResult(success=outcome != "failed", output={
                    "caption": "image", NAMESPACE: {"refs": [reference]},
                }), "staged_refs": [reference]}
            output = value.output
            if isinstance(output, dict) and NAMESPACE in output:
                refs = output[NAMESPACE]["refs"]
                assert all(loop.approved_dispatch.media_refs(origin, r) for r in refs)
                origins.append(origin)
            return {"value": value, "staged_refs": []}

        async def publish(self, refs, *, origin, disposition):
            if disposition == "retain" and origin["id"] == "parent":
                assert all(loop.approved_dispatch.media_refs(origin, r) for r in refs)
            self.publications.append((copy.deepcopy(refs), origin, disposition))
            return {"refs": refs, "publication": {"state": "published", "durability": "assets-only"}}
    media = NestedMedia()
    coordinator.register_capability(NAMESPACE, media)
    coordinator.register_capability("session.media.tool_post", fixture_boundary)
    await coordinator.mount("tools", ImageTool(), name="read")
    async def post(event, data):
        if data["tool_name"] == "read" and outcome in {"unknown", "string", "redacted"}:
            data = dict(data)
            if outcome == "unknown":
                data["result"] = {"output": data["result"]["output"]}
            elif outcome == "string":
                data["result"] = json.dumps(data["result"])
            else:
                data["result"] = None
            return HookResult(action="modify", data=data)
        return HookResult()
    coordinator.hooks.register("tool:post", post)
    async def handler(lease):
        result = await lease.call("read", {})
        origin = {"kind": "tool", "session_id": sid, "id": "parent"}
        if outcome == "revoked":
            loop.config["programmatic_allowed_tools"] = []
        checks.append(loop.approved_dispatch.media_refs(origin, actual_refs[0]))
        assert loop.approved_dispatch.media_refs(
            {**origin, "created_at": "2026-01-01T00:00:00+00:00", "expires_at": None},
            actual_refs[0]) is (outcome == "success")
        forged = copy.deepcopy(actual_refs[0])
        forged["width"] += 1
        assert not loop.approved_dispatch.media_refs(origin, forged)
        assert not loop.approved_dispatch.media_refs({**origin, "id": "other"}, actual_refs[0])
        if outcome == "success":
            return result["output"]
        return {"status": result["status"]}
    _, client = await call_client(environment, Client(coordinator, handler))
    assert checks == [outcome == "success"]
    if outcome == "success":
        assert origins[-1]["id"] == "parent"
        assert media.publications[-1][1]["id"] == "parent"
    assert not loop.approved_dispatch.media_refs(
        {"kind": "tool", "session_id": sid, "id": "parent"}, actual_refs[0])
    assert client.lease.scope.media_refs == []


@pytest.mark.parametrize("case", ["refs", "inline", "forged"])
async def test_nested_inference_is_text_only_before_selected_adapter_count_and_send(environment, case):
    from tests.test_approved_dispatch import Client
    from amplifier_core.message_models import ChatResponse, TextBlock
    _, coordinator, _, _ = environment
    media = Media()
    validations, requests, counts = [], [], []
    async def ingest(value, **kwargs):
        return {"value": value, "staged_refs": []}
    media.ingest = ingest
    def validate(value, *, purpose):
        validations.append((value, purpose))
        if case == "forged":
            raise ValueError("forged reference")
        return {"valid": True}
    media.validate = validate
    coordinator.register_capability(NAMESPACE, media)
    class Provider:
        name = "fixture"
        def get_info(self):
            return SimpleNamespace(defaults={"model": "fixture"})
        def request_budget(self, request, *, context_estimate, request_options=None):
            counts.append(request)
            assert NAMESPACE not in request.model_dump_json()
            assert PIXELS not in request.model_dump_json()
            assert "no pixels supplied" in request.model_dump_json()
            return {"estimated_input_tokens": 10, "input_limit_tokens": 1000}
        async def complete(self, request, **kwargs):
            assert counts[-1] is request
            requests.append(request)
            return ChatResponse(content=[TextBlock(text="text answer")])
    provider = Provider()
    await coordinator.mount("providers", provider, name="fixture")
    request = ChatRequest(messages=[Message(role="user", content=[
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PIXELS}}
    ] if case == "inline" else "caption", metadata=None if case == "inline" else {
        NAMESPACE: {"refs": [ref()]}, "owner": 7,
    })])
    before = request.model_dump_json()
    async def handler(lease):
        complete = coordinator.get_capability("model.complete")
        if case == "forged":
            with pytest.raises(ValueError, match="forged"):
                await complete(request)
        else:
            await complete(request)
        return {"done": True}
    await call_client(environment, Client(coordinator, handler))
    assert request.model_dump_json() == before
    assert len(requests) == (0 if case == "forged" else 1)
    assert bool(validations) is (case != "inline")