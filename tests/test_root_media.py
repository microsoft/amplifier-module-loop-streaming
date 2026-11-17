"""Offline root contract fixtures, not host activation or vendor vision proof."""

import asyncio
import base64
import copy
import hashlib
import io
import json

import pytest
from amplifier_core import HookResult, ToolResult
from amplifier_core.message_models import ToolCall
from PIL import Image

from amplifier_module_loop_streaming import SettledToolResult, StreamingOrchestrator
from amplifier_module_loop_streaming.media import NAMESPACE, effective_result
from amplifier_module_loop_streaming.media import initialize_media_consumer, report_consumer_ready
from tests.test_durable_checkpoint import BatchProvider, MockContext
from tests.test_ephemeral_cache_persist_mode import MockCoordinator, ScriptedHooks

pytestmark = pytest.mark.asyncio


class Media:
    """Structural capability spy; actual validation/storage belongs to Foundation."""
    version = 1

    def __init__(self):
        self.staged, self.publications = [], []

    def ingest(self, value, *, source_format, origin, policy):
        assert source_format == "core-tool-result" and policy is None
        refs = []
        for block in value.content or []:
            if block.type != "image":
                continue
            raw = base64.b64decode(block.source["data"])
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
            refs.append({
                "v": 1, "session_id": origin["session_id"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "mime": block.source["media_type"], "bytes": len(raw),
                "width": width, "height": height,
                "provenance": {**origin, "created_at": "2026-01-01T00:00:00+00:00",
                               "expires_at": None},
            })
        self.staged.extend(refs)
        if refs:
            value = ToolResult(success=value.success, output={
                "text": value.output, NAMESPACE: {"refs": refs},
            })
        return {"value": value, "staged_refs": refs}

    def publish(self, refs, *, origin, disposition):
        assert all(ref in self.staged for ref in refs)
        assert all(ref["provenance"]["id"] == origin["id"] for ref in refs)
        self.publications.append((copy.deepcopy(refs), disposition))
        return {"refs": refs}

    def validate(self, *args, **kwargs):
        pass

    def resolve(self, *args, **kwargs):
        pytest.fail("Loop must not resolve pixels")

    def prepare_request(self, *args, **kwargs):
        pytest.fail("Loop must not hydrate")

    def report(self, *args, **kwargs):
        pass


class Images:
    name, description = "mock_tool", "real PNG/JPEG bytes, synthetic media owner"
    input_schema = {"type": "object"}

    def __init__(self, mime):
        buffer = io.BytesIO()
        Image.new("RGB", (3, 2), "purple").save(
            buffer, format="PNG" if mime == "image/png" else "JPEG")
        self.raw, self.mime, self.calls = buffer.getvalue(), mime, 0

    async def execute(self, arguments):
        self.calls += 1
        return ToolResult(success=True, output="caption", content=[{
            "type": "image", "source": {"type": "base64", "media_type": self.mime,
                                       "data": base64.b64encode(self.raw).decode()},
        }])


class Hooks(ScriptedHooks):
    def __init__(self, action="retain"):
        super().__init__({})
        self.action, self.posts = action, 0

    async def emit(self, event, data=None):
        if event != "tool:post":
            return await super().emit(event, data)
        self.posts += 1
        assert "base64" not in json.dumps(data)
        if self.action == "null":
            return HookResult(action="modify", data={**data, "result": None})
        if self.action == "missing":
            return HookResult(action="modify", data={"tool_name": "mock_tool"})
        if self.action == "inplace":
            data["result"]["output"].pop(NAMESPACE)
        if self.action == "redact":
            return HookResult(action="modify", data={**data, "result": "redacted"})
        if self.action == "deny":
            return HookResult(action="deny", reason="fixture redaction")
        return HookResult()


async def fixture_boundary(data, *, hooks, coordinator):
    """Synthetic complete witness; real Bridge is qualified in host integration."""
    post = await hooks.emit("tool:post", data)
    post = await coordinator.process_hook_result(post, "tool:post", data["tool_name"])
    return {"complete": True, "hook_result": post,
            "result": effective_result(post, data["result"])}


def setup():
    coordinator, media = MockCoordinator(), Media()
    coordinator.session_id = "root"
    coordinator.register_capability(NAMESPACE, media)
    coordinator.register_capability("session.media.tool_post", fixture_boundary)
    return coordinator, media


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
@pytest.mark.parametrize("action", ["retain", "null", "missing", "inplace", "redact", "deny"])
async def test_final_effective_result_alone_owns_root_refs(mime, action):
    coordinator, media = setup()
    tool, hooks, loop = Images(mime), Hooks(action), StreamingOrchestrator({})
    call = ToolCall(id="call", name=tool.name, arguments={})
    result = await loop._execute_tool_only(call, {tool.name: tool}, hooks, "group", coordinator)
    assert isinstance(result, SettledToolResult)
    assert result.tool_call_id == "call" and tool.calls == hooks.posts == 1
    assert base64.b64encode(tool.raw).decode() not in json.dumps(result.model_dump())
    assert SettledToolResult.from_legacy(result.model_dump()) == result
    if action == "retain":
        refs = result.metadata[NAMESPACE]["refs"]
        assert refs[0]["mime"] == mime
        assert refs[0]["sha256"] == hashlib.sha256(tool.raw).hexdigest()
        assert media.publications == [(refs, "retain")]
    else:
        assert result.metadata == {} and media.publications[-1][1] == "drop"
        if action in {"null", "missing", "deny"}:
            assert result.content == "null"


async def test_incomplete_post_witness_cannot_promote_refs():
    coordinator, media = setup()
    async def incomplete(data, **kwargs):
        receipt = await fixture_boundary(data, **kwargs)
        return {**receipt, "complete": False}
    coordinator.register_capability("session.media.tool_post", incomplete)
    result = await StreamingOrchestrator({})._execute_tool_only(
        ToolCall(id="call", name="mock_tool", arguments={}),
        {"mock_tool": Images("image/png")}, Hooks(), "group", coordinator)
    assert not result.metadata and "Image unavailable" in result.content
    assert media.publications[-1][1] == "drop"


class CountedProvider(BatchProvider):
    def __init__(self, *, count=1, failure=None):
        super().__init__(count)
        self.counted, self.sent, self.released, self.options = [], [], [], {}
        self.failure = failure

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.counted.append(request)
        self.options[id(request)] = copy.deepcopy(request_options)
        if self.failure == "count":
            raise ValueError("synthetic count refused")
        if self.failure == "cancel":
            raise asyncio.CancelledError()
        return {"estimated_input_tokens": 10, "input_limit_tokens": 10000,
                "context_token_budget": 9000}

    async def complete(self, request, **kwargs):
        assert request is self.counted[-1] and kwargs == self.options[id(request)]
        self.sent.append(request)
        if self.failure == "send":
            raise ValueError("synthetic send refused")
        return await super().complete(request, **kwargs)

    def release_media_request(self, request, *, request_options):
        assert request_options == self.options[id(request)]
        self.released.append(request)


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
async def test_batch_checkpoint_ref_only_pair_and_cold_context_reload(mime, tmp_path):
    coordinator, media = setup()
    context, tool, provider = MockContext(), Images(mime), CountedProvider()
    path = tmp_path / "canonical.json"
    def checkpoint():
        rows = context._messages
        toolrow = next(row for row in rows if row["role"] == "tool")
        assistant = next(row for row in rows if row.get("tool_calls"))
        assert assistant["tool_calls"][0]["id"] == toolrow["tool_call_id"] == "call-0"
        assert toolrow["metadata"][NAMESPACE]["refs"]
        path.write_text(json.dumps(rows))
    coordinator.register_capability("session.durable_checkpoint", checkpoint)
    await StreamingOrchestrator({}).execute(
        "read image", context, {"main": provider}, {tool.name: tool}, Hooks(), coordinator)
    assert tool.calls == 1
    assert provider.released == provider.counted
    assert all(sent is counted for sent, counted in zip(provider.sent, provider.counted))
    saved = json.loads(path.read_text())
    assert base64.b64encode(tool.raw).decode() not in path.read_text()
    restored, second = MockContext(), CountedProvider(count=0)
    restored._messages = saved
    await StreamingOrchestrator({}).execute(
        "inspect saved image", restored, {"main": second}, {tool.name: tool}, Hooks(), coordinator)
    assert tool.calls == 1  # No replay on reload.
    assert any(message.metadata.get(NAMESPACE) for message in second.sent[0].messages
               if message.metadata)
    assert second.released == second.counted


@pytest.mark.parametrize("failure", ["count", "send", "cancel"])
async def test_owned_count_candidate_cleanup_on_refusal_or_cancellation(failure):
    coordinator, _ = setup()
    provider = CountedProvider(count=0, failure=failure)
    error = asyncio.CancelledError if failure == "cancel" else ValueError
    with pytest.raises(error):
        await StreamingOrchestrator({}).execute(
            "work", MockContext(), {"main": provider}, {}, Hooks(), coordinator)
    assert len(provider.released) == len(provider.counted) == 1
    assert provider.released[0] is provider.counted[0]


async def test_media_requires_checkpoint_but_text_only_hosts_do_not():
    coordinator, _ = setup()
    provider = CountedProvider()
    with pytest.raises(RuntimeError, match="require a durable"):
        await StreamingOrchestrator({}).execute(
            "read", MockContext(), {"main": provider},
            {"mock_tool": Images("image/png")}, Hooks(), coordinator)
    assert len(provider.sent) == 1


@pytest.mark.parametrize("changed", [False, True])
async def test_canonical_stream_tools_and_cleanup_after_iterator_close(changed):
    class Stream(CountedProvider):
        def __init__(self):
            super().__init__(count=0)
            self.support, self.closed = True, False

        def get_stream_request_support(self, **kwargs):
            return {"v": 1, "tools": "chat_request"} if self.support else None

        def request_budget(self, request, **kwargs):
            result = super().request_budget(request, **kwargs)
            if changed:
                self.support = False
            return result

        async def stream(self, request, **kwargs):
            assert "tools" not in kwargs
            assert request is self.counted[-1]
            assert request.tools[0].name == "mock_tool"
            self.sent.append(request)
            try:
                yield {"content": "stream done"}
            finally:
                self.closed = True

        def release_media_request(self, request, **kwargs):
            assert not self.sent or self.closed
            super().release_media_request(request, **kwargs)

    coordinator, _ = setup()
    provider = Stream()
    operation = StreamingOrchestrator({}).execute(
        "stream", MockContext(), {"main": provider}, {"mock_tool": Images("image/png")},
        Hooks(), coordinator)
    if changed:
        with pytest.raises(RuntimeError, match="contract changed"):
            await operation
        assert not provider.sent
    else:
        assert await operation == "stream done"
        assert provider.closed
    assert provider.released == provider.counted


class StartupMedia(Media):
    """Structural readiness spy, not a genuine Foundation construction."""
    session_id = "root"
    enabled = False

    def __init__(self):
        super().__init__()
        self.reports = []

    def report(self, component, **report):
        self.reports.append((component, report))


class StartupBoundary:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.consistent = True
        self.valid = True
        self.graph_receipt = None

    async def __call__(self, *args, **kwargs):
        pytest.fail("Initialization cannot execute a fake tool post")

    def participant(self, component):
        assert component == "loop-streaming"
        return self.coordinator

    def readiness(self):
        return {"contract": 1, "registration_consistent": self.consistent,
                "scope": "python-loader-and-explicit-host"}

    def receipt_status(self, receipt):
        return {"valid": False, "reason": "construction_not_sealed" if self.valid else "invalid"}


@pytest.mark.parametrize("missing", [
    None, "capability", "boundary", "checkpoint", "mounted",
    "foreign-session", "foreign-participant", "registration", "construction",
])
async def test_initial_readiness_requires_actual_mount_and_participation(missing):
    coordinator = MockCoordinator()
    coordinator.session_id, coordinator.hooks = "root", Hooks()
    loop, cap = StreamingOrchestrator({}), StartupMedia()
    boundary = StartupBoundary(coordinator)
    coordinator._mounts["orchestrator"] = loop
    coordinator.get = coordinator._mounts.get
    coordinator.register_capability("tools.dispatch", loop.approved_dispatch)
    coordinator.register_capability(NAMESPACE, cap)
    coordinator.register_capability("session.media.tool_post", boundary)
    coordinator.register_capability("session.durable_checkpoint", lambda: pytest.fail("no synthetic save"))
    if missing == "capability":
        coordinator.register_capability(NAMESPACE, object())
    elif missing in {"boundary", "checkpoint"}:
        coordinator.register_capability(
            "session.media.tool_post" if missing == "boundary" else "session.durable_checkpoint", None)
    elif missing == "mounted":
        coordinator._mounts["orchestrator"] = object()
    elif missing == "foreign-session":
        cap.session_id = "foreign"
    elif missing == "foreign-participant":
        boundary.coordinator = object()
    elif missing == "registration":
        boundary.consistent = False
    elif missing == "construction":
        boundary.valid = False
    registration = await initialize_media_consumer(coordinator, loop, "loop-streaming")
    assert registration._ready() is (missing is None)
    assert cap.enabled is False and not cap.staged and not cap.publications
    if missing != "capability":
        assert cap.reports[-1][1]["details"] == {
            "contract": 1, "role": "loop", "ready": missing is None, "phase": "initialization"}


async def test_readiness_withdrawal_is_not_revived_by_old_post_observation():
    coordinator = MockCoordinator()
    coordinator.session_id, coordinator.hooks = "root", Hooks()
    cap, loop = StartupMedia(), StreamingOrchestrator({})
    boundary = StartupBoundary(coordinator)
    coordinator.get = lambda kind: loop if kind == "orchestrator" else None
    coordinator.register_capability(NAMESPACE, cap)
    coordinator.register_capability("tools.dispatch", loop.approved_dispatch)
    coordinator.register_capability("session.media.tool_post", boundary)
    coordinator.register_capability("session.durable_checkpoint", lambda: None)
    registration = await initialize_media_consumer(coordinator, loop, "loop-streaming")
    assert cap.reports[-1][1]["details"]["ready"] is True
    await registration.close()
    assert cap.reports[-1][1]["details"]["ready"] is False
    await report_consumer_ready(coordinator, post_complete=True)
    await registration.refresh()
    readiness = [report for _, report in cap.reports if report["event"] == "consumer-ready"]
    assert [report["details"]["ready"] for report in readiness] == [True, False, False]


async def test_consumer_readiness_preserves_host_owned_dispatch_wrapper_and_denial():
    coordinator = MockCoordinator()
    coordinator.session_id, coordinator.hooks = "root", Hooks()
    cap, loop = StartupMedia(), StreamingOrchestrator({})
    boundary = StartupBoundary(coordinator)
    coordinator.get = lambda kind: loop if kind == "orchestrator" else None
    coordinator.register_capability(NAMESPACE, cap)
    coordinator.register_capability("session.media.tool_post", boundary)
    coordinator.register_capability("session.durable_checkpoint", lambda: None)
    calls = []
    class HostDispatch:
        def bind(self):
            calls.append("denied")
            raise PermissionError("host policy denied this dispatch")
    wrapper = HostDispatch()
    coordinator.register_capability("tools.dispatch", wrapper)
    await initialize_media_consumer(coordinator, loop, "loop-streaming")
    assert cap.reports[-1][1]["details"]["ready"] is True
    assert coordinator.get_capability("tools.dispatch") is wrapper
    with pytest.raises(PermissionError, match="host policy denied"):
        coordinator.get_capability("tools.dispatch").bind()
    assert calls == ["denied"] and loop._tool_calls_this_turn == 0