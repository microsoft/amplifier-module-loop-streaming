"""Whole streaming engine + actual F1 adapter, synthetic transport owner only."""

import base64
import copy
import hashlib
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from amplifier_core.message_models import ChatRequest, Message
from amplifier_foundation.session.media import MediaPolicy, SessionMediaStore, bind_session_media
from amplifier_foundation.session.media_provider import MediaPreparedProvider
from amplifier_module_loop_streaming import StreamingOrchestrator
from amplifier_module_loop_streaming.media import NAMESPACE
from tests.test_durable_checkpoint import MockContext
from tests.test_ephemeral_cache_persist_mode import MockCoordinator, MockResponse, ScriptedHooks
from tests.test_durable_checkpoint import OrderedTool
from tests.test_execution_suspension import Owner
from amplifier_module_loop_streaming import ExecutionSuspended
from integration.test_media_store_pairing import png, acknowledge, Images

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("tool_count,finalization", [(0, False), (1, False), (1, True)])
@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
@pytest.mark.parametrize("resume_adapter", [None, "retained", "fresh"])
async def test_whole_engine_count_send_share_actual_private_projection(
    tmp_path, tool_count, finalization, measured, image_format, resume_adapter,
):
    coordinator = MockCoordinator()
    coordinator.session_id = "stream-pair"
    context = MockContext()
    store = SessionMediaStore(
        tmp_path / "media", coordinator.session_id,
        policy=MediaPolicy(1024 * 1024, 10000, 8),
        checkpoint_verifier=__import__(
            "amplifier_foundation.session.checkpoint", fromlist=["verified_checkpoint_refs"]
        ).verified_checkpoint_refs,
    )
    cap = bind_session_media(coordinator, store=store, enabled=True)
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), "green").save(buffer, format=image_format)
    raw = buffer.getvalue()
    mime_type = "image/png" if image_format == "PNG" else "image/jpeg"
    origin = {"kind": "tool", "session_id": coordinator.session_id, "id": "original"}
    reference = (await cap.ingest(raw, source_format="image-bytes", origin=origin, policy=None))["value"]
    await cap.publish([reference], origin=origin, disposition="retain")
    await context.add_message({
        "role": "tool", "tool_call_id": "original", "name": "image", "content": "saved",
        "metadata": {NAMESPACE: {"refs": [reference]}, "owner": "preserved"},
    })
    acknowledge(store, tmp_path / "session", context._messages)
    coordinator.register_capability("session.durable_checkpoint", lambda: acknowledge(
        store, tmp_path / "session", context._messages))
    counted, sent, payloads, support = [], [], [], []
    class SyntheticStreamOwner:
        name = "synthetic-stream-owner"
        config = {}
        calls = 0
        def get_info(self):
            return SimpleNamespace(
                defaults={"model": "fixture"}, capabilities=["request_budget:provider_count"])
        def get_media_request_support(self, *, model, request_options=None):
            return {"v": 1, "positions": ["tool"], "mime_types": [mime_type]}
        def get_stream_request_support(self, *, model, request_options=None):
            support.append((model, copy.deepcopy(request_options)))
            return {"v": 1, "tools": "chat_request"}
        def request_budget(self, request, *, context_estimate, request_options=None):
            assert request_options == (
                {"extended_thinking": True} if finalization else {})
            counted.append(request)
            blocks = [block for message in request.messages if isinstance(message.content, list)
                      for block in message.content if getattr(block, "type", None) == "image"]
            assert len(blocks) == 1
            assert hashlib.sha256(base64.b64decode(blocks[0].source["data"])).hexdigest() == reference["sha256"]
            if request.tool_choice != "none":
                assert len(request.tools or []) == tool_count
            return {"estimated_input_tokens": 10, "input_limit_tokens": 10000,
                    "context_token_budget": 10000,
                    "measurement": {"kind": "provider_count", "source": "synthetic", "input_tokens": 10}}
        async def complete(self, request, **kwargs):
            assert finalization
            assert kwargs == {"extended_thinking": True}
            assert counted[-1] is request
            self.calls += 1
            sent.append(request)
            payloads.append(request.model_dump_json())
            return MockResponse("done")
        async def stream(self, request, **kwargs):
            assert kwargs == {}, "No duplicate runtime tools may appear after count"
            assert counted[-1] is request
            self.calls += 1
            sent.append(request)
            payloads.append(request.model_dump_json())
            yield {"content": "done"}
        def parse_tool_calls(self, response):
            if finalization and self.calls == 1:
                from amplifier_core.message_models import ToolCall
                return [ToolCall(id="next", name="mock_tool", arguments={"index": 0})]
            return []
    owner = SyntheticStreamOwner()
    if finalization:
        # Actual complete-only provider semantics: public stream must be absent.
        owner.stream = None
    def wrap():
        return MediaPreparedProvider(owner, cap, provider_identity={
            "provider": "synthetic-stream-owner", "account": "fixture", "generation": 1,
        }, model="fixture")
    provider = wrap()
    if measured:
        async def measured_view(*, provider, retain_contents, count_view, fit_output=None):
            view = await context.get_messages_for_request()
            attempt = await count_view(view)
            assert attempt["request_options"] == (
                {"extended_thinking": True} if finalization else {})
            assert provider.media_budget_witness(
                attempt["dispatch"], request_options=attempt["request_options"])["validated"] is True
            return {"base_view": view, "final_attempt": attempt, "count_calls": 1}
        coordinator.register_capability("context.measured_request_view", measured_view)
    class Tool(Images):
        name, description, input_schema = "image", "synthetic image", {"type": "object"}
    try:
        tools = ({"mock_tool": OrderedTool(count=1)} if finalization else
                 {"image": Tool()} if tool_count else {})
        config = {
            "max_iterations": 1 if finalization else 3, "extended_thinking": True,
        }
        loop, hooks = StreamingOrchestrator(config), ScriptedHooks({})
        if resume_adapter:
            suspension = Owner(tmp_path / "continuation.json", context)
            coordinator.register_capability("execution.suspension", suspension)
            coordinator.register_capability("execution.resume", suspension.claim)
            delay_calls = 0
            async def pause_after_count(*args, **kwargs):
                nonlocal delay_calls
                delay_calls += 1
                if delay_calls == (2 if finalization else 1):
                    suspension.hold = True
            loop._apply_rate_limit_delay = pause_after_count
            with pytest.raises(ExecutionSuspended) as stopped:
                await loop.execute("work", context, {"main": provider}, tools, hooks, coordinator)
            frame = stopped.value.continuation
            assert frame["phase"] == ("prepared_finalization" if finalization else "prepared_request")
            assert all(base64.b64encode(raw).decode() not in str(value) for value in (frame, context._messages))
            assert len(sent) == (1 if finalization else 0)
            if resume_adapter == "fresh":
                provider.close()
                provider = wrap()
            suspension.hold = False
            answer = await StreamingOrchestrator(config).resume(
                frame, context, {"main": provider}, tools, hooks, coordinator)
        else:
            answer = await loop.execute("work", context, {"main": provider}, tools, hooks, coordinator)
        assert answer == "done"
        assert len(sent) == (2 if finalization else 1)
        assert len(counted) == len(sent) + bool(resume_adapter)
        assert bool(support) is not finalization
        assert base64.b64encode(raw).decode() in payloads[0]
        assert all(base64.b64encode(raw).decode() not in str(row) for row in context._messages)
        # The adapter releases its private projected messages after stream close.
        assert all(request.messages == [] for request in sent)
    finally:
        provider.close()
        store.close()