"""Whole streaming engine + actual F1 adapter, synthetic transport owner only."""

import base64
import copy
import hashlib
from types import SimpleNamespace

import pytest
from amplifier_core.message_models import ChatRequest, Message
from amplifier_foundation.session.media import MediaPolicy, SessionMediaStore, bind_session_media
from amplifier_foundation.session.media_provider import MediaPreparedProvider
from amplifier_module_loop_streaming import StreamingOrchestrator
from amplifier_module_loop_streaming.media import NAMESPACE
from tests.test_durable_checkpoint import MockContext
from tests.test_ephemeral_cache_persist_mode import MockCoordinator, ScriptedHooks
from integration.test_media_store_pairing import png, acknowledge, Images

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("tool_count", [0, 1])
@pytest.mark.parametrize("measured", [False, True])
async def test_whole_engine_count_send_share_actual_private_projection(tmp_path, tool_count, measured):
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
    raw = png("green")
    origin = {"kind": "tool", "session_id": coordinator.session_id, "id": "original"}
    reference = (await cap.ingest(raw, source_format="image-bytes", origin=origin, policy=None))["value"]
    await cap.publish([reference], origin=origin, disposition="retain")
    await context.add_message({
        "role": "tool", "tool_call_id": "original", "name": "image", "content": "saved",
        "metadata": {NAMESPACE: {"refs": [reference]}, "owner": "preserved"},
    })
    acknowledge(store, tmp_path / "session", context._messages)
    counted, sent, payloads, support = [], [], [], []
    class SyntheticStreamOwner:
        name = "synthetic-stream-owner"
        config = {}
        def get_info(self):
            return SimpleNamespace(
                defaults={"model": "fixture"}, capabilities=["request_budget:provider_count"])
        def get_media_request_support(self, *, model, request_options=None):
            return {"v": 1, "positions": ["tool"], "mime_types": ["image/png"]}
        def get_stream_request_support(self, *, model, request_options=None):
            support.append((model, copy.deepcopy(request_options)))
            return {"v": 1, "tools": "chat_request"}
        def request_budget(self, request, *, context_estimate, request_options=None):
            assert request_options == {}
            counted.append(request)
            blocks = [block for message in request.messages if isinstance(message.content, list)
                      for block in message.content if getattr(block, "type", None) == "image"]
            assert len(blocks) == 1
            assert hashlib.sha256(base64.b64decode(blocks[0].source["data"])).hexdigest() == reference["sha256"]
            assert len(request.tools or []) == tool_count
            return {"estimated_input_tokens": 10, "input_limit_tokens": 10000,
                    "context_token_budget": 10000,
                    "measurement": {"kind": "provider_count", "source": "synthetic", "input_tokens": 10}}
        async def complete(self, request, **kwargs):
            pytest.fail("A real optional stream owner was selected")
        async def stream(self, request, **kwargs):
            assert kwargs == {}, "No duplicate runtime tools may appear after count"
            assert counted[-1] is request
            sent.append(request)
            payloads.append(request.model_dump_json())
            yield {"content": "done"}
        def parse_tool_calls(self, response):
            return []
    provider = MediaPreparedProvider(
        SyntheticStreamOwner(), cap, provider_identity={
            "provider": "synthetic-stream-owner", "account": "fixture", "generation": 1,
        }, model="fixture")
    if measured:
        async def measured_view(*, provider, retain_contents, count_view, fit_output=None):
            view = await context.get_messages_for_request()
            attempt = await count_view(view)
            assert attempt["request_options"] == {}
            assert provider.media_budget_witness(
                attempt["dispatch"], request_options=attempt["request_options"])["validated"] is True
            return {"base_view": view, "final_attempt": attempt, "count_calls": 1}
        coordinator.register_capability("context.measured_request_view", measured_view)
    class Tool(Images):
        name, description, input_schema = "image", "synthetic image", {"type": "object"}
    try:
        assert await StreamingOrchestrator({}).execute(
            "work", context, {"main": provider},
            {"image": Tool()} if tool_count else {}, ScriptedHooks({}), coordinator) == "done"
        assert len(sent) == len(counted) == 1
        assert support
        assert base64.b64encode(raw).decode() in payloads[0]
        assert all(base64.b64encode(raw).decode() not in str(row) for row in context._messages)
        # The adapter releases its private projected messages after stream close.
        assert sent[0].messages == []
    finally:
        provider.close()
        store.close()