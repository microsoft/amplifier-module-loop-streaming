"""Explicit candidate-graph checks, run separately from the Core-only suite.

Run with the exact selected Foundation source on PYTHONPATH. These tests use
real storage/checkpoint/projection but a synthetic trustworthy post boundary;
they do not qualify a host registration facade or any vendor transport.
"""

import base64
import copy
import hashlib
import io
import json

import pytest
from PIL import Image
from amplifier_core import HookResult, ToolResult
from amplifier_core.message_models import ChatRequest, Message, ToolCall
from amplifier_foundation.session.checkpoint import (
    acknowledge_legacy_checkpoint, verified_checkpoint_refs,
)
from amplifier_foundation.session.media import MediaPolicy, MediaRefusal, SessionMediaStore, bind_session_media
from amplifier_module_loop_streaming import StreamingOrchestrator
from amplifier_module_loop_streaming.media import NAMESPACE, checkpoint_settled_results, report_consumer_ready
from tests.test_media_settlement import fixture_boundary, MutatingHooks, MockContext, MockCoordinator
from tests.test_approved_dispatch import environment, call_client, Client  # noqa: F401

pytestmark = pytest.mark.asyncio


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), color).save(stream, format="PNG")
    return stream.getvalue()


class Images:
    def __init__(self):
        self.images = [png("red"), png("blue")]

    async def execute(self, arguments):
        return ToolResult(success=True, output={"useful": "preserved"}, content=[
            {"type": "text", "text": "caption"},
            *[{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                          "data": base64.b64encode(image).decode()}}
              for image in self.images],
        ])


def acknowledge(store, directory, messages):
    directory.mkdir(exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in messages), encoding="utf-8")
    receipt = acknowledge_legacy_checkpoint(
        directory, session_id=store.session_id, expected_messages=messages)
    return store.acknowledge_checkpoint(receipt)


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("keep", ["all", "one", "none"])
async def test_actual_store_final_group_publication_and_source_sealed_checkpoint(tmp_path, sequential, keep):
    coordinator = MockCoordinator()
    coordinator.session_id = "paired"
    loop, context, tool = StreamingOrchestrator({}), MockContext(), Images()
    store = SessionMediaStore(
        tmp_path / "media", "paired", policy=MediaPolicy(1024 * 1024, 10000, 8),
        checkpoint_verifier=verified_checkpoint_refs,
    )
    cap = bind_session_media(coordinator, store=store, enabled=True)
    coordinator.register_capability("session.media.tool_post", fixture_boundary)
    staged = []
    def mutation(data):
        staged.extend(copy.deepcopy(data["result"]["output"][NAMESPACE]["refs"]))
        if keep == "one":
            data["result"]["output"][NAMESPACE]["refs"] = staged[1:]
        elif keep == "none":
            data["result"]["output"].pop(NAMESPACE)
        return HookResult(action="modify", data=data)
    coordinator.register_capability("session.durable_checkpoint", lambda: acknowledge(
        store, tmp_path / "session", context._messages))
    try:
        await report_consumer_ready(coordinator)
        assert cap.report("loop", event="readiness", outcome="inspect")["consumers"]["loop-streaming"]
        call = ToolCall(id="actual-call", name="image", arguments={})
        if sequential:
            await loop._execute_tool_with_result_unscoped(
                call, {"image": tool}, context, MutatingHooks(mutation), coordinator)
        else:
            settled = await loop._execute_tool_unscoped(
                call, {"image": tool}, MutatingHooks(mutation), "group", coordinator)
            await context.add_message(settled.message())
            for reference in context._messages[0].get("metadata", {}).get(NAMESPACE, {}).get("refs", []):
                with pytest.raises(MediaRefusal):
                    cap.resolve(reference, purpose="model")
            await checkpoint_settled_results(
                coordinator, has_media=bool(settled.metadata))
        rows = context._messages
        selected = rows[0].get("metadata", {}).get(NAMESPACE, {}).get("refs", [])
        assert selected == (staged if keep == "all" else staged[1:] if keep == "one" else [])
        for reference in selected:
            with cap.resolve(reference, purpose="model") as source:
                assert hashlib.sha256(source.read()).hexdigest() == reference["sha256"]
        removed = [reference for reference in staged if reference not in selected]
        for reference in removed:
            with pytest.raises(MediaRefusal):
                cap.resolve(reference, purpose="model")
        canonical = ChatRequest(messages=[Message(**row) for row in rows])
        before = canonical.model_dump_json()
        projection = await cap.prepare_request(
            canonical, provider_identity={
                "provider": "fixture", "account": "offline", "generation": 1,
                "media_support": {"v": 1, "positions": ["tool"], "mime_types": ["image/png"]},
            }, model="fixture", request_options={}, purpose="model", execution_scope={},
        )
        try:
            projection.validate(request_options={})
            images = [block for message in projection.request.messages
                      for block in message.content if getattr(block, "type", None) == "image"] if selected else []
            assert len(images) == len(selected)
            assert canonical.model_dump_json() == before
            for image in tool.images:
                assert base64.b64encode(image).decode() not in json.dumps(rows)
            for path in (tmp_path / "media").rglob("*.json"):
                text = path.read_text()
                assert all(base64.b64encode(image).decode() not in text for image in tool.images)
        finally:
            projection.release()
    finally:
        store.close()


async def test_actual_store_nested_parent_publication_keeps_original_provenance(environment, tmp_path):
    _, coordinator, loop, _ = environment
    sid = str(coordinator.session_id)
    store = SessionMediaStore(
        tmp_path / "media", sid, policy=MediaPolicy(1024 * 1024, 10000, 8),
        checkpoint_verifier=verified_checkpoint_refs,
    )
    bind_session_media(coordinator, store=store, enabled=True)
    coordinator.register_capability("session.media.tool_post", fixture_boundary)
    coordinator.register_capability("tools.dispatch.media_refs", loop.approved_dispatch.media_refs)
    tool = Images()
    await coordinator.mount("tools", tool, name="read")
    inner = []
    async def handler(lease):
        receipt = await lease.call("read", {})
        inner.extend(copy.deepcopy(receipt["output"][NAMESPACE]["refs"]))
        return receipt["output"]
    try:
        output, _ = await call_client(environment, Client(coordinator, handler))
        assert output[NAMESPACE]["refs"] == inner and inner
        assert all(reference["provenance"]["id"] != "parent" for reference in inner)
        assert all(reference["provenance"]["session_id"] == sid for reference in inner)
        rows = [{"role": "tool", "tool_call_id": "parent", "name": "tool_exec",
                 "content": "nested", "metadata": {NAMESPACE: {"refs": inner}}}]
        acknowledge(store, tmp_path / "session", rows)
        for reference in inner:
            with store.resolve_ref(reference) as source:
                assert hashlib.sha256(source.read()).hexdigest() == reference["sha256"]
    finally:
        store.close()