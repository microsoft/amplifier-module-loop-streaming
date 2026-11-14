"""Explicit negative pairing with an immutable host-bridge candidate.

Set MEDIA_HOOK_BRIDGE_SOURCE and MEDIA_HOOK_BRIDGE_SHA256 to inspected source.
No private observation helper is called; the real public ABI is exercised.
"""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
from amplifier_core import AmplifierSession, HookResult
from amplifier_core.message_models import ToolCall
from amplifier_foundation.session.media import MediaPolicy, SessionMediaStore, bind_session_media
from amplifier_module_loop_streaming import StreamingOrchestrator
from amplifier_module_loop_streaming.media import report_consumer_ready
from amplifier_module_loop_streaming.dispatch import AttributedCoordinator, AttributedHooks
from tests.test_durable_checkpoint import MockContext

pytestmark = pytest.mark.asyncio


def bridge_module():
    source = Path(os.environ["MEDIA_HOOK_BRIDGE_SOURCE"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == os.environ["MEDIA_HOOK_BRIDGE_SHA256"]
    spec = importlib.util.spec_from_file_location("selected_media_hook_bridge", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("site", ["parallel", "sequential", "nested"])
@pytest.mark.parametrize("kind", ["image", "success", "failed", "null", "hook-refs"])
async def test_safe_public_incomplete_bridge_never_promotes_or_reports_ready(tmp_path, site, kind):
    session = AmplifierSession({"session": {
        "orchestrator": {"module": "loop-streaming"}, "context": {"module": "context-simple"},
    }, "providers": []}, session_id="bridge-pair")
    coordinator = session.coordinator
    bridge = bridge_module().install_media_hook_bridge(coordinator)
    store = SessionMediaStore(
        tmp_path / "media", coordinator.session_id,
        policy=MediaPolicy(1024 * 1024, 10000, 8),
    )
    cap = bind_session_media(coordinator, store=store, enabled=True)
    post_calls = []
    async def observe(event, data):
        post_calls.append(data)
        if kind == "null":
            return HookResult(action="modify", data={**data, "result": None})
        if kind == "hook-refs":
            return HookResult(action="modify", data={
                **data, "result": {"success": True, "output": {
                    "text": "useful", "experimental.media.v1": {"refs": [{"forged": True}]},
                }},
            })
        return HookResult()
    bridge.hooks.register("tool:post", observe, name="fixture")
    # The rich fixture uses a base64 sentinel rather than real image bytes;
    # this negative check needs valid pixels to reach the actual bridge.
    from integration.test_media_store_pairing import Images
    from amplifier_core import ToolResult
    class Tool:
        async def execute(self, arguments):
            if kind == "image":
                return await Images().execute(arguments)
            if kind == "failed":
                return ToolResult(success=False, output="useful failure", error={"message": "failed"})
            return ToolResult(success=True, output="useful text")
    loop = StreamingOrchestrator({})
    call = ToolCall(id="call", name="image", arguments={})
    context = MockContext()
    hooks, current_coordinator = coordinator.hooks, coordinator
    if site == "nested":
        observation = {}
        hooks = AttributedHooks(hooks, {"parent_tool_call_id": "outer"}, observation)
        current_coordinator = AttributedCoordinator(coordinator, observation)
    try:
        await report_consumer_ready(coordinator)
        if site != "sequential":
            result = await loop._execute_tool_unscoped(
                call, {"image": Tool()}, hooks, "group", current_coordinator)
            assert not result.metadata
            content = result.content
        else:
            await loop._execute_tool_with_result_unscoped(
                call, {"image": Tool()}, context, hooks, current_coordinator)
            assert not context._messages[0].get("metadata")
            content = context._messages[0]["content"]
        if kind == "null":
            assert content == "null"
        elif kind == "image":
            assert "preserved" in content and "Image unavailable" in content
        else:
            assert "useful" in content
        assert len(post_calls) == 1
        assert not cap.report("loop", event="readiness", outcome="inspect")["consumers"]
        assert not list((tmp_path / "media" / "assets").iterdir())
        records = list(store.journal())
        # Staging journal can retain recovery entries; they are not accepted
        # reachability. The owning record is dropped and no asset is published.
        assert bool(records) is (kind == "image")
        assert not any(record.get("state") in {"published", "committed"} for record in records)
    finally:
        store.close()