"""Focused real-loop coverage for the optional context.instructions.v1 binding.

Set AMPLIFIER_CONTEXT_SIMPLE_TEST_SOURCE to the context-simple source checkout
to exercise the integration. Production loop code keeps a structural capability
boundary and never imports that module.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from amplifier_core import HookRegistry, HookResult, ToolResult

from amplifier_module_loop_streaming import StreamingOrchestrator


_context_source = os.environ.get("AMPLIFIER_CONTEXT_SIMPLE_TEST_SOURCE")
if not _context_source:
    pytest.skip(
        "set AMPLIFIER_CONTEXT_SIMPLE_TEST_SOURCE to a context-simple source checkout",
        allow_module_level=True,
    )
_CONTEXT_SIMPLE_TEST_SOURCE = Path(_context_source)
if not _CONTEXT_SIMPLE_TEST_SOURCE.is_dir():
    pytest.skip(
        "AMPLIFIER_CONTEXT_SIMPLE_TEST_SOURCE must name an existing directory",
        allow_module_level=True,
    )
sys.path.insert(0, str(_CONTEXT_SIMPLE_TEST_SOURCE))
from amplifier_module_context_simple import mount as mount_context  # noqa: E402


class _Cancellation:
    is_cancelled = False
    is_immediate = False
    state = "running"

    async def trigger_callbacks(self) -> None:
        return None

    def register_tool_start(self, tool_call_id: str, display_name: str) -> None:
        return None

    def register_tool_complete(self, tool_call_id: str) -> None:
        return None


class _Coordinator:
    """Minimal trusted host harness with capability and normal hook ingress."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Any] = {}
        self._mounts: dict[str, Any] = {}
        self.session_state: dict[str, Any] = {}
        self.cancellation = _Cancellation()
        self.hooks = None

    async def mount(self, name: str, value: Any, *args: Any) -> None:
        self._mounts[name] = value

    def get(self, name: str, *args: Any) -> Any:
        return self._mounts.get(name)

    def register_capability(self, name: str, value: Any) -> None:
        self._capabilities[name] = value

    def get_capability(self, name: str) -> Any:
        return self._capabilities.get(name)

    def register_contributor(self, *args: Any) -> None:
        return None

    async def process_hook_result(self, result: Any, *args: Any, **kwargs: Any) -> Any:
        return result


def _provide_execution_input(
    coordinator: _Coordinator, input_id: str, origin: str = "human"
) -> None:
    """Model the trusted host binding supplied immediately before execute()."""
    coordinator.register_capability(
        "execution.input.v1",
        {"version": 1, "input_id": input_id, "origin": origin},
    )


class _Response:
    content = None
    content_blocks = None
    usage = None
    metadata = None

    def __init__(self, text: str) -> None:
        self.text = text


class _V1Provider:
    instruction_layout_version = 1
    priority = 1

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.calls = 0

    def get_info(self) -> Any:
        from amplifier_core.models import ProviderInfo

        return ProviderInfo(id="test", display_name="test", defaults={"model": "test-v1"})

    async def complete(self, request: Any, **kwargs: Any) -> _Response:
        self.calls += 1
        self.requests.append(request)
        return _Response(f"answer-{self.calls}")

    def parse_tool_calls(self, response: _Response) -> list[Any]:
        return []


class _LegacyProvider(_V1Provider):
    instruction_layout_version = 0


class _GoalText:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _GoalResponse:
    def __init__(self, text: str) -> None:
        self.content = [_GoalText(text)]


class _GoalProvider(_V1Provider):
    def __init__(self) -> None:
        super().__init__()
        self.evaluations = 0

    async def complete(self, request: Any, **kwargs: Any) -> _Response | _GoalResponse:
        self.requests.append(request)
        if request.metadata and request.metadata.get("stream") is False:
            self.evaluations += 1
            return _GoalResponse("NO\ncontinue" if self.evaluations == 1 else "YES\ncomplete")
        self.calls += 1
        return _Response(f"answer-{self.calls}")


class _ToolCall:
    def __init__(self, call_id: str, name: str) -> None:
        self.id = call_id
        self.name = name
        self.arguments: dict[str, Any] = {}


class _BatchProvider(_V1Provider):
    def parse_tool_calls(self, response: _Response) -> list[Any]:
        if self.calls == 1:
            return [_ToolCall("denied-call", "denied"), _ToolCall("missing-call", "missing")]
        return []


class _ConsecutiveBatchProvider(_V1Provider):
    def parse_tool_calls(self, response: _Response) -> list[Any]:
        if self.calls < 3:
            return [_ToolCall(f"missing-{self.calls}", "missing")]
        return []


class _SuccessfulToolProvider(_V1Provider):
    def parse_tool_calls(self, response: _Response) -> list[Any]:
        if self.calls == 1:
            return [_ToolCall("tool-call", "tool")]
        return []


class _SuccessfulTool:
    name = "tool"
    description = "test tool"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, output="tool result")


class _FailingStreamProvider(_V1Provider):
    async def stream(self, request: Any, **kwargs: Any):
        self.requests.append(request)
        raise RuntimeError("stream transport failed")
        yield  # pragma: no cover - keeps this an async generator


class _CancellingStreamProvider(_V1Provider):
    def __init__(self, cancellation: _Cancellation) -> None:
        super().__init__()
        self.cancellation = cancellation

    async def stream(self, request: Any, **kwargs: Any):
        self.requests.append(request)
        yield {"content": "partial"}
        self.cancellation.is_immediate = True
        yield {"content": "ignored"}


class _CancelAfterFinalChunkProvider(_V1Provider):
    def __init__(self, cancellation: _Cancellation) -> None:
        super().__init__()
        self.cancellation = cancellation

    async def stream(self, request: Any, **kwargs: Any):
        self.requests.append(request)
        yield {"content": "partial"}
        self.cancellation.is_cancelled = True


class _SwitchingHooks:
    def __init__(self, orchestrator: StreamingOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> HookResult:
        if event == "prompt:submit":
            self.orchestrator._pinned_provider_name = "legacy"
        return HookResult(action="continue")


class _ToolHooks:
    async def emit(self, event: str, data: dict[str, Any] | None = None) -> HookResult:
        if event == "tool:pre" and data and data["tool_call_id"] == "denied-call":
            return HookResult(action="deny", reason="policy")
        return HookResult(action="continue")


class _LegacyInjectionHooks:
    async def emit(self, event: str, data: dict[str, Any] | None = None) -> HookResult:
        injected = {
            "prompt:submit": "legacy-prompt",
            "provider:request": "legacy-request",
            "tool:pre": "legacy-pre",
            "tool:post": "legacy-post",
        }.get(event)
        if injected is not None:
            return HookResult(
                action="inject_context",
                ephemeral=True,
                context_injection=injected,
            )
        return HookResult(action="continue")


class _FinalizationInjectionHooks:
    def __init__(self) -> None:
        self.provider_requests = 0

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> HookResult:
        if event == "provider:request":
            self.provider_requests += 1
            if self.provider_requests == 2:
                return HookResult(
                    action="inject_context",
                    ephemeral=True,
                    context_injection="legacy-finalization",
                )
        return HookResult(action="continue")


async def _new_context(checkpoint: list[dict[str, Any]] | None = None):
    coordinator = _Coordinator()
    await mount_context(coordinator, {"instruction_session_id": "logical-session"})
    context = coordinator.get("context")
    if checkpoint is not None:
        await context.restore_host_checkpoint(copy.deepcopy(checkpoint))
    return coordinator, context


def _request_contents(provider: _V1Provider, index: int) -> list[str]:
    return [message.content for message in provider.requests[index].messages]


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ("human", "delegation"))
async def test_host_execution_input_binding_preserves_exact_origin_and_id(origin: str) -> None:
    """The app-owned binding is the sole source of outer-input provenance."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    scopes: list[dict[str, Any]] = []
    assembly.register(
        "test-scope",
        lambda scope: scopes.append(copy.deepcopy(scope)) or [],
    )
    _provide_execution_input(coordinator, "trusted-outer-input", origin)

    assert await StreamingOrchestrator({}).execute(
        "outer prompt", context, {"v1": _V1Provider()}, {}, HookRegistry(), coordinator
    ) == "answer-1"

    expected_anchor = {
        "input_id": "trusted-outer-input",
        "message_id": "trusted-outer-input",
        "origin": origin,
    }
    assert (await context.get_messages())[0]["metadata"]["amplifier:input"] == {
        "version": 1,
        **expected_anchor,
    }
    assert scopes[0]["input_anchor"] == expected_anchor
    assert coordinator.get_capability("execution.input.v1") is None


@pytest.mark.asyncio
async def test_no_execution_input_binding_keeps_capable_context_and_provider_legacy() -> None:
    """An unchanged app does not activate v1 merely because dependencies can."""
    coordinator, context = await _new_context()
    provider = _V1Provider()

    assert await StreamingOrchestrator({}).execute(
        "legacy H1", context, {"v1": provider}, {}, HookRegistry(), coordinator
    ) == "answer-1"

    assert [message["content"] for message in await context.get_messages()] == [
        "legacy H1",
        "answer-1",
    ]
    assert all(
        "amplifier:input" not in message.get("metadata", {})
        for message in await context.get_messages()
    )


@pytest.mark.asyncio
async def test_malformed_execution_input_fails_loud_and_cannot_be_reused() -> None:
    """The consumed capability cannot leak stale provenance into another turn."""
    coordinator, context = await _new_context()
    coordinator.register_capability(
        "execution.input.v1",
        {"version": 1, "input_id": "", "origin": "human"},
    )

    with pytest.raises(RuntimeError, match="execution.input.v1 must be exactly"):
        await StreamingOrchestrator({}).execute(
            "bad H1", context, {"v1": _V1Provider()}, {}, HookRegistry(), coordinator
        )

    assert coordinator.get_capability("execution.input.v1") is None
    provider = _V1Provider()
    assert await StreamingOrchestrator({}).execute(
        "legacy H2", context, {"v1": provider}, {}, HookRegistry(), coordinator
    ) == "answer-1"
    assert all(
        "amplifier:input" not in message.get("metadata", {})
        for message in await context.get_messages()
    )


@pytest.mark.asyncio
async def test_consumed_execution_input_is_not_reused_by_the_next_execute() -> None:
    """A host must bind each turn rather than inheriting the prior origin."""
    coordinator, context = await _new_context()
    _provide_execution_input(coordinator, "first-human-input")

    assert await StreamingOrchestrator({}).execute(
        "first H1", context, {"v1": _V1Provider()}, {}, HookRegistry(), coordinator
    ) == "answer-1"

    provider = _V1Provider()
    assert await StreamingOrchestrator({}).execute(
        "second H2", context, {"v1": provider}, {}, HookRegistry(), coordinator
    ) == "answer-1"
    messages = await context.get_messages()
    assert messages[-2]["content"] == "second H2"
    assert "amplifier:input" not in messages[-2].get("metadata", {})


@pytest.mark.asyncio
async def test_no_execution_input_refuses_marked_v1_history() -> None:
    """Restored v1 history cannot be silently dispatched on the legacy route."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    lease = assembly.register("test-fixed")
    lease.publish(
        "fixed",
        "must remain v1",
        target={"session_id": "logical-session", "kind": "conversation_head"},
        retain_history=True,
    )

    with pytest.raises(RuntimeError, match="execution.input.v1 is absent"):
        await StreamingOrchestrator({}).execute(
            "legacy H1", context, {"v1": _V1Provider()}, {}, HookRegistry(), coordinator
        )


@pytest.mark.asyncio
async def test_second_delivery_observer_failure_stops_without_another_provider_call() -> None:
    """The allowed one retry never manufactures a third delivery or LLM call."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    lease = assembly.register("test-fixed")
    lease.publish(
        "fixed",
        "fixed",
        target={"kind": "first_eligible_turn", "placement": "head"},
        retain_history=True,
    )
    original_observer = context._record_instruction_observation
    delivery_attempts = 0

    def fail_delivery(observation: dict[str, Any]) -> None:
        nonlocal delivery_attempts
        if observation.get("phase") == "delivery":
            delivery_attempts += 1
            raise RuntimeError("persistent observer failure")
        original_observer(observation)

    context._record_instruction_observation = fail_delivery
    provider = _V1Provider()
    _provide_execution_input(coordinator, "human-H1")

    with pytest.raises(RuntimeError, match="persistent observer failure"):
        await StreamingOrchestrator({}).execute(
            "H1", context, {"v1": provider}, {}, HookRegistry(), coordinator
        )

    assert delivery_attempts == 2
    assert provider.calls == 1
    assert [message["content"] for message in await context.get_messages()].count("answer-1") == 1


@pytest.mark.asyncio
async def test_goal_continuation_keeps_the_trusted_outer_anchor() -> None:
    """/goal continuations stay synthetic rather than replacing outer provenance."""
    coordinator, context = await _new_context()
    coordinator.session_state["goal"] = {
        "condition": "finish",
        "turns_used": 0,
        "cap": 2,
    }
    assembly = coordinator.get_capability("context.instructions.v1")
    scopes: list[dict[str, Any]] = []
    assembly.register(
        "test-scope",
        lambda scope: scopes.append(copy.deepcopy(scope)) or [],
    )
    _provide_execution_input(coordinator, "delegated-task", "delegation")
    provider = _GoalProvider()

    assert await StreamingOrchestrator({}).execute(
        "outer delegated task", context, {"v1": provider}, {}, HookRegistry(), coordinator
    ) == "answer-2"

    expected_anchor = {
        "input_id": "delegated-task",
        "message_id": "delegated-task",
        "origin": "delegation",
    }
    assert provider.evaluations == 2
    assert [scope["input_anchor"] for scope in scopes] == [expected_anchor, expected_anchor]
    messages = await context.get_messages()
    assert messages[0]["metadata"]["amplifier:input"] == {"version": 1, **expected_anchor}
    assert messages[2]["content"] == "continue"
    assert "amplifier:input" not in messages[2].get("metadata", {})


@pytest.mark.asyncio
async def test_three_recreated_turns_keep_fixed_h1_and_refresh_live_h2_h3() -> None:
    """A fresh host/context every turn restores fixed state but never live state."""
    checkpoint: list[dict[str, Any]] | None = None
    request_views: list[list[str]] = []

    for turn, live_text in enumerate(("live-H1", "live-H2", "live-H3"), start=1):
        coordinator, context = await _new_context(checkpoint)
        assembly = coordinator.get_capability("context.instructions.v1")
        lease = assembly.register(
            "test-live",
            lambda _scope, text=live_text: [
                {"key": "current", "content": text, "placement": "before_human"}
            ],
        )
        if turn == 1:
            lease.publish(
                "fixed-H1",
                "fixed-before-H1",
                target={"kind": "first_eligible_turn", "placement": "before_human"},
                retain_history=True,
            )
        else:
            assert (
                lease.publish(
                    "fixed-H1",
                    "fixed-before-H1",
                    target={"kind": "first_eligible_turn", "placement": "before_human"},
                    retain_history=True,
                )
                == "logical-session:test-live:fixed-H1"
            )

        provider = _V1Provider()
        _provide_execution_input(coordinator, f"human-H{turn}")
        result = await StreamingOrchestrator({}).execute(
            f"H{turn}", context, {"v1": provider}, {}, HookRegistry(), coordinator
        )
        assert result == "answer-1"
        request_views.append(_request_contents(provider, 0))
        checkpoint = await context.get_messages()

    assert request_views[0] == ["live-H1", "fixed-before-H1", "H1"]
    assert request_views[1] == ["fixed-before-H1", "H1", "answer-1", "live-H2", "H2"]
    assert request_views[2] == [
        "fixed-before-H1",
        "H1",
        "answer-1",
        "H2",
        "answer-1",
        "live-H3",
        "H3",
    ]


@pytest.mark.asyncio
async def test_complete_tool_batch_feedback_reaches_next_request_in_original_order() -> None:
    """Denied/error tool outcomes form one complete causal batch for request two."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    callback_scopes: list[dict[str, Any]] = []

    def feedback(scope: dict[str, Any]) -> list[dict[str, Any]]:
        callback_scopes.append(copy.deepcopy(scope))
        batches = scope["completed_batches"]
        if not batches:
            return []
        batch = batches[-1]
        return [
            {
                "key": "tool-feedback",
                "content": "feedback:" + ",".join(batch["call_ids"]),
                "placement": "tail",
                "after": {"after_message_id": batch["result_message_ids"][-1]},
            }
        ]

    assembly.register("test-feedback", feedback)
    provider = _BatchProvider()
    _provide_execution_input(coordinator, "human-H1")
    result = await StreamingOrchestrator({}).execute(
        "H1", context, {"v1": provider}, {}, _ToolHooks(), coordinator
    )

    assert result == "answer-2"
    assert len(provider.requests) == 2
    assert callback_scopes[0]["completed_batches"] == []
    batch = callback_scopes[1]["completed_batches"][-1]
    assert batch["call_ids"] == ["denied-call", "missing-call"]
    assert len(batch["result_message_ids"]) == 2
    second = _request_contents(provider, 1)
    assert second[-3:] == [
        "Denied by hook: policy",
        "Error: Tool 'missing' not found",
        "feedback:denied-call,missing-call",
    ]


@pytest.mark.asyncio
async def test_completed_tool_batches_do_not_reach_later_requests() -> None:
    """Each request sees only the immediately preceding completed tool batch."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    callback_scopes: list[dict[str, Any]] = []

    def feedback(scope: dict[str, Any]) -> list[dict[str, Any]]:
        callback_scopes.append(copy.deepcopy(scope))
        return []

    assembly.register("test-feedback", feedback)
    _provide_execution_input(coordinator, "human-H1")
    result = await StreamingOrchestrator({}).execute(
        "H1", context, {"v1": _ConsecutiveBatchProvider()}, {}, HookRegistry(), coordinator
    )

    assert result == "answer-3"
    assert [
        [call_id for batch in scope["completed_batches"] for call_id in batch["call_ids"]]
        for scope in callback_scopes
    ] == [[], ["missing-1"], ["missing-2"]]


@pytest.mark.asyncio
async def test_v1_finalization_keeps_the_prepared_system_view_and_accepts_response() -> None:
    """The bounded final call does not append a legacy user-role reminder in v1."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    assembly.register(
        "test-live",
        lambda _scope: [{"key": "current", "content": "live-system", "placement": "head"}],
    )
    provider = _BatchProvider()
    _provide_execution_input(coordinator, "human-H1")

    result = await StreamingOrchestrator({"max_iterations": 1}).execute(
        "H1", context, {"v1": provider}, {}, _ToolHooks(), coordinator
    )

    assert result == "answer-2"
    assert len(provider.requests) == 2
    assert _request_contents(provider, 1)[0] == "live-system"
    assert not any(
        isinstance(message.content, str) and "orchestrator-loop-limit" in message.content
        for message in provider.requests[1].messages
    )
    assert (await context.get_messages())[-1]["content"] == "answer-2"


@pytest.mark.asyncio
async def test_v1_finalization_admits_legacy_hook_feedback_before_its_view() -> None:
    """A final request scopes and prepares after its ordinary hook result."""
    coordinator, context = await _new_context()
    hooks = _FinalizationInjectionHooks()
    provider = _BatchProvider()
    _provide_execution_input(coordinator, "human-H1")

    assert await StreamingOrchestrator({"max_iterations": 1}).execute(
        "H1", context, {"v1": provider}, {}, hooks, coordinator
    ) == "answer-2"

    final_messages = provider.requests[1].messages
    legacy = next(
        message
        for message in final_messages
        if isinstance(message.content, str) and "legacy-finalization" in message.content
    )
    assert legacy.role == "system"


@pytest.mark.asyncio
async def test_unmigrated_hook_injections_remain_available_as_system_context_in_v1() -> None:
    """V1 snapshots do not discard ordinary prompt/request/post hook output."""
    coordinator, context = await _new_context()
    provider = _SuccessfulToolProvider()
    _provide_execution_input(coordinator, "human-H1")

    assert await StreamingOrchestrator({}).execute(
        "H1",
        context,
        {"v1": provider},
        {"tool": _SuccessfulTool()},
        _LegacyInjectionHooks(),
        coordinator,
    ) == "answer-2"

    first = _request_contents(provider, 0)
    second = _request_contents(provider, 1)
    assert any("legacy-prompt" in content for content in first)
    assert any("legacy-request" in content for content in first)
    assert any("legacy-pre" in content for content in second)
    assert any("legacy-post" in content for content in second)
    assert all(
        message.role == "system"
        for request in provider.requests
        for message in request.messages
        if isinstance(message.content, str) and "legacy-" in message.content
    )


@pytest.mark.asyncio
async def test_v1_acceptance_retries_the_same_stored_response_once() -> None:
    """A transient delivery observer failure does not duplicate assistant history."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    lease = assembly.register("test-fixed")
    lease.publish(
        "fixed",
        "fixed",
        target={"kind": "first_eligible_turn", "placement": "head"},
        retain_history=True,
    )
    original_observer = context._record_instruction_observation
    delivery_attempts = 0

    def fail_first_delivery(observation: dict[str, Any]) -> None:
        nonlocal delivery_attempts
        if observation.get("phase") == "delivery":
            delivery_attempts += 1
            if delivery_attempts == 1:
                raise RuntimeError("transient observer failure")
        original_observer(observation)

    context._record_instruction_observation = fail_first_delivery
    provider = _V1Provider()
    _provide_execution_input(coordinator, "human-H1")

    assert await StreamingOrchestrator({}).execute(
        "H1", context, {"v1": provider}, {}, HookRegistry(), coordinator
    ) == "answer-1"
    messages = await context.get_messages()
    assert [message["content"] for message in messages].count("answer-1") == 1
    assert delivery_attempts == 2
    assert lease.publish(
        "fixed",
        "fixed",
        target={"kind": "first_eligible_turn", "placement": "head"},
        retain_history=True,
    ) == "logical-session:test-fixed:fixed"


@pytest.mark.asyncio
async def test_v1_stream_failure_abandons_the_request_and_next_turn_can_run() -> None:
    """A stream transport failure cannot leave an overlapping request behind."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    assembly.register(
        "test-live",
        lambda _scope: [{"key": "current", "content": "live", "placement": "head"}],
    )

    _provide_execution_input(coordinator, "human-H1")
    with pytest.raises(RuntimeError, match="stream transport failed"):
        await StreamingOrchestrator({}).execute(
            "H1", context, {"v1": _FailingStreamProvider()}, {}, HookRegistry(), coordinator
        )

    _provide_execution_input(coordinator, "human-H2")
    assert await StreamingOrchestrator({}).execute(
        "H2", context, {"v1": _V1Provider()}, {}, HookRegistry(), coordinator
    ) == "answer-1"


@pytest.mark.asyncio
async def test_cancelled_v1_stream_does_not_acknowledge_partial_response() -> None:
    """Partial streaming text cannot consume pending v1 delivery on cancellation."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    lease = assembly.register("test-fixed")
    lease.publish(
        "fixed",
        "fixed",
        target={"kind": "first_eligible_turn", "placement": "head"},
        retain_history=True,
    )
    provider = _CancellingStreamProvider(coordinator.cancellation)
    _provide_execution_input(coordinator, "human-H1")

    assert await StreamingOrchestrator({}).execute(
        "H1", context, {"v1": provider}, {}, HookRegistry(), coordinator
    ) == "partial"

    messages = await context.get_messages()
    assert [message["content"] for message in messages] == ["fixed", "H1"]
    assert lease.publish(
        "fixed",
        "fixed",
        target={"kind": "first_eligible_turn", "placement": "head"},
        retain_history=True,
    ) == "logical-session:test-fixed:fixed"


@pytest.mark.asyncio
async def test_v1_stream_cancelled_after_its_last_chunk_does_not_acknowledge() -> None:
    """A cancellation race at stream exhaustion also leaves delivery pending."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    lease = assembly.register("test-fixed")
    lease.publish(
        "fixed",
        "fixed",
        target={"kind": "first_eligible_turn", "placement": "head"},
        retain_history=True,
    )

    _provide_execution_input(coordinator, "human-H1")
    assert await StreamingOrchestrator({}).execute(
        "H1",
        context,
        {"v1": _CancelAfterFinalChunkProvider(coordinator.cancellation)},
        {},
        HookRegistry(),
        coordinator,
    ) == "partial"

    assert [message["content"] for message in await context.get_messages()] == ["fixed", "H1"]
    assert lease.publish(
        "fixed",
        "fixed",
        target={"kind": "first_eligible_turn", "placement": "head"},
        retain_history=True,
    ) == "logical-session:test-fixed:fixed"


@pytest.mark.asyncio
async def test_provider_change_to_legacy_is_refused_before_dispatch() -> None:
    """A hook-time provider switch cannot lower the already-gated v1 turn."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    assembly.register(
        "test-live",
        lambda _scope: [{"key": "current", "content": "live", "placement": "head"}],
    )
    v1 = _V1Provider()
    legacy = _LegacyProvider()
    orchestrator = StreamingOrchestrator({})
    _provide_execution_input(coordinator, "human-H1")

    with pytest.raises(RuntimeError, match="changed after v1 activation"):
        await orchestrator.execute(
            "H1",
            context,
            {"v1": v1, "legacy": legacy},
            {},
            _SwitchingHooks(orchestrator),
            coordinator,
        )

    assert v1.requests == []
    assert legacy.requests == []


@pytest.mark.asyncio
async def test_old_context_or_provider_uses_unchanged_legacy_path() -> None:
    """Partial v1 compositions neither call new methods nor activate v1."""

    class OldContext:
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        async def add_message(self, message: dict[str, Any]) -> None:
            self.messages.append(message)

        async def get_messages_for_request(self, provider: Any = None) -> list[dict[str, Any]]:
            return list(self.messages)

    old_context = OldContext()
    old_provider = _V1Provider()
    assert await StreamingOrchestrator({}).execute(
        "legacy-context", old_context, {"v1": old_provider}, {}, HookRegistry(), _Coordinator()
    ) == "answer-1"
    assert [message["content"] for message in old_context.messages] == [
        "legacy-context",
        "answer-1",
    ]

    coordinator, context = await _new_context()
    legacy_provider = _LegacyProvider()
    _provide_execution_input(coordinator, "human-H1")
    assert await StreamingOrchestrator({}).execute(
        "legacy-provider", context, {"legacy": legacy_provider}, {}, HookRegistry(), coordinator
    ) == "answer-1"
    assert all(
        "amplifier:input" not in message.get("metadata", {}) for message in context.messages
    )


@pytest.mark.asyncio
async def test_incompatible_provider_refuses_marked_v1_history() -> None:
    """A selected provider may not silently lower retained v1 system records."""
    coordinator, context = await _new_context()
    assembly = coordinator.get_capability("context.instructions.v1")
    lease = assembly.register("test-fixed")
    lease.publish(
        "fixed",
        "must remain system",
        target={"session_id": "logical-session", "kind": "conversation_head"},
        retain_history=True,
    )
    provider = _LegacyProvider()

    _provide_execution_input(coordinator, "human-H1")
    with pytest.raises(RuntimeError, match="does not declare instruction_layout_version"):
        await StreamingOrchestrator({}).execute(
            "H1", context, {"legacy": provider}, {}, HookRegistry(), coordinator
        )

    assert provider.requests == []