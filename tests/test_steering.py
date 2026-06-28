"""Unit tests for mid-turn steering — orchestrator half (spec §7, tests 1-8).

Covers:
  1. Bounded overflow: SteeringQueueFull raised, never silent drop.
  2. Empty/whitespace rejection: ValueError raised, nothing enqueued.
  3. FIFO order: drain() returns messages in insertion order.
  4. Inject + event: _drain_steering adds user-role messages and emits
     orchestrator:steering_injected per message with correct payload.
  5. Top-of-iteration drain before provider call: steer enqueued during
     round 1 appears in the second provider.complete call's messages.
  6. Last-drain-edge revive: steer queued at final break causes one more
     provider call; empty queue breaks normally.
  7. Capability registered in mount: session.steer is callable and delegates
     to the orchestrator's queue.
  8. Streaming undisturbed: empty queue makes _drain_steering a no-op;
     streaming path still yields tokens.

Note: execute(), _drain_steering(), and mount() are typed against Rust-backed
types (RustHookRegistry, RustCoordinator). Duck-typed test stubs are not
structurally assignable to those types, hence the # type: ignore[arg-type]
suppressions below — this is the standard pattern for testing duck-typed
Python code against Rust extension interfaces.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Shared stubs — minimal, self-contained
# ---------------------------------------------------------------------------


class MockHookResult:
    """Minimal hook result — pass through, no deny, no injection."""

    action = "pass"
    reason = None
    ephemeral = False
    context_injection = None
    context_injection_role = "user"
    append_to_last_tool_result = False
    data = None


class MockHooks:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(
        self, event_name: str, payload: dict | None = None
    ) -> MockHookResult:
        self.emitted.append((event_name, payload or {}))
        return MockHookResult()


class MockContext:
    def __init__(self) -> None:
        self._messages: list[dict] = []

    async def add_message(self, msg: dict) -> None:
        self._messages.append(msg)

    async def get_messages_for_request(self, provider=None) -> list[dict]:
        return list(self._messages)


class MockCancellation:
    is_cancelled = False
    is_immediate = False
    state = "running"

    def register_tool_start(self, tool_call_id: str, display_name: str) -> None:
        pass

    def register_tool_complete(self, tool_call_id: str) -> None:
        pass

    async def trigger_callbacks(self) -> None:
        pass


class MockCoordinator:
    def __init__(self) -> None:
        self.cancellation = MockCancellation()
        self._capabilities: dict = {}
        self._mounts: dict = {}

    def register_contributor(self, name: str, source: str, fn) -> None:  # noqa: ANN001
        pass

    async def mount(self, name: str, obj) -> None:  # noqa: ANN001
        self._mounts[name] = obj

    def register_capability(self, name: str, value) -> None:  # noqa: ANN001
        self._capabilities[name] = value

    def get_capability(self, name: str):  # noqa: ANN201
        return self._capabilities.get(name)

    async def process_hook_result(self, result, *args, **kwargs):  # noqa: ANN001,ANN201
        return result


class MockToolCall:
    def __init__(self) -> None:
        self.id = "tc-1"
        self.name = "mock_tool"
        self.arguments: dict = {}


class MockResponse:
    """Minimal non-streaming provider response."""

    def __init__(self, text: str = "", tool_calls=None) -> None:  # noqa: ANN001
        self.text = text
        self.content = None
        self.content_blocks = None
        self.usage = None
        self.metadata = None
        # Not used directly — parse_tool_calls() decides what to return
        self._intended_tool_calls = tool_calls or []


# ---------------------------------------------------------------------------
# Test 1: Bounded overflow
# ---------------------------------------------------------------------------


class TestSteeringQueueBoundedOverflow:
    def test_overflow_raises_steering_queue_full(self) -> None:
        from amplifier_module_loop_streaming.steering import (
            SteeringQueue,
            SteeringQueueFull,
        )

        queue = SteeringQueue(maxsize=2)
        queue.steer("a")
        queue.steer("b")
        with pytest.raises(SteeringQueueFull):
            queue.steer("c")  # must reject loudly — never drop silently

    def test_rejected_message_not_enqueued(self) -> None:
        from amplifier_module_loop_streaming.steering import (
            SteeringQueue,
            SteeringQueueFull,
        )

        queue = SteeringQueue(maxsize=1)
        queue.steer("first")
        try:
            queue.steer("overflow")
        except SteeringQueueFull:
            pass
        # Only the first message should be present
        assert queue.drain() == ["first"]


# ---------------------------------------------------------------------------
# Test 2: Empty / whitespace rejection
# ---------------------------------------------------------------------------


class TestSteeringQueueValidation:
    def test_empty_string_raises_value_error(self) -> None:
        from amplifier_module_loop_streaming.steering import SteeringQueue

        with pytest.raises(ValueError):
            SteeringQueue().steer("")

    def test_whitespace_only_raises_value_error(self) -> None:
        from amplifier_module_loop_streaming.steering import SteeringQueue

        with pytest.raises(ValueError):
            SteeringQueue().steer("   ")

    def test_newline_only_raises_value_error(self) -> None:
        from amplifier_module_loop_streaming.steering import SteeringQueue

        with pytest.raises(ValueError):
            SteeringQueue().steer("\n")

    def test_invalid_messages_leave_queue_empty(self) -> None:
        from amplifier_module_loop_streaming.steering import SteeringQueue

        queue = SteeringQueue()
        for bad in ["", "   ", "\n", "\t\t"]:
            try:
                queue.steer(bad)
            except ValueError:
                pass
        assert queue.drain() == []


# ---------------------------------------------------------------------------
# Test 3: FIFO order
# ---------------------------------------------------------------------------


class TestSteeringQueueFIFO:
    def test_drain_returns_messages_in_fifo_order(self) -> None:
        from amplifier_module_loop_streaming.steering import SteeringQueue

        queue = SteeringQueue()
        queue.steer("a")
        queue.steer("b")
        queue.steer("c")
        assert queue.drain() == ["a", "b", "c"]

    def test_drain_empties_the_queue(self) -> None:
        from amplifier_module_loop_streaming.steering import SteeringQueue

        queue = SteeringQueue()
        queue.steer("x")
        queue.drain()
        assert queue.drain() == []

    def test_empty_drain_returns_empty_list(self) -> None:
        from amplifier_module_loop_streaming.steering import SteeringQueue

        assert SteeringQueue().drain() == []


# ---------------------------------------------------------------------------
# Test 4: Inject + event (_drain_steering helper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDrainSteeringInjectAndEvent:
    async def test_inject_adds_user_role_messages_fifo(self) -> None:
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        orch.steer("msg1")
        orch.steer("msg2")

        ctx = MockContext()
        hooks = MockHooks()
        # _drain_steering(context, hooks, iteration) — hooks is duck-typed
        count = await orch._drain_steering(ctx, hooks, iteration=3)  # type: ignore[arg-type]

        assert count == 2
        user_msgs = [m for m in ctx._messages if m.get("role") == "user"]
        assert [m["content"] for m in user_msgs] == ["msg1", "msg2"]

    async def test_inject_emits_one_event_per_message_with_correct_payload(
        self,
    ) -> None:
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        orch.steer("hello")
        orch.steer("world")

        ctx = MockContext()
        hooks = MockHooks()
        await orch._drain_steering(ctx, hooks, iteration=5)  # type: ignore[arg-type]

        injected = [
            (name, data)
            for name, data in hooks.emitted
            if name == "orchestrator:steering_injected"
        ]
        assert len(injected) == 2
        _, first = injected[0]
        _, second = injected[1]
        assert first["content"] == "hello"
        assert first["iteration"] == 5
        assert first["queued_remaining"] == 1
        assert second["content"] == "world"
        assert second["queued_remaining"] == 0

    async def test_drain_is_noop_on_empty_queue(self) -> None:
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        ctx = MockContext()
        hooks = MockHooks()
        count = await orch._drain_steering(ctx, hooks, iteration=1)  # type: ignore[arg-type]

        assert count == 0
        assert ctx._messages == []
        assert hooks.emitted == []


# ---------------------------------------------------------------------------
# Test 5: Top-of-iteration drain before provider call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTopOfIterationDrain:
    async def test_steer_injected_appears_in_second_provider_call_messages(
        self,
    ) -> None:
        """Steer enqueued during round-1 tool execution is in round-2 messages."""
        from amplifier_module_loop_streaming import StreamingOrchestrator
        from amplifier_core import ToolResult

        orch = StreamingOrchestrator({})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        tc = MockToolCall()
        first_resp = MockResponse(text="working…")
        second_resp = MockResponse(text="done!")

        call_count = 0

        class MockTool:
            name = "mock_tool"
            description = "test tool"
            input_schema: dict = {"type": "object", "properties": {}}

            async def execute(self, arguments):  # noqa: ANN001,ANN201
                return ToolResult(success=True, output="result")

        class NonStreamingProvider:
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Inject steer during the first provider call (mid-round 1)
                    orch.steer("redirect me!")
                    return first_resp
                return second_resp

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                if response is first_resp:
                    return [tc]
                return []

        await orch.execute(
            prompt="do work",
            context=ctx,
            providers={"main": NonStreamingProvider()},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert call_count == 2, f"Expected 2 provider calls, got {call_count}"

        # The steer must appear as a user-role message in context (which
        # is what get_messages_for_request returns to round 2's call).
        user_contents = [
            m.get("content")
            for m in ctx._messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        assert "redirect me!" in user_contents, (
            f"Steer message not found. User messages: {user_contents}"
        )

        # And the steering_injected event must have fired
        injected_events = [
            name
            for name, _ in hooks.emitted
            if name == "orchestrator:steering_injected"
        ]
        assert len(injected_events) == 1


# ---------------------------------------------------------------------------
# Test 6: Last-drain-edge revive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLastDrainEdgeRevive:
    async def test_revive_when_steer_queued_at_break_point(self) -> None:
        """Steer arriving during final generation triggers one extra provider call."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        first_resp = MockResponse(text="I am done.")
        second_resp = MockResponse(text="OK, I changed course.")

        call_count = 0

        class NonStreamingProvider:
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Inject steer DURING the final provider call
                    orch.steer("wait, redirect!")
                    return first_resp
                return second_resp

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                return []  # no tool calls in either response → hits break path

        await orch.execute(
            prompt="finish this",
            context=ctx,
            providers={"main": NonStreamingProvider()},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert call_count == 2, f"Expected 2 provider calls (revive), got {call_count}"
        # Steer message injected into context before second call
        user_contents = [
            m.get("content")
            for m in ctx._messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        assert "wait, redirect!" in user_contents

    async def test_no_revive_when_steering_queue_empty(self) -> None:
        """Normal single break when no steer is queued."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        call_count = 0

        class NonStreamingProvider:
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                nonlocal call_count
                call_count += 1
                return MockResponse(text="done")

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                return []

        await orch.execute(
            prompt="go",
            context=ctx,
            providers={"main": NonStreamingProvider()},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert call_count == 1, "Empty queue must not trigger revive"


# ---------------------------------------------------------------------------
# Test 7: Capability registered in mount
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCapabilityRegisteredInMount:
    async def test_session_steer_is_registered_after_mount(self) -> None:
        from amplifier_module_loop_streaming import mount

        coordinator = MockCoordinator()
        await mount(coordinator)  # type: ignore[arg-type]

        steer_cap = coordinator.get_capability("session.steer")
        assert steer_cap is not None
        assert callable(steer_cap)

    async def test_session_steer_delegates_to_orchestrator_queue(self) -> None:
        from amplifier_module_loop_streaming import mount, StreamingOrchestrator

        coordinator = MockCoordinator()
        await mount(coordinator)  # type: ignore[arg-type]

        steer_cap = coordinator.get_capability("session.steer")
        assert steer_cap is not None, "session.steer capability must be registered"
        steer_cap("hello from app")

        orchestrator = coordinator._mounts.get("orchestrator")
        assert isinstance(orchestrator, StreamingOrchestrator)
        assert orchestrator._steering_queue.drain() == ["hello from app"]

    async def test_session_steer_rejects_empty_input(self) -> None:
        from amplifier_module_loop_streaming import mount

        coordinator = MockCoordinator()
        await mount(coordinator)  # type: ignore[arg-type]

        steer_cap = coordinator.get_capability("session.steer")
        assert steer_cap is not None, "session.steer capability must be registered"
        with pytest.raises(ValueError):
            steer_cap("")


# ---------------------------------------------------------------------------
# Test 8: Streaming undisturbed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamingUndisturbed:
    async def test_drain_noop_on_empty_queue(self) -> None:
        """_drain_steering returns 0, no messages, no events on empty queue."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        ctx = MockContext()
        hooks = MockHooks()

        count = await orch._drain_steering(ctx, hooks, iteration=1)  # type: ignore[arg-type]

        assert count == 0
        assert ctx._messages == []
        assert hooks.emitted == []

    async def test_streaming_path_yields_tokens_with_empty_queue(self) -> None:
        """Streaming tokens emitted correctly; no steering_injected events."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        class StreamingProvider:
            async def stream(self, chat_request, tools=None):  # noqa: ANN001,ANN201
                yield {"content": "hello"}
                yield {"content": " world"}

        result = await orch.execute(
            prompt="stream this",
            context=ctx,
            providers={"main": StreamingProvider()},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # Tokens must be present in the final response
        assert "hello" in result
        assert "world" in result

        # No steering_injected events (queue was empty throughout)
        injected = [
            name
            for name, _ in hooks.emitted
            if name == "orchestrator:steering_injected"
        ]
        assert injected == []
