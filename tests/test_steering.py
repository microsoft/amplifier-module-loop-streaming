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

  SteeringQueue.clear() unit tests — drain-and-discard method.

  Hardening (steering-drain.md §4 failing-test plan):
  O1 — graceful cancel at top-of-iteration clears the queue (FAILS before fix)
  O2 — steer during cancel does not leak to next turn     (FAILS before fix)
  O3 — execute() entry clears stale queue                 (FAILS before fix)
  O4 — immediate cancel (CancelledError) clears queue     (FAILS before fix)
  O5 — burst: N steers → N separate FIFO user messages   (guard, already PASS)
  O6 — drain never injects empty/whitespace message       (guard, already PASS)
  O7 — steer queued before break-check applies via revive (guard, already PASS)
  O8 — revive uses single injection path (no double-inject) (guard, already PASS)

Note: execute(), _drain_steering(), and mount() are typed against Rust-backed
types (RustHookRegistry, RustCoordinator). Duck-typed test stubs are not
structurally assignable to those types, hence the # type: ignore[arg-type]
suppressions below — this is the standard pattern for testing duck-typed
Python code against Rust extension interfaces.

Note on multi-round tests: _has_pending_tools is a stub returning False, so
genuine multi-round mid-turn steering runs on the NON-streaming path.
All multi-round tests use a NonStreamingProvider (no .stream attribute).
"""

from __future__ import annotations

import asyncio

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


# ---------------------------------------------------------------------------
# SteeringQueue.clear() unit tests
# ---------------------------------------------------------------------------


class TestSteeringQueueClear:
    """Unit tests for the new SteeringQueue.clear() method (steering-drain.md §5.1)."""

    def test_clear_empties_queue(self) -> None:
        """clear() must discard all pending messages and return the count discarded."""
        from amplifier_module_loop_streaming.steering import SteeringQueue

        queue = SteeringQueue()
        queue.steer("a")
        queue.steer("b")
        queue.steer("c")
        assert not queue.is_empty
        count = queue.clear()
        assert count == 3, f"Expected 3 discarded, got {count}"
        assert queue.is_empty
        assert queue.drain() == []

    def test_clear_on_empty_is_noop(self) -> None:
        """clear() on an already-empty queue must return 0 and leave it empty."""
        from amplifier_module_loop_streaming.steering import SteeringQueue

        queue = SteeringQueue()
        assert queue.is_empty
        count = queue.clear()
        assert count == 0
        assert queue.is_empty


# ---------------------------------------------------------------------------
# O1-O4: Cancellation-clear tests — FAIL before fix, PASS after fix
# (steering-drain.md §3.6, §5.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCancellationClearsQueue:
    """O1-O4: Every cancellation exit path must call SteeringQueue.clear()."""

    async def test_graceful_cancel_clears_pending_steers(self) -> None:
        """O1: Graceful cancel at top-of-iteration (~line 300) must clear the queue.

        A steer enqueued AFTER execute()'s entry clear (via the execution:start
        hook) persists to the top-of-iteration cancel check.  Without the fix
        the cancel return at ~line 300 leaves the queue non-empty; with the fix
        clear() is called before that return.
        """
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        ctx = MockContext()

        class HooksThatSteerOnStart(MockHooks):
            """Injects a steer AFTER execute()-entry clear, BEFORE cancel check."""

            def __init__(self, orch_instance: StreamingOrchestrator) -> None:
                super().__init__()
                self._orch = orch_instance
                self._done = False

            async def emit(
                self, event_name: str, payload: dict | None = None
            ) -> MockHookResult:
                result = await super().emit(event_name, payload)
                if event_name == "execution:start" and not self._done:
                    self._done = True
                    # Enqueued AFTER execute() entry-clear, BEFORE cancel check
                    self._orch.steer("should_be_cleared_on_cancel_exit")
                return result

        hooks = HooksThatSteerOnStart(orch)
        coordinator = MockCoordinator()
        coordinator.cancellation.is_cancelled = True  # fires at top-of-iteration

        class NonStreamingProvider:
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                pytest.fail("provider.complete must not be reached when cancelled")

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                return []  # pragma: no cover

        await orch.execute(
            prompt="turn-O1",
            context=ctx,
            providers={"main": NonStreamingProvider()},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # Without fix: steer persists (no clear at cancel exit ~line 300)
        # With fix: clear() at cancel exit -> queue empty
        assert orch._steering_queue.is_empty, (
            "O1: graceful-cancel exit must clear the steering queue"
        )
        injected = [
            e for e, _ in hooks.emitted if e == "orchestrator:steering_injected"
        ]
        assert injected == [], "O1: cancelled turn must not inject any steers"

    async def test_steer_during_cancel_does_not_leak_into_next_turn(self) -> None:
        """O2: Steer queued before a cancelled turn must not appear in turn 2.

        Without any fix the steer persists through the cancelled turn and is
        drained at turn 2's top-of-iteration drain.  With the fix (execute()
        entry clear OR cancel-exit clear) turn 2 sees no leaked steer.
        """
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})

        # Pre-seed: simulates a steer arriving just before the turn was cancelled
        orch.steer("leaked_steer_from_cancelled_turn_O2")

        class NonStreamingProvider:
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                return MockResponse(text="done")

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                return []

        # Turn 1: cancelled immediately
        ctx1 = MockContext()
        hooks1 = MockHooks()
        coord1 = MockCoordinator()
        coord1.cancellation.is_cancelled = True
        await orch.execute(
            prompt="turn1-O2",
            context=ctx1,
            providers={"main": NonStreamingProvider()},
            tools={},
            hooks=hooks1,  # type: ignore[arg-type]
            coordinator=coord1,  # type: ignore[arg-type]
        )

        # Turn 2: fresh context, not cancelled
        ctx2 = MockContext()
        hooks2 = MockHooks()
        coord2 = MockCoordinator()
        coord2.cancellation.is_cancelled = False
        await orch.execute(
            prompt="fresh-prompt-turn2-O2",
            context=ctx2,
            providers={"main": NonStreamingProvider()},
            tools={},
            hooks=hooks2,  # type: ignore[arg-type]
            coordinator=coord2,  # type: ignore[arg-type]
        )

        # Without fix: steer appears in turn 2's user messages
        # With fix: steer discarded -> not in turn 2
        user_msgs_t2 = [
            m.get("content")
            for m in ctx2._messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        assert "leaked_steer_from_cancelled_turn_O2" not in user_msgs_t2, (
            "O2: steer from cancelled turn must not leak into turn 2"
        )

    async def test_execute_start_clears_stale_queue(self) -> None:
        """O3: execute() entry must clear the steering queue before the loop.

        A steer queued BEFORE execute() is called (stale from a prior context)
        must be cleared at execute() entry so it is never injected into this turn.
        Without the fix the stale steer is drained at iteration 1 and injected.
        """
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        # Pre-seed: simulates a steer that leaked from a previous turn
        orch.steer("stale_steer_must_not_inject_O3")

        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        class NonStreamingProvider:
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                return MockResponse(text="fresh response")

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                return []

        await orch.execute(
            prompt="fresh-prompt-O3",
            context=ctx,
            providers={"main": NonStreamingProvider()},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # Without fix: stale steer injected at iteration-1 top-of-iteration drain
        # With fix: entry clear -> stale steer discarded -> no injection
        injected_events = [
            e for e, _ in hooks.emitted if e == "orchestrator:steering_injected"
        ]
        assert injected_events == [], (
            "O3: stale steer pre-seeded before execute() must not be injected"
        )
        assert orch._steering_queue.is_empty

    async def test_immediate_cancel_clears_pending_steers(self) -> None:
        """O4: asyncio.CancelledError path (~line 777) must clear the queue.

        A steer enqueued DURING tool execution (after execute() entry clear)
        must be cleared before the CancelledError is re-raised so it cannot
        leak into a future turn.
        """
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        tc = MockToolCall()

        class ImmediateCancelTool:
            name = "mock_tool"
            description = "simulates immediate cancel via asyncio.CancelledError"
            input_schema: dict = {"type": "object", "properties": {}}

            async def execute(self, arguments):  # noqa: ANN001,ANN201
                # Enqueue AFTER execute() entry clear, then trigger immediate cancel
                orch.steer("steer_enqueued_during_tool_cancel_O4")
                raise asyncio.CancelledError("simulated immediate cancel")

        class NonStreamingProviderWithTools:
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                return MockResponse(text="doing work...")

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                return [tc]

        with pytest.raises(asyncio.CancelledError):
            await orch.execute(
                prompt="do-work-O4",
                context=ctx,
                providers={"main": NonStreamingProviderWithTools()},
                tools={"mock_tool": ImmediateCancelTool()},
                hooks=hooks,  # type: ignore[arg-type]
                coordinator=coordinator,  # type: ignore[arg-type]
            )

        # Without fix: steer persists after re-raise (no clear before ~line 777 raise)
        # With fix: clear() before re-raise -> queue empty
        assert orch._steering_queue.is_empty, (
            "O4: immediate cancel (CancelledError path) must clear the steering queue"
        )


# ---------------------------------------------------------------------------
# O5-O8: Guard tests — PASS before AND after fix
# (steering-drain.md §3.3, §3.2, §2.1, §2.3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBurstSteering:
    async def test_burst_injects_n_separate_user_messages_fifo(self) -> None:
        """O5: N steers queued during one tool round -> N separate user messages, FIFO.

        Confirms that _drain_steering loops per-message (not concatenated) and
        queued_remaining counts down correctly.  Guard passes before and after fix.
        """
        from amplifier_module_loop_streaming import StreamingOrchestrator
        from amplifier_core import ToolResult

        orch = StreamingOrchestrator({})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        tc = MockToolCall()
        call_count = 0

        class BurstSteeringTool:
            name = "mock_tool"
            description = "burst test"
            input_schema: dict = {"type": "object", "properties": {}}

            async def execute(self, arguments):  # noqa: ANN001,ANN201
                orch.steer("a")
                orch.steer("b")
                orch.steer("c")
                return ToolResult(success=True, output="burst done")

        class NonStreamingProvider:
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                nonlocal call_count
                call_count += 1
                return MockResponse(text=f"round {call_count}")

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                # Round 1 returns tool calls; round 2 terminates
                if call_count == 1:
                    return [tc]
                return []

        await orch.execute(
            prompt="burst-O5",
            context=ctx,
            providers={"main": NonStreamingProvider()},
            tools={"mock_tool": BurstSteeringTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert call_count == 2, f"O5: expected 2 provider calls, got {call_count}"

        # Three SEPARATE user messages -- not one combined
        steered_msgs = [
            m.get("content")
            for m in ctx._messages
            if isinstance(m, dict)
            and m.get("role") == "user"
            and m.get("content") in {"a", "b", "c"}
        ]
        assert steered_msgs == ["a", "b", "c"], (
            f"O5: expected ['a','b','c'] FIFO, got {steered_msgs}"
        )

        # Three steering_injected events with queued_remaining 2, 1, 0
        injected = [
            payload
            for name, payload in hooks.emitted
            if name == "orchestrator:steering_injected"
        ]
        assert len(injected) == 3, (
            f"O5: expected 3 injected events, got {len(injected)}"
        )
        assert [e["queued_remaining"] for e in injected] == [2, 1, 0], (
            "O5: queued_remaining must count down 2,1,0"
        )
        assert [e["content"] for e in injected] == ["a", "b", "c"], (
            "O5: content must be FIFO a,b,c"
        )


@pytest.mark.asyncio
class TestEmptySteerGuard:
    async def test_drain_never_injects_empty_user_message(self) -> None:
        """O6: Empty/whitespace steer raises ValueError; no empty user message injected.

        The guard lives at the orchestrator steer() boundary (SteeringQueue.steer).
        This test confirms it via the public steer() + a full execute() run.
        Guard passes before and after fix.
        """
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})

        # Attempting whitespace via the orchestrator's public steer() must raise
        with pytest.raises(ValueError):
            orch.steer("   ")

        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        class NonStreamingProvider:
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                return MockResponse(text="done")

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                return []

        await orch.execute(
            prompt="hello-O6",
            context=ctx,
            providers={"main": NonStreamingProvider()},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # No empty/whitespace user messages must ever appear in context
        empty_user_msgs = [
            m
            for m in ctx._messages
            if m.get("role") == "user" and not str(m.get("content", "")).strip()
        ]
        assert empty_user_msgs == [], (
            "O6: no empty user-role messages must ever be injected"
        )


@pytest.mark.asyncio
class TestReviveWithinExecute:
    async def test_steer_after_break_check_applies_next_turn(self) -> None:
        """O7: Steer enqueued during the final provider call applies at iteration 2.

        When a steer is queued DURING the last provider call (before the
        is_empty break-point check), the check sees a non-empty queue, the
        revive path fires, and the steer is applied at the *next* iteration
        within the same execute() call (iteration == 2).  This is the
        'correct, a turn late' contract: one iteration later within the same turn.
        The revive mechanism is unchanged by the fix.  Guard passes before and after.
        """
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
                if call_count == 1:
                    # Enqueue during the final provider call -- BEFORE is_empty check
                    orch.steer("steer_queued_before_break_check_O7")
                return MockResponse(text=f"resp-{call_count}")

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                return []  # No tools: hits break/revive path

        await orch.execute(
            prompt="O7-revive",
            context=ctx,
            providers={"main": NonStreamingProvider()},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # Revive must have fired (2 provider calls)
        assert call_count == 2, (
            f"O7: expected 2 provider calls (revive), got {call_count}"
        )

        # The steering_injected event must carry iteration == 2 (applied one iteration late)
        injected = [
            payload
            for name, payload in hooks.emitted
            if name == "orchestrator:steering_injected"
        ]
        assert len(injected) == 1, f"O7: expected 1 injected event, got {len(injected)}"
        assert injected[0]["iteration"] == 2, (
            f"O7: steer must apply at iteration 2 ('a turn late'), got {injected[0]['iteration']}"
        )

    async def test_revive_uses_single_injection_path(self) -> None:
        """O8: Revive injects the steer exactly once -- via top-of-iteration drain only.

        When the is_empty check triggers a revive (continue), the steer is
        drained at the NEXT iteration's top-of-iteration drain.  It must NOT be
        double-injected (once at break point AND once at the drain).
        Guard passes before and after fix.
        """
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
                if call_count == 1:
                    orch.steer("single_injection_steer_O8")
                return MockResponse(text=f"resp-{call_count}")

            def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
                return []

        await orch.execute(
            prompt="O8-single-inject",
            context=ctx,
            providers={"main": NonStreamingProvider()},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # Exactly ONE injection event -- not two
        injected = [
            payload
            for name, payload in hooks.emitted
            if name == "orchestrator:steering_injected"
        ]
        assert len(injected) == 1, (
            f"O8: steer must be injected exactly once, got {len(injected)}"
        )

        # Exactly one user message in context with the steer content
        steer_msgs = [
            m
            for m in ctx._messages
            if m.get("role") == "user"
            and m.get("content") == "single_injection_steer_O8"
        ]
        assert len(steer_msgs) == 1, (
            f"O8: steer must appear exactly once in context, got {len(steer_msgs)}"
        )
