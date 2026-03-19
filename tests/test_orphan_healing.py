"""Tests for orphaned tool-call healing.

When a session is interrupted mid-tool-execution, assistant messages with
tool_use blocks are persisted but their tool_result counterparts are not.
On resume the provider rejects the request:

    "tool_use ids were found without tool_result blocks immediately after"

The orchestrator should detect and repair these orphans automatically so
the session can continue without manual intervention.
"""

import json
from unittest.mock import MagicMock

import pytest

from amplifier_core import HookRegistry, HookResult
from amplifier_core.llm_errors import InvalidRequestError, LLMError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProvider:
    @property
    def name(self):
        return "mock"

    @property
    def context_window(self):
        return 100_000

    @property
    def max_output_tokens(self):
        return 4096

    def parse_tool_calls(self, response):
        return []


class SuccessProvider(MockProvider):
    """Returns a fixed text response."""

    def __init__(self, text="healed"):
        self._text = text

    async def complete(self, request, **kwargs):
        resp = MagicMock()
        resp.content = None
        resp.content_blocks = None
        resp.text = self._text
        resp.usage = None
        resp.metadata = None
        return resp


class OrphanErrorThenSuccessProvider(MockProvider):
    """First call raises orphaned-tool InvalidRequestError, second succeeds."""

    def __init__(self):
        self._calls = 0

    async def complete(self, request, **kwargs):
        self._calls += 1
        if self._calls == 1:
            raise InvalidRequestError(
                "messages.96: `tool_use` ids were found without "
                "`tool_result` blocks immediately after: toolu_abc123. "
                "Each `tool_use` block must have a corresponding "
                "`tool_result` block in the next message.",
                provider="anthropic",
                status_code=400,
            )
        resp = MagicMock()
        resp.content = None
        resp.content_blocks = None
        resp.text = "recovered"
        resp.usage = None
        resp.metadata = None
        return resp


class MockContext:
    """Tracks messages and returns them on request."""

    def __init__(self, initial_messages=None):
        self.messages = list(initial_messages or [])

    async def get_messages_for_request(self, provider=None):
        return list(self.messages)

    async def add_message(self, msg):
        self.messages.append(msg)


def create_orchestrator(**overrides):
    from amplifier_module_loop_streaming import StreamingOrchestrator

    config = {"max_iterations": 5, "stream_delay": 0, **overrides}
    return StreamingOrchestrator(config=config)


async def collect_response(
    orch, prompt, context, providers, tools, hooks, coordinator=None
):
    """Run orchestrator.execute() and return the full response text."""
    return await orch.execute(prompt, context, providers, tools, hooks, coordinator)


def _orphan_assistant_msg(tool_id="tc_orphan", tool_name="bash"):
    """Create an assistant message with an orphaned tool_call.

    Uses the kernel-normalised ``tool_call`` type (with ``input``) so the
    message survives ``Message(**msg)`` Pydantic validation inside the
    orchestrator.
    """
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "id": tool_id,
                "name": tool_name,
                "input": {},
            },
        ],
    }


# ---------------------------------------------------------------------------
# _find_orphaned_tool_calls  (unit tests)
# ---------------------------------------------------------------------------


class TestFindOrphanedToolCalls:
    """Unit tests for the static orphan-detection scanner."""

    def _find(self, messages):
        from amplifier_module_loop_streaming import StreamingOrchestrator

        return StreamingOrchestrator._find_orphaned_tool_calls(messages)

    def test_no_orphans_empty(self):
        assert self._find([]) == {}

    def test_no_orphans_balanced(self):
        """Balanced tool_use + tool_result -> no orphans."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tc_1", "name": "bash"},
                ],
                "tool_calls": [],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        assert self._find(msgs) == {}

    def test_orphan_in_content_blocks(self):
        """tool_use in content blocks without matching result -> orphan."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tc_orphan", "name": "read_file"},
                ],
                "tool_calls": [],
            },
            # No tool result follows
        ]
        result = self._find(msgs)
        assert result == {"tc_orphan": "read_file"}

    def test_orphan_in_tool_calls_array(self):
        """tool_call in tool_calls array without matching result -> orphan."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "thinking...",
                "tool_calls": [
                    {"id": "tc_arr", "name": "bash"},
                ],
            },
        ]
        result = self._find(msgs)
        assert result == {"tc_arr": "bash"}

    def test_multiple_orphans(self):
        """Multiple orphaned tool calls detected."""
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tc_1", "name": "bash"},
                    {"type": "tool_use", "id": "tc_2", "name": "read_file"},
                ],
                "tool_calls": [],
            },
            # Only one result
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
        ]
        result = self._find(msgs)
        assert result == {"tc_2": "read_file"}

    def test_tool_call_type_variant(self):
        """type: 'tool_call' (kernel-normalised) is also detected."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "tc_oai", "name": "web_search"},
                ],
            },
        ]
        result = self._find(msgs)
        assert result == {"tc_oai": "web_search"}

    def test_object_style_content_blocks(self):
        """Content blocks that are objects (not dicts) are handled."""
        block = MagicMock()
        block.type = "tool_use"
        block.id = "tc_obj"
        block.name = "grep"

        msgs = [
            {"role": "assistant", "content": [block], "tool_calls": []},
        ]
        result = self._find(msgs)
        assert result == {"tc_obj": "grep"}

    def test_object_style_tool_calls(self):
        """tool_calls that are objects (not dicts) are handled."""
        tc = MagicMock()
        tc.id = "tc_mock"
        tc.name = "edit_file"

        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [tc]},
        ]
        result = self._find(msgs)
        assert result == {"tc_mock": "edit_file"}


# ---------------------------------------------------------------------------
# _heal_orphaned_tool_calls  (unit tests)
# ---------------------------------------------------------------------------


class TestHealOrphanedToolCalls:
    """Unit tests for the async healing method."""

    @pytest.mark.asyncio
    async def test_injects_synthetic_results(self):
        orch = create_orchestrator()
        ctx = MockContext()
        hooks = HookRegistry()

        orphans = {"tc_1": "bash", "tc_2": "read_file"}
        await orch._heal_orphaned_tool_calls(orphans, ctx, hooks)

        # Two synthetic tool results should be added
        tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

        ids = {m["tool_call_id"] for m in tool_msgs}
        assert ids == {"tc_1", "tc_2"}

        # Each result should indicate interruption
        for msg in tool_msgs:
            body = json.loads(msg["content"])
            assert body["interrupted"] is True
            assert "interrupted" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_emits_healed_event(self):
        orch = create_orchestrator()
        ctx = MockContext()
        hooks = HookRegistry()

        events = []

        async def capture(event, data):
            events.append((event, data))
            return HookResult()

        hooks.register("orchestrator:tool_calls_healed", capture)

        orphans = {"tc_x": "bash"}
        await orch._heal_orphaned_tool_calls(orphans, ctx, hooks)

        healed_events = [
            (e, d) for e, d in events if e == "orchestrator:tool_calls_healed"
        ]
        assert len(healed_events) == 1
        _, data = healed_events[0]
        assert data["healed_count"] == 1
        assert "tc_x" in data["tool_call_ids"]


# ---------------------------------------------------------------------------
# Integration: proactive healing on session resume
# ---------------------------------------------------------------------------


class TestProactiveHealing:
    """Proactive healing fixes orphans before the provider call."""

    @pytest.mark.asyncio
    async def test_heals_orphaned_context_and_succeeds(self):
        """Orchestrator detects orphaned tool calls in context, injects
        synthetic results, and the provider call succeeds."""
        orch = create_orchestrator()
        provider = SuccessProvider(text="all good")

        # Context has an orphaned tool_call (simulating interrupted session).
        # Uses kernel-normalised format so Message() validation passes.
        ctx = MockContext(
            initial_messages=[
                {"role": "user", "content": "do something"},
                _orphan_assistant_msg("tc_orphan", "bash"),
                # Missing tool_result for tc_orphan
            ]
        )

        hooks = HookRegistry()
        providers = {"mock": provider}

        result = await collect_response(orch, "continue", ctx, providers, {}, hooks)

        # Should succeed (provider wasn't asked with orphaned messages)
        assert "all good" in result

        # Synthetic tool result should have been injected
        tool_results = [
            m
            for m in ctx.messages
            if m.get("role") == "tool" and m.get("tool_call_id") == "tc_orphan"
        ]
        assert len(tool_results) == 1
        body = json.loads(tool_results[0]["content"])
        assert body["interrupted"] is True


# ---------------------------------------------------------------------------
# Integration: reactive healing on provider rejection
# ---------------------------------------------------------------------------


class TestReactiveHealing:
    """Reactive healing catches provider rejection and retries."""

    @pytest.mark.asyncio
    async def test_heals_on_invalid_request_error(self):
        """When proactive healing misses (e.g. context returns different
        messages), reactive healing catches the InvalidRequestError and
        retries successfully."""
        orch = create_orchestrator()

        # Provider fails first (orphan error), succeeds second
        provider = OrphanErrorThenSuccessProvider()

        # Build an orphan message using kernel-normalised format.
        orphan_msg = _orphan_assistant_msg("tc_late", "bash")

        class LazyOrphanContext(MockContext):
            """Returns orphan only on first call (simulates race)."""

            def __init__(self):
                super().__init__()
                self._call_count = 0

            async def get_messages_for_request(self, provider=None):
                self._call_count += 1
                # Include orphan in messages so reactive scanner finds it
                base = [{"role": "user", "content": "test"}]
                if self._call_count <= 2:
                    # First two calls include the orphan
                    base.append(orphan_msg)
                else:
                    # After healing, include the synthetic result too
                    base.append(orphan_msg)
                    tool_results = [m for m in self.messages if m.get("role") == "tool"]
                    base.extend(tool_results)
                return base

        ctx = LazyOrphanContext()
        hooks = HookRegistry()
        providers = {"mock": provider}

        result = await collect_response(orch, "retry", ctx, providers, {}, hooks)

        # Should recover after healing
        assert result == "recovered"
        assert provider._calls == 2  # First call failed, second succeeded

    @pytest.mark.asyncio
    async def test_non_orphan_errors_still_raise(self):
        """LLMErrors that are NOT orphaned-tool errors propagate normally."""
        orch = create_orchestrator()

        class AuthErrorProvider(MockProvider):
            async def complete(self, request, **kwargs):
                raise LLMError(
                    "Unauthorized",
                    provider="test",
                    status_code=401,
                )

        ctx = MockContext()
        hooks = HookRegistry()
        providers = {"mock": AuthErrorProvider()}

        with pytest.raises(LLMError, match="Unauthorized"):
            await collect_response(orch, "test", ctx, providers, {}, hooks)

    @pytest.mark.asyncio
    async def test_healing_limited_to_one_attempt(self):
        """Healing only happens once per execute() -- no infinite loops."""
        orch = create_orchestrator()
        heal_count = 0

        class AlwaysOrphanErrorProvider(MockProvider):
            """Always raises orphan error (healing doesn't fix it)."""

            async def complete(self, request, **kwargs):
                raise InvalidRequestError(
                    "tool_use ids found without tool_result blocks",
                    provider="test",
                    status_code=400,
                )

        class TrackingContext(MockContext):
            """Counts add_message calls to track healing."""

            async def add_message(self, msg):
                nonlocal heal_count
                if msg.get("role") == "tool" and "interrupted" in str(
                    msg.get("content", "")
                ):
                    heal_count += 1
                await super().add_message(msg)

        ctx = TrackingContext(
            initial_messages=[
                {"role": "user", "content": "test"},
                _orphan_assistant_msg("tc_stuck", "bash"),
            ]
        )

        hooks = HookRegistry()
        providers = {"mock": AlwaysOrphanErrorProvider()}

        with pytest.raises(InvalidRequestError):
            await collect_response(orch, "test", ctx, providers, {}, hooks)

        # Healing should have been attempted (proactive), but since the
        # provider keeps failing, it should NOT loop infinitely.
        # The guard limits healing to one attempt.
        assert heal_count <= 2  # At most proactive + reactive
