"""Tests for ephemeral-injection metadata plumbing (prompt-cache breakpoint fix).

Background: amplifier-module-provider-anthropic's prompt-cache breakpoint logic
(`_count_trailing_ephemeral_messages` / `_apply_conversation_cache_control`) needs
to distinguish genuinely ephemeral, regenerated-per-turn injections (e.g. a
`<system-reminder>` block with a live timestamp/git-status) from stable,
persisted conversation history. It does this by reading
`Message.metadata["ephemeral"]`.

The orchestrator (this module) is the ONLY place that constructs the message
dict for an ephemeral hook injection, so it is the only place that can set
that flag. These tests prove:

  1. A direct ephemeral injection (from the `provider:request` hook, i.e. the
     `result` returned by the `PROVIDER_REQUEST` emit) is appended with
     `metadata={"ephemeral": True}`.
  2. A pending ephemeral injection (queued from `prompt:submit` or `tool:post`
     and applied on the next iteration via `_pending_ephemeral_injections`) is
     also appended with `metadata={"ephemeral": True}`.
  3. Stored/history messages (added via `context.add_message`, e.g. the user's
     prompt) are NOT retroactively marked ephemeral -- they carry no
     `metadata` key at all.
  4. A non-ephemeral `inject_context` result (ephemeral=False) does not cause
     any message to be appended by this code path at all (the append blocks
     are gated on `result.ephemeral` / `injection.get(...)` being truthy), so
     there is no way for a stored/history-bound injection to pick up the
     ephemeral flag through this path.
"""

from __future__ import annotations

import pytest
from amplifier_module_loop_streaming import StreamingOrchestrator

# ---------------------------------------------------------------------------
# Shared stubs -- mirrors the pattern in test_steering.py
# ---------------------------------------------------------------------------


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
        self.session_state: dict = {}

    def register_contributor(self, name: str, source: str, fn) -> None:
        pass

    async def mount(self, name: str, obj) -> None:
        self._mounts[name] = obj

    def register_capability(self, name: str, value) -> None:
        self._capabilities[name] = value

    def get_capability(self, name: str):
        return self._capabilities.get(name)

    async def process_hook_result(self, result, *args, **kwargs):
        return result


class MockResponse:
    """Minimal non-streaming provider response."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.content = None
        self.content_blocks = None
        self.usage = None
        self.metadata = None


class RequestCapturingProvider:
    """Non-streaming provider that records every ChatRequest it receives."""

    def __init__(self) -> None:
        self.requests: list = []

    async def complete(self, chat_request, **kwargs):
        self.requests.append(chat_request)
        return MockResponse(text="ok")

    def parse_tool_calls(self, response):
        return []


class ScriptedHookResult:
    """Hook result whose fields are set per-instance (not class attrs), so a
    MockHooks can hand out a different scripted result per event/iteration."""

    def __init__(
        self,
        action: str = "continue",
        ephemeral: bool = False,
        context_injection: str | None = None,
        context_injection_role: str = "system",
        append_to_last_tool_result: bool = False,
        data=None,
        reason: str | None = None,
    ) -> None:
        self.action = action
        self.ephemeral = ephemeral
        self.context_injection = context_injection
        self.context_injection_role = context_injection_role
        self.append_to_last_tool_result = append_to_last_tool_result
        self.data = data
        self.reason = reason


class ScriptedHooks:
    """Hooks stub that returns a scripted result for a given event name,
    defaulting to a pass-through 'continue' result for anything unscripted."""

    def __init__(self, scripts: dict[str, ScriptedHookResult]) -> None:
        self.scripts = scripts
        self.emitted: list[tuple[str, dict]] = []

    async def emit(
        self, event_name: str, payload: dict | None = None
    ) -> ScriptedHookResult:
        self.emitted.append((event_name, payload or {}))
        return self.scripts.get(event_name, ScriptedHookResult())


# ---------------------------------------------------------------------------
# Test 1: direct ephemeral injection (provider:request hook) gets marked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_ephemeral_injection_carries_metadata_flag() -> None:
    """A `provider:request` hook returning ephemeral inject_context must
    produce an appended message dict with metadata={"ephemeral": True}."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROVIDER_REQUEST: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="<system-reminder>2026-08-09T00:00:00Z</system-reminder>",
                context_injection_role="system",
            ),
        }
    )
    coordinator = MockCoordinator()
    # Explicit "tail" mode: this test exercises the tail-append metadata
    # stamping mechanism specifically (default flipped to "persist" --
    # see test_ephemeral_cache_persist_mode.py -- which routes this
    # injection through context.add_message() instead of a tail append).
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "tail"})

    await orch.execute(
        prompt="hello",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert len(provider.requests) == 1, "Expected exactly one provider.complete() call"
    sent_messages = provider.requests[0].messages

    # The user's prompt is stored history -- no metadata at all.
    user_msgs = [m for m in sent_messages if m.role == "user" and m.content == "hello"]
    assert len(user_msgs) == 1
    assert user_msgs[0].metadata is None, (
        "Stored user prompt must NOT carry a metadata dict"
    )

    # The ephemeral injection must be present with the ephemeral flag set.
    injected = [
        m
        for m in sent_messages
        if m.content == "<system-reminder>2026-08-09T00:00:00Z</system-reminder>"
    ]
    assert len(injected) == 1, "Expected exactly one injected ephemeral message"
    assert injected[0].metadata == {"ephemeral": True}, (
        f"Expected metadata={{'ephemeral': True}}, got {injected[0].metadata!r}"
    )
    assert injected[0].role == "system"


# ---------------------------------------------------------------------------
# Test 2: pending ephemeral injection (queued from tool:post / prompt:submit)
# also gets marked when applied on the next iteration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_ephemeral_injection_carries_metadata_flag() -> None:
    """An injection queued via `_pending_ephemeral_injections` (the
    prompt:submit / tool:post path) must also carry the ephemeral flag once
    applied to the outgoing message_dicts."""
    from amplifier_core.events import PROMPT_SUBMIT

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROMPT_SUBMIT: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="pending-ephemeral-content",
                context_injection_role="user",
            ),
        }
    )
    coordinator = MockCoordinator()
    # Explicit "tail" mode: same rationale as test 1 above -- the pending-
    # injection tail-append path is what this test targets, not the
    # persist path that is now the default.
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "tail"})

    await orch.execute(
        prompt="hi again",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert len(provider.requests) == 1
    sent_messages = provider.requests[0].messages

    injected = [m for m in sent_messages if m.content == "pending-ephemeral-content"]
    assert len(injected) == 1, "Expected the pending ephemeral injection to be applied"
    assert injected[0].metadata == {"ephemeral": True}, (
        f"Expected metadata={{'ephemeral': True}}, got {injected[0].metadata!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: non-ephemeral inject_context never gets appended (and therefore
# never gets marked) by this code path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_ephemeral_injection_is_not_appended_or_marked() -> None:
    """A hook returning action=inject_context with ephemeral=False must NOT
    cause a message to be appended by the ephemeral-injection code path (that
    is gated strictly on `.ephemeral` being truthy), so it can never pick up
    metadata={"ephemeral": True} through this mechanism."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROVIDER_REQUEST: ScriptedHookResult(
                action="inject_context",
                ephemeral=False,  # NOT ephemeral -- should be handled elsewhere, not here
                context_injection="stable-feedback-content",
                context_injection_role="system",
            ),
        }
    )
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({})

    await orch.execute(
        prompt="hello",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert len(provider.requests) == 1
    sent_messages = provider.requests[0].messages

    # Nothing in the outgoing request should carry the ephemeral-injection
    # content -- this orchestrator code path never appends non-ephemeral
    # results, so there is no message to (mis)mark.
    matching = [m for m in sent_messages if m.content == "stable-feedback-content"]
    assert matching == [], (
        "Non-ephemeral inject_context must not be appended by the "
        "ephemeral-injection append blocks"
    )

    # And no message anywhere carries the ephemeral flag.
    assert all(m.metadata is None for m in sent_messages), (
        "No message should carry metadata={'ephemeral': True} when the only "
        "hook result was non-ephemeral"
    )


# ---------------------------------------------------------------------------
# Test 4: append_to_last_tool_result path is preserved -- appending into an
# existing tool-result message's content still works (content concatenation
# unchanged), and does not spuriously add metadata to that stored message.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_to_last_tool_result_still_concatenates_content() -> None:
    """When the last message going into the NEXT provider call is a tool
    result and append_to_last_tool_result is set, the ephemeral content must
    still be concatenated onto that message's content (existing behavior
    preserved), and the stored history copy must remain untouched.

    This requires a real two-round tool-calling turn: round 1 returns a tool
    call, the tool executes, tool:post fires with an ephemeral
    append_to_last_tool_result injection, and round 2's outgoing request is
    inspected. (A single-round setup can't exercise this branch: execute()
    always adds the new user prompt to context *before* building the first
    request, so the last message is never a tool result on round 1.)
    """
    from amplifier_core import ToolResult
    from amplifier_core.events import TOOL_POST

    ctx = MockContext()

    class OneShotTool:
        name = "mock_tool"
        description = "test tool"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, arguments):
            return ToolResult(success=True, output="original tool output")

    class ToolCallStub:
        id = "tc-1"
        name = "mock_tool"
        arguments: dict = {}

    call_count = 0
    captured_requests: list = []

    class TwoRoundProvider:
        async def complete(self, chat_request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                captured_requests.append(chat_request)
            return MockResponse(text=f"round {call_count}")

        def parse_tool_calls(self, response):
            if call_count == 1:
                return [ToolCallStub()]
            return []

    provider = TwoRoundProvider()

    hooks = ScriptedHooks(
        {
            TOOL_POST: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="appended-ephemeral-note",
                context_injection_role="system",
                append_to_last_tool_result=True,
            ),
        }
    )
    coordinator = MockCoordinator()
    # Explicit "tail" mode: `append_to_last_tool_result` is only reachable
    # when mode != "persist" (persist takes priority in the orchestrator's
    # branch order -- see __init__.py). This test targets that tail-only
    # concatenation behavior specifically, which is now non-default.
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "tail"})

    await orch.execute(
        prompt="continue",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={"mock_tool": OneShotTool()},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert call_count == 2, (
        "Expected exactly two provider rounds (tool call + follow-up)"
    )
    assert len(captured_requests) == 1, "Expected round 2's request to be captured"
    sent_messages = captured_requests[0].messages

    tool_msgs = [m for m in sent_messages if m.role == "tool"]
    assert len(tool_msgs) == 1, (
        "Expected exactly one tool-role message in round 2's request"
    )
    assert "original tool output" in tool_msgs[0].content
    assert "appended-ephemeral-note" in tool_msgs[0].content, (
        "append_to_last_tool_result must still concatenate the ephemeral "
        "content onto the existing tool message"
    )

    # The stored tool message in context history must be untouched (the
    # append happens on a synthesized copy of message_dicts, never mutating
    # the object context.get_messages_for_request() returned).
    stored_tool_msgs = [m for m in ctx._messages if m.get("role") == "tool"]
    assert len(stored_tool_msgs) == 1
    assert stored_tool_msgs[0]["content"] == "original tool output", (
        "The append_to_last_tool_result path must not mutate stored history"
    )
