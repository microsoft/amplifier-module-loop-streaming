"""Tests for intermediate text handling when tool calls are present.

Covers:
- execute() returns ONLY final iteration text (intermediate text excluded)
- Content block events fire for text blocks in tool-call responses (hook path)
- Text content_block:end events fire BEFORE tool execution begins
- Pure text responses (no tool calls) still work identically (regression)
- Accumulator resets between iterations (defense-in-depth)
"""

import pytest

from amplifier_core.events import CONTENT_BLOCK_END, CONTENT_BLOCK_START
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_core.testing import EventRecorder, MockContextManager
from amplifier_core import ToolResult

from amplifier_module_loop_streaming import StreamingOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _BaseProvider:
    """Base mock provider — no `stream` attr so the orchestrator uses complete()."""

    name = "mock"


def _orch(**overrides):
    config = {"max_iterations": 5, "stream_delay": 0, **overrides}
    return StreamingOrchestrator(config)


class SimpleTool:
    """Minimal tool that returns success."""

    name = "todo"
    description = "Update todo list"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, args):
        return ToolResult(success=True, output="done")


class _MockContentBlock:
    """Mock raw SDK content block with .type.value and .to_dict() interface.

    The orchestrator's content block event emission (lines 419-443) uses
    raw SDK objects that have block.type.value (an enum) and block.to_dict().
    Our Pydantic TextBlock has type as a plain string and no to_dict().
    This mock bridges the gap for testing.
    """

    def __init__(self, block_type, text):
        self.type = type("_Enum", (), {"value": block_type})()
        self.text = text
        self.raw = None

    def to_dict(self):
        return {"type": self.type.value, "text": self.text}


def _response_with_content_blocks(text, tool_calls=None):
    """Create a ChatResponse with both content (Pydantic) and content_blocks (mock SDK)."""
    resp = ChatResponse(
        content=[TextBlock(text=text)],
        tool_calls=tool_calls,
    )
    # Set content_blocks to simulate raw SDK objects (as real providers do)
    resp.content_blocks = [_MockContentBlock("text", text)]
    return resp


# ---------------------------------------------------------------------------
# Test: execute() returns only final iteration text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_returns_only_final_iteration_text():
    """execute() must return ONLY the final iteration's text, not intermediate.

    The P1 fix yielded intermediate text into the token stream, which caused
    it to appear in the accumulated return value. After removing that yield,
    execute() returns only the final iteration's response. Intermediate text
    is still visible via content_block:end events (tested separately).
    """
    call_count = 0

    class TextThenToolProvider(_BaseProvider):
        """First call: text + tool_use. Second call: pure text (final)."""

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="Let me check that config file.")],
                    tool_calls=[
                        ToolCall(id="tc1", name="todo", arguments={"action": "list"})
                    ],
                )
            # Second call is the final response (no tool calls)
            return ChatResponse(content=[TextBlock(text="Here are the results.")])

        def parse_tool_calls(self, response):
            return getattr(response, "tool_calls", None) or []

    orchestrator = _orch()
    provider = TextThenToolProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    result = await orchestrator.execute(
        prompt="Check the config",
        context=context,
        providers={"default": provider},
        tools={"todo": SimpleTool()},
        hooks=hooks,
    )

    # Intermediate text must NOT be in the return value
    assert "Let me check that config file." not in result, (
        f"Intermediate text leaked into return value: {result!r}"
    )
    # Only the final response text should be returned
    assert "Here are the results." in result


@pytest.mark.asyncio
async def test_content_block_events_fire_for_tool_call_responses():
    """Content block events must fire for text blocks even when tool calls are present.

    The content block event emission at lines 419-443 fires for ALL responses
    (before the tool_calls branch split), but only when response.content_blocks
    is set (raw SDK objects). This test verifies the events fire correctly
    for responses that contain both text and tool calls.
    """
    call_count = 0

    class TextThenToolProvider(_BaseProvider):
        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _response_with_content_blocks(
                    "Analyzing the module.",
                    tool_calls=[
                        ToolCall(id="tc1", name="todo", arguments={"action": "list"})
                    ],
                )
            return _response_with_content_blocks("Done.")

        def parse_tool_calls(self, response):
            return getattr(response, "tool_calls", None) or []

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Analyze",
        context=context,
        providers={"default": TextThenToolProvider()},
        tools={"todo": SimpleTool()},
        hooks=hooks,
    )

    # There should be CONTENT_BLOCK_START and CONTENT_BLOCK_END events
    # from BOTH the tool-call response AND the final response
    start_events = hooks.get_events(CONTENT_BLOCK_START)
    end_events = hooks.get_events(CONTENT_BLOCK_END)

    # At minimum: 1 from the tool-call response + 1 from the final response = 2
    assert len(start_events) >= 2, (
        f"Expected >= 2 CONTENT_BLOCK_START events, got {len(start_events)}"
    )
    assert len(end_events) >= 2, (
        f"Expected >= 2 CONTENT_BLOCK_END events, got {len(end_events)}"
    )


@pytest.mark.asyncio
async def test_text_yielded_before_tool_execution():
    """Intermediate text must be yielded BEFORE tool execution begins.

    The user should see the assistant's narration before tool indicators appear.
    We verify this by checking that the text content block end event fires
    before any TOOL_PRE events.
    """
    call_count = 0

    class TextThenToolProvider(_BaseProvider):
        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _response_with_content_blocks(
                    "Checking now.",
                    tool_calls=[
                        ToolCall(id="tc1", name="todo", arguments={"action": "list"})
                    ],
                )
            return _response_with_content_blocks("All done.")

        def parse_tool_calls(self, response):
            return getattr(response, "tool_calls", None) or []

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Check",
        context=context,
        providers={"default": TextThenToolProvider()},
        tools={"todo": SimpleTool()},
        hooks=hooks,
    )

    # Get all events in order from the recorder
    # EventRecorder stores events as list of (event_name, data) tuples in .events
    all_events = hooks.events
    text_block_end_indices = [
        i
        for i, (event, data) in enumerate(all_events)
        if event == CONTENT_BLOCK_END and data.get("block", {}).get("type") == "text"
    ]
    tool_pre_indices = [
        i for i, (event, _) in enumerate(all_events) if event == "tool:pre"
    ]

    # Both event types must exist (content block events fire, tool runs)
    assert len(text_block_end_indices) >= 1, (
        "Expected at least 1 text CONTENT_BLOCK_END event"
    )
    assert len(tool_pre_indices) >= 1, "Expected at least 1 tool:pre event"

    # The first text block end must come before the first tool:pre
    assert text_block_end_indices[0] < tool_pre_indices[0], (
        "Text content block event must fire before tool:pre event"
    )


# ---------------------------------------------------------------------------
# Regression: Pure text responses (no tool calls) still work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pure_text_response_still_works():
    """Pure text responses (no tool calls) must continue to work identically.

    Regression safety net — the P1 fix must not break the no-tool-calls path.
    """

    class PureTextProvider(_BaseProvider):
        async def complete(self, request, **kwargs):
            return ChatResponse(
                content=[TextBlock(text="This is a complete response with no tools.")]
            )

        def parse_tool_calls(self, response):
            return []

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    result = await orchestrator.execute(
        prompt="Hello",
        context=context,
        providers={"default": PureTextProvider()},
        tools={},
        hooks=hooks,
    )

    assert result == "This is a complete response with no tools."


@pytest.mark.asyncio
async def test_empty_text_with_tool_calls():
    """When tool-call response has empty text, return value is final-only.

    Some LLM responses have empty text blocks before tool calls. The return
    value must contain only the final iteration's text — no empty-string
    artifacts from intermediate iterations.
    """
    call_count = 0

    class EmptyTextToolProvider(_BaseProvider):
        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="")],
                    tool_calls=[
                        ToolCall(id="tc1", name="todo", arguments={"action": "list"})
                    ],
                )
            return ChatResponse(content=[TextBlock(text="Final.")])

        def parse_tool_calls(self, response):
            return getattr(response, "tool_calls", None) or []

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    result = await orchestrator.execute(
        prompt="Do something",
        context=context,
        providers={"default": EmptyTextToolProvider()},
        tools={"todo": SimpleTool()},
        hooks=hooks,
    )

    # Return value contains only the final iteration's text
    assert result.strip() == "Final."


# ---------------------------------------------------------------------------
# Test: Accumulator reset ensures final-iteration-only return value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accumulator_resets_between_iterations():
    """execute() must reset its accumulator when the iteration number changes.

    Defense-in-depth: even if _execute_stream yields tokens from multiple
    iterations, execute() returns only text from the LAST iteration.
    This tests the iteration-boundary reset in execute().
    """
    call_count = 0

    class MultiIterationProvider(_BaseProvider):
        """Three iterations: two tool-call rounds, then a final text response."""

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content=[TextBlock(text="Starting analysis...")],
                    tool_calls=[
                        ToolCall(id="tc1", name="todo", arguments={"action": "list"})
                    ],
                )
            if call_count == 2:
                return ChatResponse(
                    content=[TextBlock(text="Running second check...")],
                    tool_calls=[
                        ToolCall(id="tc2", name="todo", arguments={"action": "list"})
                    ],
                )
            # Third call: final response (no tool calls)
            return ChatResponse(
                content=[TextBlock(text="All checks complete. Everything looks good.")]
            )

        def parse_tool_calls(self, response):
            return getattr(response, "tool_calls", None) or []

    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    result = await orchestrator.execute(
        prompt="Run full analysis",
        context=context,
        providers={"default": MultiIterationProvider()},
        tools={"todo": SimpleTool()},
        hooks=hooks,
    )

    # Return value must be ONLY the final iteration's text
    assert "All checks complete. Everything looks good." in result

    # Intermediate text from earlier iterations must NOT be present
    assert "Starting analysis" not in result, (
        f"Iteration 1 text leaked into return value: {result!r}"
    )
    assert "Running second check" not in result, (
        f"Iteration 2 text leaked into return value: {result!r}"
    )
