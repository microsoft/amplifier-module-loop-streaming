"""Tests for intermediate text visibility when tool calls are present.

Covers:
- Text blocks accompanying tool calls are yielded to the caller (P1 fix)
- Content block events fire for text blocks in tool-call responses
- Pure text responses (no tool calls) still work identically (regression)
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
# Test: Text accompanying tool calls MUST be yielded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_with_tool_calls_is_yielded():
    """When LLM returns [TEXT, TOOL_USE], the text must appear in the response.

    This is the core P1 test. Before the fix, the orchestrator silently drops
    the text — it's stored in context but never yielded to the caller.
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

    # The intermediate text "Let me check that config file." MUST be in the response
    assert "Let me check that config file." in result, (
        f"Intermediate text was not yielded. Full response: {result!r}"
    )
    # The final response text must also be present
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
    """When tool-call response has empty text, no extra tokens should be yielded.

    Some LLM responses have empty text blocks before tool calls. The fix must
    handle this gracefully — no empty string tokens yielded.
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

    # Should contain the final response but no leading garbage from empty text
    assert result.strip() == "Final."
