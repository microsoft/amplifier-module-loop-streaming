"""Regression tests for Bug A: thinking block content leaking into text extraction.

Root cause:
    There are TWO content block model systems:
      - content_models.ThinkingContent  → has .text  (was the bug hazard)
      - message_models.ThinkingBlock    → has .thinking (already safe)

    The old _extract_text_from_content used ``hasattr(block, "text")`` which
    allowed ThinkingContent objects through unchanged, leaking thinking text
    into the response string and ultimately into downstream parse_json calls.

Fix:
    Use an explicit ``block.type == "text"`` guard so only text blocks are
    included, regardless of which model system the block comes from.

RED / GREEN verification (documented here):
    Run against the *unfixed* commit (dc63368) to see test 1 FAIL.
    Run against this commit to see both tests PASS.
"""

import pytest

from amplifier_module_loop_streaming import StreamingOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orch():
    return StreamingOrchestrator({"max_iterations": 5, "stream_delay": 0})


# ---------------------------------------------------------------------------
# Fix 1 regression: _extract_text_from_content
# ---------------------------------------------------------------------------


def test_extract_text_from_content_excludes_thinking_content():
    """content_models.ThinkingContent has .text — must be filtered by type check.

    This is the primary regression test for Bug A.  Before the fix, the
    hasattr(block, "text") guard would include ThinkingContent blocks because
    they *do* have a .text attribute, just with type="thinking".

    RED (before fix):  result contains "reason" — thinking text leaked in.
    GREEN (after fix): result is ONLY the text-block JSON payload.
    """
    from amplifier_core.content_models import TextContent, ThinkingContent

    orchestrator = _orch()

    blocks = [
        ThinkingContent(text='Let me reason... {"tasks": ["t1"]}'),
        TextContent(text='{"tasks": [{"task_id": "t1"}]}'),
    ]

    result = orchestrator._extract_text_from_content(blocks)

    # Thinking text must NOT appear in the extracted string
    assert "reason" not in result, (
        f"Thinking text leaked into extraction result: {result!r}"
    )
    # Only the TextContent payload should be present
    assert result == '{"tasks": [{"task_id": "t1"}]}', (
        f"Unexpected extraction result: {result!r}"
    )


def test_extract_text_from_content_excludes_thinking_block():
    """message_models.ThinkingBlock has .thinking (not .text) — already safe.

    This test documents that the message_models variant was NOT affected by
    the bug (ThinkingBlock has no .text attribute), but it must continue to
    be excluded after the type-check refactor.
    """
    from amplifier_core.message_models import TextBlock, ThinkingBlock

    orchestrator = _orch()

    blocks = [
        ThinkingBlock(thinking="Let me reason..."),
        TextBlock(text='{"tasks": [{"task_id": "t1"}]}'),
    ]

    result = orchestrator._extract_text_from_content(blocks)

    assert "reason" not in result, (
        f"ThinkingBlock content leaked into extraction result: {result!r}"
    )
    assert result == '{"tasks": [{"task_id": "t1"}]}', (
        f"Unexpected extraction result: {result!r}"
    )


def test_extract_text_from_content_plain_string_passthrough():
    """String content is returned unchanged (backward-compat path)."""
    orchestrator = _orch()
    assert orchestrator._extract_text_from_content("hello world") == "hello world"


def test_extract_text_from_content_empty_list():
    """Empty content list returns empty string."""
    orchestrator = _orch()
    assert orchestrator._extract_text_from_content([]) == ""


def test_extract_text_from_content_multiple_text_blocks():
    """Multiple TextContent blocks are joined with double newline."""
    from amplifier_core.content_models import TextContent

    orchestrator = _orch()

    blocks = [
        TextContent(text="first"),
        TextContent(text="second"),
    ]
    result = orchestrator._extract_text_from_content(blocks)
    assert result == "first\n\nsecond"


# ---------------------------------------------------------------------------
# Fix 2 regression: _stream_from_provider chunk filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_from_provider_skips_thinking_chunks():
    """Streaming chunks with block_type='thinking' must not be yielded.

    Providers that stream extended-thinking models emit chunks tagged with
    block_type="thinking".  Without the guard these leak into full_response
    and then into downstream parse_json extraction.
    """
    from amplifier_core.testing import MockContextManager, EventRecorder

    orchestrator = _orch()

    # Minimal mock context and hooks
    context = MockContextManager()
    hooks = EventRecorder()

    # Build an async generator that mimics a provider stream with mixed chunks
    async def _mixed_stream(_request, tools=None):
        yield {"content": "thinking detail", "block_type": "thinking"}
        yield {"content": "real answer", "block_type": "text"}
        yield {"content": "more text"}  # no block_type → pass through

    class StreamProvider:
        name = "mock-stream"

        def stream(self, request, tools=None):
            return _mixed_stream(request, tools=tools)

        def parse_tool_calls(self, response):
            return []

    from amplifier_core.message_models import ChatRequest, Message

    request = ChatRequest(messages=[Message(role="user", content="hi")])

    tokens = []
    async for token in orchestrator._stream_from_provider(
        StreamProvider(),
        request,
        context,
        tools={},
        hooks=hooks,
    ):
        tokens.append(token)

    combined = "".join(tokens)

    assert "thinking detail" not in combined, (
        f"Thinking chunk leaked into stream: {combined!r}"
    )
    assert "real answer" in combined
    assert "more text" in combined
