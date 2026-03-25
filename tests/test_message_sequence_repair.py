"""Tests for _repair_message_sequence — fixes orphaned tool_use blocks.

Covers:
- No-op when messages are already correctly ordered
- No-op when there are no tool calls
- Reorders tool_result that is separated from its tool_call by a user message
- Handles multiple tool_calls in one assistant message
- Handles multiple assistant messages with tool_calls
- Handles missing tool_results (synthesises persistent results)
- Handles tool_results for IDs not in any tool_call (passthrough)
"""

from unittest.mock import AsyncMock

import pytest

from amplifier_module_loop_streaming import StreamingOrchestrator


@pytest.fixture
def orchestrator():
    return StreamingOrchestrator({"max_iterations": 5})


@pytest.fixture
def mock_context():
    """Mock context that tracks add_message calls."""
    ctx = AsyncMock()
    ctx._added = []

    async def capture(msg):
        ctx._added.append(msg)

    ctx.add_message = AsyncMock(side_effect=capture)
    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant_msg(text="", tool_calls=None):
    msg = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool_result(tool_call_id, content="ok"):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _user_msg(text="hello"):
    return {"role": "user", "content": text}


def _tool_call(tc_id, name="test_tool"):
    return {"id": tc_id, "tool": name, "arguments": {}}


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noop_when_no_tool_calls(orchestrator, mock_context):
    """Should return original list when no tool_calls exist."""
    msgs = [_user_msg("hi"), _assistant_msg("hello")]
    result = await orchestrator._repair_message_sequence(msgs, mock_context)
    assert result is msgs  # Same object, not a copy


@pytest.mark.asyncio
async def test_noop_when_already_ordered(orchestrator, mock_context):
    """Should return original list when tool_results already follow tool_calls."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg("calling tool", tool_calls=[_tool_call("tc1")]),
        _tool_result("tc1", "result"),
        _assistant_msg("done"),
    ]
    result = await orchestrator._repair_message_sequence(msgs, mock_context)
    assert result is msgs


@pytest.mark.asyncio
async def test_noop_multiple_tool_calls_already_ordered(orchestrator, mock_context):
    """Should return original when multiple tool_calls are all correctly ordered."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg(
            "calling tools",
            tool_calls=[_tool_call("tc1"), _tool_call("tc2")],
        ),
        _tool_result("tc1"),
        _tool_result("tc2"),
        _assistant_msg("done"),
    ]
    result = await orchestrator._repair_message_sequence(msgs, mock_context)
    assert result is msgs


# ---------------------------------------------------------------------------
# Reorder cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_user_msg_between_tool_call_and_result(orchestrator, mock_context):
    """Core bug: user message inserted between tool_call and tool_result."""
    msgs = [
        _user_msg("run the recipe"),
        _assistant_msg("running", tool_calls=[_tool_call("tc1", "recipes")]),
        _user_msg("actually use different params"),  # BREAKS CHAIN
        _tool_result("tc1", "recipe output"),
    ]

    result = await orchestrator._repair_message_sequence(msgs, mock_context)

    # Expected: tool_result moved to immediately after assistant
    assert result[0] == _user_msg("run the recipe")
    assert result[1] == _assistant_msg(
        "running", tool_calls=[_tool_call("tc1", "recipes")]
    )
    assert result[2] == _tool_result("tc1", "recipe output")
    assert result[3] == _user_msg("actually use different params")


@pytest.mark.asyncio
async def test_repair_multiple_interleaved(orchestrator, mock_context):
    """Multiple tool_calls with interleaved user messages and extra assistant."""
    msgs = [
        _user_msg("do stuff"),
        _assistant_msg(
            "doing",
            tool_calls=[_tool_call("tc1", "bash"), _tool_call("tc2", "grep")],
        ),
        _user_msg("wait, also check this"),  # INTERLEAVED
        _assistant_msg("new response"),  # INTERLEAVED
        _tool_result("tc1", "bash output"),
        _tool_result("tc2", "grep output"),
    ]

    result = await orchestrator._repair_message_sequence(msgs, mock_context)

    # tool_results should be moved right after the assistant with tool_calls
    assert result[0] == _user_msg("do stuff")
    assert result[1]["role"] == "assistant"
    assert result[1].get("tool_calls") is not None
    assert result[2] == _tool_result("tc1", "bash output")
    assert result[3] == _tool_result("tc2", "grep output")
    # Interleaved messages come after
    assert result[4] == _user_msg("wait, also check this")
    assert result[5] == _assistant_msg("new response")


@pytest.mark.asyncio
async def test_repair_preserves_non_tool_call_tool_messages(orchestrator, mock_context):
    """Tool results not belonging to any known tool_call should pass through."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg("calling", tool_calls=[_tool_call("tc1")]),
        _user_msg("interruption"),  # BREAKS CHAIN
        _tool_result("tc1", "result"),
        _tool_result("tc_orphan", "some orphan result"),  # Not from any tool_call
    ]

    result = await orchestrator._repair_message_sequence(msgs, mock_context)

    # tc1 result relocated after assistant
    assert result[1]["tool_calls"] is not None
    assert result[2] == _tool_result("tc1", "result")
    # Orphan tool result stays (not removed)
    assert _tool_result("tc_orphan", "some orphan result") in result


# ---------------------------------------------------------------------------
# Missing tool_result synthesis cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesises_missing_tool_result(orchestrator, mock_context):
    """Completely missing tool_results should be synthesised and persisted."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg(
            "calling",
            tool_calls=[_tool_call("tc1"), _tool_call("tc2", "recipes")],
        ),
        _tool_result("tc1", "result for tc1"),
        # tc2 result is completely missing (cancelled tool)
    ]

    result = await orchestrator._repair_message_sequence(msgs, mock_context)

    # tc2 should now have a synthetic result
    tc2_results = [m for m in result if m.get("tool_call_id") == "tc2"]
    assert len(tc2_results) == 1
    assert (
        "interrupted" in tc2_results[0]["content"].lower()
        or "cancelled" in tc2_results[0]["content"].lower()
    )

    # Should have been persisted to context
    assert mock_context.add_message.call_count == 1
    persisted = mock_context.add_message.call_args[0][0]
    assert persisted["tool_call_id"] == "tc2"
    assert persisted["role"] == "tool"


@pytest.mark.asyncio
async def test_synthesised_result_not_duplicated_across_turns(
    orchestrator, mock_context
):
    """Running repair twice should not synthesise the same result twice."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg("calling", tool_calls=[_tool_call("tc1", "recipes")]),
        # tc1 completely missing
    ]

    # Turn 1: synthesises tc1
    await orchestrator._repair_message_sequence(msgs, mock_context)
    assert mock_context.add_message.call_count == 1

    # Turn 2: same messages (context hasn't been re-read yet in this test)
    # The synthetic is now in _synthesized_tool_ids, so it shouldn't persist again
    mock_context.add_message.reset_mock()
    await orchestrator._repair_message_sequence(msgs, mock_context)
    assert mock_context.add_message.call_count == 0  # No new persistence


# ---------------------------------------------------------------------------
# Combined reorder + synthesis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_two_separate_tool_call_chains(orchestrator, mock_context):
    """Two separate assistant messages with tool_calls, both needing repair."""
    msgs = [
        _user_msg("start"),
        _assistant_msg("first call", tool_calls=[_tool_call("tc1")]),
        _user_msg("interruption 1"),
        _tool_result("tc1", "result 1"),
        _assistant_msg("second call", tool_calls=[_tool_call("tc2")]),
        _user_msg("interruption 2"),
        _tool_result("tc2", "result 2"),
    ]

    result = await orchestrator._repair_message_sequence(msgs, mock_context)

    # Both chains should be repaired — find the assistant messages
    assistant_indices = [i for i, m in enumerate(result) if m.get("tool_calls")]
    assert len(assistant_indices) == 2

    # Chain 1: assistant with tc1 followed immediately by tc1 result
    idx1 = assistant_indices[0]
    assert result[idx1 + 1] == _tool_result("tc1", "result 1")

    # Chain 2: assistant with tc2 followed immediately by tc2 result
    idx2 = assistant_indices[1]
    assert result[idx2 + 1] == _tool_result("tc2", "result 2")


@pytest.mark.asyncio
async def test_repair_idempotent(orchestrator, mock_context):
    """Running repair twice should produce the same result."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg("calling", tool_calls=[_tool_call("tc1")]),
        _user_msg("interruption"),
        _tool_result("tc1", "result"),
    ]

    result1 = await orchestrator._repair_message_sequence(msgs, mock_context)
    result2 = await orchestrator._repair_message_sequence(result1, mock_context)

    assert result1 == result2


@pytest.mark.asyncio
async def test_repair_real_world_recipe_scenario(orchestrator, mock_context):
    """Reproduces the exact scenario from the user's failed sessions.

    User runs a recipe (4-min execution), submits a new message 18 seconds in,
    assistant responds with a new tool_call, then both tool_results arrive late.
    """
    msgs = [
        _user_msg("run the daily review recipe"),
        _assistant_msg(
            "Running the recipe now",
            tool_calls=[_tool_call("toolu_AAA", "recipes")],
        ),
        # User interrupts 18 seconds in
        _user_msg("i want to use the recipe but do it for 3/24, not 3/25"),
        # Assistant responds to the new user message with a new recipe call
        _assistant_msg(
            "Running with the updated date",
            tool_calls=[_tool_call("toolu_BBB", "recipes")],
        ),
        # First recipe completes (4 min later)
        _tool_result("toolu_AAA", "daily review output"),
        # Second recipe completes
        _tool_result("toolu_BBB", "daily review for 3/24"),
    ]

    result = await orchestrator._repair_message_sequence(msgs, mock_context)

    # Verify correct ordering:
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert "toolu_AAA" in str(result[1]["tool_calls"])
    assert result[2]["role"] == "tool"
    assert result[2]["tool_call_id"] == "toolu_AAA"
    assert result[3]["role"] == "user"
    assert "3/24" in result[3]["content"]
    assert result[4]["role"] == "assistant"
    assert "toolu_BBB" in str(result[4]["tool_calls"])
    assert result[5]["role"] == "tool"
    assert result[5]["tool_call_id"] == "toolu_BBB"


@pytest.mark.asyncio
async def test_repair_cancelled_recipe_scenario(orchestrator, mock_context):
    """Reproduces the exact failure the user reported: recipe cancelled mid-execution.

    1. User starts recipe
    2. User cancels (Ctrl+C) during step 2
    3. Tool_result never written to context
    4. User sends new message
    5. Turn 1 should synthesise + reorder
    6. Turn 2 should succeed without 400 error
    """
    msgs = [
        _user_msg("run the daily review"),
        _assistant_msg(
            "running recipe",
            tool_calls=[_tool_call("toolu_CANCELLED", "recipes")],
        ),
        # No tool_result — recipe was cancelled
        _user_msg("ok i'm trying my test now"),
    ]

    # Turn 1: should synthesise missing result and reorder
    result = await orchestrator._repair_message_sequence(msgs, mock_context)

    # Synthetic result should exist and be placed after the assistant
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "tool"
    assert result[2]["tool_call_id"] == "toolu_CANCELLED"
    assert (
        "interrupted" in result[2]["content"].lower()
        or "cancelled" in result[2]["content"].lower()
    )
    assert result[3]["role"] == "user"
    assert result[3]["content"] == "ok i'm trying my test now"

    # Should have been persisted to context
    assert mock_context.add_message.call_count == 1

    # Turn 2: simulate context now including the persisted synthetic
    msgs_turn2 = [
        _user_msg("run the daily review"),
        _assistant_msg(
            "running recipe",
            tool_calls=[_tool_call("toolu_CANCELLED", "recipes")],
        ),
        result[2],  # The synthetic result is now in context
        _user_msg("ok i'm trying my test now"),
        _assistant_msg("Got it, take your time!"),
        _user_msg("ok thanks did my test"),
    ]

    mock_context.add_message.reset_mock()
    result2 = await orchestrator._repair_message_sequence(msgs_turn2, mock_context)

    # Should be correctly ordered already — no new synthesis needed
    assert result2 is msgs_turn2  # No reordering needed
    assert mock_context.add_message.call_count == 0  # No new synthesis
