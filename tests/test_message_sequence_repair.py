"""Tests for _repair_message_sequence — fixes orphaned tool_use blocks.

Covers:
- No-op when messages are already correctly ordered
- No-op when there are no tool calls
- Reorders tool_result that is separated from its tool_call by a user message
- Handles multiple tool_calls in one assistant message
- Handles multiple assistant messages with tool_calls
- Handles missing tool_results (leaves them for provider-level repair)
- Handles tool_results for IDs not in any tool_call (passthrough)
"""

import pytest

from amplifier_module_loop_streaming import StreamingOrchestrator


@pytest.fixture
def orchestrator():
    return StreamingOrchestrator({"max_iterations": 5})


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


def test_noop_when_no_tool_calls(orchestrator):
    """Should return original list when no tool_calls exist."""
    msgs = [_user_msg("hi"), _assistant_msg("hello")]
    result = orchestrator._repair_message_sequence(msgs)
    assert result is msgs  # Same object, not a copy


def test_noop_when_no_tool_results(orchestrator):
    """Should return original list when there are tool_calls but no results at all."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg("thinking", tool_calls=[_tool_call("tc1")]),
    ]
    result = orchestrator._repair_message_sequence(msgs)
    assert result is msgs


def test_noop_when_already_ordered(orchestrator):
    """Should return original list when tool_results already follow tool_calls."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg("calling tool", tool_calls=[_tool_call("tc1")]),
        _tool_result("tc1", "result"),
        _assistant_msg("done"),
    ]
    result = orchestrator._repair_message_sequence(msgs)
    assert result is msgs


def test_noop_multiple_tool_calls_already_ordered(orchestrator):
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
    result = orchestrator._repair_message_sequence(msgs)
    assert result is msgs


# ---------------------------------------------------------------------------
# Repair cases
# ---------------------------------------------------------------------------


def test_repair_user_msg_between_tool_call_and_result(orchestrator):
    """Core bug: user message inserted between tool_call and tool_result."""
    msgs = [
        _user_msg("run the recipe"),
        _assistant_msg("running", tool_calls=[_tool_call("tc1", "recipes")]),
        _user_msg("actually use different params"),  # BREAKS CHAIN
        _tool_result("tc1", "recipe output"),
    ]

    result = orchestrator._repair_message_sequence(msgs)

    # Expected: tool_result moved to immediately after assistant
    assert result[0] == _user_msg("run the recipe")
    assert result[1] == _assistant_msg(
        "running", tool_calls=[_tool_call("tc1", "recipes")]
    )
    assert result[2] == _tool_result("tc1", "recipe output")
    assert result[3] == _user_msg("actually use different params")


def test_repair_multiple_interleaved(orchestrator):
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

    result = orchestrator._repair_message_sequence(msgs)

    # tool_results should be moved right after the assistant with tool_calls
    assert result[0] == _user_msg("do stuff")
    assert result[1]["role"] == "assistant"
    assert result[1].get("tool_calls") is not None
    assert result[2] == _tool_result("tc1", "bash output")
    assert result[3] == _tool_result("tc2", "grep output")
    # Interleaved messages come after
    assert result[4] == _user_msg("wait, also check this")
    assert result[5] == _assistant_msg("new response")


def test_repair_preserves_non_tool_call_tool_messages(orchestrator):
    """Tool results not belonging to any known tool_call should pass through."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg("calling", tool_calls=[_tool_call("tc1")]),
        _user_msg("interruption"),  # BREAKS CHAIN
        _tool_result("tc1", "result"),
        _tool_result("tc_orphan", "some orphan result"),  # Not from any tool_call
    ]

    result = orchestrator._repair_message_sequence(msgs)

    # tc1 result relocated after assistant
    assert result[1]["tool_calls"] is not None
    assert result[2] == _tool_result("tc1", "result")
    # Orphan tool result stays (not removed)
    assert _tool_result("tc_orphan", "some orphan result") in result


def test_repair_missing_tool_result_not_fabricated(orchestrator):
    """Missing tool_results should not be fabricated — leave for provider repair."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg(
            "calling",
            tool_calls=[_tool_call("tc1"), _tool_call("tc2")],
        ),
        _user_msg("interruption"),
        _tool_result("tc1", "result for tc1"),
        # tc2 result is completely missing
    ]

    result = orchestrator._repair_message_sequence(msgs)

    # tc1 result should be relocated, tc2 should NOT be fabricated
    assert result[1].get("tool_calls") is not None
    assert result[2] == _tool_result("tc1", "result for tc1")
    # No synthetic tc2 result
    tc2_results = [m for m in result if m.get("tool_call_id") == "tc2"]
    assert len(tc2_results) == 0


def test_repair_two_separate_tool_call_chains(orchestrator):
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

    result = orchestrator._repair_message_sequence(msgs)

    # Both chains should be repaired — find the assistant messages
    assistant_indices = [i for i, m in enumerate(result) if m.get("tool_calls")]
    assert len(assistant_indices) == 2

    # Chain 1: assistant with tc1 followed immediately by tc1 result
    idx1 = assistant_indices[0]
    assert result[idx1 + 1] == _tool_result("tc1", "result 1")

    # Chain 2: assistant with tc2 followed immediately by tc2 result
    idx2 = assistant_indices[1]
    assert result[idx2 + 1] == _tool_result("tc2", "result 2")


def test_repair_idempotent(orchestrator):
    """Running repair twice should produce the same result."""
    msgs = [
        _user_msg("hi"),
        _assistant_msg("calling", tool_calls=[_tool_call("tc1")]),
        _user_msg("interruption"),
        _tool_result("tc1", "result"),
    ]

    result1 = orchestrator._repair_message_sequence(msgs)
    result2 = orchestrator._repair_message_sequence(result1)

    assert result1 == result2


def test_repair_real_world_recipe_scenario(orchestrator):
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

    result = orchestrator._repair_message_sequence(msgs)

    # Verify correct ordering:
    # 1. user msg
    # 2. assistant with toolu_AAA
    # 3. tool result for toolu_AAA  (relocated)
    # 4. user interruption msg
    # 5. assistant with toolu_BBB
    # 6. tool result for toolu_BBB  (relocated)
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
