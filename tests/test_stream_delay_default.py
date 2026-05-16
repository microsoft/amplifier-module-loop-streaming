"""Tests for the stream_delay default value and its behaviour in _tokenize_stream.

Regression tests for: change default stream_delay from 0.01 to 0.0 so that
headless callers (sub-sessions, automated agents) pay no synthetic per-token
latency by default. Callers that want the typing-animation UX may opt in by
explicitly setting stream_delay: 0.01 (or any positive value) in config.

RED / GREEN notes:
  Run against the *old* default (0.01) to see tests 1 and 3 FAIL.
  Run against the fixed default (0.0)  to see all four tests PASS.
"""

import pytest
from unittest.mock import AsyncMock, patch

from amplifier_module_loop_streaming import StreamingOrchestrator


# ---------------------------------------------------------------------------
# Constructor / config tests
# ---------------------------------------------------------------------------


def test_default_stream_delay_is_zero():
    """Empty config must give stream_delay == 0.0 (not the old 0.01 default)."""
    orch = StreamingOrchestrator({})
    assert orch.stream_delay == 0.0, (
        f"Expected default stream_delay=0.0, got {orch.stream_delay!r}. "
        "The constructor default must be 0.0, not 0.01."
    )


def test_explicit_stream_delay_override_is_respected():
    """Callers who explicitly set stream_delay get their value back unchanged."""
    orch = StreamingOrchestrator({"stream_delay": 0.01})
    assert orch.stream_delay == 0.01, (
        f"Expected stream_delay=0.01 when explicitly configured, "
        f"got {orch.stream_delay!r}."
    )


# ---------------------------------------------------------------------------
# _tokenize_stream behaviour tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tokenize_stream_no_sleep_with_default_config():
    """With default config (stream_delay=0.0), asyncio.sleep must never be called.

    This is the critical path for headless sub-sessions: every non-whitespace
    token used to cost 0.01 s; at 2 000–3 000 tokens that is 20–30 s of pure
    artificial tail latency.
    """
    orch = StreamingOrchestrator({})

    with patch(
        "amplifier_module_loop_streaming.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        tokens = []
        async for token in orch._tokenize_stream("hello world"):
            tokens.append(token)

    mock_sleep.assert_not_called()
    # Tokens must still be yielded correctly despite no sleep
    assert "hello" in tokens
    assert "world" in tokens


@pytest.mark.asyncio
async def test_tokenize_stream_sleep_called_per_nonwhitespace_token_when_delay_set():
    """With stream_delay=0.01, asyncio.sleep(0.01) is called once per non-whitespace token.

    "hello world" has two non-whitespace tokens ("hello", "world") and one
    whitespace token (" "). Only the non-whitespace tokens trigger a sleep.
    """
    orch = StreamingOrchestrator({"stream_delay": 0.01})

    with patch(
        "amplifier_module_loop_streaming.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        tokens = []
        async for token in orch._tokenize_stream("hello world"):
            tokens.append(token)

    # Two non-whitespace tokens → two sleep calls
    assert mock_sleep.call_count == 2, (
        f"Expected 2 asyncio.sleep calls for 2 non-whitespace tokens, "
        f"got {mock_sleep.call_count}."
    )
    mock_sleep.assert_called_with(0.01)
    # Sanity: all three raw tokens were yielded
    assert tokens == ["hello", " ", "world"]
