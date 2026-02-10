"""Tests for orchestrator rate limiting feature."""

import time
from unittest.mock import AsyncMock

import pytest


class TestRateLimitingConfig:
    """Test rate limiting configuration."""

    def test_default_disabled(self):
        """Rate limiting should be disabled by default."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator({})
        assert orchestrator.min_delay_between_calls_ms == 0

    def test_config_enabled(self):
        """Rate limiting can be enabled via config."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator({"min_delay_between_calls_ms": 500})
        assert orchestrator.min_delay_between_calls_ms == 500

    def test_tracking_initialized(self):
        """Last provider call timestamp should start as None."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator({})
        assert orchestrator._last_provider_call_end is None


@pytest.mark.asyncio
class TestRateLimitDelay:
    """Test the _apply_rate_limit_delay method."""

    async def test_no_delay_when_disabled(self):
        """No delay should be applied when rate limiting is disabled."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator({"min_delay_between_calls_ms": 0})
        hooks = AsyncMock()

        start = time.monotonic()
        await orchestrator._apply_rate_limit_delay(hooks, 1)
        elapsed = time.monotonic() - start

        assert elapsed < 0.01  # Should be nearly instant
        hooks.emit.assert_not_called()

    async def test_no_delay_on_first_call(self):
        """No delay on first call (no previous timestamp)."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator({"min_delay_between_calls_ms": 1000})
        hooks = AsyncMock()

        start = time.monotonic()
        await orchestrator._apply_rate_limit_delay(hooks, 1)
        elapsed = time.monotonic() - start

        assert elapsed < 0.01  # Should be nearly instant
        hooks.emit.assert_not_called()

    async def test_delay_applied_when_needed(self):
        """Delay should be applied when elapsed < configured."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator({"min_delay_between_calls_ms": 100})
        orchestrator._last_provider_call_end = time.monotonic()  # Just now
        hooks = AsyncMock()

        start = time.monotonic()
        await orchestrator._apply_rate_limit_delay(hooks, 2)
        elapsed = (time.monotonic() - start) * 1000  # Convert to ms

        # Should have delayed close to 100ms
        assert elapsed >= 90  # Allow some tolerance
        assert elapsed < 150
        hooks.emit.assert_called_once()

        # Verify event payload
        call_args = hooks.emit.call_args
        assert call_args[0][0] == "orchestrator:rate_limit_delay"
        assert "delay_ms" in call_args[0][1]
        assert call_args[0][1]["iteration"] == 2

    async def test_no_delay_if_enough_time_elapsed(self):
        """No delay if enough time has already passed."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator({"min_delay_between_calls_ms": 50})
        orchestrator._last_provider_call_end = time.monotonic() - 0.1  # 100ms ago
        hooks = AsyncMock()

        start = time.monotonic()
        await orchestrator._apply_rate_limit_delay(hooks, 2)
        elapsed = time.monotonic() - start

        assert elapsed < 0.01  # Should be nearly instant
        hooks.emit.assert_not_called()
