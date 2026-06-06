"""Tests for exponential backoff retry logic on retryable provider errors."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRetryConfig:
    """Test retry configuration defaults and overrides."""

    def test_default_config(self):
        """Retry should be enabled by default with sensible defaults."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator({})
        assert orchestrator.retry_max_attempts == 3
        assert orchestrator.retry_base_delay_seconds == 1.0
        assert orchestrator.retry_max_delay_seconds == 30.0

    def test_custom_config(self):
        """Retry parameters can be overridden via config."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator(
            {
                "retry_max_attempts": 5,
                "retry_base_delay_seconds": 0.5,
                "retry_max_delay_seconds": 60.0,
            }
        )
        assert orchestrator.retry_max_attempts == 5
        assert orchestrator.retry_base_delay_seconds == 0.5
        assert orchestrator.retry_max_delay_seconds == 60.0

    def test_retry_disabled(self):
        """Setting retry_max_attempts to 0 effectively disables retry."""
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orchestrator = StreamingOrchestrator({"retry_max_attempts": 0})
        assert orchestrator.retry_max_attempts == 0


@pytest.mark.asyncio
class TestCallProviderWithRetry:
    """Test the _call_provider_with_retry method."""

    def _make_orchestrator(self, **config_overrides):
        from amplifier_module_loop_streaming import StreamingOrchestrator

        config = {
            "retry_max_attempts": 3,
            "retry_base_delay_seconds": 0.01,  # Fast for tests
            "retry_max_delay_seconds": 0.1,
        }
        config.update(config_overrides)
        return StreamingOrchestrator(config)

    def _make_llm_error(self, retryable=True, retry_after=None, status_code=429):
        """Create a mock LLMError with the right attributes."""
        from amplifier_core.llm_errors import LLMError

        error = LLMError("Rate limit exceeded")
        error.retryable = retryable
        error.status_code = status_code
        if retry_after is not None:
            error.retry_after = retry_after
        return error

    async def test_success_on_first_attempt(self):
        """Should return result immediately when call succeeds."""
        orchestrator = self._make_orchestrator()
        hooks = AsyncMock()

        async def success_fn():
            return "response"

        result = await orchestrator._call_provider_with_retry(
            success_fn, hooks, "test-provider"
        )

        assert result == "response"
        hooks.emit.assert_not_called()

    async def test_retry_on_retryable_error_then_succeed(self):
        """Should retry on retryable error and return result on success."""
        orchestrator = self._make_orchestrator()
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=True)
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise error
            return "recovered"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await orchestrator._call_provider_with_retry(
                fail_then_succeed, hooks, "test-provider"
            )

        assert result == "recovered"
        assert call_count == 3
        # Should have slept twice (after attempt 1 and 2)
        assert mock_sleep.call_count == 2

    async def test_raises_after_max_retries_exhausted(self):
        """Should raise after all retry attempts are exhausted."""
        orchestrator = self._make_orchestrator(retry_max_attempts=2)
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=True)

        async def always_fail():
            raise error

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(type(error)):
                await orchestrator._call_provider_with_retry(
                    always_fail, hooks, "test-provider"
                )

    async def test_no_retry_on_non_retryable_error(self):
        """Should not retry when error has retryable=False."""
        orchestrator = self._make_orchestrator()
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=False, status_code=400)
        call_count = 0

        async def fail_non_retryable():
            nonlocal call_count
            call_count += 1
            raise error

        with pytest.raises(type(error)):
            await orchestrator._call_provider_with_retry(
                fail_non_retryable, hooks, "test-provider"
            )

        assert call_count == 1  # No retry

    async def test_no_retry_on_non_llm_exception(self):
        """Should not retry non-LLM exceptions (e.g., ValueError)."""
        orchestrator = self._make_orchestrator()
        hooks = AsyncMock()

        call_count = 0

        async def fail_with_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await orchestrator._call_provider_with_retry(
                fail_with_value_error, hooks, "test-provider"
            )

        assert call_count == 1  # No retry

    async def test_exponential_backoff_delays(self):
        """Should use exponential backoff: base * 2^attempt."""
        orchestrator = self._make_orchestrator(
            retry_max_attempts=3,
            retry_base_delay_seconds=1.0,
            retry_max_delay_seconds=30.0,
        )
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=True)

        async def always_fail():
            raise error

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(type(error)):
                await orchestrator._call_provider_with_retry(
                    always_fail, hooks, "test-provider"
                )

        # 4 attempts total (initial + 3 retries), 3 sleeps
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]  # 1*2^0, 1*2^1, 1*2^2

    async def test_delay_capped_at_max(self):
        """Delay should not exceed retry_max_delay_seconds."""
        orchestrator = self._make_orchestrator(
            retry_max_attempts=5,
            retry_base_delay_seconds=10.0,
            retry_max_delay_seconds=25.0,
        )
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=True)

        async def always_fail():
            raise error

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(type(error)):
                await orchestrator._call_provider_with_retry(
                    always_fail, hooks, "test-provider"
                )

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # 10*2^0=10, 10*2^1=20, 10*2^2=40→25(capped), 10*2^3=80→25, 10*2^4=160→25
        assert delays == [10.0, 20.0, 25.0, 25.0, 25.0]

    async def test_honors_retry_after_from_error(self):
        """Should use retry_after from the error when present."""
        orchestrator = self._make_orchestrator(
            retry_max_attempts=2,
            retry_base_delay_seconds=1.0,
        )
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=True, retry_after=5.0)

        async def always_fail():
            raise error

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(type(error)):
                await orchestrator._call_provider_with_retry(
                    always_fail, hooks, "test-provider"
                )

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # Should use retry_after (5.0) instead of exponential backoff
        assert all(d == 5.0 for d in delays)

    async def test_emits_retry_event_on_each_retry(self):
        """Should emit provider:retry event for each retry attempt."""
        orchestrator = self._make_orchestrator(retry_max_attempts=2)
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=True, status_code=429)
        call_count = 0

        async def fail_twice_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise error
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await orchestrator._call_provider_with_retry(
                fail_twice_then_succeed, hooks, "test-provider"
            )

        assert result == "ok"

        # Should have emitted 2 retry events (no error event since it succeeded)
        retry_calls = [
            call
            for call in hooks.emit.call_args_list
            if call.args[0] == "provider:retry"
        ]
        assert len(retry_calls) == 2

        # Verify retry event payload
        payload = retry_calls[0].args[1]
        assert payload["provider"] == "test-provider"
        assert payload["attempt"] == 1
        assert payload["max_retries"] == 2
        assert payload["retryable"] is True
        assert payload["status_code"] == 429
        assert "delay_seconds" in payload
        assert "error" in payload

    async def test_emits_provider_error_on_final_failure(self):
        """Should emit PROVIDER_ERROR when all retries are exhausted."""
        orchestrator = self._make_orchestrator(retry_max_attempts=1)
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=True)

        async def always_fail():
            raise error

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(type(error)):
                await orchestrator._call_provider_with_retry(
                    always_fail, hooks, "test-provider"
                )

        # Should have emitted: 1 retry event + 1 provider:error event
        event_names = [call.args[0] for call in hooks.emit.call_args_list]
        assert "provider:retry" in event_names
        assert "provider:error" in event_names

    async def test_emits_provider_error_on_non_retryable(self):
        """Should emit PROVIDER_ERROR immediately for non-retryable errors."""
        orchestrator = self._make_orchestrator()
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=False, status_code=400)

        async def fail_non_retryable():
            raise error

        with pytest.raises(type(error)):
            await orchestrator._call_provider_with_retry(
                fail_non_retryable, hooks, "test-provider"
            )

        # Only provider:error, no retry events
        event_names = [call.args[0] for call in hooks.emit.call_args_list]
        assert event_names == ["provider:error"]

    async def test_handles_sync_callable(self):
        """Should handle sync callables (e.g., provider.stream())."""
        orchestrator = self._make_orchestrator()
        hooks = AsyncMock()

        def sync_fn():
            return "stream_iterator"

        result = await orchestrator._call_provider_with_retry(
            sync_fn, hooks, "test-provider"
        )

        assert result == "stream_iterator"

    async def test_retries_sync_callable_on_error(self):
        """Should retry sync callables that raise retryable errors."""
        orchestrator = self._make_orchestrator(retry_max_attempts=2)
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=True)
        call_count = 0

        def sync_fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise error
            return "stream_iterator"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await orchestrator._call_provider_with_retry(
                sync_fail_then_succeed, hooks, "test-provider"
            )

        assert result == "stream_iterator"
        assert call_count == 2

    async def test_zero_retries_means_no_retry(self):
        """With retry_max_attempts=0, errors should propagate immediately."""
        orchestrator = self._make_orchestrator(retry_max_attempts=0)
        hooks = AsyncMock()

        error = self._make_llm_error(retryable=True)
        call_count = 0

        async def fail():
            nonlocal call_count
            call_count += 1
            raise error

        with pytest.raises(type(error)):
            await orchestrator._call_provider_with_retry(
                fail, hooks, "test-provider"
            )

        assert call_count == 1  # Single attempt, no retries

    async def test_none_provider_name_accepted(self):
        """Should accept None as provider_name."""
        orchestrator = self._make_orchestrator()
        hooks = AsyncMock()

        async def success_fn():
            return "response"

        result = await orchestrator._call_provider_with_retry(
            success_fn, hooks, None
        )

        assert result == "response"
