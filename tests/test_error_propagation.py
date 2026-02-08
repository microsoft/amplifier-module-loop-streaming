"""Tests for error propagation fix (Problem B).

Verifies that provider errors propagate as exceptions instead of being
swallowed and converted to response text.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_core import HookRegistry


# Minimal mock provider
class MockProvider:
    @property
    def name(self):
        return "mock"

    @property
    def context_window(self):
        return 100000

    @property
    def max_output_tokens(self):
        return 4096


class ErrorProvider(MockProvider):
    """Provider that raises an error on complete()."""

    def __init__(self, error: Exception):
        self._error = error

    async def complete(self, request, **kwargs):
        raise self._error

    def parse_tool_calls(self, response):
        return []


class SuccessProvider(MockProvider):
    """Provider that returns a successful response."""

    def __init__(self, text="Hello, world!"):
        self._text = text

    async def complete(self, request, **kwargs):
        response = MagicMock()
        response.content = None
        response.content_blocks = None
        response.text = self._text
        response.usage = None
        response.metadata = None
        return response

    def parse_tool_calls(self, response):
        return []


class MockContext:
    """Minimal context mock."""

    def __init__(self):
        self.messages = []

    async def get_messages_for_request(self, provider=None):
        return [{"role": "user", "content": "test"}]

    async def add_message(self, msg):
        self.messages.append(msg)


def create_orchestrator(**overrides):
    """Create a StreamingOrchestrator instance with default config."""
    from amplifier_module_loop_streaming import StreamingOrchestrator

    config = {"max_iterations": 5, "stream_delay": 0, **overrides}
    return StreamingOrchestrator(config=config)


@pytest.mark.asyncio
async def test_provider_timeout_propagates_as_exception():
    """Provider TimeoutError should propagate as an exception, not be swallowed as text."""
    orch = create_orchestrator()
    hooks = HookRegistry()
    context = MockContext()
    error_provider = ErrorProvider(TimeoutError("Request timed out after 600s"))

    with pytest.raises(TimeoutError, match="Request timed out after 600s"):
        await orch.execute(
            prompt="test",
            context=context,
            providers={"mock": error_provider},
            tools={},
            hooks=hooks,
        )


@pytest.mark.asyncio
async def test_provider_runtime_error_propagates():
    """Provider RuntimeError should propagate, not be swallowed."""
    orch = create_orchestrator()
    hooks = HookRegistry()
    context = MockContext()
    error_provider = ErrorProvider(RuntimeError("API connection failed"))

    with pytest.raises(RuntimeError, match="API connection failed"):
        await orch.execute(
            prompt="test",
            context=context,
            providers={"mock": error_provider},
            tools={},
            hooks=hooks,
        )


@pytest.mark.asyncio
async def test_orchestrator_complete_event_has_error_status():
    """orchestrator:complete should fire with status='error' when provider fails."""
    orch = create_orchestrator()
    hooks = HookRegistry()
    context = MockContext()
    error_provider = ErrorProvider(TimeoutError("timeout"))

    events = []

    async def capture_event(event, data):
        events.append((event, data))

    hooks.register("orchestrator:complete", capture_event)

    with pytest.raises(TimeoutError):
        await orch.execute(
            prompt="test",
            context=context,
            providers={"mock": error_provider},
            tools={},
            hooks=hooks,
        )

    # Event should have fired with error status
    complete_events = [(e, d) for e, d in events if e == "orchestrator:complete"]
    assert len(complete_events) == 1, f"Expected 1 orchestrator:complete event, got {len(complete_events)}"
    _, data = complete_events[0]
    assert data["status"] == "error", f"Expected status='error', got status='{data['status']}'"


@pytest.mark.asyncio
async def test_successful_execution_unchanged():
    """Normal successful execution should work exactly as before."""
    orch = create_orchestrator()
    hooks = HookRegistry()
    context = MockContext()
    provider = SuccessProvider("Hello!")

    events = []

    async def capture_event(event, data):
        events.append((event, data))

    hooks.register("orchestrator:complete", capture_event)

    result = await orch.execute(
        prompt="test",
        context=context,
        providers={"mock": provider},
        tools={},
        hooks=hooks,
    )

    assert result == "Hello!"
    complete_events = [(e, d) for e, d in events if e == "orchestrator:complete"]
    assert len(complete_events) == 1
    _, data = complete_events[0]
    assert data["status"] == "success"


@pytest.mark.asyncio
async def test_error_does_not_appear_as_response_text():
    """Errors should NOT be returned as response text (the old broken behavior)."""
    orch = create_orchestrator()
    hooks = HookRegistry()
    context = MockContext()
    error_provider = ErrorProvider(TimeoutError("Request timed out"))

    # The old behavior returned "\nError: ..." as a string.
    # The new behavior raises the exception.
    try:
        result = await orch.execute(
            prompt="test",
            context=context,
            providers={"mock": error_provider},
            tools={},
            hooks=hooks,
        )
        # If we get here, result should NOT contain error text
        assert "Error" not in result, "Error text should not appear in response"
    except TimeoutError:
        # This is the CORRECT behavior - exception propagates
        pass
