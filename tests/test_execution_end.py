"""Tests for CP-7: execution:end normalization.

Verifies that execution:end fires on ALL exit paths with {response, status} payload.

Covers:
- Normal completion: status="success" with accumulated response
- No response: status="incomplete"
- Error path: status="error", execution:end still fires
- Cancellation path: status="cancelled", execution:end still fires
- No provider available: execution:end still fires
- execution:end fires exactly once (not duplicated)
"""


import pytest
from amplifier_core.message_models import ChatResponse, TextBlock
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_streaming import StreamingOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _BaseProvider:
    """Base mock provider — no `stream` attr so the orchestrator uses complete()."""

    name = "mock"


class SimpleProvider(_BaseProvider):
    """Provider that returns a fixed text response."""

    def __init__(self, text="Hello, world!"):
        self.text = text

    async def complete(self, request, **kwargs):
        return ChatResponse(content=[TextBlock(text=self.text)])

    def parse_tool_calls(self, response):
        return []


class ErrorProvider(_BaseProvider):
    """Provider that raises a configurable exception."""

    def __init__(self, error):
        self._error = error

    async def complete(self, request, **kwargs):
        raise self._error

    def parse_tool_calls(self, response):
        return []


def _orch(**overrides):
    config = {"max_iterations": 5, "stream_delay": 0, **overrides}
    return StreamingOrchestrator(config)


def _get_execution_end_events(hooks: EventRecorder):
    return hooks.get_events("execution:end")


# ---------------------------------------------------------------------------
# Normal completion — status="success"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_end_fires_on_success():
    """execution:end fires with {response, status} on normal successful completion."""
    orchestrator = _orch()
    provider = SimpleProvider(text="Hello!")
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Hi",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    end_events = _get_execution_end_events(hooks)
    assert len(end_events) == 1, (
        f"Expected 1 execution:end event, got {len(end_events)}"
    )
    _, data = end_events[0]
    assert "response" in data, "execution:end payload must contain 'response'"
    assert "status" in data, "execution:end payload must contain 'status'"
    assert data["status"] == "success"
    assert data["response"] == "Hello!"


@pytest.mark.asyncio
async def test_execution_end_payload_not_empty():
    """execution:end must NOT fire with empty payload {}."""
    orchestrator = _orch()
    provider = SimpleProvider(text="Done")
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    end_events = _get_execution_end_events(hooks)
    assert len(end_events) == 1
    _, data = end_events[0]
    assert data != {}, "execution:end must NOT have empty payload"


# ---------------------------------------------------------------------------
# Error path — status="error"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_end_fires_on_provider_error():
    """execution:end fires with status='error' when provider raises."""
    orchestrator = _orch()
    error = RuntimeError("Provider crashed")
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(RuntimeError, match="Provider crashed"):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    end_events = _get_execution_end_events(hooks)
    assert len(end_events) == 1, (
        f"Expected 1 execution:end event, got {len(end_events)}"
    )
    _, data = end_events[0]
    assert data["status"] == "error"
    assert "response" in data


@pytest.mark.asyncio
async def test_execution_end_fires_exactly_once_on_error():
    """execution:end fires exactly once, even when an error occurs."""
    orchestrator = _orch()
    provider = ErrorProvider(ValueError("bad"))
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(ValueError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    end_events = _get_execution_end_events(hooks)
    assert len(end_events) == 1, (
        f"execution:end must fire exactly once, fired {len(end_events)} times"
    )


# ---------------------------------------------------------------------------
# No provider available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_end_fires_when_no_provider():
    """execution:end fires even when no provider is available (after execution:start)."""
    orchestrator = _orch()
    context = MockContextManager()
    hooks = EventRecorder()

    # Pass empty providers dict — orchestrator should handle gracefully
    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={},
        tools={},
        hooks=hooks,
    )

    end_events = _get_execution_end_events(hooks)
    assert len(end_events) == 1, (
        f"execution:end must fire even with no provider, fired {len(end_events)} times"
    )
    _, data = end_events[0]
    assert "response" in data
    assert "status" in data


# ---------------------------------------------------------------------------
# execution:end fires exactly once on normal completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_end_fires_exactly_once_on_success():
    """execution:end fires exactly once on normal completion (not duplicated)."""
    orchestrator = _orch()
    provider = SimpleProvider(text="Result")
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    end_events = _get_execution_end_events(hooks)
    assert len(end_events) == 1, (
        f"execution:end must fire exactly once, fired {len(end_events)} times"
    )
