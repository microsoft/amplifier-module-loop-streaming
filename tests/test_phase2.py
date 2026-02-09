"""Tests for Phase 2 orchestrator touchpoints.

Covers:
- reasoning_effort on ChatRequest from config
- LLMError emits enriched PROVIDER_ERROR with retryable and status_code
- Generic Exception emits basic PROVIDER_ERROR without enrichment
- Errors are re-raised (not swallowed)
- Max-iterations fallback path error events
"""

import pytest

from amplifier_core.events import PROVIDER_ERROR
from amplifier_core.llm_errors import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    ProviderUnavailableError,
)
from amplifier_core.message_models import ChatRequest, ChatResponse, TextBlock
from amplifier_core.testing import EventRecorder, MockContextManager

from amplifier_module_loop_streaming import StreamingOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_response(text="Mock response"):
    """Create a simple text ChatResponse."""
    return ChatResponse(content=[TextBlock(text=text)])


class _BaseProvider:
    """Base mock provider — no `stream` attr so the orchestrator uses complete()."""

    name = "mock"


class CaptureProvider(_BaseProvider):
    """Provider that captures the ChatRequest it receives."""

    def __init__(self):
        self.last_request = None
        self.last_kwargs = None

    async def complete(self, request, **kwargs):
        self.last_request = request
        self.last_kwargs = kwargs
        return _make_text_response()

    def parse_tool_calls(self, response):
        return []


class ErrorProvider(_BaseProvider):
    """Provider that raises a configurable exception on complete()."""

    def __init__(self, error):
        self._error = error

    async def complete(self, request, **kwargs):
        raise self._error

    def parse_tool_calls(self, response):
        return []


def _orch(**overrides):
    config = {"max_iterations": 5, "stream_delay": 0, **overrides}
    return StreamingOrchestrator(config)


# ---------------------------------------------------------------------------
# reasoning_effort tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_effort_high_on_chat_request():
    """Config reasoning_effort='high' flows to ChatRequest."""
    orchestrator = _orch(reasoning_effort="high")
    provider = CaptureProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert provider.last_request is not None
    assert isinstance(provider.last_request, ChatRequest)
    assert provider.last_request.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_reasoning_effort_low_on_chat_request():
    """Config reasoning_effort='low' flows to ChatRequest."""
    orchestrator = _orch(reasoning_effort="low")
    provider = CaptureProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert provider.last_request.reasoning_effort == "low"


@pytest.mark.asyncio
async def test_no_reasoning_effort_defaults_to_none():
    """Without reasoning_effort in config, ChatRequest has None."""
    orchestrator = _orch()
    provider = CaptureProvider()
    context = MockContextManager()
    hooks = EventRecorder()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={},
        hooks=hooks,
    )

    assert provider.last_request.reasoning_effort is None


@pytest.mark.asyncio
async def test_reasoning_effort_on_max_iterations_fallback():
    """reasoning_effort is also set on the max-iterations fallback ChatRequest."""
    orchestrator = _orch(reasoning_effort="high", max_iterations=1)
    context = MockContextManager()
    hooks = EventRecorder()

    call_count = 0

    class ToolCallThenTextProvider(_BaseProvider):
        """First call returns tool_calls, fallback call returns text."""

        def __init__(self):
            self.last_request = None

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from amplifier_core.message_models import ToolCall

                return ChatResponse(
                    content=[TextBlock(text="Calling tool")],
                    tool_calls=[
                        ToolCall(id="tc1", name="test_tool", arguments={"x": 1})
                    ],
                )
            # Second call is the max-iterations fallback
            self.last_request = request
            return _make_text_response("Final response")

        def parse_tool_calls(self, response):
            tc = getattr(response, "tool_calls", None)
            return tc or []

    class SimpleTool:
        name = "test_tool"
        description = "test"
        input_schema = {"type": "object", "properties": {}}

        async def execute(self, args):
            from amplifier_core import ToolResult

            return ToolResult(success=True, output="done")

    provider = ToolCallThenTextProvider()

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": provider},
        tools={"test_tool": SimpleTool()},
        hooks=hooks,
    )

    assert call_count == 2
    assert hasattr(provider, "last_request")
    assert provider.last_request.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# LLMError enriched PROVIDER_ERROR tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_error_enriches_provider_error_event():
    """LLMError populates retryable and status_code on PROVIDER_ERROR event."""
    error = RateLimitError(
        "Rate limit exceeded",
        provider="openai",
        status_code=429,
        retryable=True,
    )
    orchestrator = _orch()
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(RateLimitError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]

    assert data["provider"] == "default"
    assert data["error"]["type"] == "RateLimitError"
    assert "Rate limit exceeded" in data["error"]["msg"]
    assert data["retryable"] is True
    assert data["status_code"] == 429


@pytest.mark.asyncio
async def test_auth_error_not_retryable():
    """AuthenticationError has retryable=False in event data."""
    error = AuthenticationError(
        "Invalid API key",
        provider="anthropic",
        status_code=401,
        retryable=False,
    )
    orchestrator = _orch()
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(AuthenticationError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]

    assert data["retryable"] is False
    assert data["status_code"] == 401
    assert data["error"]["type"] == "AuthenticationError"


@pytest.mark.asyncio
async def test_llm_error_with_none_status_code():
    """LLMError with no status_code still includes the field (as None)."""
    error = LLMError("Unknown error", provider="vllm", retryable=True)
    orchestrator = _orch()
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(LLMError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]

    assert data["retryable"] is True
    assert data["status_code"] is None
    assert data["error"]["type"] == "LLMError"


@pytest.mark.asyncio
async def test_generic_exception_no_retryable_field():
    """Generic Exception produces PROVIDER_ERROR without retryable/status_code."""
    error = RuntimeError("Something broke")
    orchestrator = _orch()
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(RuntimeError):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) == 1
    _, data = error_events[0]

    assert data["provider"] == "default"
    assert data["error"]["type"] == "RuntimeError"
    assert "Something broke" in data["error"]["msg"]
    # Generic exceptions should NOT have retryable or status_code
    assert "retryable" not in data
    assert "status_code" not in data


# ---------------------------------------------------------------------------
# Re-raise tests (errors must propagate, not be swallowed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_error_still_reraises():
    """LLMError is re-raised after event emission (not swallowed)."""
    error = RateLimitError("Rate limit", provider="openai", status_code=429)
    orchestrator = _orch()
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(RateLimitError, match="Rate limit"):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )


@pytest.mark.asyncio
async def test_generic_exception_still_reraises():
    """Generic Exception is re-raised after event emission (not swallowed)."""
    error = ValueError("Bad value")
    orchestrator = _orch()
    provider = ErrorProvider(error)
    context = MockContextManager()
    hooks = EventRecorder()

    with pytest.raises(ValueError, match="Bad value"):
        await orchestrator.execute(
            prompt="Test",
            context=context,
            providers={"default": provider},
            tools={},
            hooks=hooks,
        )


# ---------------------------------------------------------------------------
# Max-iterations fallback error event tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_iterations_fallback_llm_error_event():
    """LLMError in max-iterations fallback path emits enriched PROVIDER_ERROR."""
    orchestrator = _orch(max_iterations=1)
    context = MockContextManager()
    hooks = EventRecorder()

    call_count = 0

    class ToolThenErrorProvider(_BaseProvider):
        """First call returns tool_calls, second call (fallback) raises LLMError."""

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from amplifier_core.message_models import ToolCall

                return ChatResponse(
                    content=[TextBlock(text="Calling tool")],
                    tool_calls=[
                        ToolCall(id="tc1", name="test_tool", arguments={"x": 1})
                    ],
                )
            raise ProviderUnavailableError(
                "Server down",
                provider="gemini",
                status_code=503,
            )

        def parse_tool_calls(self, response):
            tc = getattr(response, "tool_calls", None)
            return tc or []

    class SimpleTool:
        name = "test_tool"
        description = "test"
        input_schema = {"type": "object", "properties": {}}

        async def execute(self, args):
            from amplifier_core import ToolResult

            return ToolResult(success=True, output="done")

    # The fallback path catches exceptions without re-raising, so no pytest.raises
    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": ToolThenErrorProvider()},
        tools={"test_tool": SimpleTool()},
        hooks=hooks,
    )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) >= 1

    # Find the enriched event (with retryable field)
    enriched = [e for e in error_events if "retryable" in e[1]]
    assert len(enriched) == 1
    _, data = enriched[0]

    assert data["retryable"] is True
    assert data["status_code"] == 503
    assert data["error"]["type"] == "ProviderUnavailableError"


@pytest.mark.asyncio
async def test_max_iterations_fallback_generic_error_event():
    """Generic Exception in max-iterations fallback emits basic PROVIDER_ERROR."""
    orchestrator = _orch(max_iterations=1)
    context = MockContextManager()
    hooks = EventRecorder()

    call_count = 0

    class ToolThenGenericErrorProvider(_BaseProvider):
        """First call returns tool_calls, second call raises generic error."""

        async def complete(self, request, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from amplifier_core.message_models import ToolCall

                return ChatResponse(
                    content=[TextBlock(text="Calling tool")],
                    tool_calls=[
                        ToolCall(id="tc1", name="test_tool", arguments={"x": 1})
                    ],
                )
            raise RuntimeError("Unexpected failure")

        def parse_tool_calls(self, response):
            tc = getattr(response, "tool_calls", None)
            return tc or []

    class SimpleTool:
        name = "test_tool"
        description = "test"
        input_schema = {"type": "object", "properties": {}}

        async def execute(self, args):
            from amplifier_core import ToolResult

            return ToolResult(success=True, output="done")

    await orchestrator.execute(
        prompt="Test",
        context=context,
        providers={"default": ToolThenGenericErrorProvider()},
        tools={"test_tool": SimpleTool()},
        hooks=hooks,
    )

    error_events = hooks.get_events(PROVIDER_ERROR)
    assert len(error_events) >= 1

    # Find the generic error event (without retryable field)
    generic = [e for e in error_events if "retryable" not in e[1]]
    assert len(generic) == 1
    _, data = generic[0]

    assert data["error"]["type"] == "RuntimeError"
    assert "retryable" not in data
    assert "status_code" not in data
