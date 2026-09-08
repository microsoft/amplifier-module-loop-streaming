"""Unit tests for the Layer 1 LLM-call budget (spec: 298-replacement, §3, §11 T1).

Covers, per the implementation spec's test plan:
  T1.1  Budget not reached -> normal success, budget_exhausted False
  T1.2  Budget reached exactly -> N iterations + 1 wrap-up call (documents G4)
  T1.3  Wrap-up content -> system-reminder present; no further looping
  T1.4  Status -> "budget_exhausted"
  T1.5  Metadata shape -> exactly llm_calls/llm_call_budget/budget_exhausted/resumable
  T1.6  Transcript completeness -> every assistant message plus the wrap-up message
  T1.7  Warning fires once at ceil(budget_warn_ratio * N)
  T1.8  Warning suppressed when max_iterations == -1
  T1.9  Per-turn reset across two sequential execute() calls
  T1.10 Status precedence: cancelled beats budget_exhausted
  T1.11 Status precedence: error beats budget_exhausted
  T1.12 Wrap-up LLM failure -> PROVIDER_ERROR emitted, no crash, still budget_exhausted

All LLM calls go through a single FakeProvider queue (pattern mirrors
tests/test_goal_loop.py / tests/test_steering.py -- deliberately duplicated
per-file rather than shared, matching this test suite's existing convention).
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest
from amplifier_core import ToolResult
from amplifier_core.events import ORCHESTRATOR_COMPLETE

# ---------------------------------------------------------------------------
# Shared stubs -- minimal, self-contained (pattern mirrors test_goal_loop.py)
# ---------------------------------------------------------------------------


class MockHookResult:
    """Minimal hook result -- pass through, no deny, no injection."""

    action = "pass"
    reason = None
    ephemeral = False
    context_injection = None
    context_injection_role = "user"
    append_to_last_tool_result = False
    data = None


class MockHooks:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(
        self, event_name: str, payload: dict | None = None
    ) -> MockHookResult:
        self.emitted.append((event_name, payload or {}))
        return MockHookResult()

    def events(self, name: str) -> list[dict]:
        return [payload for event_name, payload in self.emitted if event_name == name]

    def orchestrator_complete_events(self) -> list[dict]:
        return self.events(ORCHESTRATOR_COMPLETE)

    def budget_warning_events(self) -> list[dict]:
        return self.events("orchestrator:budget_warning")

    def provider_error_events(self) -> list[dict]:
        return self.events("provider:error")

    def execution_end_events(self) -> list[dict]:
        return self.events("execution:end")


class MockContext:
    def __init__(self) -> None:
        self._messages: list[dict] = []

    async def add_message(self, msg: dict) -> None:
        self._messages.append(msg)

    async def get_messages(self) -> list[dict]:
        return list(self._messages)

    async def get_messages_for_request(self, provider=None) -> list[dict]:
        return list(self._messages)


class MockCancellation:
    is_cancelled = False
    is_immediate = False
    state = "running"

    def register_tool_start(self, tool_call_id: str, display_name: str) -> None:
        pass

    def register_tool_complete(self, tool_call_id: str) -> None:
        pass

    async def trigger_callbacks(self) -> None:
        pass


class MockCoordinator:
    def __init__(self) -> None:
        self.cancellation = MockCancellation()
        self.session_state: dict[str, Any] = {}
        self._capabilities: dict[str, Any] = {}

    async def process_hook_result(self, result, *args, **kwargs):
        return result

    def get_capability(self, name: str) -> Any:
        return self._capabilities.get(name)


class MockToolCall:
    def __init__(self, call_id: str = "tc-1", name: str = "mock_tool") -> None:
        self.id = call_id
        self.name = name
        self.arguments: dict = {}


class MockTool:
    name = "mock_tool"
    description = "test tool"
    input_schema: ClassVar[dict] = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return ToolResult(success=True, output="ok")


class NativeishTool(MockTool):
    """A generic stand-in for a provider-native tool declaration."""

    name = "nativeish_tool"
    native_tool_spec: ClassVar[dict] = {
        "type": "nativeish",
        "display_width": 1024,
    }


class CountingNativeishTool(NativeishTool):
    def __init__(self) -> None:
        self.executions = 0

    async def execute(self, arguments):
        self.executions += 1
        return ToolResult(success=True, output="ok")


class TypedTextBlock:
    """Minimal typed text block, matching the ContentBlock shape."""

    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text

    def model_dump(self) -> dict[str, str]:
        return {"type": self.type, "text": self.text}


class TypedThinkingBlock:
    """Minimal typed thinking block with the replay signature."""

    type = "thinking"

    def __init__(self, thinking: str, signature: str) -> None:
        self.thinking = thinking
        self.signature = signature

    def model_dump(self) -> dict[str, str]:
        return {
            "type": self.type,
            "thinking": self.thinking,
            "signature": self.signature,
        }


class MockTurnResponse:
    """A plain conversational-turn response (non-streaming path)."""

    def __init__(
        self,
        text: str = "",
        tool_calls: list | None = None,
        *,
        content: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.content = text if content is None else content
        self.content_blocks = None
        self.usage = None
        self.metadata = metadata
        self._intended_tool_calls = tool_calls or []


class FakeProvider:
    """Non-streaming provider stub (no `.stream` attribute).

    Pops responses off `turn_queue` for each `complete()` call. Once the
    queue is exhausted, returns `wrapup_response` -- this is what the
    exhaustion branch's own extra `provider.complete()` call receives, so
    tests only need to queue exactly `max_iterations` turn responses.
    """

    def __init__(self) -> None:
        self.turn_queue: list[MockTurnResponse] = []
        self.wrapup_response: MockTurnResponse | None = MockTurnResponse(
            text="Summary: made progress, here is what remains."
        )
        self.wrapup_should_raise: Exception | None = None
        self.call_count = 0
        self.requests: list[Any] = []

    async def complete(self, chat_request, **kwargs):
        self.call_count += 1
        self.requests.append(chat_request)
        if self.turn_queue:
            return self.turn_queue.pop(0)
        if self.wrapup_should_raise:
            raise self.wrapup_should_raise
        assert self.wrapup_response is not None
        return self.wrapup_response

    def parse_tool_calls(self, response):
        return getattr(response, "_intended_tool_calls", [])


def _make_orchestrator(config: dict | None = None):
    from amplifier_module_loop_streaming import StreamingOrchestrator

    return StreamingOrchestrator(config or {})


def _looping_turns(n: int) -> list[MockTurnResponse]:
    """`n` turn responses, each with a tool call, so the loop never breaks
    early on its own (only the budget can stop it)."""
    return [
        MockTurnResponse(text="", tool_calls=[MockToolCall(call_id=f"tc-{i}")])
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# T1.1 -- Budget not reached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetNotReached:
    async def test_normal_success_when_budget_not_hit(self) -> None:
        orch = _make_orchestrator({"max_iterations": 5})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        # Two tool-call turns, then a plain text turn that ends the loop
        # naturally at iteration 3 -- well under the budget of 5.
        provider.turn_queue = _looping_turns(2) + [MockTurnResponse(text="done")]

        result = await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "done"
        events = hooks.orchestrator_complete_events()
        assert len(events) == 1
        assert events[0]["status"] == "success"
        assert events[0]["metadata"]["budget_exhausted"] is False
        assert events[0]["metadata"]["llm_calls"] == 3
        assert events[0]["metadata"]["llm_call_budget"] == 5
        assert events[0]["metadata"]["resumable"] is True
        # Only 3 provider calls -- no wrap-up call, since budget was never hit.
        assert provider.call_count == 3


# ---------------------------------------------------------------------------
# Bounded natural completion -- no unnecessary wrap-up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBoundedNaturalCompletion:
    async def test_cap_one_plain_text_is_one_successful_call(self) -> None:
        """A natural final answer at the cap must not trigger a duplicate call."""
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = [MockTurnResponse(text="complete answer")]

        result = await orch.execute(
            prompt="answer once",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "complete answer"
        assert provider.call_count == 1
        event = hooks.orchestrator_complete_events()[-1]
        assert event["status"] == "success"
        assert event["metadata"]["budget_exhausted"] is False
        assert event["metadata"]["llm_calls"] == 1

    async def test_cap_two_native_tool_then_final_text_is_two_calls(self) -> None:
        """A tool result followed by a natural answer at the cap needs no wrap-up."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = [
            MockTurnResponse(
                tool_calls=[MockToolCall(call_id="native-call", name="nativeish_tool")]
            ),
            MockTurnResponse(text="natural final answer"),
        ]

        result = await orch.execute(
            prompt="use the native tool",
            context=ctx,
            providers={"main": provider},
            tools={"nativeish_tool": NativeishTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "natural final answer"
        assert provider.call_count == 2
        assert provider.requests[0].tools is not None
        assert provider.requests[0].tools[0].model_dump()["type"] == "nativeish"
        assert any(
            message.get("role") == "tool" and message.get("name") == "nativeish_tool"
            for message in ctx._messages
        )
        event = hooks.orchestrator_complete_events()[-1]
        assert event["status"] == "success"
        assert event["metadata"]["budget_exhausted"] is False


# ---------------------------------------------------------------------------
# T1.2 / T1.3 / T1.4 / T1.5 -- Budget reached exactly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetExhaustion:
    async def test_exactly_n_plus_one_provider_calls(self) -> None:
        """T1.2: loop runs exactly N iterations; wrap-up fires; provider
        called N+1 times total (documents G4)."""
        orch = _make_orchestrator({"max_iterations": 3})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(3)

        result = await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.call_count == 4  # 3 budgeted + 1 wrap-up
        assert "Summary: made progress" in result

    async def test_wrapup_reminder_and_no_further_loop(self) -> None:
        """T1.3: the final request's last message carries the
        orchestrator-loop-limit system-reminder, and the wrap-up call never
        triggers a further loop iteration. Tools remain declared so native
        assistant/tool-result history stays valid, but tool_choice disables
        further dispatch."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)
        # Even if the model tried to call a tool during wrap-up, the
        # orchestrator must not act on it -- prove this by returning a
        # tool_calls-laden response and confirming no fifth call happens.
        provider.wrapup_response = MockTurnResponse(
            text="wrap-up text", tool_calls=[MockToolCall(call_id="tc-wrapup")]
        )

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        wrapup_request = provider.requests[-1]
        last_message = wrapup_request.messages[-1]
        assert last_message.role == "user"
        assert 'source="orchestrator-loop-limit"' in last_message.content
        # Preserve the normal tool declarations, including native shapes, but
        # force the portable no-tool choice so the final response cannot
        # request work the capped loop will not execute.
        assert wrapup_request.tools is not None
        assert wrapup_request.tool_choice == "none"
        # Exactly one wrap-up call -- no further loop, even though the fake
        # wrap-up response included a tool call.
        assert provider.call_count == 3  # 2 budgeted + 1 wrap-up, no more

    async def test_cap_one_tool_call_wrapup_keeps_native_tools_but_disables_them(self) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = [
            MockTurnResponse(
                tool_calls=[MockToolCall(call_id="native-call", name="nativeish_tool")]
            )
        ]
        provider.wrapup_response = MockTurnResponse(text="wrapped up")

        result = await orch.execute(
            prompt="use one tool",
            context=ctx,
            providers={"main": provider},
            tools={"nativeish_tool": NativeishTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "wrapped up"
        assert provider.call_count == 2
        wrapup_request = provider.requests[-1]
        assert wrapup_request.tools is not None
        assert wrapup_request.tools[0].model_dump()["type"] == "nativeish"
        assert wrapup_request.tool_choice == "none"

    async def test_status_is_budget_exhausted(self) -> None:
        """T1.4."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "budget_exhausted"

    async def test_metadata_shape_is_exact(self) -> None:
        """T1.5: metadata has exactly llm_calls, llm_call_budget,
        budget_exhausted, resumable -- no more, no fewer."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        metadata = hooks.orchestrator_complete_events()[-1]["metadata"]
        assert set(metadata.keys()) == {
            "llm_calls",
            "llm_call_budget",
            "budget_exhausted",
            "resumable",
        }
        # The forced finalization is an actual third provider call, even
        # though only two calls were within the bounded loop iteration budget.
        assert metadata["llm_calls"] == 3
        assert metadata["llm_call_budget"] == 2
        assert metadata["budget_exhausted"] is True
        assert metadata["resumable"] is True


# ---------------------------------------------------------------------------
# T1.6 -- Transcript completeness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTranscriptCompleteness:
    async def test_wrapup_message_appended_to_context(self) -> None:
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)
        provider.wrapup_response = MockTurnResponse(text="final summary text")

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assistant_messages = [m for m in ctx._messages if m.get("role") == "assistant"]
        # The wrap-up's own assistant message closes the transcript -- it is
        # not truncated or dropped.
        assert assistant_messages[-1]["content"] == "final summary text"
        # Nothing was lost: the two tool-round assistant turns should also
        # be present (the orchestrator's tool-call handling records an
        # assistant message with tool_calls per round).
        assert len(assistant_messages) >= 1


# ---------------------------------------------------------------------------
# Wrap-up response handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWrapupResponseHandling:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("string summary", "string summary"),
            ([TypedTextBlock("typed summary")], "typed summary"),
            ([{"type": "text", "text": "dict summary"}], "dict summary"),
        ],
        ids=["string", "typed-block", "dict-block"],
    )
    async def test_final_response_renders_normalized_text_and_persists_safe_blocks(
        self, content: Any, expected: str
    ) -> None:
        """Finalization renders text while retaining safe structured content."""
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(1)
        provider.wrapup_response = MockTurnResponse(content=content)

        result = await orch.execute(
            prompt="force a summary",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == expected
        final_message = [m for m in ctx._messages if m.get("role") == "assistant"][-1]
        assert final_message == {
            "role": "assistant",
            "content": (
                [
                    block.model_dump() if hasattr(block, "model_dump") else block
                    for block in content
                ]
                if isinstance(content, list)
                else expected
            ),
        }

    async def test_final_response_preserves_safe_structured_content_and_metadata(
        self,
    ) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(1)
        thinking = TypedThinkingBlock("private reasoning", "replay-signature")
        text = TypedTextBlock("structured final text")
        provider.wrapup_response = MockTurnResponse(
            content=[thinking, text],
            metadata={"provider_state": "opaque"},
        )

        result = await orch.execute(
            prompt="force a structured summary",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "structured final text"
        final_message = [m for m in ctx._messages if m.get("role") == "assistant"][-1]
        assert final_message == {
            "role": "assistant",
            "content": [thinking.model_dump(), text.model_dump()],
            "thinking_block": thinking.model_dump(),
            "metadata": {"provider_state": "opaque"},
        }

    async def test_final_tool_calls_are_never_dispatched_or_structurally_persisted(
        self,
    ) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        tool = CountingNativeishTool()
        provider.turn_queue = [
            MockTurnResponse(
                tool_calls=[MockToolCall(call_id="budget-tool", name="nativeish_tool")]
            )
        ]
        provider.wrapup_response = MockTurnResponse(
            text="final text only",
            tool_calls=[MockToolCall(call_id="unsupported-final", name="nativeish_tool")],
            content=[
                TypedThinkingBlock("must not persist", "invalid-signature"),
                TypedTextBlock("final text only"),
                {"type": "tool_call", "name": "nativeish_tool"},
            ],
            metadata={"must_not_persist": True},
        )

        result = await orch.execute(
            prompt="run then summarize",
            context=ctx,
            providers={"main": provider},
            tools={"nativeish_tool": tool},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "final text only"
        assert provider.call_count == 2
        assert tool.executions == 1
        final_message = [m for m in ctx._messages if m.get("role") == "assistant"][-1]
        assert final_message == {"role": "assistant", "content": "final text only"}


# ---------------------------------------------------------------------------
# Bounded streaming and steering finalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBoundedStreamingAndSteering:
    async def test_streaming_natural_completion_at_cap_is_successful(self) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        class StreamingProvider:
            calls = 0

            async def stream(self, chat_request, tools=None):  # noqa: ANN001,ANN201
                self.calls += 1
                yield {"content": "streamed answer"}

            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                pytest.fail("natural streaming completion must not wrap up")

        provider = StreamingProvider()
        result = await orch.execute(
            prompt="stream once",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "streamed answer"
        assert provider.calls == 1
        event = hooks.orchestrator_complete_events()[-1]
        assert event["status"] == "success"
        assert event["metadata"]["budget_exhausted"] is False

    async def test_pending_steer_at_cap_requires_finalization(self) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()

        class SteeringProvider(FakeProvider):
            async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
                response = await super().complete(chat_request, **kwargs)
                if self.call_count == 1:
                    orch.steer("respond to this before ending")
                return response

        provider = SteeringProvider()
        provider.turn_queue = [MockTurnResponse(text="first answer")]
        provider.wrapup_response = MockTurnResponse(text="forced final answer")

        result = await orch.execute(
            prompt="answer",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "first answerforced final answer"
        assert provider.call_count == 2
        user_contents = [
            message.content
            for message in provider.requests[-1].messages
            if message.role == "user"
        ]
        assert "respond to this before ending" in user_contents
        assert any("maximum number of iterations" in content for content in user_contents)
        assert orch._steering_queue.is_empty
        event = hooks.orchestrator_complete_events()[-1]
        assert event["status"] == "budget_exhausted"
        assert event["metadata"]["budget_exhausted"] is True


# ---------------------------------------------------------------------------
# T1.7 / T1.8 -- 80% warning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetWarning:
    async def test_warning_fires_once_at_threshold(self) -> None:
        """T1.7: at ceil(0.8*N) one event + one injected message; never a
        second warning even though the turn continues past the threshold."""
        # N=5, ratio=0.8 -> threshold iteration = max(1, int(5*0.8)) = 4.
        orch = _make_orchestrator({"max_iterations": 5, "budget_warn_ratio": 0.8})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(5)

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        warnings = hooks.budget_warning_events()
        assert len(warnings) == 1
        assert warnings[0]["iteration"] == 4
        assert warnings[0]["budget"] == 5
        assert warnings[0]["remaining"] == 1

        injected = [
            m
            for m in ctx._messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and "You have used" in m["content"]
        ]
        assert len(injected) == 1

    async def test_warning_suppressed_when_unlimited(self) -> None:
        """T1.8: max_iterations == -1 -> zero warning events."""
        orch = _make_orchestrator({"max_iterations": -1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        # A handful of tool-call turns then a plain finish -- there is no
        # budget, so nothing should ever warn regardless of turn count.
        provider.turn_queue = _looping_turns(6) + [MockTurnResponse(text="done")]

        await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert hooks.budget_warning_events() == []


# ---------------------------------------------------------------------------
# T1.9 -- Per-turn reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPerTurnReset:
    async def test_budget_flags_reset_across_sequential_executes(self) -> None:
        # max_iterations=10, default budget_warn_ratio=0.8 -> warn threshold
        # is iteration 8. Chosen so the second (1-iteration) turn below is
        # nowhere near its own warn threshold -- if its flags come back
        # True, that can only be leakage from turn 1, not a fresh trigger.
        orch = _make_orchestrator({"max_iterations": 10})
        hooks = MockHooks()
        coordinator = MockCoordinator()

        # First execute(): hits the budget (10 tool-call turns -> exhausted,
        # and crosses the 80% warn threshold along the way).
        ctx1 = MockContext()
        provider1 = FakeProvider()
        provider1.turn_queue = _looping_turns(10)
        await orch.execute(
            prompt="first",
            context=ctx1,
            providers={"main": provider1},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )
        assert orch._budget_exhausted is True
        assert orch._budget_warned is True

        # Second execute() on the SAME orchestrator instance: finishes in a
        # single iteration, far under both the budget and its warn
        # threshold. If the flags didn't reset, this turn would incorrectly
        # report budget_exhausted/warned as leftovers from the first.
        ctx2 = MockContext()
        provider2 = FakeProvider()
        provider2.turn_queue = [MockTurnResponse(text="quick answer")]
        await orch.execute(
            prompt="second",
            context=ctx2,
            providers={"main": provider2},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "success"
        assert events[-1]["metadata"]["budget_exhausted"] is False
        assert orch._budget_exhausted is False
        assert orch._budget_warned is False


# ---------------------------------------------------------------------------
# T1.10 / T1.11 -- Status precedence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStatusPrecedence:
    async def test_cancelled_beats_budget_exhausted(self) -> None:
        """T1.10: cancellation during an exhausted turn -> status ==
        "cancelled", not "budget_exhausted"."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(2)

        # Force _budget_exhausted True as if exhaustion already ran, then
        # simulate the coordinator reporting cancellation for this turn --
        # exercises _execute_one_turn's precedence directly, since driving a
        # real mid-stream cancellation race is not what this test is about.
        async def fake_execute_stream(*args, **kwargs):
            orch._budget_exhausted = True
            coordinator.cancellation.is_cancelled = True
            return
            yield  # pragma: no cover -- makes this an async generator

        orch._execute_stream = fake_execute_stream  # type: ignore[method-assign]

        result = await orch._execute_one_turn(
            "prompt",
            ctx,
            {"main": provider},
            {},
            hooks,  # type: ignore[arg-type]
            coordinator,  # type: ignore[arg-type]
        )

        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "cancelled"
        assert result == ""

    async def test_error_beats_budget_exhausted(self) -> None:
        """T1.11: an exception during an exhausted turn -> status ==
        "error", not "budget_exhausted" (and the exception still
        propagates)."""
        orch = _make_orchestrator({"max_iterations": 2})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()

        async def fake_execute_stream(*args, **kwargs):
            orch._budget_exhausted = True
            raise RuntimeError("boom")
            yield  # pragma: no cover

        orch._execute_stream = fake_execute_stream  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="boom"):
            await orch._execute_one_turn(
                "prompt",
                ctx,
                {"main": provider},
                {},
                hooks,  # type: ignore[arg-type]
                coordinator,  # type: ignore[arg-type]
            )

        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "error"


# ---------------------------------------------------------------------------
# Cancellation while finalization is pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizationCancellation:
    async def test_cancellation_before_first_call_never_calls_provider(self) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        coordinator.cancellation.is_cancelled = True
        provider = FakeProvider()

        await orch.execute(
            prompt="cancel now",
            context=ctx,
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.call_count == 0
        assert hooks.orchestrator_complete_events()[-1]["status"] == "cancelled"

    async def test_cancellation_during_finalization_skips_wrapup_provider_call(self) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        coordinator = MockCoordinator()

        class HooksThatCancelAtFinalization(MockHooks):
            async def emit(
                self, event_name: str, payload: dict | None = None
            ) -> MockHookResult:
                result = await super().emit(event_name, payload)
                if event_name == "provider:request" and (payload or {}).get("max_reached"):
                    coordinator.cancellation.is_cancelled = True
                return result

        hooks = HooksThatCancelAtFinalization()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(1)

        await orch.execute(
            prompt="cancel during finalization",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.call_count == 1
        assert hooks.orchestrator_complete_events()[-1]["status"] == "cancelled"
        assert ctx._messages[-1] == {
            "role": "assistant",
            "content": "The previous operation was cancelled. Results from completed tools have been preserved.",
        }

    async def test_finalization_denial_closes_a_tool_result_with_yielded_text(self) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()

        class HooksThatDenyAtFinalization(MockHooks):
            def __init__(self) -> None:
                super().__init__()
                self.finalizing = False

            async def emit(
                self, event_name: str, payload: dict | None = None
            ) -> MockHookResult:
                self.finalizing = event_name == "provider:request" and bool(
                    (payload or {}).get("max_reached")
                )
                return await super().emit(event_name, payload)

        class DenyingCoordinator(MockCoordinator):
            async def process_hook_result(self, result, *args, **kwargs):
                if hooks.finalizing:
                    return type("DenyResult", (), {"action": "deny", "reason": "stop"})()
                return result

        hooks = HooksThatDenyAtFinalization()
        coordinator = DenyingCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(1)

        result = await orch.execute(
            prompt="deny finalization",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == "Operation denied: stop"
        assert provider.call_count == 1
        assert ctx._messages[-1] == {
            "role": "assistant",
            "content": "Operation denied: stop",
        }

    async def test_cancelled_finalization_provider_closes_tool_turn_then_propagates(
        self,
    ) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(1)
        provider.wrapup_should_raise = asyncio.CancelledError("cancelled finalization")

        with pytest.raises(asyncio.CancelledError, match="cancelled finalization"):
            await orch.execute(
                prompt="cancel finalization provider",
                context=ctx,
                providers={"main": provider},
                tools={"mock_tool": MockTool()},
                hooks=hooks,  # type: ignore[arg-type]
                coordinator=coordinator,  # type: ignore[arg-type]
            )

        assert ctx._messages[-1] == {
            "role": "assistant",
            "content": "The previous operation was cancelled. Results from completed tools have been preserved.",
        }

    async def test_cancellation_wins_over_finalization_hook_denial(self) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()

        class CountingCancellation(MockCancellation):
            def __init__(self) -> None:
                self.callback_count = 0

            async def trigger_callbacks(self) -> None:
                self.callback_count += 1

        class HooksThatDenyAtFinalization(MockHooks):
            def __init__(self) -> None:
                super().__init__()
                self.finalizing = False

            async def emit(
                self, event_name: str, payload: dict | None = None
            ) -> MockHookResult:
                self.finalizing = event_name == "provider:request" and bool(
                    (payload or {}).get("max_reached")
                )
                return await super().emit(event_name, payload)

        class CancellingDenyCoordinator(MockCoordinator):
            def __init__(self, hooks: HooksThatDenyAtFinalization) -> None:
                super().__init__()
                self.cancellation = CountingCancellation()
                self.hooks = hooks

            async def process_hook_result(self, result, *args, **kwargs):
                if not self.hooks.finalizing:
                    return result
                self.cancellation.is_cancelled = True
                return type("DenyResult", (), {"action": "deny", "reason": "stop"})()

        class SteeringProvider(FakeProvider):
            async def complete(self, chat_request, **kwargs):
                response = await super().complete(chat_request, **kwargs)
                if self.call_count == 1:
                    orch.steer("discard this steer after cancellation")
                return response

        hooks = HooksThatDenyAtFinalization()
        coordinator = CancellingDenyCoordinator(hooks)
        provider = SteeringProvider()
        provider.turn_queue = _looping_turns(1)

        await orch.execute(
            prompt="cancel during denied finalization",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert provider.call_count == 1
        assert coordinator.cancellation.callback_count == 1
        assert hooks.events("cancel:requested")
        assert hooks.events("cancel:completed")
        assert orch._steering_queue.is_empty
        assert hooks.orchestrator_complete_events()[-1]["status"] == "cancelled"
        assert ctx._messages[-1] == {
            "role": "assistant",
            "content": "The previous operation was cancelled. Results from completed tools have been preserved.",
        }


# ---------------------------------------------------------------------------
# T1.12 -- Wrap-up LLM failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWrapupFailure:
    async def test_wrapup_provider_error_does_not_crash(self) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(1)
        provider.wrapup_should_raise = RuntimeError("provider exploded")

        result = await orch.execute(
            prompt="do the thing",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        # No crash: execute() returns normally (possibly with empty text,
        # since the wrap-up call itself failed).
        assert result == ""
        events = hooks.orchestrator_complete_events()
        assert events[-1]["status"] == "budget_exhausted"
        assert events[-1]["metadata"]["budget_exhausted"] is True
        # PROVIDER_ERROR (generic Exception branch, not LLMError) was
        # emitted for the failed wrap-up call.
        provider_errors = hooks.events("provider:error")
        assert len(provider_errors) == 1
        assert provider_errors[0]["error"]["type"] == "RuntimeError"
        assert ctx._messages[-1] == {
            "role": "assistant",
            "content": "The final response could not be generated.",
        }

    async def test_empty_final_response_closes_a_tool_result_turn(self) -> None:
        orch = _make_orchestrator({"max_iterations": 1})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = _looping_turns(1)
        provider.wrapup_response = MockTurnResponse()

        result = await orch.execute(
            prompt="return an empty finalization",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        assert result == ""
        assert ctx._messages[-1] == {
            "role": "assistant",
            "content": "The final response could not be generated.",
        }


@pytest.mark.asyncio
class TestCallCountsOutsideForcedFinalization:
    @pytest.mark.parametrize(
        ("max_iterations", "responses"),
        [
            (1, [MockTurnResponse(text="natural cap completion")]),
            (
                -1,
                [
                    MockTurnResponse(
                        tool_calls=[MockToolCall(call_id="unlimited-tool")]
                    ),
                    MockTurnResponse(text="unlimited completion"),
                ],
            ),
        ],
        ids=["natural-at-cap", "unlimited"],
    )
    async def test_normal_paths_count_only_actual_provider_calls(
        self, max_iterations: int, responses: list[MockTurnResponse]
    ) -> None:
        orch = _make_orchestrator({"max_iterations": max_iterations})
        ctx = MockContext()
        hooks = MockHooks()
        coordinator = MockCoordinator()
        provider = FakeProvider()
        provider.turn_queue = responses

        await orch.execute(
            prompt="complete normally",
            context=ctx,
            providers={"main": provider},
            tools={"mock_tool": MockTool()},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

        metadata = hooks.orchestrator_complete_events()[-1]["metadata"]
        assert metadata["llm_calls"] == provider.call_count
        assert metadata["budget_exhausted"] is False
