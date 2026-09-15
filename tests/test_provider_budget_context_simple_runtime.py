"""Joint request-budget regression with real loop, context-simple, and OpenAI code.

This intentionally has no local path manipulation: repository-only runs skip when
the sibling modules are not installed. The root DTU installs all three local
checkouts, where this is required to execute without a skip.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from amplifier_core import ContextLengthError
from amplifier_core.message_models import ChatRequest, Message

pytest.importorskip("amplifier_module_context_simple")
pytest.importorskip("amplifier_module_provider_openai")

from amplifier_module_context_simple import SimpleContextManager
from amplifier_module_loop_streaming import StreamingOrchestrator
from amplifier_module_provider_openai import OpenAIProvider


class _Cancellation:
    is_cancelled = False
    is_immediate = False
    state = "running"


class _StableReminderHooks:
    """A real provider sees a stable persisted reminder on every assembled view."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, event: str, payload: dict | None = None):
        self.events.append((event, payload or {}))
        if event == "provider:request":
            return SimpleNamespace(
                action="inject_context",
                ephemeral=True,
                context_injection="<system-reminder>REQUIRED-REMINDER</system-reminder>",
                context_injection_role="user",
                append_to_last_tool_result=False,
                data=None,
                reason=None,
            )
        return SimpleNamespace(
            action="continue",
            ephemeral=False,
            context_injection=None,
            context_injection_role="system",
            append_to_last_tool_result=False,
            data=None,
            reason=None,
        )


class _Coordinator:
    def __init__(self, hooks: _StableReminderHooks) -> None:
        self.hooks = hooks
        self.cancellation = _Cancellation()
        self.session_state: dict = {}
        self._capabilities: dict[str, object] = {}

    def register_capability(self, name: str, capability: object) -> None:
        self._capabilities[name] = capability

    def get_capability(self, name: str):
        return self._capabilities.get(name)

    async def process_hook_result(self, result, *_args):
        return result


class _RecordingContext(SimpleContextManager):
    """The actual context-simple algorithm, with only seam-call observation added."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hard_fit_calls: list[bool] = []

    async def get_messages_for_request_retaining(
        self,
        *,
        retain_contents: list[str],
        provider=None,
        token_budget: int | None = None,
        hard_fit: bool = False,
    ) -> list[dict]:
        self.hard_fit_calls.append(hard_fit)
        return await super().get_messages_for_request_retaining(
            retain_contents=retain_contents,
            provider=provider,
            token_budget=token_budget,
            hard_fit=hard_fit,
        )


class _InMemoryResponses:
    """Completed Responses SDK fake whose usage is derived from received params."""

    def __init__(self, hard_fit_calls: list[bool]) -> None:
        self.calls: list[dict] = []
        self.hard_fit_counts_at_dispatch: list[int] = []
        self._hard_fit_calls = hard_fit_calls

    async def create(self, **params):
        serialized = json.dumps(
            params, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
        # Record exactly the SDK payload, and derive usage from that same payload
        # rather than a fixture outcome or a desired compaction size.
        self.calls.append(json.loads(serialized))
        self.hard_fit_counts_at_dispatch.append(self._hard_fit_calls.count(True))
        input_tokens = len(serialized.encode("utf-8"))
        return SimpleNamespace(
            id=f"fake-{len(self.calls)}",
            status="completed",
            model=params["model"],
            output=[
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "accepted"}],
                }
            ],
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=1),
        )


class _InMemoryClient:
    def __init__(self, hard_fit_calls: list[bool]) -> None:
        self.responses = _InMemoryResponses(hard_fit_calls)


def _payload_text(params: dict) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


@pytest.mark.asyncio
async def test_hard_fit_stays_compacted_across_real_openai_dispatches() -> None:
    hooks = _StableReminderHooks()
    coordinator = _Coordinator(hooks)
    context = _RecordingContext(
        # Initial ordinary assembly is deliberately below this context's own
        # threshold. The provider's serialized-payload preflight forces the
        # first fit, exercising the optional hard-fit seam.
        max_tokens=500_000,
        compact_threshold=0.99,
        target_usage=0.50,
        protected_recent=0.10,
        protected_tool_results=1,
        truncate_chars=64,
        compaction_notice_enabled=True,
    )
    coordinator.register_capability(
        "context.request_retention", context.get_messages_for_request_retaining
    )
    client = _InMemoryClient(context.hard_fit_calls)
    provider = OpenAIProvider(
        api_key="test-key",
        client=client,
        coordinator=coordinator,
        config={
            "default_model": "gpt-5-mini",
            "max_output_tokens": 1024,
            "max_retries": 0,
            "use_streaming": False,
        },
    )
    loop = StreamingOrchestrator({})

    bulk = "REMOVED-BULK-MARKER:" + ("x" * 800_000)
    await context.add_message({"role": "assistant", "content": bulk})

    await loop.execute(
        "ORIGINAL-HUMAN", context, {"openai": provider}, {}, hooks, coordinator
    )
    # New growth after the forced rebuild: a complete assistant/tool pair and
    # a current tool result that context-simple's protection floor must keep
    # complete.
    await context.add_message(
        {
            "role": "assistant",
            "content": "Calling the current tool.",
            "tool_calls": [
                {"id": "runtime-tool-1", "name": "current_tool", "arguments": {}}
            ],
        }
    )
    await context.add_message(
        {
            "role": "tool",
            "name": "current_tool",
            "tool_call_id": "runtime-tool-1",
            "content": "CURRENT-PROTECTED-TOOL-RESULT",
        }
    )
    for prompt in (
        "SECOND-HUMAN",
        "POST-FORCE-ORDINARY-ONE",
        "POST-FORCE-ORDINARY-TWO",
        "POST-FORCE-ORDINARY-THREE",
    ):
        await loop.execute(
            prompt, context, {"openai": provider}, {}, hooks, coordinator
        )

    # Every call in this list reached the fake SDK. The first accepted call
    # followed the provider-directed rebuild; the four subsequent calls prove
    # ordinary fetches do not resurrect the canonical bulk history.
    assert len(client.responses.calls) >= 5
    payloads = [_payload_text(params) for params in client.responses.calls]
    assert all("REMOVED-BULK-MARKER" not in payload for payload in payloads)
    first_payload = payloads[0]
    assert "ORIGINAL-HUMAN" in first_payload
    assert "REQUIRED-REMINDER" in first_payload
    assert context.hard_fit_calls[:2] == [False, True]
    assert context.hard_fit_calls.count(True) == 1
    assert client.responses.hard_fit_counts_at_dispatch == [1] * len(payloads)
    post_force_payloads = payloads[1:]
    assert post_force_payloads
    assert all("ORIGINAL-HUMAN" in payload for payload in post_force_payloads)
    assert "SECOND-HUMAN" in post_force_payloads[-1]
    assert "REQUIRED-REMINDER" in post_force_payloads[-1]
    assert "CURRENT-PROTECTED-TOOL-RESULT" in post_force_payloads[-1]
    assert 'source=\\"context-compaction\\"' in post_force_payloads[-1]

    final_input = client.responses.calls[-1]["input"]
    tool_call_index = next(
        index
        for index, item in enumerate(final_input)
        if item.get("type") == "function_call"
        and item.get("call_id") == "runtime-tool-1"
    )
    assert final_input[tool_call_index + 1] == {
        "type": "function_call_output",
        "call_id": "runtime-tool-1",
        "output": "CURRENT-PROTECTED-TOOL-RESULT",
    }

    canonical = await context.get_messages()
    assert any(message.get("content") == bulk for message in canonical)
    assert any(message.get("content") == "ORIGINAL-HUMAN" for message in canonical)
    assert any(message.get("content") == "SECOND-HUMAN" for message in canonical)
    assert any(
        message.get("content") == "CURRENT-PROTECTED-TOOL-RESULT"
        for message in canonical
    )

    # OpenAI's direct final assembled-payload guard remains the final boundary:
    # an impossible protected payload never reaches the SDK fake.
    accepted_before_guard = len(client.responses.calls)
    with pytest.raises(ContextLengthError):
        await provider.complete(
            ChatRequest(
                messages=[Message(role="user", content="IMPOSSIBLE" * 200_000)]
            )
        )
    assert len(client.responses.calls) == accepted_before_guard
