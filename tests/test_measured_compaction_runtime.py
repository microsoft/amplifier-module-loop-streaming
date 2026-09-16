"""Contract coverage for Context's optional measured request-view capability."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockCoordinator,
    MockResponse,
    RequestCapturingProvider,
    ScriptedHookResult,
    ScriptedHooks,
)
from tests.test_provider_budget_guard import BudgetContext


def _provider_count(count: int = 5) -> dict[str, object]:
    return {
        "estimated_input_tokens": count + 4,
        "input_limit_tokens": 100,
        "context_token_budget": 0,
        "measurement": {
            "kind": "provider_count",
            "source": "test.provider.count",
            "input_tokens": count,
        },
    }


class _Transaction:
    def __init__(self) -> None:
        self.committed = 0
        self.rolled_back = 0

    async def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1


class _MeasuredContext(BudgetContext):
    def __init__(self) -> None:
        super().__init__()
        self.measured_calls: list[tuple[object, list[str]]] = []
        self.transaction = _Transaction()

    async def get_measured_request_view(self, *, provider, retain_contents, count_view):
        base_view = list(self._messages)
        attempt = await count_view(base_view)
        self.measured_calls.append((provider, list(retain_contents)))
        decision = attempt["budget_decision"]
        count = decision["measurement"]["input_tokens"]
        return {
            "base_view": base_view,
            "final_attempt": attempt,
            "outcome": "not_needed",
            "measured_before": count,
            "measured_after": count,
            "policy_budget": 100,
            "trigger": 80.0,
            "target": 50,
            "count_calls": 1,
            "transaction": self.transaction,
        }


class _MeasuredProvider(RequestCapturingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.budget_calls: list[object] = []

    def get_info(self):
        return SimpleNamespace(capabilities=["request_budget:provider_count"])

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.budget_calls.append(request)
        return _provider_count()

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        response = MockResponse(text="ok")
        response.usage = SimpleNamespace(input_tokens=17, cache_write_tokens=3)
        return response


@pytest.mark.asyncio
async def test_measured_view_counts_and_dispatches_the_identical_request_once() -> None:
    context = _MeasuredContext()
    provider = _MeasuredProvider()
    coordinator = MockCoordinator()
    recorded: list[tuple[int, int]] = []
    coordinator.register_capability(
        "context.measured_request_view", context.get_measured_request_view
    )
    coordinator.register_capability(
        "context.foreground_usage",
        lambda: lambda *, input_tokens, cache_write_tokens=0: recorded.append(
            (input_tokens, cache_write_tokens)
        ),
    )
    hooks = ScriptedHooks(
        {
            "provider:request": ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="COUNTED-OVERLAY",
            )
        }
    )

    await StreamingOrchestrator(
        {"ephemeral_injection_mode": "tail", "reminder_placement": "tail"}
    ).execute("current request", context, {"main": provider}, {}, hooks, coordinator)

    assert len(context.measured_calls) == 1
    assert context.request_calls == []
    assert provider.budget_calls == [provider.requests[0]]
    assert "COUNTED-OVERLAY" in "\n".join(
        message.content for message in provider.requests[0].messages
    )
    assert context.transaction.committed == 1
    assert context.transaction.rolled_back == 0
    assert recorded == [(17, 3)]