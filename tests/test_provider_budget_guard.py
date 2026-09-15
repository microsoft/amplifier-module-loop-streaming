"""Focused coverage for the optional provider request-budget preflight."""

from __future__ import annotations

import pytest
from amplifier_core import ContextLengthError

from amplifier_module_loop_streaming import (
    StreamingOrchestrator,
    _replay_request_overlays,
    _wrap_reminders,
    mount,
)
from tests.test_ephemeral_cache_persist_mode import (
    MockContext,
    MockCoordinator,
    NRoundToolProvider,
    OneShotTool,
    RequestCapturingProvider,
    ScriptedHookResult,
    ScriptedHooks,
)


def _decision(estimated: int, limit: int, target: int) -> dict[str, int]:
    return {
        "estimated_input_tokens": estimated,
        "input_limit_tokens": limit,
        "context_token_budget": target,
    }


class BudgetProvider(RequestCapturingProvider):
    def __init__(self, decisions: list[dict[str, int]]) -> None:
        super().__init__()
        self.decisions = list(decisions)
        self.budget_calls: list[tuple[object, int]] = []

    def request_budget(self, request, *, context_estimate: int) -> dict[str, int]:
        self.budget_calls.append((request, context_estimate))
        return self.decisions.pop(0)


class BudgetContext(MockContext):
    """Context double that records ordinary and retention-budget requests."""

    def __init__(self) -> None:
        super().__init__()
        self.request_calls: list[tuple[list[str], int | None]] = []
        self.legacy_calls: list[int | None] = []

    async def get_messages(self) -> list[dict]:
        return list(self._messages)

    async def get_messages_for_request(
        self, provider=None, token_budget: int | None = None
    ) -> list[dict]:
        self.legacy_calls.append(token_budget)
        if token_budget is None:
            return list(self._messages)
        return [
            message for message in self._messages if message.get("role") != "assistant"
        ]

    async def retaining_view(
        self, *, provider=None, retain_contents: list[str], token_budget: int | None = None
    ) -> list[dict]:
        self.request_calls.append((list(retain_contents), token_budget))
        if token_budget is None:
            return list(self._messages)
        return [
            message
            for message in self._messages
            if message.get("role") != "assistant"
            or message.get("content") in retain_contents
        ]


class HardFitBudgetContext(BudgetContext):
    """Modern retention capability that records the optional hard-fit signal."""

    def __init__(self) -> None:
        super().__init__()
        self.hard_fit_calls: list[bool] = []

    async def retaining_view(
        self,
        *,
        provider=None,
        retain_contents: list[str],
        token_budget: int | None = None,
        hard_fit: bool = False,
    ) -> list[dict]:
        self.hard_fit_calls.append(hard_fit)
        return await super().retaining_view(
            provider=provider,
            retain_contents=retain_contents,
            token_budget=token_budget,
        )


class KwargsBudgetContext(BudgetContext):
    """Modern retention capability accepting future keywords through ``**kwargs``."""

    def __init__(self) -> None:
        super().__init__()
        self.hard_fit_calls: list[bool] = []

    async def retaining_view(self, **kwargs) -> list[dict]:
        self.hard_fit_calls.append(kwargs.get("hard_fit", False))
        return await super().retaining_view(
            provider=kwargs["provider"],
            retain_contents=kwargs["retain_contents"],
            token_budget=kwargs.get("token_budget"),
        )


class PositionalOnlyHardFitBudgetContext(BudgetContext):
    """Legacy retention callable whose similarly named parameter is positional-only."""

    def __init__(self) -> None:
        super().__init__()
        self.hard_fit_values: list[bool] = []

    async def retaining_view(
        self,
        hard_fit: bool = False,
        /,
        *,
        provider=None,
        retain_contents: list[str],
        token_budget: int | None = None,
    ) -> list[dict]:
        self.hard_fit_values.append(hard_fit)
        return await super().retaining_view(
            provider=provider,
            retain_contents=retain_contents,
            token_budget=token_budget,
        )


class OldSignatureBudgetContext(BudgetContext):
    """Pre-hard-fit retention capability; its call shape is the compatibility check."""

    def __init__(self) -> None:
        super().__init__()
        self.legacy_retention_calls: list[tuple[object, list[str], int | None]] = []

    async def retaining_view(
        self, *, provider=None, retain_contents: list[str], token_budget: int | None = None
    ) -> list[dict]:
        self.legacy_retention_calls.append((provider, list(retain_contents), token_budget))
        return await super().retaining_view(
            provider=provider,
            retain_contents=retain_contents,
            token_budget=token_budget,
        )


def _retaining_coordinator(context: BudgetContext) -> MockCoordinator:
    coordinator = MockCoordinator()
    coordinator.register_capability("context.request_retention", context.retaining_view)
    return coordinator


def _injection(body: str) -> ScriptedHookResult:
    return ScriptedHookResult(
        action="inject_context", ephemeral=True, context_injection=body
    )


@pytest.mark.asyncio
async def test_provider_without_budget_capability_keeps_single_normal_dispatch() -> None:
    context = MockContext()
    provider = RequestCapturingProvider()

    await StreamingOrchestrator({}).execute(
        "work", context, {"main": provider}, {}, ScriptedHooks({}), MockCoordinator()
    )

    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_provider_without_budget_capability_keeps_retained_injection() -> None:
    context = BudgetContext()
    provider = RequestCapturingProvider()
    body = "<system-reminder>LEGACY</system-reminder>"
    hooks = ScriptedHooks({"provider:request": _injection(body)})

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        hooks,
        _retaining_coordinator(context),
    )

    assert body in "\n".join(message.content for message in provider.requests[0].messages)
    assert len(provider.requests) == 1
    assert len(context.request_calls) == 1
    assert context.request_calls[0][1] is None
    assert [name for name, _ in hooks.emitted].count("provider:request") == 1


@pytest.mark.asyncio
async def test_tail_mode_without_budget_capability_stays_view_only() -> None:
    context = BudgetContext()
    provider = RequestCapturingProvider()
    body = "<system-reminder>TAIL-LEGACY</system-reminder>"
    hooks = ScriptedHooks({"provider:request": _injection(body)})

    await StreamingOrchestrator(
        {"ephemeral_injection_mode": "tail", "reminder_placement": "tail"}
    ).execute(
        "work",
        context,
        {"main": provider},
        {},
        hooks,
        _retaining_coordinator(context),
    )

    assert context.legacy_calls == []
    assert context.request_calls == [([], None)]
    assert len(provider.requests) == 1
    assert "\n".join(message.content for message in provider.requests[0].messages).count(body) == 1
    assert [name for name, _ in hooks.emitted].count("provider:request") == 1


@pytest.mark.asyncio
async def test_fitting_budget_dispatches_the_original_request_once() -> None:
    context = BudgetContext()
    provider = BudgetProvider([_decision(5, 5, 0)])

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert len(provider.requests) == 1
    assert len(provider.budget_calls) == 1
    assert context.request_calls == [([], None)]


@pytest.mark.asyncio
async def test_one_smaller_retained_view_is_rechecked_and_dispatches_once() -> None:
    context = BudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7), _decision(9, 10, 0)])
    retained_body = "<system-reminder>REQUIRED</system-reminder>"

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({"provider:request": _injection(retained_body)}),
        _retaining_coordinator(context),
    )

    assert len(provider.requests) == 1
    assert [budget for _, budget in context.request_calls] == [None, 7]
    assert context.request_calls[0][0] == context.request_calls[1][0]
    request_bodies = [message.content for message in provider.requests[0].messages]
    assert retained_body in "\n".join(request_bodies)
    assert "history" not in "\n".join(request_bodies)


@pytest.mark.asyncio
async def test_forced_normal_rebuild_forwards_hard_fit_only_to_modern_retention() -> None:
    context = HardFitBudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7), _decision(9, 10, 0)])

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    # The first ordinary request remains legacy/default behavior; exactly the
    # forced provider-directed rebuild opts into hard fitting.
    assert context.hard_fit_calls == [False, True]


@pytest.mark.asyncio
async def test_forced_rebuild_forwards_hard_fit_to_kwargs_retention() -> None:
    context = KwargsBudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7), _decision(9, 10, 0)])

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert context.hard_fit_calls == [False, True]


@pytest.mark.asyncio
async def test_forced_rebuild_treats_positional_only_hard_fit_as_legacy() -> None:
    context = PositionalOnlyHardFitBudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7), _decision(9, 10, 0)])

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    # Passing hard_fit by keyword would raise for this positional-only callable.
    # Its default on both legacy-shaped calls proves the guard withheld that keyword.
    assert context.hard_fit_values == [False, False]
    assert [budget for _, budget in context.request_calls] == [None, 7]
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_forced_rebuild_preserves_old_retention_call_signature() -> None:
    context = OldSignatureBudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7), _decision(9, 10, 0)])

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert context.legacy_retention_calls == [
        (provider, [], None),
        (provider, [], 7),
    ]


@pytest.mark.asyncio
async def test_forced_rebuild_uses_generic_legacy_context_when_retention_is_absent() -> None:
    context = BudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7), _decision(9, 10, 0)])

    await StreamingOrchestrator({}).execute(
        "work", context, {"main": provider}, {}, ScriptedHooks({}), MockCoordinator()
    )

    assert context.request_calls == []
    assert context.legacy_calls == [None, 7]


@pytest.mark.asyncio
async def test_signature_inspection_failure_keeps_safe_legacy_retention_call(monkeypatch) -> None:
    context = HardFitBudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7), _decision(9, 10, 0)])

    def unavailable_signature(_callable):
        raise ValueError("signature unavailable")

    monkeypatch.setattr(
        "amplifier_module_loop_streaming.inspect.signature", unavailable_signature
    )

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert context.hard_fit_calls == [False, False]


@pytest.mark.asyncio
async def test_retention_type_error_is_not_mistaken_for_a_signature_mismatch() -> None:
    class ExplodingHardFitContext(HardFitBudgetContext):
        async def retaining_view(self, **kwargs) -> list[dict]:
            self.hard_fit_calls.append(kwargs.get("hard_fit", False))
            if kwargs.get("hard_fit"):
                raise TypeError("retention implementation exploded")
            return await BudgetContext.retaining_view(
                self,
                provider=kwargs["provider"],
                retain_contents=kwargs["retain_contents"],
                token_budget=kwargs.get("token_budget"),
            )

    context = ExplodingHardFitContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7)])

    with pytest.raises(TypeError, match="retention implementation exploded"):
        await StreamingOrchestrator({}).execute(
            "work",
            context,
            {"main": provider},
            {},
            ScriptedHooks({}),
            _retaining_coordinator(context),
        )

    assert context.hard_fit_calls == [False, True]
    assert provider.requests == []


class StreamingBudgetProvider(BudgetProvider):
    async def stream(self, request, *, tools):
        self.requests.append(request)
        yield {"content": "streamed"}


@pytest.mark.asyncio
async def test_streaming_dispatch_uses_the_same_budget_rebuild() -> None:
    context = BudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = StreamingBudgetProvider(
        [_decision(100, 10, 7), _decision(9, 10, 0)]
    )

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert len(provider.requests) == 1
    assert len(provider.budget_calls) == 2
    # Ordinary initial view, followed by exactly one explicitly budgeted rebuild.
    assert [budget for _, budget in context.request_calls] == [None, 7]


@pytest.mark.asyncio
async def test_irreducible_budget_makes_no_sdk_call() -> None:
    context = BudgetContext()
    provider = BudgetProvider([_decision(100, 10, 0)])

    with pytest.raises(ContextLengthError, match="cannot retain"):
        await StreamingOrchestrator({}).execute(
            "work",
            context,
            {"main": provider},
            {},
            ScriptedHooks({}),
            _retaining_coordinator(context),
        )

    assert provider.requests == []


@pytest.mark.asyncio
async def test_second_oversize_after_one_rebuild_makes_no_sdk_call() -> None:
    context = BudgetContext()
    provider = BudgetProvider([_decision(100, 10, 7), _decision(50, 10, 1)])

    with pytest.raises(ContextLengthError, match="remains over budget"):
        await StreamingOrchestrator({}).execute(
            "work",
            context,
            {"main": provider},
            {},
            ScriptedHooks({}),
            _retaining_coordinator(context),
        )

    assert len(provider.budget_calls) == 2
    assert [budget for _, budget in context.request_calls] == [None, 7]
    assert context.request_calls[0][0] == context.request_calls[1][0]
    assert provider.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision",
    [
        {"estimated_input_tokens": 1},
        {"estimated_input_tokens": True, "input_limit_tokens": 10, "context_token_budget": 7},
        {"estimated_input_tokens": -1, "input_limit_tokens": 10, "context_token_budget": 7},
        {"estimated_input_tokens": float("nan"), "input_limit_tokens": 10, "context_token_budget": 7},
        {"estimated_input_tokens": "1", "input_limit_tokens": 10, "context_token_budget": 7},
        None,
    ],
)
async def test_malformed_budget_result_fails_before_dispatch(decision) -> None:
    context = BudgetContext()
    provider = BudgetProvider([decision])

    with pytest.raises(ContextLengthError, match="invalid budget decision"):
        await StreamingOrchestrator({}).execute(
            "work",
            context,
            {"main": provider},
            {},
            ScriptedHooks({}),
            _retaining_coordinator(context),
        )

    assert provider.requests == []


@pytest.mark.asyncio
async def test_budget_replay_keeps_tail_overlay_once_without_rerunning_hooks() -> None:
    context = BudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7), _decision(9, 10, 0)])
    body = "<system-reminder>ONCE</system-reminder>"
    hooks = ScriptedHooks({"provider:request": _injection(body)})

    await StreamingOrchestrator(
        {"ephemeral_injection_mode": "tail", "reminder_placement": "tail"}
    ).execute(
        "work",
        context,
        {"main": provider},
        {},
        hooks,
        _retaining_coordinator(context),
    )

    request_bodies = [message.content for message in provider.requests[0].messages]
    assert "\n".join(request_bodies).count(body) == 1
    provider_requests = [name for name, _ in hooks.emitted if name == "provider:request"]
    assert provider_requests == ["provider:request"]
    assert context.legacy_calls == []
    assert [budget for _, budget in context.request_calls] == [None, 7]


@pytest.mark.asyncio
async def test_budget_replay_keeps_pre_user_tail_overlay_in_its_original_position() -> None:
    context = BudgetContext()
    context._messages.append({"role": "assistant", "content": "history" * 200})
    provider = BudgetProvider([_decision(100, 10, 7), _decision(9, 10, 0)])
    body = "<system-reminder>PRE-USER</system-reminder>"
    hooks = ScriptedHooks({"provider:request": _injection(body)})

    await StreamingOrchestrator({"ephemeral_injection_mode": "tail"}).execute(
        "work",
        context,
        {"main": provider},
        {},
        hooks,
        _retaining_coordinator(context),
    )

    request_messages = provider.requests[0].messages
    body_index = next(
        index for index, message in enumerate(request_messages) if body in message.content
    )
    assert request_messages[body_index + 1].content == "work"
    assert sum(body in message.content for message in request_messages) == 1
    assert [name for name, _ in hooks.emitted].count("provider:request") == 1
    assert context.legacy_calls == []
    assert [budget for _, budget in context.request_calls] == [None, 7]


def test_replayed_pending_overlays_keep_tool_adjacency_and_bodies_once() -> None:
    messages = _replay_request_overlays(
        [{"role": "tool", "content": "tool output"}],
        turn_start_view_block=None,
        request_injection=None,
        pending_injections=[
            {"content": "TOOL-REMINDER", "append_to_last_tool_result": True},
            {"content": "MESSAGE-REMINDER", "append_to_last_tool_result": False},
        ],
    )

    assert messages[0]["role"] == "tool"
    assert messages[0]["content"].count("TOOL-REMINDER") == 1
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].count("MESSAGE-REMINDER") == 1


class BudgetToolProvider(NRoundToolProvider):
    def __init__(self, decisions: list[dict[str, int]]) -> None:
        super().__init__(n_tool_rounds=1)
        self.decisions = list(decisions)
        self.budget_calls: list[object] = []

    def request_budget(self, request, *, context_estimate: int) -> dict[str, int]:
        self.budget_calls.append(request)
        return self.decisions.pop(0)


@pytest.mark.asyncio
async def test_pending_tool_overlay_is_replayed_once_after_budget_rebuild() -> None:
    context = BudgetContext()
    provider = BudgetToolProvider(
        [_decision(1, 10, 0), _decision(100, 10, 7), _decision(1, 10, 0)]
    )
    pending = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="PENDING-TOOL",
        append_to_last_tool_result=True,
    )

    loop = StreamingOrchestrator(
        {"ephemeral_injection_mode": "tail", "reminder_placement": "tail"}
    )
    await loop.execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({"tool:post": pending}),
        _retaining_coordinator(context),
    )

    second_request = provider.requests[1]
    tool_messages = [message for message in second_request.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content.count("PENDING-TOOL") == 1
    assert loop._pending_ephemeral_injections == []


class FinalizingBudgetProvider(NRoundToolProvider):
    def __init__(self, decisions: list[dict[str, int]] | None = None) -> None:
        super().__init__(n_tool_rounds=1)
        self.budget_calls: list[object] = []
        self.decisions = decisions or [_decision(1, 10, 0), _decision(1, 10, 0)]

    def request_budget(self, request, *, context_estimate: int) -> dict[str, int]:
        self.budget_calls.append(request)
        return self.decisions.pop(0)


@pytest.mark.asyncio
async def test_finalization_request_is_budget_checked_before_dispatch() -> None:
    context = BudgetContext()
    provider = FinalizingBudgetProvider()

    await StreamingOrchestrator({"max_iterations": 1}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert len(provider.requests) == 2
    assert len(provider.budget_calls) == 2
    assert provider.requests[-1].tool_choice == "none"


@pytest.mark.asyncio
async def test_forced_finalization_rebuild_forwards_hard_fit_only_at_rebuild() -> None:
    context = HardFitBudgetContext()
    provider = FinalizingBudgetProvider(
        [_decision(1, 10, 0), _decision(100, 10, 7), _decision(1, 10, 0)]
    )

    await StreamingOrchestrator({"max_iterations": 1}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({}),
        _retaining_coordinator(context),
    )

    assert context.hard_fit_calls == [False, False, True]


class AnthropicStyleAssemblyProvider(RequestCapturingProvider):
    """Non-budget control: preserve ordinary assembled requests for other providers."""

    name = "anthropic"


@pytest.mark.asyncio
async def test_non_budget_anthropic_style_provider_keeps_ordinary_request_assembly() -> None:
    context = BudgetContext()
    provider = AnthropicStyleAssemblyProvider()
    body = "<system-reminder>ASSEMBLY-CONTROL</system-reminder>"

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {},
        ScriptedHooks({"provider:request": _injection(body)}),
        _retaining_coordinator(context),
    )

    assert len(provider.requests) == 1
    assert context.request_calls == [([_wrap_reminders(body, tail=False)], None)]
    assert body in "\n".join(message.content for message in provider.requests[0].messages)


class ContributorSpy:
    """Mount-level coordinator double retaining real contributor callables."""

    def __init__(self) -> None:
        self.contributors: list[tuple[str, str, object]] = []
        self.capabilities: dict[str, object] = {}

    def register_contributor(self, channel: str, name: str, callback) -> None:
        self.contributors.append((channel, name, callback))

    async def mount(self, _name: str, _module: object) -> None:
        pass

    def register_capability(self, name: str, capability: object) -> None:
        self.capabilities[name] = capability


@pytest.mark.asyncio
async def test_mount_discovers_provider_budget_observability_event() -> None:
    coordinator = ContributorSpy()

    await mount(coordinator, {})

    events_contributor = next(
        callback
        for channel, name, callback in coordinator.contributors
        if (channel, name) == ("observability.events", "loop-streaming")
    )
    events = events_contributor()
    assert events.count("orchestrator:provider_budget") == 1
    assert {
        "execution:start",
        "execution:end",
        "orchestrator:steering_injected",
        "orchestrator:goal_progress",
        "orchestrator:budget_warning",
    }.issubset(events)


@pytest.mark.asyncio
async def test_finalization_irreducible_budget_skips_its_sdk_dispatch() -> None:
    context = BudgetContext()
    provider = FinalizingBudgetProvider(
        [_decision(1, 10, 0), _decision(100, 10, 0)]
    )

    with pytest.raises(ContextLengthError, match="cannot retain"):
        await StreamingOrchestrator({"max_iterations": 1}).execute(
            "work",
            context,
            {"main": provider},
            {"mock_tool": OneShotTool()},
            ScriptedHooks({}),
            _retaining_coordinator(context),
        )

    assert len(provider.requests) == 1
    assert len(provider.budget_calls) == 2


@pytest.mark.asyncio
async def test_finalization_replays_current_and_pending_tail_overlays_once() -> None:
    context = BudgetContext()
    provider = FinalizingBudgetProvider(
        [_decision(1, 10, 0), _decision(100, 10, 7), _decision(1, 10, 0)]
    )
    direct = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="FINAL-DIRECT",
        append_to_last_tool_result=True,
    )
    pending = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="FINAL-PENDING",
        append_to_last_tool_result=True,
    )
    loop = StreamingOrchestrator(
        {
            "max_iterations": 1,
            "ephemeral_injection_mode": "tail",
            "reminder_placement": "tail",
        }
    )

    await loop.execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        ScriptedHooks({"provider:request": direct, "tool:post": pending}),
        _retaining_coordinator(context),
    )

    final_bodies = "\n".join(message.content for message in provider.requests[-1].messages)
    assert final_bodies.count("FINAL-DIRECT") == 1
    assert final_bodies.count("FINAL-PENDING") == 1
    assert loop._pending_ephemeral_injections == []