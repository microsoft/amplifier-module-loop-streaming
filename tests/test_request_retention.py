"""Exercise delivery via an optional retaining context, including fallback."""

import pytest

from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockContext,
    MockCoordinator,
    NRoundToolProvider,
    OneShotTool,
    RequestCapturingProvider,
    ScriptedHookResult,
    ScriptedHooks,
    SequencedHooks,
)


class ReducingContext(MockContext):
    """Models a context which can reduce storage in the outgoing view."""

    def __init__(self):
        super().__init__()
        self.requirements = []

    async def get_messages(self):
        return list(self._messages)

    async def retaining_view(self, *, provider=None, retain_contents):
        self.requirements.append(list(retain_contents))
        assert all(
            any(m["content"] == body for m in self._messages)
            for body in retain_contents
        )
        return [
            dict(m)
            if not (m.get("metadata") or {}).get("persisted")
            or m["content"] in retain_contents
            else {**m, "content": "[compacted]"}
            for m in self._messages
        ]


def injection(body):
    return ScriptedHookResult(
        action="inject_context", ephemeral=True, context_injection=body
    )


def coordinator_for(context):
    coordinator = MockCoordinator()
    coordinator.register_capability("context.request_retention", context.retaining_view)
    return coordinator


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["pre_user", "tail"])
async def test_unchanged_hook_is_retained_each_iteration_without_extra_writes(
    placement,
):
    context = ReducingContext()
    provider = NRoundToolProvider(n_tool_rounds=3)
    hooks = ScriptedHooks({"provider:request": injection("ACTIVE_FACT")})
    loop = StreamingOrchestrator({"reminder_placement": placement})
    await loop.execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        hooks,
        coordinator_for(context),
    )
    assert len(provider.requests) == 4
    assert all(
        any("ACTIVE_FACT" in str(m.content) for m in r.messages)
        for r in provider.requests
    )
    assert (
        len([m for m in context.add_message_calls if "ACTIVE_FACT" in m["content"]])
        == 1
    )
    assert all(len(required) == 1 for required in context.requirements)


@pytest.mark.asyncio
async def test_changed_and_withdrawn_hook_do_not_retain_old_snapshot():
    context = ReducingContext()
    provider = NRoundToolProvider(n_tool_rounds=2)
    hooks = SequencedHooks(
        {
            "provider:request": [
                injection("OLD_FACT"),
                injection("NEW_FACT"),
                ScriptedHookResult(),
            ]
        }
    )
    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        hooks,
        coordinator_for(context),
    )
    assert any("OLD_FACT" in str(m.content) for m in provider.requests[0].messages)
    assert any("NEW_FACT" in str(m.content) for m in provider.requests[1].messages)
    assert not any("OLD_FACT" in str(m.content) for m in provider.requests[1].messages)
    assert context.requirements[-1] == []
    assert not any("NEW_FACT" in str(m.content) for m in provider.requests[2].messages)


@pytest.mark.asyncio
async def test_provider_and_pending_hook_are_both_retained_in_final_request():
    context = ReducingContext()
    provider = NRoundToolProvider(n_tool_rounds=1)
    hooks = ScriptedHooks(
        {
            "provider:request": injection("PROVIDER_FACT"),
            "tool:post": injection("TOOL_FACT"),
        }
    )
    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        hooks,
        coordinator_for(context),
    )
    final = provider.requests[-1]
    assert any("PROVIDER_FACT" in str(m.content) for m in final.messages)
    assert any("TOOL_FACT" in str(m.content) for m in final.messages)
    assert len(context.requirements[-1]) == 2


@pytest.mark.asyncio
async def test_prompt_submit_and_turn_start_hook_are_retained_together():
    context = ReducingContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            "provider:request": injection("PROVIDER_FACT"),
            "prompt:submit": injection("PROMPT_FACT"),
        }
    )
    await StreamingOrchestrator({}).execute(
        "work", context, {"main": provider}, {}, hooks, coordinator_for(context)
    )
    assert any(
        "PROVIDER_FACT" in str(m.content) and "PROMPT_FACT" in str(m.content)
        for m in provider.requests[-1].messages
    )


@pytest.mark.asyncio
async def test_context_replacement_readmits_unchanged_hook():
    context = ReducingContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks({"provider:request": injection("ACTIVE_FACT")})
    loop = StreamingOrchestrator({})
    coordinator = coordinator_for(context)
    await loop.execute("first", context, {"main": provider}, {}, hooks, coordinator)
    context._messages.clear()
    await loop.execute("second", context, {"main": provider}, {}, hooks, coordinator)
    assert any("ACTIVE_FACT" in str(m.content) for m in provider.requests[-1].messages)


@pytest.mark.asyncio
async def test_tail_mode_does_not_require_the_retaining_capability():
    context = ReducingContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks({"provider:request": injection("ACTIVE_FACT")})
    await StreamingOrchestrator({"ephemeral_injection_mode": "tail"}).execute(
        "work", context, {"main": provider}, {}, hooks, coordinator_for(context)
    )
    assert not context.requirements
    assert any("ACTIVE_FACT" in str(m.content) for m in provider.requests[-1].messages)
