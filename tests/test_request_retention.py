"""Retention-aware request assembly and resume-deduplication coverage."""

from __future__ import annotations

import logging

import pytest
from amplifier_core import ContextLengthError

from amplifier_module_loop_streaming import StreamingOrchestrator, _wrap_reminders
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


class RetainingContext(MockContext):
    """A context whose request view compacts persisted messages not retained."""

    def __init__(self, *, fail_retention: bool = False) -> None:
        super().__init__()
        self.requirements: list[list[str]] = []
        self.fail_retention = fail_retention

    async def get_messages(self) -> list[dict]:
        return list(self._messages)

    async def retaining_view(self, *, provider=None, retain_contents: list[str]) -> list[dict]:
        self.requirements.append(list(retain_contents))
        if self.fail_retention:
            raise ContextLengthError("retention cannot fit")
        return [
            dict(message)
            if not (message.get("metadata") or {}).get("persisted")
            or message["content"] in retain_contents
            else {**message, "content": "[compacted]"}
            for message in self._messages
        ]


def injection(body: str) -> ScriptedHookResult:
    return ScriptedHookResult(
        action="inject_context", ephemeral=True, context_injection=body
    )


def coordinator_for(context: RetainingContext) -> MockCoordinator:
    coordinator = MockCoordinator()
    coordinator.register_capability("context.request_retention", context.retaining_view)
    return coordinator


def reminder_contents(request) -> list[str]:
    return [
        message.content
        for message in request.messages
        if isinstance(message.content, str)
        and message.content.startswith("<system-reminders>")
    ]


def persisted(context: RetainingContext) -> list[dict]:
    return [
        message
        for message in context.add_message_calls
        if (message.get("metadata") or {}).get("persisted") is True
    ]


@pytest.mark.asyncio
async def test_changed_content_is_retained_and_sent_as_the_current_envelope() -> None:
    old_body = "<system-reminder>OLD</system-reminder>"
    new_body = "<system-reminder>NEW</system-reminder>"
    old = _wrap_reminders(old_body, tail=False)
    new = _wrap_reminders(new_body, tail=True, header=False)
    context = RetainingContext()
    provider = NRoundToolProvider(n_tool_rounds=1)
    hooks = SequencedHooks({"provider:request": [injection(old_body), injection(new_body)]})

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        hooks,
        coordinator_for(context),
    )

    assert context.requirements == [[old], [new]]
    assert [reminder_contents(request) for request in provider.requests] == [[old], [new]]
    assert "OLD" not in "\n".join(message.content for message in provider.requests[-1].messages)


@pytest.mark.asyncio
async def test_unchanged_content_reuses_its_admitted_envelope_each_request() -> None:
    body = "<system-reminder>STABLE</system-reminder>"
    admitted = _wrap_reminders(body, tail=False)
    context = RetainingContext()
    provider = NRoundToolProvider(n_tool_rounds=2)
    hooks = ScriptedHooks({"provider:request": injection(body)})

    await StreamingOrchestrator({}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        hooks,
        coordinator_for(context),
    )

    assert context.requirements == [[admitted], [admitted], [admitted]]
    assert [reminder_contents(request) for request in provider.requests] == [
        [admitted],
        [admitted],
        [admitted],
    ]
    assert [message["content"] for message in persisted(context)] == [admitted]


@pytest.mark.asyncio
async def test_pending_tool_injection_drains_into_the_retained_request() -> None:
    provider_body = "<system-reminder>PROVIDER</system-reminder>"
    tool_body = "<system-reminder>TOOL</system-reminder>"
    provider_envelope = _wrap_reminders(provider_body, tail=False)
    tool_envelope = _wrap_reminders(tool_body, tail=True, header=False)
    context = RetainingContext()
    provider = NRoundToolProvider(n_tool_rounds=1)
    hooks = ScriptedHooks(
        {
            "provider:request": injection(provider_body),
            "tool:post": injection(tool_body),
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

    assert context.requirements == [
        [provider_envelope],
        [provider_envelope],
        [provider_envelope, tool_envelope],
    ]
    assert reminder_contents(provider.requests[-1]) == [provider_envelope, tool_envelope]


@pytest.mark.asyncio
async def test_finalization_retains_current_hook_and_pending_tool_injection() -> None:
    initial_body = "<system-reminder>INITIAL</system-reminder>"
    final_body = "<system-reminder>FINAL</system-reminder>"
    tool_body = "<system-reminder>TOOL</system-reminder>"
    initial = _wrap_reminders(initial_body, tail=False)
    final = _wrap_reminders(final_body, tail=True, header=False)
    tool = _wrap_reminders(tool_body, tail=True, header=False)
    context = RetainingContext()
    provider = NRoundToolProvider(n_tool_rounds=1)
    hooks = SequencedHooks(
        {
            "provider:request": [injection(initial_body), injection(final_body)],
            "tool:post": [injection(tool_body)],
        }
    )

    await StreamingOrchestrator({"max_iterations": 1}).execute(
        "work",
        context,
        {"main": provider},
        {"mock_tool": OneShotTool()},
        hooks,
        coordinator_for(context),
    )

    assert len(provider.requests) == 2
    assert context.requirements == [[initial], [final, tool]]
    final_request_reminders = reminder_contents(provider.requests[-1])
    assert final_request_reminders[:2] == [final, tool]
    assert 'source="orchestrator-loop-limit"' in final_request_reminders[-1]


@pytest.mark.asyncio
async def test_new_orchestrator_resumes_an_admitted_reminder_without_writing_again() -> None:
    body = "<system-reminder>RESUME</system-reminder>"
    admitted = _wrap_reminders(body, tail=False)
    context = RetainingContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks({"provider:request": injection(body)})
    coordinator = coordinator_for(context)

    await StreamingOrchestrator({}).execute(
        "first", context, {"main": provider}, {}, hooks, coordinator
    )
    await StreamingOrchestrator({}).execute(
        "second", context, {"main": provider}, {}, hooks, coordinator
    )

    assert [message["content"] for message in persisted(context)] == [admitted]
    assert context.requirements == [[admitted], [admitted]]
    assert reminder_contents(provider.requests[-1]) == [admitted]


@pytest.mark.asyncio
async def test_context_reset_readmits_but_a_human_xml_quote_does_not_deduplicate() -> None:
    body = "<system-reminder>QUOTE</system-reminder>"
    admitted = _wrap_reminders(body, tail=False)
    context = RetainingContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks({"provider:request": injection(body)})
    loop = StreamingOrchestrator({})
    coordinator = coordinator_for(context)

    await loop.execute("first", context, {"main": provider}, {}, hooks, coordinator)
    context._messages.clear()
    await loop.execute("second", context, {"main": provider}, {}, hooks, coordinator)
    assert [message["content"] for message in persisted(context)] == [admitted, admitted]

    quoted_context = RetainingContext()
    quoted_context._messages.append({"role": "user", "content": admitted})
    await StreamingOrchestrator({}).execute(
        "quoted", quoted_context, {"main": RequestCapturingProvider()}, {}, hooks,
        coordinator_for(quoted_context),
    )
    assert [message["content"] for message in persisted(quoted_context)] == [admitted]
    assert sum(message["content"] == admitted for message in quoted_context._messages) == 2


@pytest.mark.asyncio
async def test_missing_retention_capability_warns_once_and_tail_remains_unmodified(
    caplog,
) -> None:
    body = "<system-reminder>LEGACY</system-reminder>"
    context = MockContext()
    provider = RequestCapturingProvider()
    with caplog.at_level(logging.WARNING):
        await StreamingOrchestrator({}).execute(
            "work",
            context,
            {"main": provider},
            {},
            ScriptedHooks({"provider:request": injection(body)}),
            MockCoordinator(),
        )
    warnings = [
        record
        for record in caplog.records
        if "retention guarantee unavailable" in record.message
    ]
    assert len(warnings) == 1
    assert "legacy context request fallback" in warnings[0].message

    tail_context = RetainingContext()
    tail_provider = RequestCapturingProvider()
    await StreamingOrchestrator({"ephemeral_injection_mode": "tail"}).execute(
        "work",
        tail_context,
        {"main": tail_provider},
        {},
        ScriptedHooks({"provider:request": injection(body)}),
        coordinator_for(tail_context),
    )
    assert tail_context.requirements == []
    assert persisted(tail_context) == []
    assert reminder_contents(tail_provider.requests[0]) == [_wrap_reminders(body, tail=False)]


@pytest.mark.asyncio
async def test_context_length_during_retention_makes_no_provider_call() -> None:
    context = RetainingContext(fail_retention=True)
    provider = RequestCapturingProvider()

    with pytest.raises(ContextLengthError, match="retention cannot fit"):
        await StreamingOrchestrator({}).execute(
            "work",
            context,
            {"main": provider},
            {},
            ScriptedHooks({"provider:request": injection("<system-reminder>FIT</system-reminder>")}),
            coordinator_for(context),
        )

    assert context.requirements
    assert provider.requests == []