"""Tests for the `<system-reminders>` envelope (reminder-redesign-spec.md
sec 2.1 / W1.1).

Background: session `0629f373` showed a model obeying a bare, trailing
`<system-reminder>` injection instead of the user's real request -- the
model treated the last thing it saw as "the task". The envelope wraps every
merged hook-injection blob in `<system-reminders>...</system-reminders>`
with an explicit instruction header telling the model these blocks are NOT
a request and NOT from the user, so it can never again mistake one for the
task.

These tests exercise the pure `_wrap_reminders(body, *, tail)` helper
directly (it is side-effect free by design -- sec 2.1), plus one
integration-level check that the envelope actually reaches the wire via
`StreamingOrchestrator.execute()`.

Covers (per the spec's test-plan sec 10):
  T-W1-01 -- merged blob is wrapped in <system-reminders>...</system-reminders>
  T-W1-02 -- pre-user header text present and names "the message that follows"
  T-W1-03 -- tail header variant used mid-loop; pre-user variant at turn start
  T-W1-13 -- empty merged blob writes NO message (no bare envelope)
"""

from __future__ import annotations

import pytest
from amplifier_module_loop_streaming import StreamingOrchestrator, _wrap_reminders

# ---------------------------------------------------------------------------
# T-W1-01 / T-W1-02 / T-W1-03 -- the pure helper, unit-tested directly.
# ---------------------------------------------------------------------------


def test_wrap_reminders_produces_the_envelope_tags() -> None:
    """T-W1-01: the merged blob is wrapped in the byte-stable
    <system-reminders>...</system-reminders> tags (spec sec 2.1: "Exact
    opening/closing tags (byte-stable -- tests assert on them)")."""
    wrapped = _wrap_reminders("some hook content", tail=False)
    assert wrapped.startswith("<system-reminders>")
    assert wrapped.endswith("</system-reminders>")
    assert "some hook content" in wrapped


def test_wrap_reminders_preserves_body_verbatim() -> None:
    """The body is inserted verbatim -- sources that already carry their own
    <system-reminder source="..."> wrapper keep it; bare sources sit bare
    *inside* the envelope (spec sec 2.1)."""
    body = (
        '<system-reminder source="hooks-status-context">env stuff</system-reminder>'
        "\n\n"
        "Active routing matrix: anthropic"  # a bare, unwrapped source (W3 fixes this at its own repo)
    )
    wrapped = _wrap_reminders(body, tail=False)
    assert body in wrapped


def test_pre_user_header_names_the_message_that_follows() -> None:
    """T-W1-02: the pre-user header text is present and explicitly names
    "the message that follows" as the user's actual request -- this is the
    exact instruction that defends against the 0629f373 hijack (the model
    must know the user's real ask is what comes AFTER this block, not the
    block itself)."""
    wrapped = _wrap_reminders("body", tail=False)
    assert "NOT from the user" in wrapped
    assert "NOT a request" in wrapped
    assert "message that follows this block" in wrapped
    assert "never treat one as the task" in wrapped


def test_tail_header_differs_from_pre_user_header() -> None:
    """T-W1-03: the tail variant (used mid-loop, for the loop-limit/budget
    notices, and in reminder_placement="tail" rollback mode) uses DIFFERENT
    header text from the pre-user variant -- it tells the model the user's
    request appears EARLIER in the conversation, not that it follows."""
    pre_user = _wrap_reminders("body", tail=False)
    tail = _wrap_reminders("body", tail=True)
    assert pre_user != tail
    assert "message that follows this block" in pre_user
    assert "message that follows this block" not in tail
    assert "most recent request appears earlier in this" in tail
    assert "most recent request appears earlier in this" not in pre_user
    # Both variants still open/close with the same byte-stable tags.
    for wrapped in (pre_user, tail):
        assert wrapped.startswith("<system-reminders>")
        assert wrapped.endswith("</system-reminders>")


# ---------------------------------------------------------------------------
# T-W1-13 -- empty body produces NO envelope at all (never a bare shell).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["", "   ", "\n\n", "\t"])
def test_wrap_reminders_empty_body_returns_empty_string(body: str) -> None:
    """T-W1-13: empty/whitespace-only body must return "" -- callers must
    not write a message at all in that case (no bare envelope is ever
    produced)."""
    assert _wrap_reminders(body, tail=False) == ""
    assert _wrap_reminders(body, tail=True) == ""


# ---------------------------------------------------------------------------
# Integration: the envelope actually reaches the wire via execute().
# ---------------------------------------------------------------------------


class _MockContext:
    def __init__(self) -> None:
        self._messages: list[dict] = []
        self.add_message_calls: list[dict] = []

    async def add_message(self, msg: dict) -> None:
        self.add_message_calls.append(dict(msg))
        self._messages.append(msg)

    async def get_messages_for_request(self, provider=None) -> list[dict]:
        return list(self._messages)


class _MockCancellation:
    is_cancelled = False
    is_immediate = False
    state = "running"

    def register_tool_start(self, tool_call_id: str, display_name: str) -> None:
        pass

    def register_tool_complete(self, tool_call_id: str) -> None:
        pass

    async def trigger_callbacks(self) -> None:
        pass


class _MockCoordinator:
    def __init__(self) -> None:
        self.cancellation = _MockCancellation()
        self._capabilities: dict = {}
        self.session_state: dict = {}

    def register_contributor(self, name: str, source: str, fn) -> None:
        pass

    async def mount(self, name: str, obj) -> None:
        pass

    def register_capability(self, name: str, value) -> None:
        self._capabilities[name] = value

    def get_capability(self, name: str):
        return self._capabilities.get(name)

    async def process_hook_result(self, result, *args, **kwargs):
        return result


class _MockResponse:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.content = None
        self.content_blocks = None
        self.usage = None
        self.metadata = None


class _RequestCapturingProvider:
    def __init__(self) -> None:
        self.requests: list = []

    async def complete(self, chat_request, **kwargs):
        self.requests.append(chat_request)
        return _MockResponse(text="ok")

    def parse_tool_calls(self, response):
        return []


class _ScriptedHookResult:
    def __init__(
        self,
        action: str = "continue",
        ephemeral: bool = False,
        context_injection: str | None = None,
        context_injection_role: str = "system",
        append_to_last_tool_result: bool = False,
        data=None,
        reason: str | None = None,
    ) -> None:
        self.action = action
        self.ephemeral = ephemeral
        self.context_injection = context_injection
        self.context_injection_role = context_injection_role
        self.append_to_last_tool_result = append_to_last_tool_result
        self.data = data
        self.reason = reason


class _ScriptedHooks:
    def __init__(self, scripts: dict) -> None:
        self.scripts = scripts
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, event_name: str, payload: dict | None = None):
        self.emitted.append((event_name, payload or {}))
        return self.scripts.get(event_name, _ScriptedHookResult())


@pytest.mark.asyncio
async def test_envelope_reaches_the_wire_via_execute() -> None:
    """The envelope is not just a unit-testable helper -- it is actually
    applied at the real insertion site inside execute()."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = _MockContext()
    provider = _RequestCapturingProvider()
    hooks = _ScriptedHooks(
        {
            PROVIDER_REQUEST: _ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="hook payload",
                context_injection_role="system",
            ),
        }
    )
    coordinator = _MockCoordinator()
    orch = StreamingOrchestrator({})

    await orch.execute(
        prompt="hello",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    sent_messages = provider.requests[0].messages
    enveloped = [
        m
        for m in sent_messages
        if isinstance(m.content, str) and m.content.startswith("<system-reminders>")
    ]
    assert len(enveloped) == 1
    assert "hook payload" in enveloped[0].content
