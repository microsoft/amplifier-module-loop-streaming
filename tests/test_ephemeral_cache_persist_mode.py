"""Tests for `ephemeral_injection_mode` (ephemeral-cache-fix-spec.md §5/§6).

Background: OpenAI's implicit prompt cache reuses only the longest TRUE
PREFIX of a prior request. The orchestrator's per-iteration ephemeral tail
injections (hooks-status-context, hooks-todo-reminder, ...) get regenerated
and re-appended at the tail on every iteration, positionally displacing the
assistant/tool turn that follows them -- so request N is never a true prefix
of request N+1, and the cache never advances past the static system prompt.

The fix adds a config-gated third mode:

    ephemeral_injection_mode: "tail" | "persist"   (default "persist")

In "persist" mode, an injection is written into CANONICAL context via
`context.add_message(...)` only when its text differs from the last text
this orchestrator persisted. When unchanged, nothing is added, and the
request is a pure append of the new assistant/tool turn -- append-only,
i.e. request N IS a true prefix of request N+1.

NOTE (default flip): #44 shipped this behind `default "tail"` because
persist was unvalidated on Anthropic at the time. Both providers are now
validated (OpenAI: 30+ in-vivo DTU runs; Anthropic: n=3 DTU S1 flip-gate
runs, favorable on cache-read share, cost, wall time, and quality), so the
default flipped to "persist" here. "tail" remains a fully supported,
byte-identical-to-original explicit opt-out.

These tests prove (per spec §6, in file order):

  1. test_tail_mode_is_byte_identical_to_original_behavior -- the regression
     guard on the explicit opt-out: the "tail" path is untouched
     byte-for-byte from the module's original pre-#44 behavior.
  2. test_persist_mode_adds_message_to_canonical_context
  3. test_persist_mode_suppresses_unchanged_injection -- the change-gate.
  4. test_persist_mode_emits_on_changed_injection
  5. test_persist_mode_request_is_append_only -- the contract itself:
     request N is a strict prefix of request N+1, element-wise.
  6. test_persist_mode_injection_is_budgeted -- the persisted message is
     visible to `get_messages_for_request` (and therefore to any real
     context module's token accounting) on the very next call, not just
     "eventually".
  7. test_unknown_mode_falls_back_to_persist_with_warning
  8. test_default_mode_is_now_persist -- the default-flip guard: with NO
     `ephemeral_injection_mode` key in config at all, the default is now
     "persist", not "tail".
"""

from __future__ import annotations

import logging

import pytest

from amplifier_module_loop_streaming import StreamingOrchestrator

# ---------------------------------------------------------------------------
# Shared stubs -- same pattern as test_ephemeral_cache_metadata.py, plus a
# SequencedHooks double that can return a DIFFERENT scripted result per call
# to the same event (needed to simulate an injection that changes value on
# a later iteration -- ScriptedHooks in the sibling file always returns the
# same fixed result for a given event name).
# ---------------------------------------------------------------------------


class MockContext:
    def __init__(self) -> None:
        self._messages: list[dict] = []
        self.add_message_calls: list[dict] = []

    async def add_message(self, msg: dict) -> None:
        self.add_message_calls.append(dict(msg))
        self._messages.append(msg)

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
        self._capabilities: dict = {}
        self._mounts: dict = {}
        self.session_state: dict = {}

    def register_contributor(self, name: str, source: str, fn) -> None:
        pass

    async def mount(self, name: str, obj) -> None:
        self._mounts[name] = obj

    def register_capability(self, name: str, value) -> None:
        self._capabilities[name] = value

    def get_capability(self, name: str):
        return self._capabilities.get(name)

    async def process_hook_result(self, result, *args, **kwargs):
        return result


class MockResponse:
    """Minimal non-streaming provider response."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.content = None
        self.content_blocks = None
        self.usage = None
        self.metadata = None


class RequestCapturingProvider:
    """Non-streaming provider that records every ChatRequest it receives."""

    def __init__(self) -> None:
        self.requests: list = []

    async def complete(self, chat_request, **kwargs):
        self.requests.append(chat_request)
        return MockResponse(text="ok")

    def parse_tool_calls(self, response):
        return []


class ScriptedHookResult:
    """Hook result whose fields are set per-instance (mirrors the sibling
    test file's fixture of the same name)."""

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


class ScriptedHooks:
    """Hooks stub returning a FIXED scripted result per event name (same
    result every call) -- sufficient when the injection text never changes
    across iterations within one test."""

    def __init__(self, scripts: dict[str, ScriptedHookResult]) -> None:
        self.scripts = scripts
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, event_name: str, payload: dict | None = None):
        self.emitted.append((event_name, payload or {}))
        return self.scripts.get(event_name, ScriptedHookResult())


class SequencedHooks:
    """Hooks stub that pops the NEXT scripted result off a per-event queue on
    each call, repeating the last one once the queue is exhausted. Needed to
    simulate hooks-status-context's real shape: byte-stable for several
    iterations, then genuinely changing once (e.g. a compaction escalation),
    matching probe arm E's own text-variation schedule.
    """

    def __init__(self, sequences: dict[str, list[ScriptedHookResult]]) -> None:
        self.sequences = {k: list(v) for k, v in sequences.items()}
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, event_name: str, payload: dict | None = None):
        self.emitted.append((event_name, payload or {}))
        seq = self.sequences.get(event_name)
        if not seq:
            return ScriptedHookResult()
        if len(seq) > 1:
            return seq.pop(0)
        return seq[0]


class OneShotTool:
    name = "mock_tool"
    description = "test tool"
    input_schema: dict = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        from amplifier_core import ToolResult

        return ToolResult(success=True, output="tool output")


class ToolCallStub:
    def __init__(self, call_id: str = "tc-1") -> None:
        self.id = call_id
        self.name = "mock_tool"
        self.arguments: dict = {}


class NRoundToolProvider:
    """Provider that returns a tool call for the first `n_tool_rounds`
    completions, then a plain-text (no tool call) response forever after --
    driving `n_tool_rounds + 1` total provider.complete() calls within one
    execute(), and capturing every outgoing ChatRequest along the way."""

    def __init__(self, n_tool_rounds: int) -> None:
        self.n_tool_rounds = n_tool_rounds
        self.call_count = 0
        self.requests: list = []

    async def complete(self, chat_request, **kwargs):
        self.call_count += 1
        self.requests.append(chat_request)
        return MockResponse(text=f"round {self.call_count}")

    def parse_tool_calls(self, response):
        if self.call_count <= self.n_tool_rounds:
            return [ToolCallStub(call_id=f"tc-{self.call_count}")]
        return []


# ---------------------------------------------------------------------------
# 1. Explicit "tail" opt-out is byte-identical to the module's original
#    pre-#44 behavior (the regression guard on the now-non-default path).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tail_mode_is_byte_identical_to_original_behavior() -> None:
    """With `ephemeral_injection_mode: "tail"` AND `reminder_placement:
    "tail"` BOTH explicitly configured (the full legacy opt-out combination
    -- reminder-redesign-spec.md sec 9's 2x2 mode table, "fully pre-spec"
    row), the injected message must be a per-request TAIL append (never
    routed through `context.add_message`), appearing AFTER the user's own
    message -- the pre-spec ORDERING is exactly reproduced (T-W1-11).

    The envelope, role pin, and `reminder_placement` metadata tag are NOT
    behind a flag (sec 9: "The envelope itself is also not behind a flag
    ... Rollback for the envelope is `reminder_placement: tail` plus a
    revert, not a toggle") -- so those are still present here. What
    `reminder_placement: "tail"` restores is the ORDERING only: no
    turn-start emit, block after the user message, exactly as pre-spec."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROVIDER_REQUEST: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="<system-reminder>static</system-reminder>",
                context_injection_role="system",
            ),
        }
    )
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator(
        {"ephemeral_injection_mode": "tail", "reminder_placement": "tail"}
    )

    assert orch._ephemeral_injection_mode == "tail"
    assert orch._reminder_placement == "tail"

    await orch.execute(
        prompt="hello",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    # Never persisted into canonical context. (execute() itself always
    # calls context.add_message() for the user's own prompt and the
    # assistant's reply -- that is normal, unrelated orchestrator behavior;
    # what must NOT happen is the ephemeral injection's own content being
    # routed through add_message at all.)
    injected_via_add_message = [
        m
        for m in ctx.add_message_calls
        if isinstance(m.get("content"), str)
        and "<system-reminder>static</system-reminder>" in m["content"]
    ]
    assert injected_via_add_message == [], (
        "Explicit tail mode must never call context.add_message() for the "
        f"injection; got {ctx.add_message_calls!r}"
    )

    sent_messages = provider.requests[0].messages
    injected = [
        m
        for m in sent_messages
        if isinstance(m.content, str)
        and "<system-reminder>static</system-reminder>" in m.content
    ]
    assert len(injected) == 1
    assert injected[0].content.startswith("<system-reminders>"), (
        "Injected content must be wrapped in the <system-reminders> envelope "
        "-- the envelope is not behind reminder_placement"
    )
    assert injected[0].role == "user", (
        "Role must be pinned to 'user' regardless of the hook-requested "
        "context_injection_role='system' -- the role pin is not behind "
        "reminder_placement either"
    )
    assert injected[0].metadata == {
        "ephemeral": True,
        "reminder_placement": "tail",
    }, (
        f"Expected metadata={{'ephemeral': True, 'reminder_placement': 'tail'}} "
        f"(no 'persisted' key), got {injected[0].metadata!r}"
    )

    # ORDERING: the block appears AFTER the user's own message -- the
    # pre-spec order, reproduced exactly (T-W1-11).
    user_msgs = [m for m in sent_messages if m.role == "user" and m.content == "hello"]
    assert len(user_msgs) == 1
    injected_idx = sent_messages.index(injected[0])
    user_idx = sent_messages.index(user_msgs[0])
    assert injected_idx > user_idx, (
        "reminder_placement='tail' must reproduce pre-spec ordering: block "
        "AFTER the user's message"
    )


@pytest.mark.asyncio
async def test_default_mode_is_now_persist() -> None:
    """With NO `ephemeral_injection_mode` key (and no `reminder_placement`
    key either) in config at all, both defaults apply: "persist" (default
    flip, both providers validated) and "pre_user" (reminder-redesign-
    spec.md, W1) -- the injection must be written into canonical context
    via `context.add_message(...)`, enveloped, role-pinned to "user", tagged
    `reminder_placement: "pre_user"`, and land BEFORE the user's own message
    (turn-start placement, not a per-request tail append)."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROVIDER_REQUEST: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="<system-reminder>v1</system-reminder>",
                context_injection_role="user",
            ),
        }
    )
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({})  # no config keys at all

    assert orch._ephemeral_injection_mode == "persist"
    assert orch._reminder_placement == "pre_user"

    await orch.execute(
        prompt="hello",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    persisted = [
        m
        for m in ctx.add_message_calls
        if isinstance(m.get("content"), str)
        and "<system-reminder>v1</system-reminder>" in m["content"]
    ]
    assert len(persisted) == 1, (
        "Default (no config keys) must now persist the injection via "
        f"context.add_message(); got {ctx.add_message_calls!r}"
    )
    assert persisted[0]["content"].startswith("<system-reminders>")
    assert persisted[0]["metadata"] == {
        "ephemeral": True,
        "persisted": True,
        "reminder_placement": "pre_user",
    }

    sent_messages = provider.requests[0].messages
    injected = [
        m
        for m in sent_messages
        if isinstance(m.content, str)
        and "<system-reminder>v1</system-reminder>" in m.content
    ]
    assert len(injected) == 1
    assert injected[0].role == "user"

    # Block precedes the user's own message (turn-start placement, T-W1-04).
    user_msgs = [m for m in sent_messages if m.role == "user" and m.content == "hello"]
    assert len(user_msgs) == 1
    assert sent_messages.index(injected[0]) < sent_messages.index(user_msgs[0])


# ---------------------------------------------------------------------------
# 2. Persist mode adds the message to canonical context.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_mode_adds_message_to_canonical_context() -> None:
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROVIDER_REQUEST: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="<system-reminder>v1</system-reminder>",
                context_injection_role="user",
            ),
        }
    )
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "persist"})

    assert orch._ephemeral_injection_mode == "persist"

    await orch.execute(
        prompt="hello",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    persisted = [
        m
        for m in ctx.add_message_calls
        if isinstance(m.get("content"), str)
        and "<system-reminder>v1</system-reminder>" in m["content"]
    ]
    assert len(persisted) == 1, (
        f"Expected exactly one context.add_message() call for the injection, "
        f"got {ctx.add_message_calls!r}"
    )
    assert persisted[0]["content"].startswith("<system-reminders>")
    assert persisted[0]["metadata"] == {
        "ephemeral": True,
        "persisted": True,
        "reminder_placement": "pre_user",
    }

    # And it must actually be present in what the (only) request sent.
    sent_messages = provider.requests[0].messages
    injected = [
        m
        for m in sent_messages
        if isinstance(m.content, str)
        and "<system-reminder>v1</system-reminder>" in m.content
    ]
    assert len(injected) == 1


# ---------------------------------------------------------------------------
# 3 & 4. The change-gate: unchanged text -> one add_message call ever;
# changed text -> a second add_message call.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_mode_suppresses_unchanged_injection() -> None:
    """Three tool-loop iterations, SAME injection text every time -> exactly
    ONE context.add_message() call for the injection.

    Under reminder_placement="pre_user" (default), iteration 1 is now the
    TURN-START emit (consumes the first scripted result, persists it
    pre-user-wrapped). The in-loop iteration-1 emit is skipped entirely
    (Option D). Iterations 2 and 3 are genuinely in-loop (mid-turn,
    tail-wrapped) -- the change-gate compares the RAW (pre-envelope) body,
    so "unchanging" at iteration 2/3 still matches the RAW body persisted
    at turn start even though the two iterations wrap it in DIFFERENT
    envelope header variants (pre_user vs tail) -- this is exactly why the
    change-gate compares raw text, not the enveloped string (see the
    orchestrator's own comment at the persist-mode append site)."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=2)  # 3 total completions
    same_result = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="<system-reminder>unchanging</system-reminder>",
        context_injection_role="user",
    )
    hooks = SequencedHooks({PROVIDER_REQUEST: [same_result, same_result, same_result]})
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "persist"})

    await orch.execute(
        prompt="start",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={"mock_tool": OneShotTool()},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert provider.call_count == 3, "Expected 3 provider.complete() rounds"
    persisted = [
        m
        for m in ctx.add_message_calls
        if isinstance(m.get("content"), str)
        and "<system-reminder>unchanging</system-reminder>" in m["content"]
    ]
    assert len(persisted) == 1, (
        f"Change-gate must suppress re-persisting unchanged text (compared "
        f"on the raw body, across the pre_user/tail header-variant "
        f"boundary); expected 1 add_message call, got {len(persisted)}: "
        f"{ctx.add_message_calls!r}"
    )
    assert persisted[0]["metadata"]["reminder_placement"] == "pre_user", (
        "The one persist must be the TURN-START one (pre_user variant)"
    )


@pytest.mark.asyncio
async def test_persist_mode_emits_on_changed_injection() -> None:
    """Three tool-loop iterations, injection text changes on iteration 2 and
    stays changed on iteration 3 -> exactly TWO context.add_message() calls:
    turn-start's "v1" (pre_user-wrapped) and mid-loop iteration 2's "v2"
    (tail-wrapped); iteration 3 repeats "v2", so it is suppressed by the
    change-gate (raw-body comparison) same as the unchanged-injection test."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=2)  # 3 total completions
    r_v1 = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="<system-reminder>v1</system-reminder>",
        context_injection_role="user",
    )
    r_v2 = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="<system-reminder>v2</system-reminder>",
        context_injection_role="user",
    )
    hooks = SequencedHooks({PROVIDER_REQUEST: [r_v1, r_v2, r_v2]})
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "persist"})

    await orch.execute(
        prompt="start",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={"mock_tool": OneShotTool()},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert provider.call_count == 3
    persisted_calls = [
        m for m in ctx.add_message_calls if "system-reminder" in m["content"]
    ]
    assert len(persisted_calls) == 2, (
        f"Expected exactly 2 add_message calls (v1 then v2, v2 repeat "
        f"suppressed), got {len(persisted_calls)}: {persisted_calls!r}"
    )
    assert "<system-reminder>v1</system-reminder>" in persisted_calls[0]["content"]
    assert persisted_calls[0]["metadata"]["reminder_placement"] == "pre_user", (
        "First persist is the TURN-START block"
    )
    assert "<system-reminder>v2</system-reminder>" in persisted_calls[1]["content"]
    assert persisted_calls[1]["metadata"]["reminder_placement"] == "tail", (
        "Second persist is a MID-LOOP block (iteration 2)"
    )


# ---------------------------------------------------------------------------
# 5. The contract itself: request N is a strict prefix of request N+1.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_mode_request_is_append_only() -> None:
    """Across 3 iterations WITHIN one turn (injection unchanged throughout,
    the common case), each outgoing request's message list must be a strict
    element-wise prefix of the next request's -- the actual contract this
    mode exists to guarantee (request N \u2282 request N+1), not an
    implementation detail of how it's achieved."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=2)
    stable = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="<system-reminder>stable</system-reminder>",
        context_injection_role="user",
    )
    hooks = SequencedHooks({PROVIDER_REQUEST: [stable, stable, stable]})
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "persist"})

    await orch.execute(
        prompt="start",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={"mock_tool": OneShotTool()},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert provider.call_count == 3
    requests = [r.messages for r in provider.requests]

    def sig(msg) -> tuple:
        return (
            msg.role,
            msg.content if isinstance(msg.content, str) else str(msg.content),
        )

    for i in range(len(requests) - 1):
        shorter = [sig(m) for m in requests[i]]
        longer = [sig(m) for m in requests[i + 1]]
        assert len(shorter) < len(longer), (
            f"request {i} must be strictly shorter than request {i + 1}"
        )
        assert longer[: len(shorter)] == shorter, (
            f"request {i}'s messages must be an exact element-wise prefix of "
            f"request {i + 1}'s (append-only contract). request {i}="
            f"{shorter!r} is not a prefix of request {i + 1}={longer!r}"
        )


@pytest.mark.asyncio
async def test_persist_mode_request_is_append_only_across_turns() -> None:
    """The append-only contract extended ACROSS TURNS (reminder-redesign-
    spec.md, W1.2's scorecard: "cross-turn append-only" is constraint (2) in
    the owner's hierarchy) -- request N (the LAST request of turn 1) must be
    a strict element-wise prefix of request N+1 (the FIRST request of turn
    2), even though turn 2 re-runs the turn-start reminder assembly. With
    the injection text STABLE across turns, the change-gate (compared on
    the raw body) suppresses turn 2's persist entirely, so nothing is
    inserted between turn 1's last message and turn 2's new user message."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    stable = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="<system-reminder>stable</system-reminder>",
        context_injection_role="user",
    )
    hooks = ScriptedHooks({PROVIDER_REQUEST: stable})
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "persist"})

    await orch.execute(
        prompt="first",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )
    await orch.execute(
        prompt="second",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert len(provider.requests) == 2, "Expected exactly one request per turn"

    def sig(msg) -> tuple:
        return (
            msg.role,
            msg.content if isinstance(msg.content, str) else str(msg.content),
        )

    turn1 = [sig(m) for m in provider.requests[0].messages]
    turn2 = [sig(m) for m in provider.requests[1].messages]
    assert len(turn1) < len(turn2), "turn 2's request must be strictly longer"
    assert turn2[: len(turn1)] == turn1, (
        f"turn 1's request must be an exact element-wise prefix of turn 2's "
        f"(cross-turn append-only). turn1={turn1!r} is not a prefix of "
        f"turn2={turn2!r}"
    )
    # Only ONE persisted reminder-block message across both turns (the
    # change-gate suppresses turn 2's identical-text re-persist).
    persisted_blocks = [
        m for m in ctx.add_message_calls if "system-reminder" in m["content"]
    ]
    assert len(persisted_blocks) == 1


# ---------------------------------------------------------------------------
# 6. The persisted injection is visible to get_messages_for_request on the
# very next call -- the structural prerequisite for a real context module's
# token budgeting to count it (ef1 in the spec's risk table).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_mode_injection_is_budgeted() -> None:
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=1)  # 2 total completions
    injected = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="<system-reminder>budget-me</system-reminder>",
        context_injection_role="user",
    )
    hooks = SequencedHooks({PROVIDER_REQUEST: [injected, injected]})
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "persist"})

    await orch.execute(
        prompt="start",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={"mock_tool": OneShotTool()},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert provider.call_count == 2

    # It must show up in context.get_messages_for_request() itself (not just
    # in the outgoing request's synthesized copy) -- that is what makes it
    # visible to a REAL context module's token-budget accounting on every
    # subsequent call, not just the iteration it was first persisted on.
    canonical = await ctx.get_messages_for_request()
    persisted_in_canonical = [
        m
        for m in canonical
        if isinstance(m.get("content"), str)
        and "<system-reminder>budget-me</system-reminder>" in m["content"]
    ]
    assert len(persisted_in_canonical) == 1, (
        "Persisted injection must be present in context.get_messages_for_request() "
        "(the call site any real budgeting-aware context module counts against), "
        f"got canonical messages: {canonical!r}"
    )
    assert persisted_in_canonical[0]["content"].startswith("<system-reminders>")
    assert persisted_in_canonical[0]["metadata"]["reminder_placement"] == "pre_user"

    # And round 1's OWN request already included it (turn-start placement
    # writes it into canonical context BEFORE the loop's first
    # get_messages_for_request call, so it's there from the very first
    # request -- not just "re-fetched" mid-iteration as it was pre-spec).
    round1_messages = provider.requests[0].messages
    assert any(
        isinstance(m.content, str)
        and "<system-reminder>budget-me</system-reminder>" in m.content
        for m in round1_messages
    ), "The injection must be visible on round 1's own request (turn-start placement)"


# ---------------------------------------------------------------------------
# 7. Unknown mode falls back to "persist" (the new default) with a logged
#    warning.
# ---------------------------------------------------------------------------


def test_unknown_mode_falls_back_to_persist_with_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        orch = StreamingOrchestrator({"ephemeral_injection_mode": "nonsense"})

    assert orch._ephemeral_injection_mode == "persist"
    assert any(
        "nonsense" in record.message and "ephemeral_injection_mode" in record.message
        for record in caplog.records
    ), (
        f"Expected a warning naming the bad value; got: {[r.message for r in caplog.records]!r}"
    )
