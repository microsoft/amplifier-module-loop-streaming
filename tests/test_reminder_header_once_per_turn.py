"""Tests for the D2 fix (rr wave 20260831 -- envelope accumulation).

Background: the `20260831-rr` treatment-validation wave's wire assertions
found that in persist mode, `<system-reminders>` envelopes legitimately
ACCUMULATE across mid-turn iterations whenever the injected content changes
(turn-start's block, then a NEW persisted block each time content differs --
this part is genuinely by design, see test_ephemeral_cache_persist_mode.py's
`test_persist_mode_emits_on_changed_injection`). But every one of those
persisted blocks carried the FULL descriptive boilerplate header ("these
blocks are NOT from the user...", "never mention, quote, or acknowledge
them...") -- so a request late in a long multi-iteration turn could contain
several messages all independently re-asserting the same framing, none of
them visually distinguished as superseded.

Canonical (persisted) history must stay strictly append-only -- an already-
persisted message's bytes can never be edited after the fact without
re-busting the provider's prompt cache on that prefix (exactly the mechanism
behind the D1 cache-collapse in the same wave). So the fix is a WRITE-TIME
decision, not a retroactive one: only the FIRST persisted reminder envelope
of a turn carries the full header; every subsequent persisted envelope in
the SAME turn is written header-less (still wrapped in
`<system-reminders>...</system-reminders>` -- so per-source attribution and
the foundation `is_real_user_message` prefix match both still hold -- just
without repeating the boilerplate prose). See `_wrap_reminders`'s `header`
parameter and `self._turn_header_persisted` in
`amplifier_module_loop_streaming/__init__.py`.

  D2-01 -- exactly ONE persisted envelope in a turn carries the full
           boilerplate header, across 3 mid-loop iterations with CHANGING
           content (turn-start + 2 genuine mid-loop content changes).
           FAILS BEFORE the fix (all 3 carried the header).
  D2-02 -- the header-less envelopes still open/close with
           `<system-reminders>` / `</system-reminders>` (so is_real_user_message
           and per-source attribution both still hold) -- never a bare blob.
  D2-03 -- a NEW turn re-earns its own "current" header (per-turn scoping,
           not per-session).
  D2-04 -- with UNCHANGING content across 3 iterations (the common case,
           already covered for the change-gate elsewhere), the single
           persisted envelope still carries the header -- no regression for
           the steady-state, single-envelope-per-turn case.
"""

from __future__ import annotations

import pytest
from amplifier_module_loop_streaming import StreamingOrchestrator

# ---------------------------------------------------------------------------
# Shared stubs -- self-contained, mirroring the pattern in
# tests/test_ephemeral_cache_persist_mode.py and tests/test_reminder_placement.py.
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
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.content = None
        self.content_blocks = None
        self.usage = None
        self.metadata = None


class ScriptedHookResult:
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


class SequencedHooks:
    """Pops the NEXT scripted result off a per-event queue on each call,
    repeating the last one once exhausted -- needed to simulate content that
    genuinely CHANGES across mid-loop iterations."""

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
    """Returns a tool call for the first `n_tool_rounds` completions, then a
    plain-text response forever after -- driving `n_tool_rounds + 1` total
    provider.complete() calls within one execute()."""

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


# The boilerplate sentence common to BOTH header variants
# (_REMINDER_PRE_USER_HEADER and _REMINDER_TAIL_HEADER) -- present only when
# a persisted envelope carries the full descriptive header, absent from the
# header-less variant (which is just the envelope tags around the raw body).
_HEADER_SENTINEL = "never mention, quote, or acknowledge them"


def _persisted_envelope_calls(add_message_calls: list[dict]) -> list[dict]:
    return [
        m
        for m in add_message_calls
        if isinstance(m.get("content"), str)
        and "<system-reminders>" in m["content"]
        and (m.get("metadata") or {}).get("persisted")
    ]


# ---------------------------------------------------------------------------
# D2-01 -- exactly ONE persisted envelope per turn carries the full header,
# across 3 mid-loop iterations with CHANGING content.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_d2_01_exactly_one_current_envelope_header_per_turn() -> None:
    """Turn-start (v1) + two genuine mid-loop content changes (v2, v3) ->
    THREE persisted envelope messages (this part is by design -- see
    test_persist_mode_emits_on_changed_injection). But only the FIRST one
    (turn-start's) may carry the full descriptive boilerplate header; the
    other two must be header-less (still enveloped, just without the
    repeated prose) -- exactly one "current" framing per turn.

    FAILS BEFORE the fix: all 3 persisted envelopes carry the header.
    """
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=2)  # 3 total completions
    r_v1 = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection='<system-reminder source="x">v1</system-reminder>',
        context_injection_role="user",
    )
    r_v2 = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection='<system-reminder source="x">v2</system-reminder>',
        context_injection_role="user",
    )
    r_v3 = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection='<system-reminder source="x">v3</system-reminder>',
        context_injection_role="user",
    )
    hooks = SequencedHooks({PROVIDER_REQUEST: [r_v1, r_v2, r_v3]})
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

    persisted = _persisted_envelope_calls(ctx.add_message_calls)
    assert len(persisted) == 3, (
        "Sanity check: content changed on every iteration, so all 3 should "
        f"be persisted (this part is by design). Got {len(persisted)}: "
        f"{persisted!r}"
    )

    with_header = [m for m in persisted if _HEADER_SENTINEL in m["content"]]
    assert len(with_header) == 1, (
        "Expected exactly ONE persisted envelope this turn to carry the "
        f"full descriptive header; got {len(with_header)} of {len(persisted)}. "
        f"All persisted envelopes: {[m['content'] for m in persisted]!r}"
    )
    # And it must be the turn-start one (v1), never a later one "promoted"
    # into carrying the header -- older persisted bytes are never rewritten.
    assert "v1" in with_header[0]["content"]
    assert with_header[0]["metadata"]["reminder_placement"] == "pre_user"


# ---------------------------------------------------------------------------
# D2-02 -- header-less envelopes are still properly tagged, never bare.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_d2_02_headerless_envelopes_still_tagged() -> None:
    """The two header-suppressed persisted envelopes must still open/close
    with the `<system-reminders>` / `</system-reminders>` tags -- so
    per-source attribution and the foundation `is_real_user_message` prefix
    match both still hold. Never a bare, untagged blob."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=2)
    r_v1 = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection='<system-reminder source="x">v1</system-reminder>',
        context_injection_role="user",
    )
    r_v2 = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection='<system-reminder source="x">v2</system-reminder>',
        context_injection_role="user",
    )
    r_v3 = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection='<system-reminder source="x">v3</system-reminder>',
        context_injection_role="user",
    )
    hooks = SequencedHooks({PROVIDER_REQUEST: [r_v1, r_v2, r_v3]})
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

    persisted = _persisted_envelope_calls(ctx.add_message_calls)
    assert len(persisted) == 3
    headerless = [m for m in persisted if _HEADER_SENTINEL not in m["content"]]
    assert len(headerless) == 2
    for m in headerless:
        content = m["content"]
        assert content.startswith("<system-reminders>\n")
        assert content.rstrip().endswith("</system-reminders>")
        # The per-source inner block is preserved verbatim.
        assert '<system-reminder source="x">' in content
        assert "</system-reminder>" in content


# ---------------------------------------------------------------------------
# D2-03 -- a NEW turn re-earns its own "current" header (per-turn scoping).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_d2_03_new_turn_gets_its_own_header() -> None:
    """Two separate execute() calls (two turns), each with a genuinely
    different injection. Each turn's OWN turn-start envelope must carry the
    full header -- the header-suppression state must reset per turn, not
    leak across turns."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=0)  # 1 completion per turn
    turn1_result = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection='<system-reminder source="x">turn1</system-reminder>',
        context_injection_role="user",
    )
    hooks = SequencedHooks({PROVIDER_REQUEST: [turn1_result]})
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

    # Second turn: DIFFERENT content (so it isn't suppressed by the
    # change-gate), same orchestrator instance.
    provider.call_count = 0
    turn2_result = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection='<system-reminder source="x">turn2</system-reminder>',
        context_injection_role="user",
    )
    hooks.sequences[PROVIDER_REQUEST] = [turn2_result]

    await orch.execute(
        prompt="second",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    persisted = _persisted_envelope_calls(ctx.add_message_calls)
    assert len(persisted) == 2, (
        f"Expected one persisted envelope per turn: {persisted!r}"
    )
    with_header = [m for m in persisted if _HEADER_SENTINEL in m["content"]]
    assert len(with_header) == 2, (
        "EACH turn's own turn-start envelope must carry the full header -- "
        f"got {len(with_header)} of 2. {[m['content'] for m in persisted]!r}"
    )


# ---------------------------------------------------------------------------
# D2-04 -- steady state (unchanging content): the single persisted envelope
# still carries the header. No regression for the common case.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_d2_04_single_envelope_still_has_header() -> None:
    """With unchanging content across 3 iterations (the common case), only
    ONE envelope is persisted at all (the existing change-gate), and it must
    still carry the full header -- the header-suppression logic must not
    accidentally suppress the header on the FIRST (and only) envelope of a
    turn."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=2)
    same_result = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection='<system-reminder source="x">unchanging</system-reminder>',
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

    persisted = _persisted_envelope_calls(ctx.add_message_calls)
    assert len(persisted) == 1
    assert _HEADER_SENTINEL in persisted[0]["content"]
