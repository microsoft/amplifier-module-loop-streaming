"""Tests for `reminder_placement` (reminder-redesign-spec.md, W1.2 -- Option D).

Background: session `0629f373` showed the merged reminder blob landing
AFTER the user's real request in the wire order -- the model obeyed the
trailing conditional directive instead of doing the actual work. Option D
fixes this by hoisting iteration 1's `provider:request` emit to TURN START,
before the user prompt is appended, while preserving cross-turn append-only
canonical history (constraint (2) in the owner's hierarchy, spec sec 0).

Config: `reminder_placement: "pre_user" | "tail"`, default `"pre_user"`.
`"tail"` is the rollback lever -- pre-spec ordering, byte-identical.

Covers (per the spec's test-plan sec 10), the IDs not already exercised as
a side effect of `tests/test_ephemeral_cache_persist_mode.py` /
`tests/test_ephemeral_cache_metadata.py`'s own updates:

  T-W1-04 -- block precedes the user message at iteration 1 (persist mode)
  T-W1-05 -- same, in tail injection mode via the view splice
  T-W1-06 -- canonical history is append-only across 3 TURNS
  T-W1-07 -- provider:request fires exactly once per LLM call (no double-fire)
  T-W1-08 -- no reminder message is ever inserted between an assistant
             tool_use and its tool_result batch
  T-W1-09 -- budget-warning message carries metadata.ephemeral (latent bug)
  T-W1-10 -- role is "user" even when every contributing hook asked for "system"
  T-W1-11 -- reminder_placement="tail" reproduces pre-spec ordering exactly
  T-W1-12 -- unknown reminder_placement warns and falls back to "pre_user"
  T-W1-14 -- prompt:submit content precedes provider:request content in the block
  T-W1-15 -- append_to_last_tool_result stash is not consumed at turn start
             and still concatenates in-loop
"""

from __future__ import annotations

import logging

import pytest
from amplifier_module_loop_streaming import StreamingOrchestrator

# ---------------------------------------------------------------------------
# Shared stubs -- self-contained, mirroring the pattern in
# tests/test_ephemeral_cache_persist_mode.py and tests/test_provider_pin.py.
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


class RequestCapturingProvider:
    def __init__(self) -> None:
        self.requests: list = []

    async def complete(self, chat_request, **kwargs):
        self.requests.append(chat_request)
        return MockResponse(text="ok")

    def parse_tool_calls(self, response):
        return []


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


class ScriptedHooks:
    """Fixed scripted result per event name (same result every call)."""

    def __init__(self, scripts: dict[str, ScriptedHookResult]) -> None:
        self.scripts = scripts
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, event_name: str, payload: dict | None = None):
        self.emitted.append((event_name, payload or {}))
        return self.scripts.get(event_name, ScriptedHookResult())


class SequencedHooks:
    """Pops the NEXT scripted result off a per-event queue on each call,
    repeating the last one once exhausted."""

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
    plain-text response forever after."""

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
# T-W1-04 -- block precedes the user message at iteration 1 (persist mode).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_04_block_precedes_user_message_in_persist_mode() -> None:
    """The headline regression test: in persist mode (the default), the
    turn-start reminder block must appear BEFORE the user's message in BOTH
    canonical history (ctx.add_message_calls) and the request sent to the
    provider -- fixing the exact defect from session 0629f373 (the block
    followed the user message, and the model obeyed it instead)."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROVIDER_REQUEST: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="Active routing matrix: anthropic",
                context_injection_role="system",
            ),
        }
    )
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({})  # defaults: persist + pre_user

    await orch.execute(
        prompt="what does auth.py do?",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    # Canonical history (add_message call order).
    canonical_roles_contents = [
        (m.get("role"), m.get("content")) for m in ctx.add_message_calls
    ]
    block_idx = next(
        i
        for i, (_, c) in enumerate(canonical_roles_contents)
        if isinstance(c, str) and "Active routing matrix" in c
    )
    user_idx = next(
        i
        for i, (_, c) in enumerate(canonical_roles_contents)
        if c == "what does auth.py do?"
    )
    assert block_idx < user_idx, (
        "In CANONICAL history, the reminder block must precede the user's "
        f"message. Got order: {canonical_roles_contents!r}"
    )

    # The request view (what the model actually sees on the wire).
    sent = provider.requests[0].messages
    request_block_idx = next(
        i
        for i, m in enumerate(sent)
        if isinstance(m.content, str) and "Active routing matrix" in m.content
    )
    request_user_idx = next(
        i for i, m in enumerate(sent) if m.content == "what does auth.py do?"
    )
    assert request_block_idx < request_user_idx, (
        "In the REQUEST VIEW, the reminder block must precede the user's message"
    )


# ---------------------------------------------------------------------------
# T-W1-05 -- same, tail injection mode via the view splice.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_05_block_precedes_user_message_in_tail_mode_via_splice() -> None:
    """In ephemeral_injection_mode="tail" (never persisted), the turn-start
    block is spliced into the request VIEW immediately before the user's
    message -- never routed through context.add_message."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROVIDER_REQUEST: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="live env snapshot",
                context_injection_role="system",
            ),
        }
    )
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "tail"})

    await orch.execute(
        prompt="fix the bug",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    # Never persisted.
    assert not any(
        isinstance(m.get("content"), str) and "live env snapshot" in m["content"]
        for m in ctx.add_message_calls
    )

    sent = provider.requests[0].messages
    block_idx = next(
        i
        for i, m in enumerate(sent)
        if isinstance(m.content, str) and "live env snapshot" in m.content
    )
    user_idx = next(i for i, m in enumerate(sent) if m.content == "fix the bug")
    assert block_idx < user_idx


# ---------------------------------------------------------------------------
# T-W1-06 -- canonical history is append-only across THREE turns.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_06_append_only_across_three_turns() -> None:
    """Extended prefix-property check across THREE separate execute() calls
    (three turns), with a CHANGING injection each turn -- so each turn does
    persist a new block, and the property must still hold: request N is a
    strict element-wise prefix of request N+1, for all three turns."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "persist"})

    versions = ["v1", "v2", "v3"]
    for i, prompt in enumerate(["first", "second", "third"]):
        hooks = ScriptedHooks(
            {
                PROVIDER_REQUEST: ScriptedHookResult(
                    action="inject_context",
                    ephemeral=True,
                    context_injection=f"<system-reminder>{versions[i]}</system-reminder>",
                    context_injection_role="user",
                ),
            }
        )
        await orch.execute(
            prompt=prompt,
            context=ctx,  # type: ignore[arg-type]
            providers={"main": provider},
            tools={},
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )

    assert len(provider.requests) == 3

    def sig(msg) -> tuple:
        return (
            msg.role,
            msg.content if isinstance(msg.content, str) else str(msg.content),
        )

    reqs = [[sig(m) for m in r.messages] for r in provider.requests]
    for i in range(len(reqs) - 1):
        shorter, longer = reqs[i], reqs[i + 1]
        assert len(shorter) < len(longer), f"turn {i} must be shorter than turn {i + 1}"
        assert longer[: len(shorter)] == shorter, (
            f"turn {i}'s request must be an exact element-wise prefix of "
            f"turn {i + 1}'s. turn{i}={shorter!r} turn{i + 1}={longer!r}"
        )


# ---------------------------------------------------------------------------
# T-W1-07 -- provider:request fires exactly once per LLM call.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_07_provider_request_fires_exactly_once_per_llm_call() -> None:
    """The turn-start emit (Option D) must not cause a SECOND
    provider:request emission for iteration 1 -- the in-loop emit is
    skipped via self._turn_start_request_spent. Across a 3-round
    tool-calling turn (3 LLM calls total), provider:request must be
    observed exactly 3 times, not 4."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=2)  # 3 total completions
    result = ScriptedHookResult(
        action="inject_context",
        ephemeral=True,
        context_injection="<system-reminder>stable</system-reminder>",
        context_injection_role="user",
    )
    hooks = SequencedHooks({PROVIDER_REQUEST: [result, result, result]})
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({})

    await orch.execute(
        prompt="start",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={"mock_tool": OneShotTool()},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert provider.call_count == 3
    provider_request_emits = [
        payload for name, payload in hooks.emitted if name == PROVIDER_REQUEST
    ]
    assert len(provider_request_emits) == 3, (
        f"Expected exactly 3 provider:request emits (one per LLM call), got "
        f"{len(provider_request_emits)}: {provider_request_emits!r}"
    )
    # The first emit carries the turn_start discriminator.
    assert provider_request_emits[0].get("phase") == "turn_start"
    assert provider_request_emits[0].get("iteration") == 1
    # Subsequent emits are the normal in-loop shape (no phase key).
    assert "phase" not in provider_request_emits[1]
    assert "phase" not in provider_request_emits[2]


# ---------------------------------------------------------------------------
# T-W1-08 -- no reminder message ever inserted between tool_use/tool_result.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_08_never_splits_tool_use_and_tool_result() -> None:
    """Across a multi-round tool-calling turn with a CHANGING mid-loop
    injection (arriving via tool:post -> _pending_ephemeral_injections),
    canonical history must never place a reminder message between an
    assistant message carrying tool_calls and its corresponding tool-result
    message(s)."""
    from amplifier_core import ToolResult
    from amplifier_core.events import TOOL_POST

    ctx = MockContext()
    provider = NRoundToolProvider(n_tool_rounds=2)  # 3 total completions

    class _CountingTool:
        name = "mock_tool"
        description = "test tool"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, arguments):
            return ToolResult(success=True, output="tool output")

    call_n = 0

    class _ChangingToolPostHooks:
        def __init__(self) -> None:
            self.emitted: list[tuple[str, dict]] = []

        async def emit(self, event_name: str, payload: dict | None = None):
            nonlocal call_n
            self.emitted.append((event_name, payload or {}))
            if event_name == TOOL_POST:
                call_n += 1
                return ScriptedHookResult(
                    action="inject_context",
                    ephemeral=True,
                    context_injection=f"<system-reminder>note-{call_n}</system-reminder>",
                    context_injection_role="user",
                )
            return ScriptedHookResult()

    hooks = _ChangingToolPostHooks()
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "persist"})

    await orch.execute(
        prompt="start",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={"mock_tool": _CountingTool()},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert provider.call_count == 3

    # Walk canonical history: whenever we see an assistant message carrying
    # tool_calls, the IMMEDIATELY following message(s) up to the next
    # non-tool message must ALL be role=="tool" -- never a reminder.
    history = ctx.add_message_calls
    i = 0
    saw_a_tool_round = False
    while i < len(history):
        msg = history[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            saw_a_tool_round = True
            j = i + 1
            # At least one tool-result message must follow immediately.
            assert j < len(history) and history[j].get("role") == "tool", (
                f"Expected a tool-result message immediately after the "
                f"tool_calls-bearing assistant message at index {i}; got "
                f"{history[j] if j < len(history) else '<end of history>'!r}"
            )
            while j < len(history) and history[j].get("role") == "tool":
                j += 1
            i = j
        else:
            i += 1
    assert saw_a_tool_round, "Test setup must actually exercise a tool round"


# ---------------------------------------------------------------------------
# T-W1-09 -- budget-warning message carries metadata.ephemeral (latent bug).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_09_budget_warning_message_carries_ephemeral_metadata() -> None:
    """FAILS BEFORE this spec's fix: pre-spec, the budget-warning message
    (`orchestrator-loop-limit`) was written with metadata=None. OpenAI's
    reasoning cutoff (`max(idx for non-ephemeral user)`) would then count it
    as a REAL user turn and collapse the reasoning-replay window for the
    rest of the turn. This test pins the fix: the message must carry
    metadata={"ephemeral": True, "persisted": True, "reminder_placement":
    "tail"} and must be wrapped in the envelope."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    # max_iterations=3, budget_warn_ratio default 0.8 -> warning fires once
    # iteration >= max(1, int(3*0.8)) == max(1,2) == 2.
    provider = NRoundToolProvider(n_tool_rounds=10)  # keep calling tools
    hooks = ScriptedHooks({PROVIDER_REQUEST: ScriptedHookResult()})
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator(
        {"max_iterations": 3, "ephemeral_injection_mode": "persist"}
    )

    await orch.execute(
        prompt="start",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={"mock_tool": OneShotTool()},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    budget_msgs = [
        m
        for m in ctx.add_message_calls
        if isinstance(m.get("content"), str)
        and "orchestrator-loop-limit" in m["content"]
    ]
    assert len(budget_msgs) == 1, (
        f"Expected exactly one budget-warning message; got {budget_msgs!r}"
    )
    assert budget_msgs[0]["content"].startswith("<system-reminders>"), (
        "Budget-warning message must be wrapped in the envelope"
    )
    assert budget_msgs[0]["metadata"] == {
        "ephemeral": True,
        "persisted": True,
        "reminder_placement": "tail",
    }, (
        "Budget-warning message must carry metadata.ephemeral=True (the "
        f"T-W1-09 fix) -- got {budget_msgs[0].get('metadata')!r}"
    )


# ---------------------------------------------------------------------------
# T-W1-10 -- role is "user" even when every contributing hook asked for
# "system".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_10_role_pinned_to_user_despite_all_system_requests() -> None:
    """Reproduces the exact defect chain from spec sec 1.2: if EVERY
    contributing hook requests context_injection_role="system" (the
    HookResult default, and what a bare/unfixed contributor like
    routing-matrix gets by omission), the orchestrator must still pin the
    role to "user" -- defusing the cache-prefix-rewrite bomb on
    anthropic/openai where a system-role message mid-conversation gets
    hoisted into the cached system block."""
    from amplifier_core.events import PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROVIDER_REQUEST: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="Active routing matrix: anthropic",
                context_injection_role="system",  # the accident (spec sec 1.2)
            ),
        }
    )
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({})

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
        if isinstance(m.get("content"), str) and "Active routing matrix" in m["content"]
    ]
    assert len(persisted) == 1
    assert persisted[0]["role"] == "user", (
        f"Role must be pinned to 'user' regardless of the system-role "
        f"request; got role={persisted[0]['role']!r}"
    )

    sent = provider.requests[0].messages
    injected = [
        m
        for m in sent
        if isinstance(m.content, str) and "Active routing matrix" in m.content
    ]
    assert len(injected) == 1
    assert injected[0].role == "user"


# ---------------------------------------------------------------------------
# T-W1-11 -- reminder_placement="tail" reproduces pre-spec ordering exactly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_11_tail_placement_reproduces_pre_spec_ordering() -> None:
    """With `reminder_placement: "tail"` explicitly set, the turn-start
    reminder assembly must be SKIPPED entirely: no turn_start emit, no
    turn-start persist/splice -- the in-loop iteration-1 provider:request
    emit fires normally, AFTER the user message, exactly as pre-spec."""
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
    orch = StreamingOrchestrator(
        {"ephemeral_injection_mode": "persist", "reminder_placement": "tail"}
    )

    await orch.execute(
        prompt="hello",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    # Exactly ONE provider:request emit -- the in-loop one (no turn-start
    # emit at all when reminder_placement == "tail").
    provider_request_emits = [
        payload for name, payload in hooks.emitted if name == PROVIDER_REQUEST
    ]
    assert len(provider_request_emits) == 1
    assert "phase" not in provider_request_emits[0], (
        "No turn_start phase discriminator should appear when "
        "reminder_placement == 'tail'"
    )

    canonical = [(m.get("role"), m.get("content")) for m in ctx.add_message_calls]
    block_idx = next(
        i for i, (_, c) in enumerate(canonical) if isinstance(c, str) and "v1" in c
    )
    user_idx = next(i for i, (_, c) in enumerate(canonical) if c == "hello")
    assert block_idx > user_idx, (
        "reminder_placement='tail' must reproduce pre-spec ordering: block "
        f"AFTER the user's message. Got order: {canonical!r}"
    )


# ---------------------------------------------------------------------------
# T-W1-12 -- unknown reminder_placement warns and falls back to "pre_user".
# ---------------------------------------------------------------------------


def test_t_w1_12_unknown_reminder_placement_falls_back_with_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        orch = StreamingOrchestrator({"reminder_placement": "nonsense"})

    assert orch._reminder_placement == "pre_user"
    assert any(
        "nonsense" in record.message and "reminder_placement" in record.message
        for record in caplog.records
    ), (
        f"Expected a warning naming the bad value; got: "
        f"{[r.message for r in caplog.records]!r}"
    )


def test_reminder_placement_defaults_to_pre_user() -> None:
    orch = StreamingOrchestrator({})
    assert orch._reminder_placement == "pre_user"


def test_reminder_placement_accepts_explicit_tail() -> None:
    orch = StreamingOrchestrator({"reminder_placement": "tail"})
    assert orch._reminder_placement == "tail"


# ---------------------------------------------------------------------------
# T-W1-14 -- prompt:submit content precedes provider:request content in the
# merged turn-start block (fixes the sec 1.3 order inversion).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_14_prompt_submit_content_precedes_provider_request_content() -> (
    None
):
    """Reproduces and fixes the order inversion from session 0629f373 (sec
    1.3): prompt:submit fires FIRST (at :2754, pre-spec) but its content
    used to land LAST, because the stash drained after the tail-appended
    provider:request block. After Option D, both are merged at turn start
    with prompt:submit's content FIRST -- matching emission order."""
    from amplifier_core.events import PROMPT_SUBMIT, PROVIDER_REQUEST

    ctx = MockContext()
    provider = RequestCapturingProvider()

    class _MultiEventHooks:
        def __init__(self) -> None:
            self.emitted: list[tuple[str, dict]] = []

        async def emit(self, event_name: str, payload: dict | None = None):
            self.emitted.append((event_name, payload or {}))
            if event_name == PROMPT_SUBMIT:
                return ScriptedHookResult(
                    action="inject_context",
                    ephemeral=True,
                    context_injection="WAYFINDER-CONTENT",
                    context_injection_role="user",
                )
            if event_name == PROVIDER_REQUEST:
                return ScriptedHookResult(
                    action="inject_context",
                    ephemeral=True,
                    context_injection="ROUTING-MATRIX-CONTENT",
                    context_injection_role="system",
                )
            return ScriptedHookResult()

    hooks = _MultiEventHooks()
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({})

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
        and "WAYFINDER-CONTENT" in m["content"]
        and "ROUTING-MATRIX-CONTENT" in m["content"]
    ]
    assert len(persisted) == 1, (
        "Expected ONE merged turn-start block containing both sources, got "
        f"{ctx.add_message_calls!r}"
    )
    block = persisted[0]["content"]
    wayfinder_pos = block.index("WAYFINDER-CONTENT")
    routing_pos = block.index("ROUTING-MATRIX-CONTENT")
    assert wayfinder_pos < routing_pos, (
        "prompt:submit content (WAYFINDER-CONTENT) must precede "
        "provider:request content (ROUTING-MATRIX-CONTENT) in the merged "
        f"block -- fixing the sec 1.3 order inversion. Block: {block!r}"
    )


# ---------------------------------------------------------------------------
# T-W1-15 -- append_to_last_tool_result stash is NOT consumed at turn start
# and still concatenates in-loop.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t_w1_15_append_to_last_tool_result_stash_survives_turn_start() -> None:
    """A prompt:submit result with append_to_last_tool_result=True has no
    tool result to attach to at turn start (no tool round has happened yet
    this turn) -- the turn-start merge must NOT swallow it. It stays queued
    in self._pending_ephemeral_injections and is handled by the normal
    in-loop pending-drain, which (per the module's pre-existing, unchanged
    fallback rule) appends it as its own message when no tool-result
    message exists yet to concatenate into -- exactly the same fallback
    documented at the direct provider:request append site.

    (`ephemeral_injection_mode="tail"` -- append_to_last_tool_result is
    only a tail-mode concept; persist mode folds every pending injection
    into one canonical message regardless of the flag, see the
    orchestrator's own comment at that call site.)"""
    from amplifier_core.events import PROMPT_SUBMIT

    ctx = MockContext()
    provider = RequestCapturingProvider()
    hooks = ScriptedHooks(
        {
            PROMPT_SUBMIT: ScriptedHookResult(
                action="inject_context",
                ephemeral=True,
                context_injection="append-me-to-tool-result",
                context_injection_role="user",
                append_to_last_tool_result=True,
            ),
        }
    )
    coordinator = MockCoordinator()
    orch = StreamingOrchestrator({"ephemeral_injection_mode": "tail"})

    await orch.execute(
        prompt="continue",
        context=ctx,  # type: ignore[arg-type]
        providers={"main": provider},
        tools={},
        hooks=hooks,  # type: ignore[arg-type]
        coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert len(provider.requests) == 1
    sent = provider.requests[0].messages

    # The turn-start merge (if any) must NOT contain this content -- it
    # only ever appears via the SEPARATE in-loop pending-drain fallback,
    # never folded into the pre_user turn-start block.
    non_tool_result_hits = [
        m
        for m in sent
        if isinstance(m.content, str) and "append-me-to-tool-result" in m.content
    ]
    assert len(non_tool_result_hits) == 1, (
        f"Expected the stash content to appear exactly once (via the "
        f"in-loop pending-drain, not the turn-start merge); got {sent!r}"
    )
    # It must be enveloped and never have been silently dropped.
    assert non_tool_result_hits[0].content.startswith("<system-reminders>")

    # self._pending_ephemeral_injections must be drained (empty) afterward
    # -- proven indirectly: exactly one occurrence total, never duplicated
    # on a hypothetical next call. (Direct attribute check for clarity.)
    assert orch._pending_ephemeral_injections == []
