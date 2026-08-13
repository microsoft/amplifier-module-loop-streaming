"""Tests for the conversation-scope provider pin.

Covers the `conversation.provider_pin` coordinator capability
(ConversationProviderPin) and the `_select_provider` behavior it drives:

  1. mount() registers the capability (an app can detect its presence).
  2. Capability surface: available/current/pin/unpin round-trip.
  3. Fail-loud path (a): pinning an UNMOUNTED name raises at PIN TIME,
     synchronously, naming what IS mounted.
  4. Fail-loud path (b): a pin that was valid at pin time but whose provider
     is GONE at turn time raises rather than silently falling back to
     priority ordering.
  5. Unpinned selection is unchanged (priority ordering still wins).
  6. `provider:resolve` reports basis="pinned" vs basis="priority".
  7. SCOPE: the pin does not leak into `_resolve_goal_model` (goal loop /
     stall judge / summarizer), and nothing is unmounted.
"""

from __future__ import annotations

from typing import Any

import pytest
from amplifier_core import HookRegistry

from amplifier_module_loop_streaming import (
    ConversationProviderPin,
    StreamingOrchestrator,
)

# ---------------------------------------------------------------------------
# Stubs -- minimal and self-contained (pattern mirrors tests/test_steering.py
# and tests/test_error_propagation.py).
# ---------------------------------------------------------------------------


class StubProvider:
    """Provider stub conforming to the bits of the kernel Provider contract
    this feature touches.

    `priority` is what `_select_provider`'s unpinned path sorts on (lower
    wins), so tests can construct an unambiguous "the pin overrode the
    priority winner" scenario.

    `vendor` is what `get_info().id` reports -- the kernel's vendor
    identity, which the cross-vendor guard compares. It defaults to
    "anthropic" for every stub so that, unless a test says otherwise, all
    providers are SAME-VENDOR and exercise the supported switching case.
    """

    def __init__(
        self,
        label: str,
        priority: int = 100,
        text: str = "ok",
        vendor: str = "anthropic",
        default_model: str | None = None,
    ) -> None:
        self.label = label
        self.priority = priority
        self._text = text
        self._vendor = vendor
        self._default_model = default_model

    @property
    def name(self) -> str:
        return self.label

    def get_info(self) -> Any:
        from amplifier_core.models import ProviderInfo

        return ProviderInfo(
            id=self._vendor,
            display_name=self.label,
            defaults=(
                {"model": self._default_model}
                if self._default_model is not None
                else {}
            ),
        )

    async def complete(self, request: Any, **kwargs: Any) -> Any:
        class _Resp:
            content = None
            content_blocks = None
            usage = None
            metadata = None

        resp = _Resp()
        resp.text = self._text  # type: ignore[attr-defined]
        return resp

    def parse_tool_calls(self, response: Any) -> list[Any]:
        return []


class NoGetInfoProvider(StubProvider):
    """Provider whose vendor identity cannot be established at all."""

    def get_info(self) -> Any:
        raise RuntimeError("get_info exploded")


class BlankVendorProvider(StubProvider):
    """Provider whose get_info() works but yields no usable vendor id."""

    def get_info(self) -> Any:
        from amplifier_core.models import ProviderInfo

        return ProviderInfo(id="   ", display_name=self.label)


class StubContext:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def get_messages_for_request(self, provider: Any = None) -> list[dict]:
        return [{"role": "user", "content": "test"}]

    async def add_message(self, msg: dict) -> None:
        self.messages.append(msg)


class StubCoordinator:
    """Duck-typed coordinator carrying the surface this feature touches:
    the `providers` mount point (read by ConversationProviderPin.available),
    capability register/get, and the bits execute() reads.
    """

    def __init__(self, providers: dict[str, Any] | None = None) -> None:
        # Holds the SAME dict object the test passed in, not a copy -- the
        # real `providers` mount point is live, so a test that simulates an
        # unmount by mutating its own dict must be reflected here too.
        self._mounts: dict[str, Any] = {
            "providers": providers if providers is not None else {}
        }
        self._capabilities: dict[str, Any] = {}
        self.session_state: dict[str, Any] = {}
        self.contributors: list[tuple] = []

    # --- mount points ---
    async def mount(self, mount_point: str, module: Any, name: str | None = None):
        if name is None:
            self._mounts[mount_point] = module
        else:
            self._mounts.setdefault(mount_point, {})[name] = module

    def get(self, mount_point: str, name: str | None = None) -> Any:
        entry = self._mounts.get(mount_point)
        if name is None:
            return entry
        return (entry or {}).get(name)

    # --- capabilities ---
    def register_capability(self, name: str, value: Any) -> None:
        self._capabilities[name] = value

    def get_capability(self, name: str) -> Any:
        return self._capabilities.get(name)

    # --- misc ---
    def register_contributor(self, channel: str, name: str, callback) -> None:
        self.contributors.append((channel, name, callback))

    async def process_hook_result(self, result, *args, **kwargs):
        return result


def _make_pin(providers: dict[str, Any]) -> tuple[StreamingOrchestrator, Any]:
    orch = StreamingOrchestrator({"max_iterations": 5, "stream_delay": 0})
    coordinator = StubCoordinator(providers)
    pin = ConversationProviderPin(orch, coordinator)
    return orch, pin


def _resolve_events(hooks_events: list[tuple[str, dict]]) -> list[dict]:
    return [d for e, d in hooks_events if e == "provider:resolve"]


# ---------------------------------------------------------------------------
# 1. mount() registers the capability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mount_registers_conversation_provider_pin_capability() -> None:
    """An app must be able to FIND the capability -- that is what lets it
    detect absence under another orchestrator and refuse loudly."""
    from amplifier_module_loop_streaming import mount

    coordinator = StubCoordinator({"anthropic": StubProvider("anthropic")})
    await mount(coordinator, {})

    cap = coordinator.get_capability("conversation.provider_pin")
    assert cap is not None
    assert isinstance(cap, ConversationProviderPin)
    # Pre-existing capability still registered (no regression).
    assert coordinator.get_capability("session.steer") is not None


# ---------------------------------------------------------------------------
# 2. Capability surface round-trip
# ---------------------------------------------------------------------------


def test_available_lists_mounted_provider_names_sorted() -> None:
    _orch, pin = _make_pin(
        {"openai-gpt5": StubProvider("openai"), "anthropic": StubProvider("anthropic")}
    )
    assert pin.available() == ["anthropic", "openai-gpt5"]


def test_pin_unpin_current_round_trip() -> None:
    orch, pin = _make_pin(
        {"anthropic": StubProvider("anthropic"), "openai": StubProvider("openai")}
    )

    assert pin.current() is None, "starts unpinned"

    assert pin.pin("openai") == "openai"
    assert pin.current() == "openai"
    assert orch._pinned_provider_name == "openai"

    assert pin.unpin() == "openai", "unpin returns what it cleared"
    assert pin.current() is None
    assert orch._pinned_provider_name is None


def test_unpin_is_idempotent_and_not_an_error() -> None:
    _orch, pin = _make_pin({"anthropic": StubProvider("anthropic")})
    assert pin.unpin() is None
    assert pin.unpin() is None
    assert pin.current() is None


def test_pin_survives_across_execute_calls() -> None:
    """A pin is a session-lifetime decision -- execute() resets the goal
    model cache but must NOT reset the pin."""
    orch, pin = _make_pin({"anthropic": StubProvider("anthropic")})
    pin.pin("anthropic")
    # Simulate what execute() resets at the top of a run.
    orch._goal_model_cache = None
    orch._goal_model_basis = None
    assert pin.current() == "anthropic", "pin must survive a turn boundary"


# ---------------------------------------------------------------------------
# 3. Fail-loud (a): pin time, synchronous
# ---------------------------------------------------------------------------


def test_pin_unmounted_provider_raises_at_pin_time_naming_what_is_mounted() -> None:
    _orch, pin = _make_pin(
        {"anthropic": StubProvider("anthropic"), "openai": StubProvider("openai")}
    )

    with pytest.raises(ValueError) as exc:
        pin.pin("does-not-exist")

    msg = str(exc.value)
    assert "does-not-exist" in msg
    # Must name what IS mounted so the caller can correct itself.
    assert "anthropic" in msg
    assert "openai" in msg
    # And must NOT have quietly pinned anything.
    assert pin.current() is None


def test_pin_empty_name_raises() -> None:
    _orch, pin = _make_pin({"anthropic": StubProvider("anthropic")})
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            pin.pin(bad)  # type: ignore[arg-type]
    assert pin.current() is None


def test_failed_pin_does_not_disturb_an_existing_pin() -> None:
    _orch, pin = _make_pin(
        {"anthropic": StubProvider("anthropic"), "openai": StubProvider("openai")}
    )
    pin.pin("anthropic")
    with pytest.raises(ValueError):
        pin.pin("nope")
    assert pin.current() == "anthropic", "a rejected pin must leave state alone"


# ---------------------------------------------------------------------------
# 4 + 5. Selection: pin wins over priority; unpinned unchanged; gone => raise
# ---------------------------------------------------------------------------


def test_unpinned_selection_still_picks_lowest_priority_number() -> None:
    """Regression guard: unpinned behavior must be what it was before."""
    orch = StreamingOrchestrator({})
    best = StubProvider("best", priority=1)
    worst = StubProvider("worst", priority=99)
    providers = {"worst": worst, "best": best}

    assert orch._select_provider(providers) is best


def test_unpinned_empty_providers_still_returns_none() -> None:
    orch = StreamingOrchestrator({})
    assert orch._select_provider({}) is None


def test_pin_overrides_priority_ordering() -> None:
    """The whole point: the pinned provider wins even though another
    provider has a better (lower) priority number."""
    best = StubProvider("best", priority=1)
    pinned_but_worse = StubProvider("worse", priority=99)
    providers = {"best": best, "worse": pinned_but_worse}

    orch, pin = _make_pin(providers)
    pin.pin("worse")

    assert orch._select_provider(providers) is pinned_but_worse


def test_pinned_provider_gone_at_turn_time_raises_and_does_not_fall_back() -> None:
    """Fail-loud (b): mounted at pin time, unmounted before the turn.

    Silently answering on a different provider is the exact failure this
    feature exists to prevent, so this must raise -- NOT return the
    priority winner.
    """
    survivor = StubProvider("survivor", priority=1)
    doomed = StubProvider("doomed", priority=50)

    orch, pin = _make_pin({"survivor": survivor, "doomed": doomed})
    pin.pin("doomed")  # valid at pin time

    # ... someone unmounts it; the turn sees only `survivor`.
    with pytest.raises(RuntimeError) as exc:
        orch._select_provider({"survivor": survivor})

    msg = str(exc.value)
    assert "doomed" in msg
    assert "survivor" in msg, "names what IS mounted now"
    assert "conversation.provider_pin" in msg, "tells the caller how to recover"


def test_pinned_provider_with_no_providers_at_all_raises() -> None:
    orch, pin = _make_pin({"only": StubProvider("only")})
    pin.pin("only")
    with pytest.raises(RuntimeError):
        orch._select_provider({})


# ---------------------------------------------------------------------------
# 6. provider:resolve basis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_resolve_reports_basis_pinned() -> None:
    best = StubProvider("best", priority=1, text="from-best")
    pinned = StubProvider("pinned", priority=99, text="from-pinned")
    providers = {"best": best, "pinned": pinned}

    orch, pin = _make_pin(providers)
    pin.pin("pinned")

    hooks = HookRegistry()
    events: list[tuple[str, dict]] = []

    async def capture(event, data):
        events.append((event, data))

    hooks.register("provider:resolve", capture)

    result = await orch.execute(
        prompt="hi",
        context=StubContext(),
        providers=providers,
        tools={},
        hooks=hooks,
    )

    assert result == "from-pinned", "the pinned provider actually answered"
    resolves = _resolve_events(events)
    conversation = [d for d in resolves if d.get("scope") == "conversation"]
    assert len(conversation) == 1
    assert conversation[0]["basis"] == "pinned"
    assert conversation[0]["provider"] == "pinned"


@pytest.mark.asyncio
async def test_provider_resolve_reports_basis_priority_when_unpinned() -> None:
    best = StubProvider("best", priority=1, text="from-best")
    other = StubProvider("other", priority=99, text="from-other")
    providers = {"best": best, "other": other}

    orch = StreamingOrchestrator({"max_iterations": 5, "stream_delay": 0})

    hooks = HookRegistry()
    events: list[tuple[str, dict]] = []

    async def capture(event, data):
        events.append((event, data))

    hooks.register("provider:resolve", capture)

    result = await orch.execute(
        prompt="hi",
        context=StubContext(),
        providers=providers,
        tools={},
        hooks=hooks,
    )

    assert result == "from-best"
    conversation = [
        d for d in _resolve_events(events) if d.get("scope") == "conversation"
    ]
    assert len(conversation) == 1
    assert conversation[0]["basis"] == "priority"
    assert conversation[0]["provider"] == "best"


@pytest.mark.asyncio
async def test_stale_pin_surfaces_as_error_through_execute() -> None:
    """The RuntimeError from _select_provider must propagate out of
    execute() (same fail-loud path as a provider error), not be swallowed
    into response text."""
    survivor = StubProvider("survivor", priority=1)
    orch, pin = _make_pin({"survivor": survivor, "doomed": StubProvider("doomed")})
    pin.pin("doomed")

    hooks = HookRegistry()
    events: list[tuple[str, dict]] = []

    async def capture(event, data):
        events.append((event, data))

    hooks.register("orchestrator:complete", capture)

    with pytest.raises(RuntimeError, match="doomed"):
        await orch.execute(
            prompt="hi",
            context=StubContext(),
            providers={"survivor": survivor},
            tools={},
            hooks=hooks,
        )

    complete = [d for e, d in events if e == "orchestrator:complete"]
    assert len(complete) == 1
    assert complete[0]["status"] == "error"


# ---------------------------------------------------------------------------
# 7. SCOPE: goal-loop resolution and the mounted set are untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pin_does_not_affect_resolve_goal_model() -> None:
    """HARD REQUIREMENT: the pin is conversation-scope ONLY.

    With no role resolver and no preferences, `_resolve_goal_model` falls
    back to the session default = the FIRST-listed provider. Pinning a
    DIFFERENT provider must not change that answer.
    """
    default = StubProvider("default", priority=99)
    pinned = StubProvider("pinned", priority=1)
    providers = {"default": default, "pinned": pinned}  # insertion order matters

    orch, pin = _make_pin(providers)
    coordinator = StubCoordinator(providers)  # no model_role_resolver registered

    (
        before_name,
        before_provider,
        before_model,
        before_config,
    ) = await orch._resolve_goal_model(providers, coordinator)
    assert before_name == "default"
    assert before_provider is default

    # Now pin the OTHER provider and re-resolve from scratch.
    pin.pin("pinned")
    orch._goal_model_cache = None
    orch._goal_model_basis = None

    (
        after_name,
        after_provider,
        after_model,
        after_config,
    ) = await orch._resolve_goal_model(providers, coordinator)

    assert after_name == before_name == "default", (
        "goal-loop resolution must ignore the conversation pin"
    )
    assert after_provider is default
    assert after_model == before_model
    assert after_config == before_config
    assert orch._goal_model_basis == "session_default_fallback"


def test_pin_does_not_unmount_anything() -> None:
    """The pin changes SELECTION, not the mounted set -- every provider
    stays available to role resolution and sub-agent spawning."""
    providers = {"a": StubProvider("a"), "b": StubProvider("b")}
    _orch, pin = _make_pin(providers)

    pin.pin("b")

    assert set(providers) == {"a", "b"}, "mounted dict untouched"
    assert pin.available() == ["a", "b"], "both still mounted per the coordinator"


# ---------------------------------------------------------------------------
# 8. CROSS-VENDOR GUARD
#
# Shipping blocker, measured in a real DTU cross-vendor test:
#   anthropic -> openai   worked
#   openai    -> gemini   HTTP 400 (missing thought_signature)
#   gemini    -> anthropic HTTP 400 (bad thinking.signature) AND the
#                          conversation was permanently wedged thereafter.
# ---------------------------------------------------------------------------


def test_same_vendor_pin_across_different_mount_names_is_allowed() -> None:
    """THE PRIMARY, PROVEN USE CASE -- must not regress.

    Three distinct mount names, one vendor id. Switching among them is
    exactly what the pin is for, and none of the mount names contain a
    parseable vendor token (`anthropic-fable`), which is why identity comes
    from get_info().id and not from the name.
    """
    providers = {
        "anthropic-sonnet": StubProvider("sonnet", priority=1, vendor="anthropic"),
        "anthropic-haiku": StubProvider("haiku", priority=2, vendor="anthropic"),
        "anthropic-fable": StubProvider("fable", priority=3, vendor="anthropic"),
    }
    orch, pin = _make_pin(providers)

    assert pin.pin("anthropic-haiku") == "anthropic-haiku"
    assert pin.current() == "anthropic-haiku"
    assert orch._select_provider(providers) is providers["anthropic-haiku"]

    # And again, pin-to-pin within the same vendor.
    assert pin.pin("anthropic-fable") == "anthropic-fable"
    assert orch._select_provider(providers) is providers["anthropic-fable"]


def test_cross_vendor_pin_is_refused_with_an_explanatory_message() -> None:
    """Refused at PIN TIME, and the message must explain WHY -- naming both
    vendors and the wedging mechanism, not just saying 'no'."""
    providers = {
        "anthropic-sonnet": StubProvider("sonnet", priority=1, vendor="anthropic"),
        "gemini-pro": StubProvider("gemini", priority=2, vendor="gemini"),
    }
    _orch, pin = _make_pin(providers)

    with pytest.raises(ValueError) as exc:
        pin.pin("gemini-pro")

    msg = str(exc.value)
    assert "anthropic" in msg, "names the vendor being switched FROM"
    assert "gemini" in msg, "names the vendor being switched TO"
    assert "not supported yet" in msg
    # Explains the mechanism, so the user understands it is not arbitrary.
    assert "thought_signature" in msg or "thinking.signature" in msg
    assert "wedg" in msg.lower(), "says what the consequence would be"
    # Nothing was pinned.
    assert pin.current() is None


def test_cross_vendor_refusal_lists_same_vendor_alternatives() -> None:
    """The refusal points at what the caller CAN do."""
    providers = {
        "anthropic-sonnet": StubProvider("sonnet", priority=1, vendor="anthropic"),
        "anthropic-haiku": StubProvider("haiku", priority=2, vendor="anthropic"),
        "openai-gpt5": StubProvider("gpt5", priority=3, vendor="openai"),
    }
    _orch, pin = _make_pin(providers)

    with pytest.raises(ValueError) as exc:
        pin.pin("openai-gpt5")

    msg = str(exc.value)
    assert "anthropic-sonnet" in msg
    assert "anthropic-haiku" in msg
    assert "openai-gpt5" not in msg.split("same-vendor options here:")[-1]


def test_cross_vendor_refused_even_with_no_conversation_history() -> None:
    """DECISION UNDER TEST: refuse UNCONDITIONALLY, with no empty-history
    exception.

    A brand-new orchestrator that has never run a turn still refuses. See
    ConversationProviderPin.pin's docstring: allowing this would make the
    NEXT state unsafe -- once conversed through, unpin() becomes the
    cross-vendor switch, and guarding unpin would trap the user inside the
    pin with no exit.
    """
    providers = {
        "anthropic-sonnet": StubProvider("sonnet", priority=1, vendor="anthropic"),
        "gemini-pro": StubProvider("gemini", priority=2, vendor="gemini"),
    }
    orch, pin = _make_pin(providers)

    # Nothing has happened at all: no turns, no messages, no pin.
    assert pin.current() is None
    assert orch._pinned_provider_name is None

    with pytest.raises(ValueError, match="not supported yet"):
        pin.pin("gemini-pro")


def test_cross_vendor_refused_from_an_existing_pin_too() -> None:
    """The reference vendor is the CURRENT conversation provider -- which is
    the pin when one is set, not the priority winner."""
    providers = {
        "openai-gpt5": StubProvider("gpt5", priority=99, vendor="openai"),
        "anthropic-sonnet": StubProvider("sonnet", priority=1, vendor="anthropic"),
        "anthropic-haiku": StubProvider("haiku", priority=2, vendor="anthropic"),
    }
    _orch, pin = _make_pin(providers)

    # Pin within anthropic (priority winner is anthropic-sonnet).
    pin.pin("anthropic-haiku")
    # Now attempting to cross to openai must still be refused, measured
    # against the PIN (anthropic-haiku), not the priority winner.
    with pytest.raises(ValueError) as exc:
        pin.pin("openai-gpt5")
    assert "anthropic-haiku" in str(exc.value)
    assert pin.current() == "anthropic-haiku", "refusal left the pin intact"


def test_repinning_the_currently_effective_provider_is_always_allowed() -> None:
    """Re-pinning whatever is already answering changes no vendor, so it is
    allowed even when that provider cannot report its identity."""
    providers = {"broken": NoGetInfoProvider("broken", priority=1)}
    _orch, pin = _make_pin(providers)

    assert pin.pin("broken") == "broken"
    assert pin.current() == "broken"
    # Idempotent re-pin still fine.
    assert pin.pin("broken") == "broken"


def test_unverifiable_vendor_is_refused_not_guessed() -> None:
    """get_info() raising is NOT permission to proceed."""
    providers = {
        "anthropic-sonnet": StubProvider("sonnet", priority=1, vendor="anthropic"),
        "mystery": NoGetInfoProvider("mystery", priority=2),
    }
    _orch, pin = _make_pin(providers)

    with pytest.raises(ValueError) as exc:
        pin.pin("mystery")

    msg = str(exc.value)
    assert "mystery" in msg
    assert "get_info()" in msg
    assert "guess" in msg.lower(), "says it is refusing rather than guessing"
    assert pin.current() is None


def test_blank_vendor_id_is_refused_not_guessed() -> None:
    """get_info() succeeding but yielding no usable id is also refused."""
    providers = {
        "anthropic-sonnet": StubProvider("sonnet", priority=1, vendor="anthropic"),
        "blank": BlankVendorProvider("blank", priority=2),
    }
    _orch, pin = _make_pin(providers)

    with pytest.raises(ValueError) as exc:
        pin.pin("blank")

    assert "no usable vendor id" in str(exc.value)
    assert pin.current() is None


def test_vendor_comparison_is_case_insensitive() -> None:
    """Same vendor reported with different casing is still the same vendor
    -- must not be refused on a cosmetic difference."""
    providers = {
        "a": StubProvider("a", priority=1, vendor="anthropic"),
        "b": StubProvider("b", priority=2, vendor="Anthropic"),
    }
    _orch, pin = _make_pin(providers)

    assert pin.pin("b") == "b"


def test_stale_pin_refuses_new_pin_and_points_at_unpin() -> None:
    """When the pinned provider is gone, the vendor owning the transcript
    cannot be verified, so a NEW pin is refused -- and the message names
    unpin() as the escape (unpin is unguarded and always works)."""
    providers = {
        "anthropic-sonnet": StubProvider("sonnet", priority=1, vendor="anthropic"),
        "doomed": StubProvider("doomed", priority=2, vendor="anthropic"),
    }
    orch, pin = _make_pin(providers)
    pin.pin("doomed")

    # Simulate the provider being unmounted underneath the pin.
    del providers["doomed"]

    with pytest.raises(ValueError) as exc:
        pin.pin("anthropic-sonnet")
    assert "unpin()" in str(exc.value)

    # The documented escape still works, unguarded.
    assert pin.unpin() == "doomed"
    assert orch._select_provider(providers) is providers["anthropic-sonnet"]


def test_not_mounted_check_precedes_vendor_check() -> None:
    """An unmounted name reports THAT, not a confusing vendor error."""
    providers = {"anthropic-sonnet": StubProvider("sonnet", vendor="anthropic")}
    _orch, pin = _make_pin(providers)

    with pytest.raises(ValueError, match="not mounted"):
        pin.pin("gemini-pro")
