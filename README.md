# Amplifier Streaming Loop Orchestrator Module

Token-level streaming orchestration for real-time response delivery.

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Purpose

Provides streaming orchestration that delivers LLM responses token-by-token for improved perceived performance and user experience.

## Contract

**Module Type:** Orchestrator
**Mount Point:** `orchestrators`
**Entry Point:** `amplifier_module_loop_streaming:mount`

## Behavior

- Token-level streaming from provider
- Real-time response delivery
- **Parallel tool execution**: Multiple tool calls execute concurrently
- Deterministic context updates: Results added in original order
- Progressive rendering
- Interruptible generation

## Configuration

```toml
[[orchestrators]]
module = "loop-streaming"
name = "streaming"
config = {
    max_iterations = -1,             # Maximum LLM calls for a single execute() turn
                                      # (-1 = unlimited, default). This is also the
                                      # mechanism a delegated child session's call
                                      # budget is enforced through -- see "Delegated-
                                      # session call budget (Layer 1)" below.
    budget_warn_ratio = 0.8,         # Fraction of max_iterations at which a one-shot
                                      # "start converging" system-reminder is injected
                                      # (see below). Inert when max_iterations is -1.
    goal_stall_threshold = 3,        # /goal: candidate threshold before a bounded
                                      # evidence judge allows one recovery turn
    goal_model_role = "fast",        # /goal: routing-matrix model role requested for
                                      # the evaluator/stall-judge/summary calls, via
                                      # the model_role_resolver coordinator capability
    goal_provider_preferences = [    # /goal: ordered {provider, model, config?}
        {provider = "anthropic", model = "claude-haiku-*"},        # fallback list, consulted ONLY when
        {provider = "openai", model = "gpt-?.?-luna*"},            # goal_model_role routing above didn't
        {provider = "openai", model = "gpt-?.?-mini*"},            # yield a usable, mounted provider (no
        {provider = "gemini", model = "gemini-*-flash-preview"},   # routing bundle installed, resolver
        {provider = "github-copilot", model = "claude-haiku-4.5"}, # returned no candidates, or resolved
        {provider = "github-copilot", model = "gpt-5.4-mini"},     # provider not mounted). Without this,
        {provider = "ollama", model = "*"},                        # that case falls through to the
    ],                                # session's expensive default model for every
                                      # evaluator call (one per turn) -- a cost
                                      # regression. Models are GLOB patterns, not
                                      # pinned versions, so a new release (e.g. the
                                      # next Haiku point release) is picked up
                                      # automatically the moment a provider lists it,
                                      # with no config change here. Shown above is
                                      # the built-in default (the routing matrix's
                                      # own "fast"-role membership) -- override to
                                      # change it.
    stream_delay = 0.0,              # Per-token artificial delay (seconds), for
                                      # human-facing typing animation (0.0 = off)
    extended_thinking = false,       # Enable extended thinking on the main
                                      # conversational turns (not the /goal internal
                                      # calls, which always disable it)
    min_delay_between_calls_ms = 0,  # Minimum delay between provider calls (rate
                                      # limiting; 0 = disabled)
}
```

## Ephemeral hook injections

```toml
config = {
    ephemeral_injection_mode = "persist",  # legacy-provider compatibility setting
}
```

When `context.instructions.v1` is available, its assembly advertises
`instruction_layout_authority_v1 = true`, and the selected provider opts into
both `instruction_layout_version = 1` and
`instruction_layout_authority_v1 = true`, ephemeral hook output is staged in
an execution-local instruction source. This authority extension is negotiated
at both the context and provider boundaries. It is included in exactly one
prepared request and is discarded when that response is accepted or abandoned;
it is never added to canonical conversation history.

The hook's original `context_injection_role` is preserved as instruction
authority: `system` is authoritative, while `user` and `assistant` are
advisory. Assistant-role output also logs a migration warning. All staged
records remain canonical system instructions, so a provider can preserve the
declared authority without mistaking advisory hook output for user input.

Legacy providers, or providers without the authority opt-in, retain the
existing request-view behavior. A context with marked v1 instruction history
refuses that downgrade rather than silently lowering authority.

## System-reminder envelope and placement (reminder-redesign-spec.md, W1)

```toml
config = {
    reminder_placement = "pre_user",  # "pre_user" | "tail" (default "pre_user")
}
```

**Background:** a captured production session showed a model obeying a
bare, trailing `<system-reminder>` injection instead of the user's real
request -- the reminder landed AFTER the user's message on the wire, and
the model treated the last thing it saw as "the task" rather than
supporting context. Two independent fixes address this:

1. **The envelope.** Every merged hook-injection blob this orchestrator
   writes is wrapped in `<system-reminders>...</system-reminders>` with an
   explicit instruction header telling the model these blocks are NOT from
   the user and NOT a request, and must never be treated as the task. This
   is **not** behind a flag -- it is the fix, and a flag would mean
   shipping a knob whose "off" position is the known-bad behavior. The
   envelope tag is deliberately `<system-reminders>` (not e.g.
   `<injected-context>`) so it shares the `"<system-reminder"` prefix that
   `amplifier-foundation`'s `is_real_user_message` matcher (and
   `amplifier-module-provider-openai`'s FM3 repair) already use -- one
   prefix match covers both the per-source blocks and this outer envelope.

2. **Placement (`reminder_placement`).** On the legacy route, by default
   (`"pre_user"`), the turn's reminder block is written **before** the
   user's prompt -- in canonical history for
   `ephemeral_injection_mode = "persist"` (the block precedes the user
   message as real, append-only history), or spliced into the request VIEW
   for `ephemeral_injection_mode = "tail"` (nothing persisted; the splice
   happens once, at iteration 1, and is never repeated). The v1 route
   stages reminders as request-local tail instructions instead. Legacy
   placement is achieved by hoisting iteration 1's
   `provider:request` emit to TURN START, before `context.add_message` adds
   the user's prompt -- the event payload carries `"phase": "turn_start"`
   so a hook that cares can discriminate; hooks that ignore the key behave
   exactly as before. `reminder_placement = "tail"` is the **rollback
   lever**: it skips the turn-start assembly entirely and restores the
   pre-this-feature ordering (block after the user message) -- the
   envelope, role pin, and metadata tag are still applied in `"tail"` mode;
   only the ORDERING reverts. An unknown value falls back to `"pre_user"`
   with a logged warning.

**Role preservation.** On the v1 route, the original
`context_injection_role` is explicit authority metadata: `system` remains
authoritative; `user` and `assistant` are advisory. The source records are
canonical system instructions, and never persisted. The assistant mapping
emits a migration warning. The legacy route retains its existing
user-carrier behavior for compatibility.

**"Before", not "immediately before".** On the legacy persist route, the change-gate
suppresses re-persisting an unchanged reminder block. On a turn where
nothing changed, canonical history looks like
`[block N] [user N] [assistant] [tool] [assistant] [user N+1]` -- the
block is several messages back, not adjacent to `user N+1`. **This is
correct and intended**, and is the same cache-prefix property
`ephemeral_injection_mode = "persist"` exists to guarantee (see above).
Do not "fix" this by disabling the change-gate -- that would defeat the
whole cache-prefix benefit this mode provides.

**Change-gate comparison basis.** On the legacy persist route, the change-gate always compares the RAW
(pre-envelope) merged body against the last persisted body, never the
enveloped string. This matters because the turn-start block uses the
pre-user header variant and a later mid-loop block (same iteration's
change-gate lineage) uses the tail variant -- two different headers
wrapping potentially-identical content. Comparing enveloped strings would
falsely detect a "change" the first time a turn transitions from its
turn-start block to a mid-loop one, forcing a spurious extra persisted
message on every multi-iteration turn.

**Legacy mid-loop (iterations >= 2) placement is unchanged**: new content is
still written at the tail (tail-variant envelope), the change-gate still
suppresses unchanged content, and the pending-injection drain (from
`tool:post` / a stashed `prompt:submit` result with
`append_to_last_tool_result`) still joins ALL pending injections for one
drain into a SINGLE enveloped message (or a single concatenation) rather
than one message per injection.

**Intended successor (not implemented here):** the clean end state is a
dedicated `turn:reminders` event that reminder-contributing hooks register
on explicitly, replacing the current re-use of `provider:request` with a
`phase` discriminator. That would require editing every reminder hook in
the ecosystem; hoisting the existing `provider:request` emit (as done
here) delivers the identical wire result today with zero hook edits. A
future module version may introduce `turn:reminders` as the registration
point of record.

## Delegated-session call budget (Layer 1)

`max_iterations` doubles as the enforcement mechanism for a per-session-leg
LLM-call budget (see `microsoft/amplifier-foundation`'s `tool-delegate`
module, which injects a value here via `orchestrator_config` when it spawns
a child session -- this module has no concept of "delegation" itself; it
only counts main-loop LLM calls against whatever `max_iterations` it was
given, root session or child).

**Exhaustion is a normal turn ending, not an error.** A response that ends
naturally (no tool call and no pending steer) returns immediately, including
on iteration `max_iterations`; it makes no duplicate provider call and is
not marked budget-exhausted. Only when the hard limit prevents a required
continuation does the loop make one additional `provider.complete()` call
with an injected `<system-reminder source="orchestrator-loop-limit">` asking
the agent to wrap up and summarize. Thus a budget of `N` permits at most
`N + 1` main-loop provider calls, while a natural completion uses exactly the
calls it needed.

That final request retains the ordinary generic tool declarations (including
provider-native declarations needed to validate preceding tool history), but
sets the portable `tool_choice="none"`. A compliant final response retains
safe text/thinking blocks and provider metadata in the transcript while
rendering normalized text; any unexpected tool call is neither dispatched nor
persisted structurally, so finalization cannot extend the iteration budget or
leave unpaired tool state.

`ORCHESTRATOR_COMPLETE`'s payload always carries a `metadata` bag:

```python
{
    "llm_calls": 301,             # actual main-loop provider calls, including any one finalization call (not goal-loop internal calls -- see below)
    "llm_call_budget": 300,       # the max_iterations this turn ran under, or None if unlimited
    "budget_exhausted": True,     # whether the budget prevented a needed continuation
    "resumable": True,            # whether this exit path guarantees the transcript was persisted
}
```

`status` gains a new value, `"budget_exhausted"`, with precedence
`error > cancelled > budget_exhausted > success/incomplete` -- budget
exhaustion sits above `success` because the wrap-up call fills the response
with the agent's own summary text, which would otherwise look identical to
an ordinary completed turn.

At `budget_warn_ratio` (default 80%) of `max_iterations`, the loop emits
`orchestrator:budget_warning` once per turn and injects a
`<system-reminder>` telling the agent how many calls remain and to start
converging. This is a single flat threshold, not an escalation ladder --
unlike `hooks-progress-monitor` (which escalates because it is *guessing*
the agent is stuck), the budget here is a known fact the agent can act on
directly. Both this message and the max-iteration wrap-up reminder above
are wrapped in the `<system-reminders>` envelope (see above) and carry
`metadata.ephemeral = True` -- without it, OpenAI's reasoning-replay cutoff
(`max(idx for non-ephemeral user)`) would count either message as a REAL
user turn and collapse the reasoning-replay window for the rest of the
turn. The budget-warning message also carries `metadata.persisted = True`
(it is written via `context.add_message`, genuine history); the
max-iteration reminder does not (it is view-only, appended to the outgoing
request but never persisted).

The `/goal` auto-continue loop's own internal calls (evaluator, stall
judge, run summary -- emitted with `iteration: 0`) are **not** counted
against `max_iterations`; they are separately bounded by
`goal_stall_threshold` and are ~3 calls per goal turn. This keeps the
budget coupled to real conversational turns, not goal-loop internals.

Both `max_iterations` and `budget_warn_ratio` default to today's behavior
(unlimited, and an inert ratio) -- this feature is fully opt-in and ships
with zero effect until a caller sets a budget.

## Usage

```python
# In amplifier configuration
[session]
orchestrator = "loop-streaming"
```

Perfect for:

- Interactive CLI applications
- Web UIs with progressive rendering
- Long-form content generation

## Dependencies

- `amplifier-core>=1.0.0`

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
