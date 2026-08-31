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
    goal_stall_threshold = 3,        # /goal: consecutive no-tool continuation turns
                                      # before the stall judge is consulted
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

## Ephemeral injection mode (prompt-cache prefix fix)

```toml
config = {
    ephemeral_injection_mode = "persist",  # "tail" | "persist" (default "persist")
}
```

Per-iteration ephemeral tail messages -- `hooks-status-context`,
`hooks-todo-reminder`, the compaction notice, and any other
`inject_context` hook result -- are, by default (`"persist"`), written into
canonical context via `context.add_message(...)`, and only when the text
differs from the last text this orchestrator persisted. When unchanged,
nothing is added, so request N is a true, append-only prefix of request
N+1. This matters because OpenAI's (and most providers') implicit/explicit
prompt cache reuses only the longest true prefix of a prior request: the
original `"tail"` behavior re-generates and re-appends these messages at
the tail of every request, positionally displacing the assistant/tool turn
that follows and truncating the reusable prefix -- pinning cache-hit share
near the static system-prompt boundary and re-billing the entire growing
transcript as a fresh cache write on every call.

**Evidence for the default:**

- **OpenAI**: a pre-registered 9-arm live probe found only the persist
  design (change-gated, canonical-context write) heals prefix reuse
  (98.9%); byte-stable tails and folding into the tool result do not heal
  it (both are still positional, not content, mismatches). In-vivo across
  4 DTU eval waves (30+ runs), cache-read share recovered from ~9-11% to
  89-97% on every persist run, cache-write dropped ~10x, and task quality
  was unchanged (all runs correct/passing). A real 6-turn session with
  `"tail"` (the old default) showed `cache_read` pinned flat at 63,060
  tokens across every call while `cache_write` climbed monotonically --
  77.9M cache-write vs. 24.0M cache-read (3.24:1, inverted) -- an
  estimated $250-380 of that session's $452 total was avoidable re-write
  spend.
- **Anthropic** (the flip gate): n=3 DTU S1 runs with persist mode on,
  `claude-opus-4` @ xhigh: 3/3 correct; cache-read share 90.9-92.5% (mean
  91.9%) vs. the `"tail"` baseline's 83.1-87.8% (mean 86.2%) -- +5pts,
  favorable; cost $1.29 vs. $1.90 mean; wall time 257s vs. 331s mean; wire
  contract clean (append-only message list, persisted injections
  re-emitted only on change, valid `cache_control` breakpoints, zero
  provider errors, no thinking-block interaction issues).

**The tradeoff this mode accepts (spec §5.2):** once an injection is
persisted, it is no longer "removed next turn" -- it becomes real,
bounded history from that point on (still marked
`metadata.ephemeral=True`, now meaning "machine-generated per-turn
scaffolding", not "guaranteed absent next turn"), and it is re-emitted
(persisted again) only when its content actually changes. Operators who
need the original single-ephemeral-tail-message contract -- e.g. a custom
hook whose injection text must never accumulate in history -- should set
`ephemeral_injection_mode = "tail"` explicitly; that path remains fully
supported and is byte-identical to the module's original, pre-this-feature
behavior. An unknown value falls back to `"persist"` (the current default)
with a logged warning.

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

2. **Placement (`reminder_placement`).** By default (`"pre_user"`), the
   turn's reminder block is written **before** the user's prompt -- in
   canonical history for `ephemeral_injection_mode = "persist"` (the
   block precedes the user message as real, append-only history), or
   spliced into the request VIEW for `ephemeral_injection_mode = "tail"`
   (nothing persisted; the splice happens once, at iteration 1, and is
   never repeated). This is achieved by hoisting iteration 1's
   `provider:request` emit to TURN START, before `context.add_message` adds
   the user's prompt -- the event payload carries `"phase": "turn_start"`
   so a hook that cares can discriminate; hooks that ignore the key behave
   exactly as before. `reminder_placement = "tail"` is the **rollback
   lever**: it skips the turn-start assembly entirely and restores the
   pre-this-feature ordering (block after the user message) -- the
   envelope, role pin, and metadata tag are still applied in `"tail"` mode;
   only the ORDERING reverts. An unknown value falls back to `"pre_user"`
   with a logged warning.

**Role pinning.** Every reminder message this orchestrator writes uses the
literal `role = "user"`, regardless of what a contributing hook requested
via `context_injection_role`. `HookResult.context_injection_role` defaults
to `"system"`, and a hook that never sets it explicitly contributes
system-role content; if the one hook in a chain that *does* set `"user"`
is ever unmounted, deprioritized, or out-registered, the kernel's
`merge_inject_context_results` ("first result wins" for role) would let a
system-role blob through. `amplifier-module-provider-anthropic` hoists
every `role == "system"` message into the single cached system block, and
`amplifier-module-provider-openai` folds system content into
`instructions` -- a per-turn-changing blob in that position rewrites the
system prefix on every single turn. Pinning to `"user"` defuses this.

**Known, accepted contract narrowing:** a hook that requests
`context_injection_role = "assistant"` (simulating agent self-talk) is
now overridden to `"user"` like everything else. No hook shipped in this
ecosystem does this today; if one needs to in the future, it needs a
different mechanism than `provider:request` inject_context (this
orchestrator logs at `debug` when a non-`"user"`/`"system"` role is
overridden, so the narrowing is observable rather than invisible).

**"Before", not "immediately before".** In persist mode, the change-gate
suppresses re-persisting an unchanged reminder block. On a turn where
nothing changed, canonical history looks like
`[block N] [user N] [assistant] [tool] [assistant] [user N+1]` -- the
block is several messages back, not adjacent to `user N+1`. **This is
correct and intended**, and is the same cache-prefix property
`ephemeral_injection_mode = "persist"` exists to guarantee (see above).
Do not "fix" this by disabling the change-gate -- that would defeat the
whole cache-prefix benefit this mode provides.

**Change-gate comparison basis.** The change-gate always compares the RAW
(pre-envelope) merged body against the last persisted body, never the
enveloped string. This matters because the turn-start block uses the
pre-user header variant and a later mid-loop block (same iteration's
change-gate lineage) uses the tail variant -- two different headers
wrapping potentially-identical content. Comparing enveloped strings would
falsely detect a "change" the first time a turn transitions from its
turn-start block to a mid-loop one, forcing a spurious extra persisted
message on every multi-iteration turn.

**Mid-loop (iterations >= 2) placement is unchanged**: new content is
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

**Exhaustion is a normal turn ending, not an error.** When `iteration`
reaches `max_iterations`, the loop makes **one additional** tool-less
`provider.complete()` call with an injected
`<system-reminder source="orchestrator-loop-limit">` asking the agent to
wrap up and summarize -- so a budget of `N` permits at most `N + 1`
main-loop provider calls, not `N`. This wrap-up call ends the turn with a
normal return (`execution:end` fires, no exception raised), so the caller's
usual persistence path runs and the resulting transcript is complete and
resumable.

`ORCHESTRATOR_COMPLETE`'s payload always carries a `metadata` bag:

```python
{
    "llm_calls": 300,             # main-loop iterations actually used (not goal-loop internal calls -- see below)
    "llm_call_budget": 300,       # the max_iterations this turn ran under, or None if unlimited
    "budget_exhausted": True,     # whether this turn hit the budget (vs. finishing early)
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
