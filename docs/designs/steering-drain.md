# Hardening Mid-Turn Steering Drain Semantics

**Status:** Spec for implementation (failing-test-first)
**Repos:** `amplifier-module-loop-streaming` (orchestrator), `amplifier-app-cli` (UX/counter)
**Scope guard:** Mid-turn steering is an INTENTIONAL feature. A steer takes effect
*after the next tool call completes* — between tool rounds, NOT at end-of-turn.
This spec hardens the *timing, edges, and races* of that mid-turn drain. It does
**not** redesign it into an end-of-turn drain. The post-turn "next message" concept
(industry "drain race") is a **separate, DEFERRED `FollowUpQueue`** and is explicitly
out of scope here.

---

## 0. TL;DR

- **Current drain is mechanically correct for live turns.** A steer enqueued at any
  point during a running turn — up to the synchronous `is_empty` check at each
  break point — is drained and acted upon. No live-turn steer is dropped.
- **The user-observed "landed a turn late" is CORRECT behavior, not a defect.** The
  second steer was enqueued *after* the turn's final break-check (no provider call
  remained to inject into). It was correctly persisted and applied next turn. The
  fix is **UX/communication**, not a semantics change.
- **Real hardening targets** (genuine defects / fragilities): (1) the instance-scoped
  queue is **never cleared on cancellation**, so a steer queued during/after Ctrl-C
  leaks into an unrelated future turn; (2) the cross-turn carry is **silent and
  surprising** (stale steer rides the next prompt with no user feedback); (3) the
  app-cli badge counter is a **parallel counter** that can desync (phantom "1 queued").
- **Minimal fix:** clear the queue on cancellation; make the cross-turn carry visible
  ("will apply next turn") at the steer ack; derive the badge from a monotonic
  `(enqueued, injected)` pair instead of a lone decrementing counter. Empty/whitespace
  and overflow are already fail-loud — add regression guards only.

---

## 1. Precise current behavior

All line references are `amplifier_module_loop_streaming/__init__.py` unless noted.

### 1.1 The capability and the queue

- `mount()` registers `session.steer` → `orchestrator.steer` (`__init__.py:59`).
- `steer(message)` delegates to `self._steering_queue.steer(message)` (`__init__.py:121-127`).
- The queue is **instance-scoped**, created once in `__init__` (`__init__.py:88`) and
  **never cleared** for the life of the orchestrator instance.
- `SteeringQueue` (`steering.py`) is a bounded `asyncio.Queue` (`DEFAULT_MAXSIZE = 100`,
  `steering.py:24`):
  - `steer()` (`steering.py:29-40`) — raises `ValueError` on empty/whitespace
    (`steering.py:35-36`), `SteeringQueueFull` on overflow (`steering.py:39-40`).
    Non-blocking `put_nowait`. **Synchronous** (no `await`).
  - `drain()` (`steering.py:42-50`) — FIFO; dequeues all currently-present messages.
  - `is_empty` (`steering.py:52-54`) — synchronous snapshot.

### 1.2 Drain site #1 — top-of-iteration (the actual injection)

```python
# __init__.py:302-308
iteration += 1
# Mid-turn steering: drain queued user messages BEFORE building the request,
# so they are part of this iteration's provider call. ...
await self._drain_steering(context, hooks, iteration)
```

`_drain_steering` (`__init__.py:129-153`) drains FIFO and, **per message**, appends a
`{"role": "user", "content": msg}` to context (`__init__.py:142`) and emits one
`orchestrator:steering_injected` event (`__init__.py:143-152`) carrying
`content`, `iteration`, `queued_remaining`, `metadata`. Returns the count; `0` is a
true no-op (no messages, no events — streaming undisturbed).

This is the single boundary where steers actually enter the conversation. At
`iteration == 1` it is "before the first LLM call"; at `iteration > 1` it is "after the
prior tool round, before the next provider call" — i.e. the mid-turn boundary the
feature exists to serve.

### 1.3 Drain-and-revive site A — streaming path break point

```python
# __init__.py:462-468
else:
    # Last-drain edge: if a steer arrived during the final generation,
    # loop once more so the model acts on it this turn. The top-of-
    # iteration drain performs the actual injection.
    if not self._steering_queue.is_empty:
        continue
    break
```

### 1.4 Drain-and-revive site B — non-streaming path break point

```python
# __init__.py:631-637
await context.add_message(assistant_msg)
# Last-drain edge: if a steer arrived during the final generation,
# loop once more so the model acts on it this turn. The top-of-
# iteration drain performs the actual injection.
if not self._steering_queue.is_empty:
    continue
break
```

Both break points are reached only when the model returns **no tool calls** (turn is
ending). The `is_empty` check is read **synchronously** immediately after the last
`await` on that path (`context.add_message` non-streaming; stream exhaustion +
`_has_pending_tools` streaming). If a steer was enqueued before that read, the queue is
non-empty → `continue` → the top-of-iteration drain injects it → one more provider call
"this turn." If empty → `break` → `execute()` returns.

### 1.5 Important current-behavior caveat: the streaming path is single-call

`_has_pending_tools` is a stub that **always returns `False`** (`__init__.py:1449-1452`)
and `_process_tools` is a no-op (`__init__.py:1454-1457`). Therefore, in the streaming
branch (provider exposes `.stream`), the outer loop never iterates on tool rounds via
that mechanism — after the stream it goes straight to the break-or-revive at
`__init__.py:462-468`. **Genuine multi-round mid-turn steering is exercised by the
non-streaming path** (`provider.complete` returning tool calls; loop continues at the
bottom of the `while` after appending tool results, `__init__.py:829-839`, back to the
top-of-iteration drain). **All multi-round orchestrator tests in this spec use a
non-streaming stub provider** (no `.stream` attribute).

### 1.6 What happens at the turn-ending edge today

- No tool calls + queue empty → `break` → `execute()` returns the full response.
- No tool calls + queue **non-empty** → `continue` (revive) → one more provider call;
  the steer is injected at the next top-of-iteration drain. **This is correct and is
  retained.**
- The queue persists across `execute()` calls (instance-scoped, never reset at
  `__init__.py:170-173`). A steer enqueued after the final `is_empty` check survives to
  the next `execute()` and is drained at *that* turn's top-of-iteration.

---

## 2. The turn-ending edge — definition and the "landed a turn late" verdict

### 2.1 Definition of the edge (retained, confirmed correct)

> When the loop would break (model returned no tool calls) and `is_empty` is `False`,
> the orchestrator performs exactly one additional iteration: the top-of-iteration drain
> injects the pending steer(s) as user messages and a further provider call is made so
> the model acts on them within the same `execute()` invocation.

This is the correct "one more provider call" guarantee and **must be preserved**. The
revive is a `continue`, not a special drain — the actual injection always happens at the
single top-of-iteration drain site (§1.2), keeping one injection path.

### 2.2 The race window, precisely

Enqueue (`put_nowait`) and the break-point `is_empty` read both run on the **same event
loop**; `steer()` is fully synchronous. So:

- A steer enqueued **before** the synchronous `is_empty` read on the final path → queue
  non-empty → revive → injected this turn. **Cannot be missed.**
- A steer enqueued **after** that read (the loop has already chosen `break`, or
  `execute()` has already returned) → not seen this turn → persists in the
  instance-scoped queue → drained at the **next** `execute()`'s top-of-iteration.

There is no interleaving that lets a steer slip *between* "queue observed empty" and
"provider call that could have carried it," because once `is_empty` reads empty there is
no further provider call on that path — the only honest place for that steer is the next
turn.

### 2.3 The observed scenario, traced

1. User types `pause` mid-turn (during a tool round). Enqueued.
2. Loop returns to top, drains `pause` (§1.2), provider call → model acknowledges and
   returns **no tool calls**.
3. Non-streaming break path (§1.4): `is_empty` is `True` → `break`. `execute()` returns.
   The turn ends; app renders the response (`main.py:2853-2872`).
4. User types a **subsequent** steer. The steering reader (`SteeringInputManager.run`)
   is still alive in the window between `await execute_task` returning
   (`main.py:2853`) and the `finally` teardown `stop_event.set()` (`main.py:2925`). The
   reader forwards it → `steer_cap(text)` → enqueued. **But `execute()` already
   returned.** The steer sits in the instance-scoped queue.
5. Next REPL prompt → next `_execute_with_interrupt` → next `execute()` → top-of-iteration
   drain injects the leftover steer — "a turn late."

### 2.4 Verdict: (a) CORRECT, with a UX gap and a latent hardening issue

**The steer "landing a turn late" is correct behavior.** Evidence:

- The synchronous `is_empty` read (`__init__.py:466`, `:635`) provably cannot miss a
  steer enqueued before it (single event loop, synchronous `put_nowait`, no reordering
  `await`). So no *live-turn* steer was dropped.
- The leftover steer was enqueued *after* the turn's last provider opportunity; there
  was no in-turn injection point left. Persisting it to the next turn (rather than
  dropping it) is the fail-loud-preserving choice.

**Therefore the fix for the reported symptom is UX/communication, not semantics:** when a
steer is accepted while no in-turn provider call remains (turn ending/ended), the user
must be told it will apply next turn, instead of receiving the same "⧗ queued" ack that
implies in-turn application.

**Two distinct, real issues are exposed by the same trace** (these ARE worth fixing and
ARE the substance of the diagnosis-to-change the council asked for):

- **H1 — Silent cross-turn carry.** The leftover steer is injected at the next turn's
  top-of-iteration **after** that turn's fresh prompt (the new prompt is added at
  `__init__.py:255` before the loop; the steer is drained at `:308` in iteration 1).
  The user gets a stale instruction silently appended to an unrelated prompt, with no
  acknowledgment that this is what happened.
- **H2 — Cancellation leak (see §3.6).** The same persistence mechanism means a steer
  queued during/after Ctrl-C survives into a later, unrelated turn.

---

## 3. Edge-case hardening — intended behavior + test per case

For each: **Intended behavior**, **Current code**, **Defect?**, **Test**.

### 3.1 Bounded-queue overflow at the cap

- **Intended:** `steer()` at the cap fails loud and never silently drops. The rejected
  message is not enqueued; previously-queued messages are intact. The app surfaces a
  visible rejection to the user.
- **Current:** `steering.py:39-40` raises `SteeringQueueFull`; the failed `put_nowait`
  does not enqueue (`test_rejected_message_not_enqueued` already pins this). App side:
  `steering_input.py:170-171` prints `Steering rejected: ...`.
- **Defect?** No. Behavior is correct. **Add regression guards only.**
- **Test (orchestrator, exists):** `test_overflow_raises_steering_queue_full`,
  `test_rejected_message_not_enqueued`.
- **Test (app-cli, NEW):** `test_overflow_shows_visible_rejection` — stub `steer_cap`
  raising `SteeringQueueFull`; assert `_enqueue` prints a red "rejected" line and does
  **not** increment the badge.

### 3.2 Empty / whitespace steer

- **Intended:** A null/whitespace steer must never inject a user-role turn. The guard
  must live at the **orchestrator `steer()` boundary** (capability surface), not only in
  the app.
- **Current:** `SteeringQueue.steer` raises `ValueError` on `None`/whitespace
  (`steering.py:35-36`); `orchestrator.steer` → that queue, so the capability surface is
  guarded. App also silently ignores blank lines before calling (`steering_input.py:149-150`).
- **Defect?** No. **Add regression guards only.**
- **Test (orchestrator, exists):** `TestSteeringQueueValidation` +
  `test_session_steer_rejects_empty_input` (asserts via the registered capability).
- **Test (orchestrator, NEW, hardening):** `test_drain_never_injects_empty_user_message`
  — drive a full `execute()` where the app-equivalent passes whitespace through the
  capability; assert `ValueError` and that **no** `{"role":"user"}` message with empty
  content ever reaches context.

### 3.3 Burst: N steers queued during one tool round

- **Intended:** FIFO. Each steer injects as a **separate** user message (N messages,
  not one combined), each with its own `orchestrator:steering_injected` event, in
  enqueue order.
- **Current:** `_drain_steering` loops over the drained list, one `add_message` + one
  event per message (`__init__.py:141-152`). Confirmed: **N separate user messages**,
  FIFO. `queued_remaining` counts down `N-1 … 0`.
- **Defect?** No. **Add an explicit guard** (existing tests assert order but not the
  "N separate messages, not combined" property end-to-end through `execute()`).
- **Test (orchestrator, NEW):** `test_burst_injects_n_separate_user_messages_fifo` —
  non-streaming stub; during round 1 enqueue `["a","b","c"]`; assert round-2 context
  contains exactly three user messages `["a","b","c"]` in order (not one `"a\nb\nc"`),
  and exactly three `steering_injected` events with `queued_remaining` `2,1,0`.

### 3.4 Counter decrement race (app-cli badge)

- **Intended:** The badge ("N queued") can never go negative and can never stick at a
  phantom value. It reflects "accepted-but-not-yet-injected" with a single source of
  truth.
- **Current:** Two parallel mutations — increment on enqueue (`steering_input.py:161`),
  decrement (guarded `> 0`) on each injected event (`steering_input.py:111-112`). Within
  one manager's lifetime, increments == injected events, so it balances and the `> 0`
  guard prevents negatives. **But** the counter is a separate accumulator, not derived
  from the authoritative drain stream:
  - A lost/dropped `steering_injected` event (e.g. hook registration race at
    `main.py:2814-2821`, or a steer that carries to the next turn whose event is consumed
    by a *different* manager) leaves the badge stuck at a phantom "1 queued."
  - Cross-turn: a leftover steer (H1) is injected during the *next* turn; that turn's
    manager (`main.py:2804`, fresh per turn) receives the event with `enqueued == 0`,
    decrements guarded at 0 — harmless, but it confirms the counters are not coupled to a
    single truth.
- **Defect?** Fragile, yes — **fix the discipline.**
- **Discipline (the fix):** Replace the lone decrementing counter with a **monotonic
  pair** owned by the manager: `enqueued_total` (++ on each accepted `steer_cap` call)
  and `injected_total` (++ on each `steering_injected` event). Badge =
  `max(0, enqueued_total - injected_total)`. Monotonic counters are order-insensitive:
  the badge is always a pure function of two non-decreasing integers, so it cannot go
  negative and self-heals once events catch up. (The event already carries
  `queued_remaining`; it MAY be used as a corroborating signal but the authoritative
  badge is the monotonic difference, clamped at 0.)
- **Test (app-cli, NEW):**
  - `test_badge_is_enqueued_minus_injected` — drive enqueue/inject in interleaved orders;
    assert badge == `max(0, enqueued - injected)` at each step.
  - `test_extra_injected_event_cannot_drive_badge_negative` — fire more injected events
    than enqueues; badge stays `0`.
  - `test_lost_injected_event_does_not_stick_phantom` (xfail→fix) — under the **old**
    decrement model, dropping one event sticks the badge at 1 forever; under the
    monotonic model, a subsequent matching injected event drives it back to 0. Asserts
    the monotonic contract (this test FAILS against current code, passes after fix).

### 3.5 `patch_stdout(raw=True)` splitting an ANSI/OSC escape across writes

- **Context:** The whole turn runs inside `with patch_stdout(raw=True)` (`main.py:2838`).
  `raw=True` is deliberate (`main.py:2831-2837`) — `raw=False` would strip ESC bytes and
  mangle Rich output. The concern is the *opposite*: with raw passthrough, if a single
  ANSI/OSC sequence is split across two `write()` calls and a pinned-prompt redraw
  (`app.invalidate()` from `on_steering_injected`/`_enqueue`) interleaves between the
  halves, the terminal could see a corrupted/half-applied escape.
- **Assessment:** **Medium-likelihood, unproven.** Rich generally emits a styled segment
  (with its SGR escapes) in a single `write`, and prompt_toolkit's `run_in_terminal`
  owns prompt erase/restore around app redraws, which makes mid-escape interleave
  unlikely but not provably impossible during token-by-token streaming. This is an
  **investigate-first** item, not a confirmed corruption.
- **Defect?** Unconfirmed. **Do not ship a fix blind.** Write the probe test first; only
  if it reproduces corruption do we specify a remedy (e.g. buffer writes to escape
  boundaries, or gate `invalidate()` so it cannot fire mid-write).
- **Test (app-cli, NEW, probe):** `test_split_ansi_escape_survives_prompt_redraw` —
  feed a write of the first half of an SGR/OSC sequence (e.g. `"\x1b[38;5"`), trigger
  `_pt_session.app.invalidate()` (or the `on_steering_injected` redraw) via a fake
  session capturing raw writes, then feed the remainder (`";5m…\x1b[0m"`); assert the
  captured byte stream reassembles the intact escape (no redraw bytes injected between
  the ESC and its terminator). If it cannot be reproduced with the StdoutProxy harness,
  mark `xfail(reason="unreproduced; tracking")` so the risk stays visible rather than
  silently closed.

### 3.6 Cancellation interplay (Ctrl-C, graceful/immediate) — REAL DEFECT

- **Intended:** Cancellation means "stop / the user is taking control." Any steer queued
  but not yet injected at the moment of cancellation must be **discarded** (cleared from
  the queue), and a steer arriving *after* cancellation is detected must not silently
  ride into a future, unrelated turn. The badge resets.
- **Current:** On graceful cancel detected at top-of-iteration, the loop emits cancel
  events and `return`s **without draining or clearing** the queue (`__init__.py:273-300`).
  Graceful cancel after tools (`__init__.py:780-827`) and immediate cancel
  (`__init__.py:728-777`) likewise never touch `_steering_queue`. Because the queue is
  instance-scoped and never reset (`__init__.py:88`, `:170-173`), a steer queued during
  cancellation **leaks into the next `execute()`** and is injected at its top-of-iteration
  drain — into an unrelated turn (this is H2, the cancellation arm of the §2.4 carry).
- **Defect?** **Yes.** Silent cross-turn injection after a cancel is a correctness bug.
- **Fix:** Clear `_steering_queue` at every cancellation exit (the three sites above)
  and at the **start of `execute()`** (defensive, in the reset block at
  `__init__.py:170-173`) so no turn ever begins with steers from a previous turn it
  didn't observe. Clearing at `execute()` start also resolves H1's silent-carry by
  policy: **steers do not cross turn boundaries inside the orchestrator.** Cross-turn
  intent is the deferred `FollowUpQueue`'s job, not this queue's. (See §5 for why this
  is consistent with the §2.4 "correct" verdict: the *symptom* was correct given today's
  carry; the *hardened policy* is "no carry" + explicit next-turn UX, which is strictly
  clearer.)
- **Add to `SteeringQueue`:** a `clear()` method (drain-and-discard) so the orchestrator
  has a first-class, fail-loud way to empty the queue.
- **Tests (orchestrator, NEW):**
  - `test_graceful_cancel_clears_pending_steers` — enqueue a steer, set
    `cancellation.is_cancelled` before the top-of-iteration check; assert that after
    `execute()` returns the queue `is_empty` and the steer was **not** injected.
  - `test_steer_during_cancel_does_not_leak_into_next_turn` — cancel turn 1 with a steer
    queued; run a second `execute()` with a fresh prompt; assert the turn-1 steer never
    appears as a user message in turn 2's context.
  - `test_execute_start_clears_stale_queue` — pre-seed the queue, call `execute()`;
    assert no stale steer is injected (queue cleared at entry).
  - `test_immediate_cancel_clears_pending_steers` — same as graceful but via the
    `asyncio.CancelledError` path (`__init__.py:728`).

---

## 4. Failing-test plan (write these FIRST)

Ordering: write the test, watch it fail (or pass-as-guard, noted), then implement §5.
Harness reuse: extend the existing stubs in
`amplifier-module-loop-streaming/tests/test_steering.py` (`MockHooks`, `MockContext`,
`MockCoordinator`, `MockCancellation`, `NonStreamingProvider` pattern). The
`NonStreamingProvider` returning tool calls across N rounds + a concurrent
`orch.steer(...)` inside `complete()` is the canonical multi-round mid-turn harness
(already used by `TestTopOfIterationDrain`).

### 4.1 Orchestrator-repo tests (`amplifier-module-loop-streaming/tests/test_steering.py`)

| # | Test | Reproduces | Status vs current code |
|---|------|------------|------------------------|
| O1 | `test_graceful_cancel_clears_pending_steers` | §3.6 leak | **FAILS** (no clear) |
| O2 | `test_steer_during_cancel_does_not_leak_into_next_turn` | §3.6 / H2 | **FAILS** |
| O3 | `test_execute_start_clears_stale_queue` | §3.6 / H1 carry | **FAILS** |
| O4 | `test_immediate_cancel_clears_pending_steers` | §3.6 immediate path | **FAILS** |
| O5 | `test_burst_injects_n_separate_user_messages_fifo` | §3.3 | PASS (guard) |
| O6 | `test_drain_never_injects_empty_user_message` | §3.2 | PASS (guard) |
| O7 | `test_steer_after_break_check_applies_next_turn` | §2.3 verdict | PASS (pins the "correct, a turn late" contract) |
| O8 | `test_revive_uses_single_injection_path` | §2.1 | PASS (guard; assert injection only via top-of-iteration drain — one event per steer, no double-inject) |

`SteeringQueue.clear()` unit tests (steering.py): `test_clear_empties_queue`,
`test_clear_on_empty_is_noop`.

### 4.2 App-cli tests (`amplifier-app-cli/tests/test_steering.py`)

| # | Test | Reproduces | Status vs current code |
|---|------|------------|------------------------|
| A1 | `test_badge_is_enqueued_minus_injected` | §3.4 discipline | **FAILS** (no monotonic pair) |
| A2 | `test_extra_injected_event_cannot_drive_badge_negative` | §3.4 | PASS (guard, `>0`) but re-pin under new model |
| A3 | `test_lost_injected_event_does_not_stick_phantom` | §3.4 phantom | **FAILS** under decrement model; passes after monotonic |
| A4 | `test_overflow_shows_visible_rejection` | §3.1 | PASS (guard) |
| A5 | `test_next_turn_steer_ack_communicates_deferral` | §2.4 UX | **FAILS** (no such message today) |
| A6 | `test_split_ansi_escape_survives_prompt_redraw` | §3.5 probe | xfail/investigate |

A5 covers the actual reported symptom's fix: when a steer is accepted while the turn is
ending/ended, the ack text must communicate next-turn application rather than the
in-turn "⧗ queued". (Mechanism: see §5.3.)

---

## 5. The minimal fix

Smallest change set that makes timing predictable and edges defined. Each item maps to
tests above.

### 5.1 `SteeringQueue.clear()` (orchestrator, `steering.py`)

Add a fail-loud drain-and-discard:

```python
def clear(self) -> int:
    """Discard all pending steers (e.g. on cancellation). Returns count discarded."""
    return len(self.drain())
```

(Unit tests: `test_clear_*`.)

### 5.2 No steer crosses a turn boundary (orchestrator, `__init__.py`)

- **At `execute()` entry** (reset block `__init__.py:170-173`): `self._steering_queue.clear()`.
  Guarantees a turn never begins with steers it never observed. (O3.)
- **At each cancellation exit** — top-of-iteration return (`__init__.py:300`), graceful
  post-tool return (`__init__.py:827`), immediate-cancel re-raise (`__init__.py:777`):
  `self._steering_queue.clear()` before returning/raising. (O1, O2, O4.)

Leave drain site (§1.2) and both revive sites (§1.3–1.4) **unchanged** — the mid-turn
drain and the one-more-call revive are correct and retained.

### 5.3 UX: communicate next-turn application (app-cli)

The reported "landed a turn late" is fixed by *telling the user*, not by changing
semantics. In `SteeringInputManager._enqueue` (or at the call boundary), when the steer
is accepted while the turn is no longer able to consume it in-turn, the ack must say so.

Pragmatic, low-coupling implementation: the manager already runs only for the duration of
a turn and is torn down at turn end (`main.py:2925-2930`). A steer accepted in the
teardown window (`await execute_task` returned, `stop_event` not yet set) is exactly the
"next turn" case. Gate the ack on `self._stop_event.is_set()`:

- If `stop_event` is not set (turn live): keep `⧗ queued: {text}`.
- If `stop_event` is set (turn ending/ended): print `⧗ queued for next turn: {text}`.

This needs no orchestrator change and directly addresses the symptom. (A5.)

> Note: with §5.2 in force, a steer accepted after the turn has fully ended and the queue
> cleared at the next `execute()` entry would be dropped. To avoid silently dropping such
> a steer, the app SHOULD route a post-turn steer into the **next prompt** path rather
> than `session.steer()` once `stop_event` is set — i.e. treat it as next-turn input
> explicitly. The minimal cut for THIS spec is the honest ack (A5) + the orchestrator
> clear (§5.2); wiring the post-turn steer into the next prompt is a small follow-up
> noted here so the behavior is defined, not discovered. Implementer: confirm which of
> the two (ack-only vs ack+reroute) the app wants; ack-only is the floor.

### 5.4 Badge from a monotonic pair (app-cli, `steering_input.py`)

Replace `_pending_count` decrement logic:

- `_enqueued_total += 1` on each successful `steer_cap` (`steering_input.py:161` site).
- `_injected_total += 1` in `on_steering_injected` (`steering_input.py:111-112` site),
  drop the `> 0` decrement.
- `pending_count` property → `max(0, self._enqueued_total - self._injected_total)`.
- `_prompt_message` (`steering_input.py:123-137`) reads the derived property unchanged.

(A1–A3.)

---

## 6. Explicitly out of scope (parked)

- **Dual-mode queue** (a separate post-turn / "next message" `FollowUpQueue`). The
  industry "drain race" pain belongs to that deferred concept, not this `SteeringQueue`.
  Not built here.
- **Fixed-input TUI** (a persistent bottom input box independent of per-turn managers).
  Parked.
- **§3.5 ANSI/OSC remedy** beyond the probe test — only specify a fix if the probe
  reproduces corruption.
- **§5.3 post-turn-steer→next-prompt rerouting** — defined above as a follow-up; floor
  is the honest ack.

---

## 7. Platform note

DTU proof for this work is **Linux only** (the workspace DTU runs Incus on Linux).
**Windows is unproven** for: `prompt_toolkit` `patch_stdout(raw=True)` escape handling
(§3.5), SIGINT/Ctrl-C semantics in `_execute_with_interrupt` (`main.py:2761-2795`), and
terminal redraw timing. Any Windows behavior must be flagged as unverified until tested
on a native Windows Python host.

---

## 8. Verification gradient (hand-off to implementer)

1. **Unit** (`steering.py`): `SteeringQueue.clear()` + existing queue tests.
2. **Orchestrator integration** (`tests/test_steering.py`): O1–O8 via non-streaming
   multi-round stub provider. Run: `pytest tests/test_steering.py` (asyncio strict,
   `--import-mode=importlib`, per `pyproject.toml`).
3. **App-cli unit** (`tests/test_steering.py`): A1–A6 via injected `_input_provider` /
   fake console — no real TTY (existing pattern).
4. **Manual / DTU (Linux)**: reproduce the `pause` → subsequent-steer scenario; confirm
   the ack now reads "queued for next turn" and that a Ctrl-C with a queued steer leaves
   the next turn clean.

Success = every O*/A* test green (A3 and A6 flips/handled as specified), no change to the
mid-turn drain or revive guarantees, and the cancellation leak closed.
