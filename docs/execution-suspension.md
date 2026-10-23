# Cooperative execution suspension

This opt-in public interface separates suspension from cancellation and normal
completion. With no `execution.suspension` capability, ordinary execution is
unchanged. A host supplying the capability owns durable storage, admission, and
the explicit decision to resume. The engine never chooses a storage path or
automatically resumes after process loss.

The capability object supplies:

- `requested() -> bool`: a synchronous, side-effect-free hold check.
- `async checkpoint(frame) -> True`: retain the frame and matching canonical
  history durably, or raise/refuse. Only `True` acknowledges persistence.

The engine waits for an in-flight model response or ordinary tool invocation to
settle, then checkpoints before another model call or tool effect. Concurrent
tool batches settle every started sibling, retain ordered outcomes, and preserve
unstarted calls. It does not cancel an arbitrary tool and call that a safe pause.
A hung external operation can remain waiting indefinitely.

`ExecutionSuspended` is a `BaseException` carrying the acknowledged frame. It is
not a completed response or a failed tool result. Hosts handle this outcome
separately and release resources only after canonical checkpoint acknowledgment.

Mount the same configuration with the saved canonical messages and call
`await orchestrator.resume(frame, context, providers, tools, hooks, coordinator)`.
Register `execution.resume` as an async one-time admission callback accepting that
exact frame. History/configuration, cancellation, goal state and callback
authorization are checked before effects. A used or uncertain claim is never
retried implicitly. The host rechecks current permission, budget, task and
ownership constraints; old configuration is not permission to bypass controls.

Frame version 1 contains explicit engine program state, not a coroutine stack.
It records the phase (`provider`, `prepared_request`, `tools`, `settling_tools`,
`prepared_finalization`, or `goal`), original iteration/accounting, ordered tool
outcomes and history/configuration fingerprints. Prepared requests retain
completed hook/context/rate-limit preparation. Goal assessment, judgment and
continuation state retain obtained results and accounting. Frames are private
execution state and can contain full model requests and tool outputs.

A cooperating long-lived tool can raise `ToolExecutionSuspended` with its own
versioned continuation. Its mounted tool implements
`async resume_execution(continuation)`; the engine calls that original completion
owner instead of `execute(arguments)`. There is no redispatch fallback. The tool
owns child/session/result attribution and timeout/admission checks.

Resumed batches validate every required completion owner before scheduling any
sibling. A refusal after host admission does not release or reuse that claim.
An unexpected ordinary dispatch failure during a resumed batch is an unknown,
non-retryable outcome; started siblings still settle and retain their known
results. Cancellation remains distinct from this failure path.

True-stream input-overflow recovery checks for suspension after recovery,
preflight and rate-limit waits, before sending the replacement request. Its
prepared checkpoint retains that exact request, spent one-shot recovery
allowance and accounting. Authorized resume does not repeat preparation or grant
another recovery attempt. Any provider chunk, including a non-text chunk, still
forbids overflow recovery.

This interface does not undo remote effects, make opaque provider-native
execution suspendable, or authorize serialization of custom private component
state. Hosts must report unsupported or unconfirmed cases honestly.
