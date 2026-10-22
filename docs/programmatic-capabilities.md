# Programmatic execution capabilities (development candidate)

`tools.dispatch` and `model.complete` are optional capabilities mounted by the
streaming loop and reused by loop-live. They contain no UI-use or vendor logic.
Consumers bind during an admitted parent tool call. The lease is scoped to that
session, run and parent, expires on completion, and cannot be used from an idle
session. Nested actions re-enter the ordinary tool hooks and approval processor.
They retain parent/call identities, serialize actual execution, and report denied,
failed and unknown outcomes separately. Cancellation never implies rollback.

Root configuration can explicitly enable `programmatic_dispatch` and select
`programmatic_callers` / `programmatic_allowed_tools`, as before. A portable tool
behavior can instead register a named entry in `tools.dispatch.registrations`:

```python
{"controller_name": {"tools": ["permitted_surface_tool"]}}
```

This is a trusted module declaration made at mount, not an argument supplied by
a model. Explicit host `programmatic_dispatch: false` disables it; an explicit
caller list constrains it; a host tool allowlist intersects it. Removing the
declaration revokes it. A module cleans up only the declaration it owns. Nested
delegation, background execution and recursive controllers remain unavailable.
Automatic enablement admits only registered callers; it does not enable an
unregistered `tool_exec`. That legacy caller still requires the host's explicit
`programmatic_dispatch: true`, or inclusion in a host caller list when dispatch
is enabled.

`model.complete(ChatRequest)` accepts a stateless tool-free request only during
such an active call. The selected ordinary provider and its account are reused;
the caller cannot select a different model through this service. Provider request
hooks may deny the call, input budgets are checked, streaming is disabled, and
cancellation reaches the in-flight provider task. Provider usage/cost reporting
remains active. A `model:complete` receipt adds parent attribution without copying
the prompt or result into diagnostics. These calls increment the turn's actual
`llm_calls`; they do not mutate history or its context-occupancy estimate.

`programmatic_max_model_calls` bounds these utility calls per turn (default 24,
zero disables them). It is separate from the existing foreground iteration cap,
not a claim that routing, goal evaluation, decision endpoints and financial
budgets are already governed by a single meter. A broader inference service can
be extracted later without putting that policy in a UI controller.
