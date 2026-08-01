"""
Streaming orchestrator module for Amplifier.
Provides token-by-token streaming responses.
"""

# Amplifier module metadata
__amplifier_module_type__ = "orchestrator"

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from amplifier_core import HookRegistry, ModuleCoordinator, ToolResult
from amplifier_core.events import (
    CANCEL_COMPLETED,
    CANCEL_REQUESTED,
    CONTENT_BLOCK_END,
    CONTENT_BLOCK_START,
    ORCHESTRATOR_COMPLETE,
    PROMPT_SUBMIT,
    PROVIDER_ERROR,
    PROVIDER_REQUEST,
    TOOL_ERROR,
    TOOL_POST,
    TOOL_PRE,
)
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from .steering import SteeringQueue

logger = logging.getLogger(__name__)


def _flatten_message_for_evaluator(msg: dict) -> str:
    """Render one stored (dict) message as plain text for the goal evaluator.

    Ported verbatim from amplifier-app-cli's `_flatten_message_for_evaluator`
    (see docs/designs/goal-command.md). Kept as a module-level function since
    it needs no orchestrator state.
    """
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_call":
                parts.append(f"[called tool: {block.get('name', '?')}]")
            elif btype == "tool_result":
                parts.append("[tool result omitted]")
            # thinking/redacted_thinking/reasoning blocks are intentionally
            # skipped -- the evaluator only judges what was surfaced.
        text = "\n".join(p for p in parts if p)
    else:
        text = str(content)
    return f"{role}: {text}" if text else ""


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the streaming orchestrator module."""
    config = config or {}

    # Declare observable lifecycle events for this module
    # (hooks-logging will auto-discover and log these)
    coordinator.register_contributor(
        "observability.events",
        "loop-streaming",
        lambda: [
            "execution:start",  # When orchestrator execution begins
            "execution:end",  # When orchestrator execution completes
            "orchestrator:steering_injected",  # When a steer message is injected mid-turn
            "orchestrator:goal_progress",  # /goal auto-continue loop progress (see docs/designs/goal-command.md)
        ],
    )

    orchestrator = StreamingOrchestrator(config)
    await coordinator.mount("orchestrator", orchestrator)
    coordinator.register_capability("session.steer", orchestrator.steer)
    logger.info("Mounted StreamingOrchestrator with steering capability")


class StreamingOrchestrator:
    """
    Streaming implementation of the agent loop.
    Yields tokens as they're generated for real-time display.
    """

    # === /goal support (spike: docs/designs/goal-command.md) ===
    #
    # Ported from amplifier-app-cli's app-layer spike (main.py's
    # `_evaluate_goal` / `_execute_with_interrupt_and_goal`). Moving it here
    # means any orchestrator caller -- interactive REPL or headless
    # `--mode single` -- gets the auto-continue loop for free, because both
    # paths funnel through `execute()`.
    _GOAL_EVALUATOR_SYSTEM_PROMPT = (
        "You are a strict, tool-less evaluator. You will be shown a GOAL "
        "CONDITION and a transcript of an assistant's work so far in a "
        "coding/agent session. Decide whether the GOAL CONDITION has been "
        "satisfied by that work.\n\n"
        "Respond with EXACTLY two lines and nothing else:\n"
        "Line 1: the single word YES or NO (verbatim, nothing else)\n"
        "Line 2: one sentence explaining why\n"
    )

    # Best-effort cheap-model hints per provider-name substring. Spike-level
    # heuristic only -- mirrors amplifier-app-cli's hint table.
    #
    # NOTE on model-role routing (investigated for the stall-detection work,
    # docs/designs/goal-command.md): Amplifier's routing-matrix `fast` role
    # is a delegation-time concept -- it's only reachable through the
    # `delegate` tool's sub-session spawning (`model_role` param). It is not
    # exposed as a coordinator capability (see amplifier-core's
    # CAPABILITY_REGISTRY.md -- the only standard capabilities are
    # `session.spawn` / `session.resume`), so there is nothing for a module
    # making a direct `provider.complete()` call, like this one, to reach.
    # If a `model.route`-style capability is ever registered, this hint
    # table (and `_select_cheap_model` below) is the place to swap it in.
    _GOAL_CHEAP_MODEL_HINTS: ClassVar[dict[str, str]] = {
        "anthropic": "claude-haiku-4-5",
        "openai": "gpt-5-mini",
        "azure-openai": "gpt-5-mini",
        "azure_openai": "gpt-5-mini",
    }

    _GOAL_MAX_TRANSCRIPT_MESSAGES = 40

    # Stall-detection judge (see execute()'s stall-detection block). Only
    # invoked when the mechanical no-tool-turns condition already holds, so
    # it fires rarely and stays cheap.
    _GOAL_STALL_SYSTEM_PROMPT = (
        "You are a strict, tool-less judge. You will be shown a short history "
        "of reasons an evaluator gave, across consecutive turns, for why a "
        "goal condition was not yet satisfied -- during a stretch where the "
        "assistant took no tool actions at all. Decide whether these reasons "
        "describe the SAME unresolved blocker recurring with no sign of new "
        "progress (a stall), as opposed to reasons that, even with no tools "
        "run, show the assistant narrowing down, ruling things out, or making "
        "genuine incremental progress toward the condition.\n\n"
        "Respond with EXACTLY two lines and nothing else:\n"
        "Line 1: the single word YES (this is a stall) or NO (not a stall), "
        "verbatim\n"
        "Line 2: one sentence explaining why\n"
    )

    # Fast-model run-summary prompts (terminal goal_progress states only),
    # one per terminal state. The reader watched the entire run happen
    # live, so the summary is never a recap of what they already saw --
    # it's the one-sentence answer to the single question that particular
    # ending raises (what blocked it / what's left / what broke). Replaces
    # the earlier single 2-4 sentence generic-recap prompt, which restated
    # things the CLI had already rendered turn by turn.
    #
    # ``achieved`` deliberately has no entry here: it never reaches this
    # dict at all (see _goal_run_needs_summary). ``cancelled`` is
    # short-circuited before any summary call in execute().
    _GOAL_SUMMARY_SYSTEM_PROMPTS: ClassVar[dict[str, str]] = {
        "stalled": (
            "You write a single, short line for a developer who just watched "
            "an automated goal-pursuit run end with no progress. They saw "
            "every turn happen live -- do not recap the run or narrate "
            "attempts. You will be given the goal condition and the sequence "
            "of reasons an evaluator gave, turn by turn, for why the "
            "condition kept going unmet. Name the ONE thing that blocked "
            "progress -- lead with the blocker itself. If the goal is "
            'impossible as stated, say so plainly. Never begin with "The '
            'assistant" or "The evaluator" (both are implied), and never '
            "restate the goal condition. Write in present tense. Respond "
            "with exactly one sentence, no more than about 120 characters, "
            "no preamble, no quotation marks."
        ),
        "cap_hit": (
            "You write a single, short line for a developer whose automated "
            "goal-pursuit run ran out of turns before an evaluator ever "
            "confirmed the goal condition was met. They watched every turn "
            "happen live -- do not recap the run. You will be given the "
            "goal condition and the sequence of reasons the evaluator gave. "
            "State only what REMAINS UNDONE -- frame it as remaining work, "
            "never as history, and never assert or imply the goal was met. "
            "If the work looked complete but was simply never confirmed, "
            "say exactly that. This sentence is the sole input to the "
            "developer's next decision -- rerun with a bigger cap, or "
            "change approach -- so make the remaining work concrete. Never "
            'begin with "The assistant" or "The evaluator", and never '
            "restate the goal condition. Write in present tense. Respond "
            "with exactly one sentence, no more than about 120 characters, "
            "no preamble, no quotation marks."
        ),
        "error": (
            "You write a single, short line for a developer whose automated "
            "goal-pursuit run crashed because the evaluator itself failed. "
            "You will be given the error that was raised. State only what "
            "failed in the evaluator. No apology, no advice, no restating "
            "the goal condition, no narrating the run. Never begin with "
            '"The assistant" or "The evaluator" (both are implied). '
            "Write in present tense. Respond with exactly one sentence, no "
            "more than about 120 characters, no preamble, no quotation "
            "marks."
        ),
    }

    # Hard ceiling enforced in code on every generated run summary (see
    # _enforce_summary_length below) -- the prompts above ask the model for
    # ~120 characters, but a model instruction is not a guarantee.
    _GOAL_SUMMARY_MAX_CHARS = 120

    def __init__(self, config: dict[str, Any]):
        self.config = config
        # -1 means unlimited iterations (default)
        max_iter_config = config.get("max_iterations", -1)
        self.max_iterations = int(max_iter_config) if max_iter_config != -1 else -1
        # /goal stall detection: consecutive continuation turns with zero tool
        # calls required before the stall judge is even consulted (see
        # execute()'s stall-detection block and _judge_stall).
        self.goal_stall_threshold = int(config.get("goal_stall_threshold", 3))
        # Per-token artificial delay (seconds) injected after each non-whitespace
        # token in _tokenize_stream(). Default 0.0 so headless callers (sub-sessions,
        # automated agents) pay no synthetic latency. Set to e.g. 0.01 to opt in to
        # token-by-token typing animation for human-facing terminal UX.
        self.stream_delay = config.get("stream_delay", 0.0)
        self.extended_thinking = config.get("extended_thinking", False)
        self.min_delay_between_calls_ms = config.get("min_delay_between_calls_ms", 0)
        self._last_provider_call_end: float | None = None  # Timestamp tracking
        # Per-turn tool-call counter (see _execute_tool_only /
        # _execute_tool_with_result and execute()'s stall detection).
        # Initialized here (not just reset in _execute_one_turn) so direct
        # callers of the tool-execution methods never hit an
        # AttributeError.
        self._tool_calls_this_turn: int = 0
        # Store ephemeral injections from tool:post hooks for next iteration
        self._pending_ephemeral_injections: list[dict[str, Any]] = []
        # Track whether cancel:requested has been emitted for the current execution
        self._cancel_requested_emitted: bool = False
        # Bounded queue for mid-turn steering messages (session.steer capability)
        self._steering_queue = SteeringQueue()
        # Deferred ORCHESTRATOR_COMPLETE payload for the goal auto-continue loop
        # (spike: docs/designs/goal-command.md). See _execute_one_turn's
        # `goal_turn` param and _flush_pending_complete() for why this is
        # deferred rather than emitted immediately.
        self._pending_orchestrator_complete: (
            tuple[HookRegistry, dict[str, Any]] | None
        ) = None

    async def _apply_rate_limit_delay(
        self, hooks: HookRegistry, iteration: int
    ) -> None:
        """Apply rate limit delay if configured and needed.

        Only delays if:
        - min_delay_between_calls_ms > 0 (enabled)
        - This is not the first call (has previous timestamp)
        - Elapsed time < configured minimum
        """
        if self.min_delay_between_calls_ms <= 0:
            return  # Disabled

        if self._last_provider_call_end is None:
            return  # First call, no delay needed

        elapsed_ms = (time.monotonic() - self._last_provider_call_end) * 1000
        remaining_ms = self.min_delay_between_calls_ms - elapsed_ms

        if remaining_ms > 0:
            await hooks.emit(
                "orchestrator:rate_limit_delay",
                {
                    "delay_ms": remaining_ms,
                    "configured_ms": self.min_delay_between_calls_ms,
                    "elapsed_ms": elapsed_ms,
                    "iteration": iteration,
                },
            )
            await asyncio.sleep(remaining_ms / 1000)

    def steer(self, message: str) -> None:
        """Queue a steering message for injection at the next iteration boundary.

        Non-blocking. Raises ValueError (empty/whitespace) or SteeringQueueFull.
        This is the target of the ``session.steer`` coordinator capability.
        """
        self._steering_queue.steer(message)

    async def _drain_steering(self, context, hooks, iteration: int) -> int:
        """Drain queued steering messages into context as user-role messages.

        FIFO. Each message is appended via context.add_message({"role":"user",...})
        so the very next get_messages_for_request() picks it up, and an
        orchestrator:steering_injected event is emitted per message. Returns the
        number of messages injected (0 = no-op, no events, streaming undisturbed).
        """
        messages = self._steering_queue.drain()
        if not messages:
            return 0
        total = len(messages)
        for idx, msg in enumerate(messages):
            await context.add_message({"role": "user", "content": msg})
            await hooks.emit(
                "orchestrator:steering_injected",
                {
                    "orchestrator": "loop-streaming",
                    "content": msg,
                    "iteration": iteration,
                    "queued_remaining": total - idx - 1,
                    "metadata": None,
                },
            )
        return total

    async def execute(
        self,
        prompt: str,
        context,
        providers: dict[str, Any],
        tools: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> str:
        """
        Execute with streaming - returns full response but could be modified to stream.

        Note: This is a simplified version. A real streaming implementation would
        need to modify the core interfaces to support AsyncIterator returns.

        SPIKE (docs/designs/goal-command.md): after the first turn, if
        ``coordinator.session_state["goal"]`` is set, this method pursues the
        goal itself -- evaluating after each turn and automatically running
        another turn when not yet satisfied -- so the auto-continue loop
        works for *any* caller (interactive REPL or headless `--mode single`),
        not just the app layer that happens to wrap it in a REPL. When no
        goal is active this is a single pass-through call: zero behavior
        change from the pre-goal implementation.
        """
        # Peek at goal state *before* the first turn. Goal state can only be
        # set (by the app layer's /goal command) before execute() is called,
        # and can only be cleared (never newly set) from within this method's
        # own goal loop below -- so whether a goal is active for this whole
        # execute() invocation is a stable fact determinable once, up front.
        # This lets us tell _execute_one_turn whether ORCHESTRATOR_COMPLETE
        # emission must be deferred (see its `goal_turn` param and
        # _flush_pending_complete below) -- deferred because "was this turn
        # the *final* one" isn't knowable until the evaluator judges its
        # result, which happens only after the turn (and its completion
        # event) would otherwise already have fired.
        initial_goal = coordinator.session_state.get("goal") if coordinator else None
        if initial_goal:
            self._ensure_goal_defaults(initial_goal)
        goal_turn = (initial_goal["turns_used"] + 1) if initial_goal else None

        full_response = await self._execute_one_turn(
            prompt, context, providers, tools, hooks, coordinator, goal_turn=goal_turn
        )

        if coordinator is None:
            return full_response

        # Tracks whether the turn just completed (and about to be evaluated)
        # was itself a continuation turn. Stall bookkeeping (no_tool_turns)
        # only ever counts continuation turns -- the very first turn is never
        # part of that count, since it isn't a re-prompt. Set True right
        # after each continuation _execute_one_turn call below.
        is_continuation_turn = False

        while True:
            goal = coordinator.session_state.get("goal")
            if not goal:
                # No goal active (either never was, or cleared elsewhere
                # mid-turn). Flush any deferred completion as final; a no-op
                # when nothing is pending (the common, non-goal path -- zero
                # behavior change).
                await self._flush_pending_complete(goal_final=True)
                return full_response

            if coordinator.cancellation.is_cancelled:
                coordinator.session_state["goal"] = None
                await self._flush_pending_complete(goal_final=True)
                # No fast-model summary for a user-initiated cancellation --
                # nothing to explain.
                await hooks.emit(
                    "orchestrator:goal_progress",
                    self._goal_progress_payload(
                        goal,
                        state="cancelled",
                        reason=f"condition was: {goal['condition']}",
                    ),
                )
                return full_response

            goal["turns_used"] += 1
            # cap is optional: None/0/absent means unlimited (parity default).
            # A positive int is a hard, Python-enforced mechanical cap. This
            # is only a *flag* here -- checked further down, AFTER
            # evaluation and stall bookkeeping have both run for this turn.
            #
            # DEFECT 1 (fixed): this used to short-circuit the rest of the
            # loop body immediately, before stall bookkeeping ever ran. That
            # meant whenever the turn that would complete a *second*
            # zero-tool streak also happened to hit the cap, the stall
            # judge was never re-consulted and the run silently rode to
            # "cap_hit" instead of "stalled" -- exactly what happened in
            # real session 48adf75a (cap=8): escalate at turn 5, one tool
            # call resets the streak, turns 6/7/8 spin again with
            # near-identical evaluator reasons, and turn 8 -- which should
            # have re-tripped the judge -- was also the cap, so the cap
            # check won the race and the judge was never asked. The
            # mechanical re-arm itself (reset-on-tool-use, reaccumulate)
            # was already correct; the bug was purely in the ordering
            # against the cap check.
            cap = goal.get("cap") or None
            cap_hit = bool(cap and goal["turns_used"] >= cap)

            try:
                satisfied, reason = await self._evaluate_goal(
                    goal["condition"], context, providers, hooks, coordinator
                )
            except Exception as e:
                # FAIL LOUD: never silently keep going, never silently
                # declare success -- regardless of whether this turn also
                # hit the cap. (Previously, an eval failure exactly at the
                # cap boundary was swallowed into a bare "cap_hit" with no
                # reason via a separate "final evaluation at cap" call;
                # now there's only one evaluation call per turn, so a
                # failure here is always reported honestly as "error".)
                coordinator.session_state["goal"] = None
                await self._flush_pending_complete(goal_final=True)
                summary = (
                    await self._summarize_goal_run(
                        goal,
                        providers,
                        hooks,
                        coordinator,
                        "error",
                        error_detail=str(e),
                    )
                    if self._goal_run_needs_summary("error")
                    else None
                )
                await hooks.emit(
                    "orchestrator:goal_progress",
                    self._goal_progress_payload(
                        goal, state="error", reason=str(e), summary=summary
                    ),
                )
                return full_response

            goal["last_reason"] = reason
            goal["reasons"].append(reason)

            if satisfied:
                # Achieved regardless of cap_hit -- the cap merely stops the
                # loop from re-checking again, it never fails a goal that
                # was, in fact, satisfied on its last permitted turn.
                coordinator.session_state["goal"] = None
                await self._flush_pending_complete(goal_final=True)
                summary = (
                    await self._summarize_goal_run(
                        goal, providers, hooks, coordinator, "achieved"
                    )
                    if self._goal_run_needs_summary("achieved")
                    else None
                )
                await hooks.emit(
                    "orchestrator:goal_progress",
                    self._goal_progress_payload(
                        goal, state="achieved", reason=reason, summary=summary
                    ),
                )
                return full_response

            # Stall bookkeeping runs for every completed continuation turn
            # -- INCLUDING the turn that also happens to hit the cap. This
            # is the DEFECT 1 fix: the zero-tool streak, and the judge
            # consultation it can trigger, must re-arm after an escalation
            # regardless of where the cap happens to fall, or the detector
            # can be silently starved by an unlucky cap value (see the
            # comment above `cap_hit`).
            is_stalled = False
            stall_detail: str | None = None
            if is_continuation_turn:
                if self._tool_calls_this_turn == 0:
                    goal["no_tool_turns"] += 1
                else:
                    goal["no_tool_turns"] = 0

                if goal["no_tool_turns"] >= self.goal_stall_threshold:
                    # Condition (a) -- absence of action -- holds. Only now
                    # do we pay for the (rare) stall-judge call to check
                    # condition (b): is this a static, unresolved blocker, or
                    # does it just look repetitive while genuinely
                    # progressing? Both conditions are required to ever trip
                    # -- text-similarity/repetition alone is never enough,
                    # since legitimate agent work (e.g. re-running a test
                    # after a fix) can look repetitive too.
                    try:
                        is_stalled, stall_detail = await self._judge_stall(
                            goal, providers, hooks, coordinator
                        )
                    except Exception as e:
                        # Fail open: a flaky judge call must never itself
                        # manufacture a false stall.
                        logger.warning(
                            f"/goal: stall judge failed, continuing normally: {e}"
                        )
                        is_stalled, stall_detail = False, None

            if is_stalled and (goal["escalated"] or cap_hit):
                # Either this is the second trip (escalation already used
                # and it stalled again), or it's the first trip but there's
                # no cap budget left to offer the one-shot rescue turn.
                # Either way: hard stop, reported as "stalled" rather than
                # "cap_hit" -- we now know definitively the run is stuck,
                # which is more informative than "ran out of turns", even
                # when the cap also happened to run out on this same turn.
                # This is a LOUD failure state, never mistakable for success.
                coordinator.session_state["goal"] = None
                await self._flush_pending_complete(goal_final=True)
                summary = (
                    await self._summarize_goal_run(
                        goal, providers, hooks, coordinator, "stalled"
                    )
                    if self._goal_run_needs_summary("stalled")
                    else None
                )
                await hooks.emit(
                    "orchestrator:goal_progress",
                    self._goal_progress_payload(
                        goal,
                        state="stalled",
                        reason=reason,
                        stall_detail=stall_detail,
                        summary=summary,
                    ),
                )
                return full_response

            if is_stalled:
                # First trip, and cap budget remains: one-shot escalation --
                # give the agent a single explicit chance to change approach
                # or admit the goal can't be met as defined, before
                # hard-stopping.
                goal["escalated"] = True
                await self._flush_pending_complete(goal_final=False)
                await hooks.emit(
                    "orchestrator:goal_progress",
                    self._goal_progress_payload(
                        goal, state="continuing", reason=reason
                    ),
                )
                goal["continuations"] += 1
                stall_prompt = (
                    f"You've been asked to work toward this goal: "
                    f"{goal['condition']}\n\n"
                    f"For the last {goal['no_tool_turns']} turns you "
                    "haven't taken any actions (no tool calls), and "
                    "the evaluator keeps reporting the same kind of "
                    f"blocker: {reason}\n\n"
                    "You appear stuck. Either try a genuinely "
                    "different approach to make progress, or, if you "
                    "believe this goal cannot be achieved as it's "
                    "currently defined, say so plainly and explain "
                    "specifically why -- don't just repeat what "
                    "you've already said."
                )
                full_response = await self._execute_one_turn(
                    stall_prompt,
                    context,
                    providers,
                    tools,
                    hooks,
                    coordinator,
                    goal_turn=goal["turns_used"] + 1,
                )
                is_continuation_turn = True
                continue

            if cap_hit:
                # Not stalled (or the mechanical streak hadn't reached
                # threshold this turn) -- the cap simply ran out. `reason`
                # is already known from the evaluation above; no separate
                # "final" evaluation call is needed since evaluation now
                # always runs before the cap is checked.
                coordinator.session_state["goal"] = None
                await self._flush_pending_complete(goal_final=True)
                summary = await self._summarize_goal_run(
                    goal, providers, hooks, coordinator, "cap_hit"
                )
                await hooks.emit(
                    "orchestrator:goal_progress",
                    self._goal_progress_payload(
                        goal, state="cap_hit", reason=reason, summary=summary
                    ),
                )
                return full_response

            await self._flush_pending_complete(goal_final=False)
            await hooks.emit(
                "orchestrator:goal_progress",
                self._goal_progress_payload(goal, state="continuing", reason=reason),
            )

            goal["continuations"] += 1
            full_response = await self._execute_one_turn(
                reason,
                context,
                providers,
                tools,
                hooks,
                coordinator,
                goal_turn=goal["turns_used"] + 1,
            )
            is_continuation_turn = True

    async def _execute_one_turn(
        self,
        prompt: str,
        context,
        providers: dict[str, Any],
        tools: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
        *,
        goal_turn: int | None = None,
    ) -> str:
        """Run exactly one turn (the pre-goal-loop behavior of ``execute()``).

        This is the unmodified inner execution: accumulate ``_execute_stream``
        output, emit ``ORCHESTRATOR_COMPLETE``, return the response. The goal
        loop in ``execute()`` calls this once per turn, including the
        evaluator-driven auto-continue turns.

        Args:
            goal_turn: When this turn is part of an active /goal auto-continue
                pursuit (spike: docs/designs/goal-command.md), the 1-based
                goal-turn number; ``None`` when no goal is active. When
                ``None`` (the default), ``ORCHESTRATOR_COMPLETE`` is emitted
                immediately as before -- zero behavior change. When set, the
                caller (``execute()``'s goal loop) doesn't yet know whether
                this is the *final* turn of the pursuit (that depends on the
                evaluator judging this turn's result, which hasn't happened
                yet), so emission is deferred via
                ``self._pending_orchestrator_complete`` -- the caller must
                call ``self._flush_pending_complete(goal_final=...)`` once it
                knows.
        """
        # Reset cancellation event tracking for this execution
        self._cancel_requested_emitted = False
        # Clear any steering messages that accumulated before this turn.
        # Steers do not cross turn boundaries — a stale steer from a prior turn or
        # a cancelled turn must never silently ride into a fresh turn. (spec §5.2)
        self._steering_queue.clear()
        # Reset the per-turn tool-call counter (see _execute_tool_only /
        # _execute_tool_with_result, the actual tool-execution paths that
        # increment it) so execute()'s stall detection can read an accurate
        # "did this turn run any tools" count once this method returns.
        self._tool_calls_this_turn = 0
        full_response = ""
        iteration_count = 0
        error: Exception | None = None

        try:
            async for token, iteration in self._execute_stream(
                prompt, context, providers, tools, hooks, coordinator
            ):
                full_response += token
                iteration_count = iteration
        except Exception as e:
            error = e

        # Always emit orchestrator complete event (observability)
        if error:
            status = "error"
        elif coordinator and coordinator.cancellation.is_cancelled:
            status = "cancelled"
        else:
            status = "success" if full_response else "incomplete"

        # Read the active goal's continuations count (if any) for the
        # payload below. None when no goal is active -- same discriminator
        # pattern as goal_turn/goal_final.
        goal_state = coordinator.session_state.get("goal") if coordinator else None

        payload = {
            "orchestrator": "loop-streaming",
            "turn_count": iteration_count,
            "status": status,
            # Discriminator fields (spike: docs/designs/goal-command.md) so
            # consumers that treat ORCHESTRATOR_COMPLETE as "end of turn"
            # (e.g. amplifierd's MetadataSaveHook, amplifier-voice's
            # delegate_agent_completed mapping) can filter to the single
            # emission that corresponds to a real user turn, instead of
            # counting every goal-continuation iteration.
            "goal_turn": goal_turn,
            "goal_final": goal_turn is None,
            # Times sent back to the assistant so far in the active /goal
            # pursuit; None when no goal is active (see goal_state lookup
            # above -- additive field, same discriminator pattern as
            # goal_turn/goal_final).
            "continuations": goal_state.get("continuations") if goal_state else None,
        }
        if goal_turn is None:
            await hooks.emit(ORCHESTRATOR_COMPLETE, payload)
        else:
            self._pending_orchestrator_complete = (hooks, payload)

        if error:
            raise error

        return full_response

    async def _flush_pending_complete(self, *, goal_final: bool) -> None:
        """Emit a deferred ``ORCHESTRATOR_COMPLETE`` from ``_execute_one_turn``.

        No-op when nothing is pending (the common, non-goal path). See the
        ``goal_turn`` param of ``_execute_one_turn`` for why emission is
        deferred for goal-driven turns.
        """
        pending = self._pending_orchestrator_complete
        if pending is None:
            return
        self._pending_orchestrator_complete = None
        hooks, payload = pending
        payload["goal_final"] = goal_final
        await hooks.emit(ORCHESTRATOR_COMPLETE, payload)

    @staticmethod
    def _ensure_goal_defaults(goal: dict[str, Any]) -> None:
        """Backfill goal-state keys added after the original 4 (condition,
        turns_used, last_reason, cap) via ``setdefault``.

        This is version tolerance on a plain dict -- an older caller (e.g. an
        app-cli built against the original spike) that only sets the
        original 4 keys must keep working unmodified. It is not a fallback
        hiding an error: every key here has a well-defined zero value.
        """
        goal.setdefault("reasons", [])
        goal.setdefault("continuations", 0)
        goal.setdefault("no_tool_turns", 0)
        goal.setdefault("escalated", False)

    def _select_cheap_model(self, provider_name: str) -> str | None:
        """Best-effort cheap/fast model selection for goal-loop LLM calls
        (evaluator, stall judge, run summary).

        Uses the hardcoded substring hint table because no coordinator
        capability for model-role routing is reachable from here -- see the
        note above ``_GOAL_CHEAP_MODEL_HINTS`` for what was investigated.
        Logs a WARNING on a miss so silently falling back to the provider's
        (potentially expensive) default model is visible instead of silent.
        """
        for hint_key, hint_model in self._GOAL_CHEAP_MODEL_HINTS.items():
            if hint_key in provider_name.lower():
                return hint_model
        logger.warning(
            f"/goal: no cheap-model hint for provider '{provider_name}' -- "
            "falling back to the provider's default model for this "
            "evaluator/stall-judge/summary call, which may be more expensive "
            "than intended. Add a hint to _GOAL_CHEAP_MODEL_HINTS."
        )
        return None

    @staticmethod
    def _goal_run_needs_summary(final_state: str) -> bool:
        """Whether the terminal ``goal_progress`` event's ``summary`` field
        is worth generating for this final state.

        ``achieved`` never needs one, regardless of how many continuations
        the run took (extended DEFECT 3 fix): the CLI renders no prose at
        all on success -- the user watched the whole run happen live -- so
        paying for the fast-model summary call there is pure latency (5-12s
        measured) and cost for a value that's never displayed. This used to
        only skip the zero-continuations case; it now skips unconditionally
        for ``achieved``, since even a multi-continuation success has
        nothing worth saying that the structured `continuations` field
        doesn't already say.

        Every other terminal state -- ``stalled``, ``cap_hit``, ``error`` --
        is exactly the case where the summary tells the developer something
        they could not see live (what blocked it / what's left / what
        broke), so those always get one -- falling back to a deterministic
        per-state string (see ``_goal_summary_fallback``) if generation
        itself fails. (``cancelled`` never reaches this: it's
        short-circuited before any summary call, same as before.)
        """
        return final_state != "achieved"

    @staticmethod
    def _normalize_reason(reason: str) -> str:
        """Exact-ish normalization of one evaluator reason: collapse
        whitespace, strip, lowercase. Deliberately NOT fuzzy/similarity
        based -- shared by ``_dedupe_consecutive_reasons`` (rendering) and
        ``_distinct_blocker_count`` (the give-up justification count), so
        both agree on what counts as "the same" reason.
        """
        return " ".join(reason.split()).strip().lower()

    @staticmethod
    def _dedupe_consecutive_reasons(reasons: list[str]) -> list[str]:
        """Collapse consecutive runs of identical/near-identical evaluator
        reasons into a single entry annotated with a repeat count, for the
        rendered ``goal_progress`` payload.

        An unsatisfiable-goal run can produce 5-8 near-identical sentences
        in a row (the evaluator keeps reporting the same blocker every
        turn); consumers render ``reasons`` as a list, so N verbatim copies
        is noise, not signal -- the signal is *that* it repeated and *how
        many* times. "Near-identical" here means equal after collapsing
        whitespace and case (see ``_normalize_reason``); the first
        occurrence's original text is kept verbatim in the collapsed entry.

        This only affects what gets rendered in the payload -- the raw,
        uncollapsed per-turn history is untouched on ``goal["reasons"]``
        itself (read by ``_judge_stall``, ``_summarize_goal_run``, and
        anything else that wants the full detail).
        """
        if not reasons:
            return []

        _norm = StreamingOrchestrator._normalize_reason

        collapsed: list[str] = []
        run_text = reasons[0]
        run_norm = _norm(reasons[0])
        run_count = 1

        for r in reasons[1:]:
            if _norm(r) == run_norm:
                run_count += 1
            else:
                collapsed.append(
                    run_text
                    if run_count == 1
                    else f"{run_text} (repeated {run_count}x)"
                )
                run_text = r
                run_norm = _norm(r)
                run_count = 1

        collapsed.append(
            run_text if run_count == 1 else f"{run_text} (repeated {run_count}x)"
        )
        return collapsed

    @staticmethod
    def _distinct_blocker_count(reasons: list[str]) -> int:
        """Count of distinct evaluator-reason signatures across the
        *entire* run, after the same exact-ish normalization used by
        ``_dedupe_consecutive_reasons`` (whitespace/case only -- NOT fuzzy
        similarity).

        This is the sole justification a consumer has for distinguishing
        "hit a wall" (the evaluator reported the same blocker every turn --
        count of 1) from "flailing" (a different blocker each turn -- count
        approaching the number of turns). Getting this number wrong is
        worse than not reporting it at all, since it would assert a false
        narrative about *why* the run gave up -- hence exact normalization
        only, never a similarity heuristic that could quietly merge two
        genuinely different blockers or split one blocker's rephrasing
        into two.

        Unlike ``_dedupe_consecutive_reasons`` (which only collapses
        *consecutive* repeats for rendering), this counts distinct
        signatures across the whole list regardless of position -- an
        A-B-A pattern is 2 distinct blockers, not 3.
        """
        if not reasons:
            return 0
        return len({StreamingOrchestrator._normalize_reason(r) for r in reasons})

    def _goal_progress_payload(
        self,
        goal: dict[str, Any],
        *,
        state: str,
        reason: str | None,
        stall_detail: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Build the exact ``orchestrator:goal_progress`` payload contract.

        Centralized so every emission site (continuing/achieved/cap_hit/
        cancelled/error/stalled) produces an identical shape -- app-cli is
        built in parallel against this exact contract.
        """
        reasons = list(goal.get("reasons", []))
        return {
            "orchestrator": "loop-streaming",
            "state": state,
            "turn": goal["turns_used"],
            "continuations": goal.get("continuations", 0),
            "cap": goal.get("cap") or None,
            "reason": reason,
            "reasons": self._dedupe_consecutive_reasons(reasons),
            "stall_detail": stall_detail,
            "summary": summary,
            # Additive field (see _distinct_blocker_count): count of
            # distinct evaluator-reason signatures across the whole run --
            # distinguishes "hit a wall" (1) from "flailing" (N).
            "distinct_blockers": self._distinct_blocker_count(reasons),
            "metadata": None,
        }

    async def _judge_stall(
        self,
        goal: dict[str, Any],
        providers: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> tuple[bool, str | None]:
        """Ask a cheap, tool-less model whether the recent run of no-tool-turn
        evaluator reasons describes a static, unresolved blocker (a stall) or
        genuine incremental progress despite no tool calls.

        Only called from execute()'s stall-detection block, and only once the
        mechanical condition (``no_tool_turns >= self.goal_stall_threshold``)
        already holds -- that's what keeps this rare and cheap. Returns
        (is_stalled, detail). Raises on failure; the caller treats a raised
        exception as "not stalled" (fail open on the judge -- a flaky judge
        call must never itself manufacture a false stall; the mechanical
        absence-of-action condition is what keeps stall detection safe).
        """
        recent_reasons = (
            goal["reasons"][-goal["no_tool_turns"] :] if goal.get("reasons") else []
        )
        if not recent_reasons:
            return False, None

        if not providers:
            raise RuntimeError("no provider mounted for stall judgment")
        provider_name, provider = next(iter(providers.items()))
        model_override = self._select_cheap_model(provider_name)

        history_text = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(recent_reasons))
        user_prompt = (
            f"GOAL CONDITION:\n{goal['condition']}\n\n"
            f"EVALUATOR REASONS across the last {len(recent_reasons)} turns, "
            "during which the assistant took no tool actions at all:\n"
            f"{history_text}\n\n"
            "Is this a stall? Respond in the required two-line format."
        )

        judge_messages = [
            Message(role="system", content=self._GOAL_STALL_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]
        chat_request = ChatRequest(
            messages=judge_messages, tools=None, model=model_override
        )

        request_result = await hooks.emit(
            PROVIDER_REQUEST, {"provider": provider_name, "iteration": 0}
        )
        if coordinator:
            request_result = await coordinator.process_hook_result(
                request_result, "provider:request", "orchestrator"
            )
            if request_result.action == "deny":
                raise RuntimeError(
                    f"stall judge call denied by hook: {request_result.reason}"
                )

        # DEFECT 2 fix: `model_override` was previously only set on
        # `ChatRequest.model`, never passed as a `model=` kwarg to
        # `provider.complete()`. The installed Anthropic provider's
        # `complete()` / `_complete_chat_request()` reads the effective
        # model from `kwargs.get("model", self.default_model)` -- it never
        # reads `request.model` at all -- so the override was silently
        # dropped and this call ran on the session's default (expensive)
        # model every time. Passing it as a kwarg too, in addition to the
        # ChatRequest field, is correct regardless of which convention a
        # given provider implementation happens to honor.
        complete_kwargs: dict[str, Any] = (
            {"model": model_override} if model_override else {}
        )
        try:
            response = await provider.complete(chat_request, **complete_kwargs)
        except LLMError as e:
            await hooks.emit(
                PROVIDER_ERROR,
                {
                    "provider": provider_name,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                    "retryable": e.retryable,
                    "status_code": e.status_code,
                },
            )
            raise
        except Exception as e:
            await hooks.emit(
                PROVIDER_ERROR,
                {
                    "provider": provider_name,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                },
            )
            raise

        response_text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                response_text += getattr(block, "text", "")

        lines = [
            line.strip() for line in response_text.strip().splitlines() if line.strip()
        ]
        if not lines:
            raise ValueError("stall judge returned an empty response")

        verdict = lines[0].strip().upper()
        if verdict not in ("YES", "NO"):
            raise ValueError(f"unparseable stall judge verdict: {lines[0]!r}")

        detail = lines[1] if len(lines) > 1 else "(judge gave no reason)"
        return verdict == "YES", detail

    @staticmethod
    def _enforce_summary_length(text: str, max_chars: int | None = None) -> str:
        """Hard-truncate ``text`` to at most ``max_chars`` (default
        ``_GOAL_SUMMARY_MAX_CHARS``), breaking at the last whole word rather
        than mid-word.

        The per-state system prompts ask the model for "no more than about
        120 characters", but a model instruction is not a guarantee -- this
        is the code-level backstop that actually enforces it.
        """
        cap = (
            max_chars
            if max_chars is not None
            else StreamingOrchestrator._GOAL_SUMMARY_MAX_CHARS
        )
        text = text.strip()
        if len(text) <= cap:
            return text
        truncated = text[:cap]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated.rstrip()

    @staticmethod
    def _goal_summary_fallback(goal: dict[str, Any], final_state: str) -> str | None:
        """Deterministic per-state fallback string used when summary
        generation fails outright, or the model returns empty text, so the
        terminal ``goal_progress`` event's ``summary`` field is never
        silently ``None`` for a state where the developer genuinely needs
        an explanation.

        ``achieved`` never reaches ``_summarize_goal_run`` at all (see
        ``_goal_run_needs_summary``) so it has no fallback entry here; a
        final state outside the known three returns ``None`` as a safety
        net rather than inventing text for a state that shouldn't be
        calling this in the first place.
        """
        if final_state == "stalled":
            no_tool_turns = goal.get("no_tool_turns", 0)
            return f"no progress across the last {no_tool_turns} turns"
        if final_state == "cap_hit":
            return "completion not confirmed before the cap"
        if final_state == "error":
            return "evaluator failed"
        return None

    async def _summarize_goal_run(
        self,
        goal: dict[str, Any],
        providers: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None,
        final_state: str,
        *,
        error_detail: str | None = None,
    ) -> str | None:
        """Best-effort fast-model, one-sentence explanation of a completed
        /goal run, for the terminal ``goal_progress`` event's ``summary``
        field.

        Uses the per-state prompt from ``_GOAL_SUMMARY_SYSTEM_PROMPTS`` --
        ``final_state`` must be one of ``"stalled"``, ``"cap_hit"``, or
        ``"error"`` (``achieved`` never reaches this method at all, see
        ``_goal_run_needs_summary``; ``cancelled`` is short-circuited in
        ``execute()`` before any summary call). ``error_detail`` is the
        stringified exception, required for the ``"error"`` state's prompt.

        Never raises and never returns ``None`` for a known final_state --
        on any failure, or an empty model response, this falls back to a
        deterministic per-state string (``_goal_summary_fallback``) so the
        payload's ``summary`` field always carries *something* actionable
        rather than silently going missing. The length cap is enforced in
        code (``_enforce_summary_length``) regardless of what the model
        actually returned.
        """
        try:
            if not providers:
                raise RuntimeError("no provider mounted for summary")
            provider_name, provider = next(iter(providers.items()))
            model_override = self._select_cheap_model(provider_name)

            system_prompt = self._GOAL_SUMMARY_SYSTEM_PROMPTS[final_state]

            if final_state == "error":
                user_prompt = (
                    f"GOAL CONDITION:\n{goal['condition']}\n\n"
                    "ERROR RAISED BY THE EVALUATOR:\n"
                    f"{error_detail or '(no error detail captured)'}\n\n"
                    "Write the one-sentence line now."
                )
            else:
                reasons = goal.get("reasons", [])
                reasons_text = (
                    "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reasons))
                    if reasons
                    else "(no evaluator reasons recorded)"
                )
                user_prompt = (
                    f"GOAL CONDITION:\n{goal['condition']}\n\n"
                    f"Continuations (times sent back): {goal.get('continuations', 0)}\n"
                    f"EVALUATOR REASONS in order:\n{reasons_text}\n\n"
                    "Write the one-sentence line now."
                )

            summary_messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            chat_request = ChatRequest(
                messages=summary_messages, tools=None, model=model_override
            )

            request_result = await hooks.emit(
                PROVIDER_REQUEST, {"provider": provider_name, "iteration": 0}
            )
            if coordinator:
                request_result = await coordinator.process_hook_result(
                    request_result, "provider:request", "orchestrator"
                )
                if request_result.action == "deny":
                    raise RuntimeError(
                        f"summary call denied by hook: {request_result.reason}"
                    )

            # DEFECT 2 fix: see the matching comment in _judge_stall -- the
            # ChatRequest.model field alone is not honored by the installed
            # provider; the model must also be passed as a kwarg.
            complete_kwargs: dict[str, Any] = (
                {"model": model_override} if model_override else {}
            )
            response = await provider.complete(chat_request, **complete_kwargs)
            summary_text = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    summary_text += getattr(block, "text", "")
            summary_text = summary_text.strip()
            if not summary_text:
                return self._goal_summary_fallback(goal, final_state)
            return self._enforce_summary_length(summary_text)
        except Exception as e:
            logger.warning(
                f"/goal: summary generation failed, falling back to a "
                f"deterministic message: {e}"
            )
            return self._goal_summary_fallback(goal, final_state)

    async def _evaluate_goal(
        self,
        condition: str,
        context,
        providers: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> tuple[bool, str]:
        """Ask a cheap, tool-less model whether ``condition`` is satisfied.

        Ported from amplifier-app-cli's `_evaluate_goal` (see
        docs/designs/goal-command.md). Returns (satisfied, reason). Raises on
        any failure -- no fallbacks; the caller (execute()'s goal loop) stops
        the goal loop loudly on any exception here.

        Mirrors ``_execute_stream``'s own provider-call instrumentation
        (PROVIDER_REQUEST + process_hook_result deny-check, PROVIDER_ERROR on
        failure -- see :~605-609, :~732-750) around this method's provider
        call, so hooks that gate LLM calls (approval, cost-aware routing,
        rate limiting) see and can govern the evaluator's call too, instead
        of it silently bypassing the hook pipeline entirely.
        """
        if not context or not hasattr(context, "get_messages"):
            raise RuntimeError("no context manager available to read the conversation")
        all_messages = await context.get_messages()

        truncated = False
        messages = all_messages
        if len(messages) > self._GOAL_MAX_TRANSCRIPT_MESSAGES:
            messages = messages[-self._GOAL_MAX_TRANSCRIPT_MESSAGES :]
            truncated = True

        transcript_text = "\n\n".join(
            t for t in (_flatten_message_for_evaluator(m) for m in messages) if t
        )

        if not providers:
            raise RuntimeError("no provider mounted for evaluation")
        provider_name, provider = next(iter(providers.items()))

        model_override = self._select_cheap_model(provider_name)

        user_prompt = (
            f"GOAL CONDITION:\n{condition}\n\n"
            f"CONVERSATION SO FAR"
            f"{' (truncated, most recent messages shown)' if truncated else ''}:\n"
            f"{transcript_text}\n\n"
            "Has the GOAL CONDITION been satisfied? Respond in the required "
            "two-line format."
        )

        eval_messages = [
            Message(role="system", content=self._GOAL_EVALUATOR_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]
        chat_request = ChatRequest(
            messages=eval_messages, tools=None, model=model_override
        )

        # Mirror _execute_stream's provider:request instrumentation so hooks
        # that gate LLM calls (approval, cost-aware routing, rate limiting)
        # see and can govern the evaluator's call too.
        request_result = await hooks.emit(
            PROVIDER_REQUEST, {"provider": provider_name, "iteration": 0}
        )
        if coordinator:
            request_result = await coordinator.process_hook_result(
                request_result, "provider:request", "orchestrator"
            )
            if request_result.action == "deny":
                raise RuntimeError(
                    f"evaluator call denied by hook: {request_result.reason}"
                )

        # DEFECT 2 fix: see the matching comment in _judge_stall -- the
        # ChatRequest.model field alone is not honored by the installed
        # provider; the model must also be passed as a kwarg.
        complete_kwargs: dict[str, Any] = (
            {"model": model_override} if model_override else {}
        )
        try:
            response = await provider.complete(chat_request, **complete_kwargs)
        except LLMError as e:
            await hooks.emit(
                PROVIDER_ERROR,
                {
                    "provider": provider_name,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                    "retryable": e.retryable,
                    "status_code": e.status_code,
                },
            )
            raise
        except Exception as e:
            await hooks.emit(
                PROVIDER_ERROR,
                {
                    "provider": provider_name,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                },
            )
            raise

        response_text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                response_text += getattr(block, "text", "")

        lines = [
            line.strip() for line in response_text.strip().splitlines() if line.strip()
        ]
        if not lines:
            raise ValueError("evaluator returned an empty response")

        verdict = lines[0].strip().upper()
        if verdict not in ("YES", "NO"):
            raise ValueError(f"unparseable evaluator verdict: {lines[0]!r}")

        reason = lines[1] if len(lines) > 1 else "(evaluator gave no reason)"
        return verdict == "YES", reason

    async def _execute_stream(
        self,
        prompt: str,
        context,
        providers: dict[str, Any],
        tools: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> AsyncIterator[tuple[str, int]]:
        """
        Internal streaming execution.
        Yields tuples of (token, iteration) as they're generated.
        """
        # Emit and process prompt submit (allows hooks to inject context before processing)
        prompt_submit_result = await hooks.emit(PROMPT_SUBMIT, {"prompt": prompt})
        if coordinator:
            prompt_submit_result = await coordinator.process_hook_result(
                prompt_submit_result, "prompt:submit", "orchestrator"
            )
            if prompt_submit_result.action == "deny":
                yield (f"Operation denied: {prompt_submit_result.reason}", 0)
                return

        # Store ephemeral injection from prompt:submit for use in the loop
        # (must be stored before provider:request overwrites 'result')
        if (
            prompt_submit_result.action == "inject_context"
            and prompt_submit_result.ephemeral
            and prompt_submit_result.context_injection
        ):
            self._pending_ephemeral_injections.append(
                {
                    "role": prompt_submit_result.context_injection_role,
                    "content": prompt_submit_result.context_injection,
                    "append_to_last_tool_result": prompt_submit_result.append_to_last_tool_result,
                }
            )
            logger.debug(
                "Stored ephemeral injection from prompt:submit for first iteration"
            )

        # Emit execution start
        await hooks.emit("execution:start", {"prompt": prompt})

        # Reset rate limit tracking for new session
        self._last_provider_call_end = None

        # Add user message
        await context.add_message({"role": "user", "content": prompt})

        # Select provider
        provider = self._select_provider(providers)
        if not provider:
            yield ("Error: No providers available", 0)
            return

        # Find provider name for event emission
        provider_name = None
        for name, prov in providers.items():
            if prov is provider:
                provider_name = name
                break

        iteration = 0

        while self.max_iterations == -1 or iteration < self.max_iterations:
            # Check for cancellation at iteration start
            if coordinator and coordinator.cancellation.is_cancelled:
                # Emit cancel:requested on first detection and trigger cleanup callbacks
                if not self._cancel_requested_emitted:
                    self._cancel_requested_emitted = True
                    await hooks.emit(
                        CANCEL_REQUESTED,
                        {
                            "orchestrator": "loop-streaming",
                            "state": str(coordinator.cancellation.state),
                            "turn_count": iteration,
                        },
                    )
                    try:
                        await coordinator.cancellation.trigger_callbacks()
                    except Exception as e:
                        logger.warning(f"Error in cancellation callbacks: {e}")
                # Emit cancel:completed — orchestrator is exiting due to cancellation
                await hooks.emit(
                    CANCEL_COMPLETED,
                    {
                        "orchestrator": "loop-streaming",
                        "was_immediate": coordinator.cancellation.is_immediate,
                        "turn_count": iteration,
                    },
                )
                # Don't yield more content, just exit.
                # Clear any pending steers so they cannot leak into the next turn
                # (cancellation means "stop now" — stale steers have no next injection
                # point and must not silently ride a future, unrelated turn). (spec §5.2)
                self._steering_queue.clear()
                return

            iteration += 1

            # Mid-turn steering: drain queued user messages BEFORE building the request,
            # so they are part of this iteration's provider call. At iteration 1 this is
            # "before the first LLM call"; at iteration N>1 this is "after the prior tool
            # round, before the next provider call" — the single natural boundary.
            await self._drain_steering(context, hooks, iteration)

            # Emit provider request BEFORE getting messages (allows hook injections)
            result = await hooks.emit(
                PROVIDER_REQUEST, {"provider": provider_name, "iteration": iteration}
            )
            if coordinator:
                result = await coordinator.process_hook_result(
                    result, "provider:request", "orchestrator"
                )
                if result.action == "deny":
                    yield (f"Operation denied: {result.reason}", iteration)
                    return

            # Get messages for LLM request (context handles compaction internally)
            # Pass provider for dynamic budget calculation based on model's context window
            message_dicts = await context.get_messages_for_request(provider=provider)
            message_dicts = list(message_dicts)  # Convert to list for modification

            # Append ephemeral injection if present (temporary, not stored)
            if (
                result.action == "inject_context"
                and result.ephemeral
                and result.context_injection
            ):
                # Check if we should append to last tool result
                if result.append_to_last_tool_result and len(message_dicts) > 0:
                    last_msg = message_dicts[-1]
                    # Append to last message if it's a tool result
                    if last_msg.get("role") == "tool":
                        # Append to existing content
                        original_content = last_msg.get("content", "")
                        message_dicts[-1] = {
                            **last_msg,
                            "content": f"{original_content}\n\n{result.context_injection}",
                        }
                        logger.debug(
                            "Appended ephemeral injection to last tool result message"
                        )
                    else:
                        # Fall back to new message if last message isn't a tool result
                        message_dicts.append(
                            {
                                "role": result.context_injection_role,
                                "content": result.context_injection,
                            }
                        )
                        logger.debug(
                            f"Last message role is '{last_msg.get('role')}', not 'tool' - "
                            "created new message for injection"
                        )
                else:
                    # Default behavior: append as new message
                    message_dicts.append(
                        {
                            "role": result.context_injection_role,
                            "content": result.context_injection,
                        }
                    )

            # Apply pending ephemeral injections from tool:post hooks
            if self._pending_ephemeral_injections:
                for injection in self._pending_ephemeral_injections:
                    if (
                        injection.get("append_to_last_tool_result")
                        and len(message_dicts) > 0
                    ):
                        last_msg = message_dicts[-1]
                        if last_msg.get("role") == "tool":
                            original_content = last_msg.get("content", "")
                            message_dicts[-1] = {
                                **last_msg,
                                "content": f"{original_content}\n\n{injection['content']}",
                            }
                            logger.debug(
                                "Applied pending ephemeral injection to last tool result"
                            )
                        else:
                            message_dicts.append(
                                {
                                    "role": injection["role"],
                                    "content": injection["content"],
                                }
                            )
                            logger.debug(
                                "Last message not a tool result, created new message for injection"
                            )
                    else:
                        message_dicts.append(
                            {"role": injection["role"], "content": injection["content"]}
                        )
                        logger.debug(
                            "Applied pending ephemeral injection as new message"
                        )
                # Clear pending injections after applying
                self._pending_ephemeral_injections = []

            # Convert dicts to ChatRequest for provider
            messages_objects = [Message(**msg) for msg in message_dicts]

            # Convert tools to ToolSpec format for ChatRequest
            tools_list = None
            if tools:
                tools_list = [
                    ToolSpec(
                        name=t.name,
                        description=t.description,
                        parameters=t.input_schema,
                    )
                    for t in tools.values()
                ]

            chat_request = ChatRequest(
                messages=messages_objects,
                tools=tools_list,
                reasoning_effort=self.config.get("reasoning_effort"),
            )
            logger.info(
                f"[ORCHESTRATOR] ChatRequest created with {len(tools_list) if tools_list else 0} tools"
            )
            if tools_list:
                logger.debug(
                    f"[ORCHESTRATOR] Tool names: {[t.name for t in tools_list]}"
                )

            # Apply rate limit delay before provider call
            await self._apply_rate_limit_delay(hooks, iteration)

            # Check if provider supports streaming
            if hasattr(provider, "stream"):
                # Use streaming if available
                async for chunk in self._stream_from_provider(
                    provider,
                    chat_request,
                    context,
                    tools,
                    hooks,
                    coordinator,
                    provider_name=provider_name,
                ):
                    # Check for immediate cancellation between chunks
                    if coordinator and coordinator.cancellation.is_immediate:
                        # Clear pending steers: immediate cancellation ends the turn,
                        # and any steer queued during streaming must not leak into a
                        # future turn — matching the other cancellation exits. (spec §5.2)
                        self._steering_queue.clear()
                        return
                    yield (chunk, iteration)

                # Update rate limit timestamp after streaming completes
                self._last_provider_call_end = time.monotonic()

                # Check for tool calls after streaming
                # This is simplified - real implementation would parse during stream
                if await self._has_pending_tools(context):
                    # Process tools
                    await self._process_tools(context, tools, hooks)
                    continue
                else:
                    # Last-drain edge: if a steer arrived during the final generation,
                    # loop once more so the model acts on it this turn. The top-of-
                    # iteration drain performs the actual injection.
                    if not self._steering_queue.is_empty:
                        continue
                    break
            else:
                # Fallback to non-streaming
                # Build kwargs for provider
                kwargs = {}
                if self.extended_thinking:
                    kwargs["extended_thinking"] = True
                try:
                    response = await provider.complete(chat_request, **kwargs)
                except LLMError as e:
                    await hooks.emit(
                        PROVIDER_ERROR,
                        {
                            "provider": provider_name,
                            "error": {"type": type(e).__name__, "msg": str(e)},
                            "retryable": e.retryable,
                            "status_code": e.status_code,
                        },
                    )
                    raise
                except Exception as e:
                    await hooks.emit(
                        PROVIDER_ERROR,
                        {
                            "provider": provider_name,
                            "error": {"type": type(e).__name__, "msg": str(e)},
                        },
                    )
                    raise

                # Update rate limit timestamp after non-streaming response
                self._last_provider_call_end = time.monotonic()

                # Emit content block events if present
                content_blocks = getattr(response, "content_blocks", None)
                if content_blocks:
                    total_blocks = len(content_blocks)
                    for idx, block in enumerate(content_blocks):
                        # Emit block start
                        await hooks.emit(
                            CONTENT_BLOCK_START,
                            {
                                "block_type": block.type.value,
                                "block_index": idx,
                                "total_blocks": total_blocks,
                                "metadata": getattr(block, "raw", None),
                            },
                        )

                        # Emit block end with complete block, usage, and total count
                        event_data = {
                            "block_index": idx,
                            "total_blocks": total_blocks,
                            "block": block.to_dict(),
                        }
                        if response.usage:
                            event_data["usage"] = response.usage.model_dump()
                        await hooks.emit(CONTENT_BLOCK_END, event_data)
                elif response.content and isinstance(response.content, list):
                    # Fallback for providers that populate response.content
                    # (Pydantic ContentBlock models) but not content_blocks
                    # (raw SDK objects). Synthesize content_block events so
                    # downstream hooks (e.g. streaming-ui token usage) fire.
                    total_blocks = len(response.content)
                    for idx, block in enumerate(response.content):
                        block_dict = (
                            block.model_dump()
                            if hasattr(block, "model_dump")
                            else block
                        )
                        block_type = (
                            block_dict.get("type", "text")
                            if isinstance(block_dict, dict)
                            else "text"
                        )
                        await hooks.emit(
                            CONTENT_BLOCK_START,
                            {
                                "block_type": block_type,
                                "block_index": idx,
                                "total_blocks": total_blocks,
                            },
                        )
                        event_data = {
                            "block_index": idx,
                            "total_blocks": total_blocks,
                            "block": block_dict,
                        }
                        if response.usage:
                            event_data["usage"] = response.usage.model_dump()
                        await hooks.emit(CONTENT_BLOCK_END, event_data)

                # Parse tool calls
                tool_calls = provider.parse_tool_calls(response)

                if not tool_calls:
                    # Extract text content from response for streaming
                    # Use .text field if available (e.g., OpenAI provider), otherwise extract from content blocks
                    if hasattr(response, "text") and response.text:
                        response_text = response.text
                    else:
                        response_text = self._extract_text_from_content(
                            response.content
                        )

                    # Stream the final response token by token
                    async for token in self._tokenize_stream(response_text):
                        yield (token, iteration)

                    # Store structured content from response.content (our Pydantic models)
                    # This preserves reasoning state, thinking blocks, etc.
                    # response.content = list of our ContentBlock models (TextBlock, ThinkingBlock, etc.)
                    # response.content_blocks = raw SDK objects (for streaming events only)
                    response_content = getattr(response, "content", None)
                    if response_content and isinstance(response_content, list):
                        # Convert ContentBlock objects to dicts for serialization
                        content_dicts = [
                            block.model_dump()
                            if hasattr(block, "model_dump")
                            else block
                            for block in response_content
                        ]
                        logger.info(
                            f"[ORCHESTRATOR] Storing {len(content_dicts)} content blocks"
                        )
                        for i, block_dict in enumerate(content_dicts):
                            logger.info(
                                f"[ORCHESTRATOR]   Block {i}: type={block_dict.get('type')}, has_content={'content' in block_dict}"
                            )
                        assistant_msg = {
                            "role": "assistant",
                            "content": content_dicts,
                        }
                    else:
                        assistant_msg = {
                            "role": "assistant",
                            "content": response_text,
                        }

                    # Preserve thinking blocks for Anthropic extended thinking (backward compat)
                    # Use response_content (our Pydantic models) not content_blocks (raw SDK objects)
                    if response_content and isinstance(response_content, list):
                        for block in response_content:
                            block_type = getattr(block, "type", None)
                            type_value = (
                                getattr(block_type, "value", block_type)
                                if block_type
                                else None
                            )
                            if type_value == "thinking":
                                # Store the thinking block as dict to preserve signature
                                assistant_msg["thinking_block"] = (
                                    block.model_dump()
                                    if hasattr(block, "model_dump")
                                    else None
                                )
                                break

                    # Preserve provider metadata (provider-agnostic passthrough)
                    # This enables providers to maintain state across steps (e.g., OpenAI reasoning items)
                    if hasattr(response, "metadata") and response.metadata:
                        assistant_msg["metadata"] = response.metadata

                    await context.add_message(assistant_msg)
                    # Last-drain edge: if a steer arrived during the final generation,
                    # loop once more so the model acts on it this turn. The top-of-
                    # iteration drain performs the actual injection.
                    if not self._steering_queue.is_empty:
                        continue
                    break

                # Add assistant message with tool calls
                # Store structured content blocks (preserves reasoning state, thinking blocks, etc.)
                # Extract text for display/logging only
                if hasattr(response, "text") and response.text:
                    response_text = response.text
                else:
                    response_text = (
                        self._extract_text_from_content(response.content)
                        if response.content
                        else ""
                    )

                # Store structured content from response.content (our Pydantic models)
                response_content = getattr(response, "content", None)
                if response_content and isinstance(response_content, list):
                    assistant_msg = {
                        "role": "assistant",
                        "content": [
                            block.model_dump()
                            if hasattr(block, "model_dump")
                            else block
                            for block in response_content
                        ],
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "tool": tc.name,
                                "arguments": tc.arguments,
                            }
                            for tc in tool_calls
                        ],
                    }
                else:
                    assistant_msg = {
                        "role": "assistant",
                        "content": response_text,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "tool": tc.name,
                                "arguments": tc.arguments,
                            }
                            for tc in tool_calls
                        ],
                    }

                # Preserve thinking blocks for Anthropic extended thinking (backward compat)
                # Use response_content (our Pydantic models) not content_blocks (raw SDK objects)
                if response_content and isinstance(response_content, list):
                    for block in response_content:
                        block_type = getattr(block, "type", None)
                        type_value = (
                            getattr(block_type, "value", block_type)
                            if block_type
                            else None
                        )
                        if type_value == "thinking":
                            # Store the thinking block as dict to preserve signature
                            assistant_msg["thinking_block"] = (
                                block.model_dump()
                                if hasattr(block, "model_dump")
                                else None
                            )
                            break

                # Preserve provider metadata (provider-agnostic passthrough)
                # This enables providers to maintain state across steps (e.g., OpenAI reasoning items)
                if hasattr(response, "metadata") and response.metadata:
                    assistant_msg["metadata"] = response.metadata

                await context.add_message(assistant_msg)

                # Process tool calls in parallel (user guidance: assume parallel intent)
                # Execute tools concurrently, but add results to context sequentially for determinism
                import uuid

                parallel_group_id = str(uuid.uuid4())

                # Execute all tools in parallel (no context updates inside)
                # Wrap in try/except for CancelledError to handle immediate cancellation
                tool_tasks = [
                    self._execute_tool_only(
                        tc, tools, hooks, parallel_group_id, coordinator
                    )
                    for tc in tool_calls
                ]

                try:
                    tool_results = await asyncio.gather(*tool_tasks)
                except asyncio.CancelledError:
                    # Immediate cancellation (second Ctrl+C) - synthesize cancelled results
                    # for ALL tool_calls to maintain tool_use/tool_result pairing
                    logger.info(
                        "Tool execution cancelled - synthesizing cancelled results"
                    )
                    for tc in tool_calls:
                        await context.add_message(
                            {
                                "role": "tool",
                                "name": tc.name,
                                "tool_call_id": tc.id,
                                "content": f'{{"error": "Tool execution was cancelled by user", "cancelled": true, "tool": "{tc.name}"}}',
                            }
                        )
                    # Emit cancel events before re-raising so hooks receive them
                    if coordinator and not self._cancel_requested_emitted:
                        self._cancel_requested_emitted = True
                        await hooks.emit(
                            CANCEL_REQUESTED,
                            {
                                "orchestrator": "loop-streaming",
                                "state": str(coordinator.cancellation.state),
                                "turn_count": iteration,
                            },
                        )
                        try:
                            await coordinator.cancellation.trigger_callbacks()
                        except Exception as e:
                            logger.warning(f"Error in cancellation callbacks: {e}")
                    if coordinator:
                        await hooks.emit(
                            CANCEL_COMPLETED,
                            {
                                "orchestrator": "loop-streaming",
                                "was_immediate": coordinator.cancellation.is_immediate,
                                "turn_count": iteration,
                            },
                        )
                    # Write synthetic assistant message to close the turn.
                    # Without this, transcript has tool_results without a closing assistant
                    # message, triggering FM3 (incomplete_assistant_turn) on resume.
                    await context.add_message(
                        {
                            "role": "assistant",
                            "content": "The previous operation was cancelled. Results from completed tools have been preserved.",
                        }
                    )
                    # Re-raise to let the cancellation propagate.
                    # Clear pending steers first — a steer queued during tool
                    # execution must not leak into any future turn. (spec §5.2)
                    self._steering_queue.clear()
                    raise

                # Check for cancellation after tools complete (graceful cancellation)
                if coordinator and coordinator.cancellation.is_cancelled:
                    # MUST add tool results to context before returning
                    # Otherwise we leave orphaned tool_calls without matching tool_results
                    # which violates provider API contracts (Anthropic, OpenAI)
                    for tool_call_id, tool_name, content in tool_results:
                        await context.add_message(
                            {
                                "role": "tool",
                                "name": tool_name,
                                "tool_call_id": tool_call_id,
                                "content": content,
                            }
                        )
                    # Emit cancel:requested on first detection and trigger cleanup callbacks
                    if not self._cancel_requested_emitted:
                        self._cancel_requested_emitted = True
                        await hooks.emit(
                            CANCEL_REQUESTED,
                            {
                                "orchestrator": "loop-streaming",
                                "state": str(coordinator.cancellation.state),
                                "turn_count": iteration,
                            },
                        )
                        try:
                            await coordinator.cancellation.trigger_callbacks()
                        except Exception as e:
                            logger.warning(f"Error in cancellation callbacks: {e}")
                    # Emit cancel:completed — orchestrator is exiting due to cancellation
                    await hooks.emit(
                        CANCEL_COMPLETED,
                        {
                            "orchestrator": "loop-streaming",
                            "was_immediate": coordinator.cancellation.is_immediate,
                            "turn_count": iteration,
                        },
                    )
                    # Write synthetic assistant message to close the turn.
                    # Without this, transcript has tool_results without a closing assistant
                    # message, triggering FM3 (incomplete_assistant_turn) on resume.
                    await context.add_message(
                        {
                            "role": "assistant",
                            "content": "The previous operation was cancelled. Results from completed tools have been preserved.",
                        }
                    )
                    # Exit the loop - orchestrator complete event will be emitted in execute().
                    # Clear pending steers: cancellation closes the turn; any steer that
                    # arrived after the last injection point must not ride a future turn. (spec §5.2)
                    self._steering_queue.clear()
                    return

                # Add all results to context in original order (sequential, deterministic)
                # Note: Context manager handles compaction internally when get_messages_for_request() is called
                for tool_call_id, tool_name, content in tool_results:
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "tool_call_id": tool_call_id,
                            "content": content,
                        }
                    )

        # Check if we exceeded max iterations (only if not unlimited)
        if self.max_iterations != -1 and iteration >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached")

            # Inject system reminder to agent before returning
            await hooks.emit(
                PROVIDER_REQUEST,
                {
                    "provider": provider_name,
                    "iteration": iteration,
                    "max_reached": True,
                },
            )

            # Get one final response with the reminder (via _execute_stream helper)
            message_dicts = await context.get_messages_for_request(provider=provider)
            message_dicts = list(message_dicts)
            message_dicts.append(
                {
                    "role": "user",
                    "content": """<system-reminder source="orchestrator-loop-limit">
You have reached the maximum number of iterations for this turn. Please provide a response to the user now, summarizing your progress and noting what remains to be done. You can continue in the next turn if needed.

DO NOT mention this iteration limit or reminder to the user explicitly. Simply wrap up naturally.
</system-reminder>""",
                }
            )

            try:
                # Convert dicts to ChatRequest
                messages_objects = [Message(**msg) for msg in message_dicts]

                # Convert tools to ToolSpec format for ChatRequest
                tools_list = None
                if tools:
                    tools_list = [
                        ToolSpec(
                            name=t.name,
                            description=t.description,
                            parameters=t.input_schema,
                        )
                        for t in tools.values()
                    ]

                max_iter_chat_request = ChatRequest(
                    messages=messages_objects,
                    tools=tools_list,
                    reasoning_effort=self.config.get("reasoning_effort"),
                )

                kwargs = {}
                if self.extended_thinking:
                    kwargs["extended_thinking"] = True

                response = await provider.complete(max_iter_chat_request, **kwargs)
                content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                if content:
                    # Yield the final response
                    async for token in self._tokenize_stream(content):
                        yield (token, iteration)

                    # Add to context
                    await context.add_message({"role": "assistant", "content": content})

            except LLMError as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {
                        "provider": provider_name,
                        "error": {"type": type(e).__name__, "msg": str(e)},
                        "retryable": e.retryable,
                        "status_code": e.status_code,
                    },
                )
                logger.error(f"Error getting final response after max iterations: {e}")
            except Exception as e:
                await hooks.emit(
                    PROVIDER_ERROR,
                    {
                        "provider": provider_name,
                        "error": {"type": type(e).__name__, "msg": str(e)},
                    },
                )
                logger.error(f"Error getting final response after max iterations: {e}")

        # Emit execution end
        await hooks.emit("execution:end", {})

    async def _stream_from_provider(
        self,
        provider,
        chat_request,
        context,
        tools,
        hooks,
        coordinator=None,
        provider_name=None,
    ) -> AsyncIterator[str]:
        """Stream tokens from provider that supports streaming.

        Args:
            provider: The provider to stream from
            chat_request: The chat request to send
            context: The context manager
            tools: Available tools
            hooks: Hook registry
            coordinator: Optional coordinator for cancellation support
            provider_name: Name of the provider for event emission
        """
        # This is a simplified example
        # Real implementation would handle streaming tool calls

        full_response = ""

        # Convert tools dict to list for provider
        tools_list = list(tools.values()) if tools else []
        try:
            stream_iter = provider.stream(chat_request, tools=tools_list)
        except LLMError as e:
            await hooks.emit(
                PROVIDER_ERROR,
                {
                    "provider": provider_name,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                    "retryable": e.retryable,
                    "status_code": e.status_code,
                },
            )
            raise
        except Exception as e:
            await hooks.emit(
                PROVIDER_ERROR,
                {
                    "provider": provider_name,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                },
            )
            raise

        async for chunk in stream_iter:
            # Check for immediate cancellation between chunks
            if coordinator and coordinator.cancellation.is_immediate:
                # Add partial response to context before exiting
                if full_response:
                    await context.add_message(
                        {"role": "assistant", "content": full_response}
                    )
                return

            # Skip non-text block deltas (e.g. thinking block streaming chunks).
            # Providers that stream extended-thinking models include a block_type
            # field so callers can distinguish thinking deltas from text deltas.
            # Without this guard, thinking content leaks into full_response and
            # ultimately into parse_json extraction downstream.
            chunk_block_type = chunk.get("block_type")
            if chunk_block_type and chunk_block_type != "text":
                continue

            token = chunk.get("content", "")
            if token:
                yield token
                full_response += token
                if self.stream_delay:
                    await asyncio.sleep(self.stream_delay)

        # Add complete message to context
        if full_response:
            await context.add_message({"role": "assistant", "content": full_response})

    def _extract_text_from_content(self, content) -> str:
        """Extract text from content blocks.

        Args:
            content: Either a string or list of ContentBlock objects

        Returns:
            Extracted text as string
        """
        if isinstance(content, str):
            return content

        if not content:
            return ""

        # Extract text from content blocks.
        # IMPORTANT: Use explicit block.type check, NOT hasattr(block, "text").
        # content_models.ThinkingContent has a .text attribute (distinct from
        # message_models.ThinkingBlock which uses .thinking). The hasattr-based
        # filter was letting thinking text leak into the response string, which
        # pollutes parse_json extraction in downstream recipe steps.
        text_parts = []
        for block in content:
            # Explicit type check — works for both enum (ContentBlockType.TEXT)
            # and plain-str "type" fields (e.g., message_models.TextBlock).
            block_type = getattr(block, "type", None)
            # Handle both enum (block_type.value == "text") and raw str ("text")
            type_value = (
                getattr(block_type, "value", block_type) if block_type else None
            )
            if type_value == "text" and hasattr(block, "text"):
                text_parts.append(block.text)
            # Thinking blocks, tool_use blocks, etc. are all correctly excluded.

        return "\n\n".join(text_parts)

    async def _tokenize_stream(self, text: str) -> AsyncIterator[str]:
        """
        Yield text token-by-token, optionally throttled by ``self.stream_delay``
        for human-facing typing-animation UX.

        When ``stream_delay == 0.0`` (the default), this is a near-zero-overhead
        pass-through used to satisfy callers that consume the orchestrator's
        output stream incrementally. When ``stream_delay > 0.0``, each
        non-whitespace token is followed by ``await asyncio.sleep(stream_delay)``
        — used by the streaming-ui hook to animate output in interactive
        terminals.

        This is invoked from the non-streaming code path (when ``provider`` has
        no ``stream`` method), after the full response is already in hand. It
        does NOT do real streaming from the provider; for that, see
        ``_stream_from_provider``.

        Preserves:
        - Leading whitespace (critical for code block indentation)
        - Multiple consecutive spaces (critical for ASCII art alignment)
        - Newlines between lines
        """
        lines = text.split("\n")

        for line_idx, line in enumerate(lines):
            # Split into tokens while PRESERVING all whitespace runs
            # \S+ matches non-whitespace sequences, \s+ matches whitespace sequences
            # This ensures multiple spaces are preserved (e.g., for ASCII art tables)
            tokens = re.findall(r"\S+|\s+", line)

            for token in tokens:
                yield token
                # Only delay on non-whitespace tokens for natural streaming effect;
                # skip entirely when stream_delay is 0.0 (the default) so headless
                # callers pay no event-loop overhead.
                if token.strip() and self.stream_delay:
                    await asyncio.sleep(self.stream_delay)

            # Yield newline after each line except the last
            if line_idx < len(lines) - 1:
                yield "\n"

    async def _execute_tool(
        self,
        tool_call,
        tools: dict[str, Any],
        context,
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> None:
        """Execute a single tool call (legacy method for compatibility)."""
        await self._execute_tool_with_result(
            tool_call, tools, context, hooks, coordinator
        )

    async def _execute_tool_only(
        self,
        tool_call,
        tools: dict[str, Any],
        hooks: HookRegistry,
        parallel_group_id: str,
        coordinator: ModuleCoordinator | None = None,
    ) -> tuple[str, str, str]:
        """Execute a single tool in parallel without adding to context.

        Returns (tool_call_id, name, content) tuple.
        Never raises - errors become error messages.
        """
        try:
            # Pre-tool hook
            pre_result = await hooks.emit(
                TOOL_PRE,
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "tool_input": tool_call.arguments,
                    "parallel_group_id": parallel_group_id,
                },
            )
            if coordinator:
                pre_result = await coordinator.process_hook_result(
                    pre_result, "tool:pre", tool_call.name
                )
                if pre_result.action == "deny":
                    return (
                        tool_call.id,
                        tool_call.name,
                        f"Denied by hook: {pre_result.reason}",
                    )

            # Get tool
            tool = tools.get(tool_call.name)
            if not tool:
                error_msg = f"Error: Tool '{tool_call.name}' not found"
                await hooks.emit(
                    TOOL_ERROR,
                    {
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "error": {"type": "RuntimeError", "msg": error_msg},
                        "parallel_group_id": parallel_group_id,
                    },
                )
                return (tool_call.id, tool_call.name, error_msg)

            # Register tool with cancellation token for visibility
            if coordinator:
                # Build semantic display name for delegate calls
                display_name = tool_call.name
                if tool_call.name == "delegate":
                    try:
                        _args = (
                            tool_call.arguments
                            if isinstance(tool_call.arguments, dict)
                            else json.loads(tool_call.arguments)
                        )
                        _agent = _args.get("agent", "")
                        if _agent:
                            display_name = _agent
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
                coordinator.cancellation.register_tool_start(tool_call.id, display_name)

            # Set per-task dispatch context so delegate tools can read the
            # calling tool_call_id and parallel_group_id during execute().
            # A task-keyed dict avoids races when multiple delegates run
            # concurrently inside asyncio.gather().
            if coordinator:
                if not hasattr(coordinator, "_tool_dispatch_contexts"):
                    coordinator._tool_dispatch_contexts = {}
                _dispatch_task = asyncio.current_task()
                coordinator._tool_dispatch_contexts[_dispatch_task] = {
                    "tool_call_id": tool_call.id,
                    "parallel_group_id": parallel_group_id,
                }

            # Execute
            # NB: incremented here, not on tool lookup / pre-hook denial above
            # -- this is where a tool actually runs (see execute()'s stall
            # detection, which reads self._tool_calls_this_turn per turn).
            self._tool_calls_this_turn += 1
            try:
                result = await tool.execute(tool_call.arguments)
            except Exception as e:
                result = ToolResult(success=False, error={"message": str(e)})
            finally:
                # Always unregister tool from cancellation token
                if coordinator:
                    coordinator.cancellation.register_tool_complete(tool_call.id)
                    # Clear per-task dispatch context so completed tasks don't linger
                    if hasattr(coordinator, "_tool_dispatch_contexts"):
                        coordinator._tool_dispatch_contexts.pop(
                            asyncio.current_task(), None
                        )

            # Serialize result for logging
            result_data = (
                result.model_dump() if hasattr(result, "model_dump") else str(result)
            )

            # Post-tool hook
            post_result = await hooks.emit(
                TOOL_POST,
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "tool_input": tool_call.arguments,
                    "result": result_data,
                    "parallel_group_id": parallel_group_id,
                },
            )
            if coordinator:
                await coordinator.process_hook_result(
                    post_result, "tool:post", tool_call.name
                )

            # Store ephemeral injection from tool:post for next iteration
            if (
                post_result.action == "inject_context"
                and post_result.ephemeral
                and post_result.context_injection
            ):
                self._pending_ephemeral_injections.append(
                    {
                        "role": post_result.context_injection_role,
                        "content": post_result.context_injection,
                        "append_to_last_tool_result": post_result.append_to_last_tool_result,
                    }
                )
                logger.debug(
                    f"Stored ephemeral injection from tool:post ({tool_call.name}) for next iteration"
                )

            # Check if a hook modified the tool result.
            # hooks.emit() chains modify actions: when a hook
            # returns action="modify", the data dict is replaced.
            # We detect this by checking if the returned "result"
            # is a different object than what we originally sent.
            modified_result = None
            if post_result and post_result.data is not None:
                returned_result = post_result.data.get("result")
                if returned_result is not None and returned_result is not result_data:
                    modified_result = returned_result

            if modified_result is not None:
                if isinstance(modified_result, (dict, list)):
                    content = json.dumps(modified_result)
                else:
                    content = str(modified_result)
            else:
                content = result.get_serialized_output()
            return (tool_call.id, tool_call.name, content)

        except Exception as e:
            # Safety net: errors become error messages
            logger.error(f"Tool {tool_call.name} failed: {e}")
            error_msg = f"Internal error executing tool: {e!s}"
            await hooks.emit(
                TOOL_ERROR,
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "error": {"type": type(e).__name__, "msg": str(e)},
                    "parallel_group_id": parallel_group_id,
                },
            )
            return (tool_call.id, tool_call.name, error_msg)

    async def _execute_tool_with_result(
        self,
        tool_call,
        tools: dict[str, Any],
        context,
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
    ) -> dict:
        """Execute a single tool call and return result info.

        Guarantees that a tool response is always added to context, even if errors occur.
        This prevents orphaned tool calls that corrupt conversation state.
        """
        response_added = False

        try:
            # Pre-tool hook
            pre_result = await hooks.emit(
                TOOL_PRE,
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "tool_input": tool_call.arguments,
                },
            )
            if coordinator:
                pre_result = await coordinator.process_hook_result(
                    pre_result, "tool:pre", tool_call.name
                )
                if pre_result.action == "deny":
                    # Add tool_result message (not system) so Anthropic API accepts it
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "content": f"Tool execution denied: {pre_result.reason}",
                        }
                    )
                    response_added = True
                    return {"success": False, "error": f"Denied: {pre_result.reason}"}

            # Get tool
            tool = tools.get(tool_call.name)
            if not tool:
                # Add tool_result message (not system) so Anthropic API accepts it
                await context.add_message(
                    {
                        "role": "tool",
                        "name": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "content": f"Error: Tool '{tool_call.name}' not found",
                    }
                )
                response_added = True
                return {"success": False, "error": "Tool not found"}

            # Set per-task dispatch context so delegate tools can read the
            # calling tool_call_id during execute() (sequential path has no
            # parallel_group_id so that field is None here).
            if coordinator:
                if not hasattr(coordinator, "_tool_dispatch_contexts"):
                    coordinator._tool_dispatch_contexts = {}
                _dispatch_task = asyncio.current_task()
                coordinator._tool_dispatch_contexts[_dispatch_task] = {
                    "tool_call_id": tool_call.id,
                    "parallel_group_id": None,
                }

            # Execute
            # NB: incremented here, not on tool lookup / pre-hook denial above
            # -- this is where a tool actually runs (see execute()'s stall
            # detection, which reads self._tool_calls_this_turn per turn).
            self._tool_calls_this_turn += 1
            try:
                result = await tool.execute(tool_call.arguments)
            except Exception as e:
                result = ToolResult(success=False, error={"message": str(e)})
            finally:
                # Clear per-task dispatch context so completed tasks don't linger
                if coordinator and hasattr(coordinator, "_tool_dispatch_contexts"):
                    coordinator._tool_dispatch_contexts.pop(
                        asyncio.current_task(), None
                    )

            # Serialize result for logging
            result_data = (
                result.model_dump() if hasattr(result, "model_dump") else str(result)
            )

            # Post-tool hook
            post_result = await hooks.emit(
                TOOL_POST,
                {
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "tool_input": tool_call.arguments,
                    "result": result_data,
                },
            )
            if coordinator:
                await coordinator.process_hook_result(
                    post_result, "tool:post", tool_call.name
                )

            # Store ephemeral injection from tool:post for next iteration
            if (
                post_result.action == "inject_context"
                and post_result.ephemeral
                and post_result.context_injection
            ):
                self._pending_ephemeral_injections.append(
                    {
                        "role": post_result.context_injection_role,
                        "content": post_result.context_injection,
                        "append_to_last_tool_result": post_result.append_to_last_tool_result,
                    }
                )
                logger.debug(
                    f"Stored ephemeral injection from tool:post ({tool_call.name}) for next iteration"
                )

            # Check if a hook modified the tool result.
            # hooks.emit() chains modify actions: when a hook
            # returns action="modify", the data dict is replaced.
            # We detect this by checking if the returned "result"
            # is a different object than what we originally sent.
            modified_result = None
            if post_result and post_result.data is not None:
                returned_result = post_result.data.get("result")
                if returned_result is not None and returned_result is not result_data:
                    modified_result = returned_result

            if modified_result is not None:
                if isinstance(modified_result, (dict, list)):
                    tool_content = json.dumps(modified_result)
                else:
                    tool_content = str(modified_result)
            else:
                tool_content = result.get_serialized_output()

            await context.add_message(
                {
                    "role": "tool",
                    "name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                }
            )
            response_added = True

            return {
                "success": result.success,
                "error": result.error if not result.success else None,
            }

        except Exception as e:
            # Safety net: Ensure a tool response is ALWAYS added to prevent orphaned tool calls
            logger.error(
                f"Unexpected error executing tool {tool_call.name}: {e}", exc_info=True
            )

            if not response_added:
                try:
                    await context.add_message(
                        {
                            "role": "tool",
                            "name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "content": f"Internal error executing tool: {e!s}",
                        }
                    )
                except Exception as inner_e:
                    # Critical failure: Even adding error response failed
                    logger.error(
                        f"Critical: Failed to add error response for tool_call_id {tool_call.id}: {inner_e}"
                    )

            return {"success": False, "error": str(e)}

    async def _has_pending_tools(self, context) -> bool:
        """Check if there are pending tool calls."""
        # Simplified - would need to track tool calls properly
        return False

    async def _process_tools(self, context, tools, hooks) -> None:
        """Process any pending tool calls."""
        # Simplified - would process tracked tool calls

    def _select_provider(self, providers: dict[str, Any]) -> Any:
        """Select a provider based on priority."""
        if not providers:
            return None

        # Collect providers with their priority (default priority is 100)
        provider_list = []
        for name, provider in providers.items():
            # Try to get priority from provider's config or attributes
            priority = 100  # Default priority
            if hasattr(provider, "priority"):
                priority = provider.priority
            elif hasattr(provider, "config") and isinstance(provider.config, dict):
                priority = provider.config.get("priority", 100)

            provider_list.append((priority, name, provider))

        # Sort by priority (lower number = higher priority)
        provider_list.sort(key=lambda x: x[0])

        # Return the highest priority provider
        if provider_list:
            return provider_list[0][2]

        return None
