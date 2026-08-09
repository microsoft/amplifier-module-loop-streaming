"""
Streaming orchestrator module for Amplifier.
Provides token-by-token streaming responses.
"""

# Amplifier module metadata
__amplifier_module_type__ = "orchestrator"

import asyncio
import fnmatch
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


def _build_tool_spec(tool: Any) -> ToolSpec:
    """Build a `ToolSpec` for one mounted tool, preserving model-native form.

    Most tools are plain function tools: name + description + JSON schema.

    A *model-native* tool (Anthropic's `computer_20251124`, `web_search_*`,
    ...) is instead a server-side definition whose shape is fixed by the
    provider. It is declared on the wire as `{"type": "<tool_type>", ...}`,
    and the model is post-trained on that exact definition - a lookalike
    function tool measurably degrades its behaviour, so the native form has
    to survive the trip to the provider.

    Building the spec from only (name, description, parameters) silently
    discarded that form, and no error surfaced: the request was still valid,
    the tool still appeared, and the model simply got the weaker definition.
    Providers already accept the native shape (`type != "function"` is passed
    through untouched) - it just never reached them.

    A tool opts in by exposing `native_tool_spec`: a dict of the provider's
    native definition. `ToolSpec` is declared `extra="allow"`, so those keys
    ride along as extras and reach the provider intact. Tools without the
    attribute are unaffected.
    """
    spec_source = getattr(type(tool), "native_tool_spec", None)
    if spec_source is not None:
        try:
            native = tool.native_tool_spec
        except Exception:
            # A tool that cannot describe its native form is still a usable
            # function tool. Degrade to the function shape rather than failing
            # the whole request, but say so - a silent downgrade here is what
            # made the original bug invisible.
            logger.warning(
                "tool %r raised while reading native_tool_spec; "
                "falling back to its function-tool definition",
                getattr(tool, "name", "<unknown>"),
                exc_info=True,
            )
        else:
            if isinstance(native, dict) and native.get("type"):
                return ToolSpec(
                    name=native.get("name") or tool.name,
                    description=tool.description,
                    parameters=tool.input_schema,
                    **{k: v for k, v in native.items() if k not in ("name",)},
                )

    return ToolSpec(
        name=tool.name,
        description=tool.description,
        parameters=tool.input_schema,
    )


# --- Evaluator input hygiene defaults (see _flatten_message_for_evaluator,
# StreamingOrchestrator._evaluate_goal) ---
#
# Before this, the evaluator's transcript had exactly one bound: the
# 40-message window (`_GOAL_MAX_TRANSCRIPT_MESSAGES`). A single large tool
# result (file read, log dump) inside those 40 messages shipped to the
# evaluator IN FULL, once per turn -- a cost problem and a signal-burial
# problem. These module-level defaults back the `self.goal_*` config knobs
# read in `StreamingOrchestrator.__init__` (same override pattern as
# `goal_stall_threshold`).
#
# Per-tool-result / tool-call-argument clip, in characters. 2000 chars
# (~500 tokens) is generous for a typical tool result while bounding the
# worst case (a multi-MB file dump) to a fixed, cheap-to-evaluate size.
_GOAL_DEFAULT_TOOL_CONTENT_CLIP_CHARS = 2000


def _clip_head_tail(text: str, limit: int) -> str:
    """Clip ``text`` to roughly ``limit`` chars, keeping BOTH the head and
    the tail with a marker naming how much was dropped in between.

    Head+tail, not head-only: verdict-bearing detail in tool output
    commonly sits at BOTH ends -- the command/args echo at the head, the
    exit status or summary line at the tail. A naive head-only clip loses
    exactly the part that usually decides the verdict.

    ``limit <= 0`` disables clipping entirely (returns ``text`` unchanged).
    """
    if limit <= 0 or len(text) <= limit:
        return text
    # First-pass estimate of the marker's own size (using the naive
    # dropped-char count) so it can be subtracted from `limit` before
    # splitting head/tail. The exact dropped count is recomputed below from
    # the actual split chosen, so this estimate only affects how much of
    # `limit` the marker itself consumes -- not correctness.
    estimated_marker = f"\n...[{len(text) - limit} chars truncated]...\n"
    keep = limit - len(estimated_marker)
    if keep < 20:
        # `limit` too tight for a head+tail split plus a marker -- fall
        # back to a hard cut rather than producing a nonsensical clip.
        return text[:limit]
    head_len = keep // 2
    tail_len = keep - head_len
    dropped = len(text) - head_len - tail_len
    marker = f"\n...[{dropped} chars truncated]...\n"
    return text[:head_len] + marker + text[-tail_len:]


def _flatten_message_for_evaluator(
    msg: dict, tool_content_clip_chars: int = _GOAL_DEFAULT_TOOL_CONTENT_CLIP_CHARS
) -> str:
    """Render one stored (dict) message as plain text for the goal evaluator.

    Ported (with input-hygiene added -- see below) from amplifier-app-cli's
    `_flatten_message_for_evaluator` (see docs/designs/goal-command.md).
    Kept as a module-level function -- ``tool_content_clip_chars`` is a
    plain parameter, not orchestrator state, so the function stays pure and
    directly testable; `_evaluate_goal` passes its configured
    `self.goal_tool_content_clip_chars`.

    Tool results and tool-call arguments remain FULLY VISIBLE to the
    evaluator -- they are just BOUNDED. A short tool result still appears
    verbatim; an overlong one appears head+tail clipped (see
    `_clip_head_tail`) with an explicit marker naming what was dropped, so
    the evaluator knows it's looking at a clip, not the whole story.
    """
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    if isinstance(content, str):
        text = content
        if role == "tool":
            text = _clip_head_tail(text, tool_content_clip_chars)
    elif isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_call":
                name = block.get("name", "?")
                # `input` is the tool-call-argument dict per ToolCallBlock's
                # schema (amplifier_core.message_models). Previously
                # dropped entirely -- the evaluator could see WHICH tool ran
                # but never WHAT it was asked to do. Clipped for the same
                # cost-bound reason as tool results above.
                args = block.get("input")
                if args:
                    args_text = _clip_head_tail(
                        json.dumps(args, default=str), tool_content_clip_chars
                    )
                    parts.append(f"[called tool: {name} args: {args_text}]")
                else:
                    parts.append(f"[called tool: {name}]")
            elif btype == "tool_result":
                parts.append("[tool result omitted]")
            # thinking/redacted_thinking/reasoning blocks are intentionally
            # skipped -- the evaluator only judges what was surfaced.
        text = "\n".join(p for p in parts if p)
    else:
        text = str(content)
    return f"{role}: {text}" if text else ""


# --- goal_provider_preferences glob ranking --------------------------------
#
# `goal_provider_preferences` (see `StreamingOrchestrator.__init__` and
# `_resolve_goal_model`) resolves glob model patterns the same way the
# routing-matrix `hooks-routing` bundle does, so a new model release is
# picked up automatically without a code change here. This is a deliberate,
# from-scratch reimplementation of hooks-routing's private
# `resolver._version_sort_key` / `resolver._resolve_glob` ranking logic --
# NOT an import of either:
#
#   * `amplifier_module_hooks_routing` -- this fallback list exists
#     precisely for sessions where that bundle is NOT installed, so
#     depending on it here would be circular.
#   * `amplifier_foundation.spawn_utils.resolve_model_pattern` -- its
#     ranking is a plain `matched.sort(reverse=True)` lexicographic sort,
#     which is verifiably WRONG in two cases the routing matrix gets right:
#     it prefers a date-pinned snapshot (`claude-haiku-4-5-20251001`) over
#     the clean alias (`claude-haiku-4-5`), and it ranks `claude-opus-4-7`
#     above `claude-opus-4-10` because multi-digit version segments compare
#     as strings, not integers. Do not "simplify" this back to
#     `resolve_model_pattern` -- that would reintroduce both defects.
_GOAL_PREF_DATE_SUFFIX_RE = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$")
_GOAL_PREF_DIGIT_RUN_RE = re.compile(r"(\d+)")


def _goal_pref_is_glob(pattern: str) -> bool:
    """Whether *pattern* contains glob wildcard characters."""
    return any(c in pattern for c in "*?[")


def _goal_pref_version_sort_key(name: str) -> tuple[list[Any], int]:
    """Natural-sort key matching the routing matrix's ranking exactly (see
    module comment above).

    1. Strip a trailing ``-YYYYMMDD``/``-YYYY-MM-DD`` snapshot-date suffix
       so clean aliases (``claude-haiku-4-5``) outrank pinned snapshots
       (``claude-haiku-4-5-20251001``).
    2. Split on digit runs and compare them as integers, not strings, so
       ``claude-opus-4-10`` correctly outranks ``claude-opus-4-7``.
    3. Tie-break on ``-len(name)`` so a shorter alias wins when the primary
       key is otherwise equal.
    """
    stripped = _GOAL_PREF_DATE_SUFFIX_RE.sub("", name)
    primary: list[Any] = [
        int(part) if part.isdigit() else part
        for part in _GOAL_PREF_DIGIT_RUN_RE.split(stripped)
    ]
    return (primary, -len(name))


async def _resolve_goal_pref_glob(pattern: str, provider: Any) -> str | None:
    """Resolve one ``goal_provider_preferences`` glob pattern against a
    single provider's ``list_models()``, ranked via
    ``_goal_pref_version_sort_key`` (see module comment above).

    Returns the highest-ranked match, or ``None`` when ``list_models()``
    raises or nothing matches -- both cases mean the caller should advance
    to the next preference in the list, not abort the whole fallback.
    """
    try:
        available = await provider.list_models()
    except Exception:
        logger.info(
            "/goal: goal_provider_preferences glob '%s' -- provider."
            "list_models() raised, advancing to next preference",
            pattern,
            exc_info=True,
        )
        return None

    model_names = [
        m if isinstance(m, str) else getattr(m, "id", str(m)) for m in available
    ]
    pattern_lower = pattern.lower()
    matched = [m for m in model_names if fnmatch.fnmatch(m.lower(), pattern_lower)]
    if not matched:
        return None

    matched.sort(key=_goal_pref_version_sort_key, reverse=True)
    return matched[0]


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
        "What you can and cannot see: tool results and tool-call arguments "
        "are shown to you, but each is bounded in size -- an overlong one "
        "is clipped and marked with '...[N chars truncated]...' showing "
        "exactly how much was cut. Very old turns may be dropped entirely "
        "if the transcript is marked truncated. When content is marked "
        "clipped or the transcript is marked truncated, say so plainly in "
        "your reasoning rather than inferring or guessing at what the "
        "elided content contained.\n\n"
        "Respond with EXACTLY two lines and nothing else:\n"
        "Line 1: the single word YES or NO (verbatim, nothing else)\n"
        "Line 2: one sentence explaining why\n"
    )

    # NOTE on model-role routing (investigated for the stall-detection work,
    # docs/designs/goal-command.md): the routing-matrix `fast` role IS
    # reachable from here. `hooks-routing` registers a coordinator capability
    # named `model_role_resolver` (contract: `async def resolve(model_role:
    # str | list[str]) -> list[ProviderPreference]`, see amplifier_foundation's
    # `ProviderPreference`). This orchestrator looks that capability up
    # lazily -- inside `_resolve_goal_model`, never at mount()/__init__ time
    # -- because hooks mount *after* the orchestrator does, so the capability
    # is not yet registered when this class is constructed. See
    # `_resolve_goal_model` below for the resolution + caching + fallback
    # logic, matching the pattern established by tool-delegate's own
    # `model_role_resolver` consumption.

    _GOAL_MAX_TRANSCRIPT_MESSAGES = 40

    # Total transcript character budget: a backstop ABOVE the per-message
    # clip (`_GOAL_DEFAULT_TOOL_CONTENT_CLIP_CHARS`) and the message-count
    # window above. Applied NEWEST-FIRST in `_evaluate_goal` so that when
    # the budget binds, the most recent (most verdict-relevant) turns
    # survive and older messages within the window are what gets dropped.
    # 20000 chars (~5000 tokens) keeps the evaluator call cheap even in the
    # worst case of all 40 messages sitting at the per-message clip ceiling
    # (40 * 2000 = 80000 chars before this budget applies).
    _GOAL_DEFAULT_TRANSCRIPT_CHAR_BUDGET: ClassVar[int] = 20000

    # Stall-detection judge (see execute()'s stall-detection block). Only
    # invoked when one of the two mechanical pre-filters already holds --
    # trigger (a) "idle" (goal["no_tool_turns"] >= goal_stall_threshold) or
    # trigger (b) "busy" (_busy_stall_pretrip) -- so it fires rarely and
    # stays cheap. See _GOAL_STALL_SYSTEM_PROMPT_IDLE / _BUSY below for why
    # there are two prompts rather than one.
    #
    # Shared taxonomy block: the judge no longer answers a bare YES/NO. A
    # locked verdict tells the *user* which kind of dead end they wrote --
    # that's the whole point of naming it (see GOAL-HARDENING-DESIGN.md sec
    # 1.2): "is this a stall?" doesn't say what to fix, "history-locked"
    # does.
    _GOAL_STALL_TAXONOMY_BLOCK: ClassVar[str] = (
        "Classify the situation using EXACTLY one of these four words:\n"
        "RESOLVABLE -- more work could plausibly close this; there is no "
        "structural reason it cannot be resolved with further turns.\n"
        "TIME-LOCKED -- the condition requires elapsed wall-clock time "
        "(e.g. a soak period, waiting for an external event) that cannot "
        "pass within this session no matter what the assistant does.\n"
        "STRUCTURE-LOCKED -- the condition applies a universal "
        'requirement over a set that contains a member which is '
        'structurally exempt or unreachable (e.g. "all N sites" when one '
        "site cannot produce the required measurement).\n"
        "HISTORY-LOCKED -- the condition constrains the transcript's own "
        'past (e.g. "proof must precede the claim") in a way that cannot '
        "be retroactively repaired regardless of what happens from here.\n"
        "\n"
        "Respond with EXACTLY two lines and nothing else:\n"
        "Line 1: the single word above, verbatim\n"
        "Line 2: one sentence explaining why\n"
    )

    # Idle framing: used when trigger="idle" (goal["no_tool_turns"] reached
    # threshold -- the assistant took NO tool actions for that whole
    # streak). This is the original (pre-taxonomy) trigger.
    _GOAL_STALL_SYSTEM_PROMPT_IDLE: ClassVar[str] = (
        "You are a strict, tool-less judge. You will be shown a short "
        "history of reasons an evaluator gave, across consecutive turns, "
        "for why a goal condition was not yet satisfied -- during a "
        "stretch where the assistant took NO TOOL ACTIONS AT ALL. Decide "
        "whether the goal is durably stuck (and if so, why), as opposed to "
        "reasons that, even with no tools run, show the assistant "
        "narrowing down, ruling things out, or making genuine incremental "
        "progress toward the condition.\n\n" + _GOAL_STALL_TAXONOMY_BLOCK
    )

    # Busy framing: used when trigger="busy" (_busy_stall_pretrip tripped --
    # the assistant HAS been taking tool actions every turn, but the same
    # kind of blocker keeps recurring regardless). Real long stalls
    # (a3126f2f, 6e64b3db -- see GOAL-HARDENING-DESIGN.md sec 1.2) were all
    # busy every turn; feeding that situation into the idle prompt above
    # would hand the judge a false premise ("took no tool actions"), so
    # this is a genuinely separate prompt, not a parameter substitution
    # into the same string.
    _GOAL_STALL_SYSTEM_PROMPT_BUSY: ClassVar[str] = (
        "You are a strict, tool-less judge. You will be shown a short "
        "history of reasons an evaluator gave, across consecutive turns, "
        "for why a goal condition was not yet satisfied -- during a "
        "stretch where the assistant WAS ACTIVELY TAKING TOOL ACTIONS "
        "EVERY TURN, but the same kind of blocker kept recurring "
        "regardless of that activity. Decide whether the goal is durably "
        "stuck (and if so, why), as opposed to reasons that, despite "
        "looking repetitive, show the assistant making genuine incremental "
        "progress toward the condition (e.g. re-running a test after a "
        "real fix).\n\n" + _GOAL_STALL_TAXONOMY_BLOCK
    )

    # Verdicts a judge call may return (see _judge_stall). Any verdict other
    # than "resolvable" trips is_stalled -- i.e. falls through into the
    # existing (already tool-agnostic) escalate-then-hard-stop machinery in
    # execute(), unchanged from before the taxonomy existed.
    _GOAL_STALL_LOCKED_VERDICTS: ClassVar[frozenset[str]] = frozenset(
        {"time-locked", "structure-locked", "history-locked"}
    )

    # Wire-word (as instructed in _GOAL_STALL_TAXONOMY_BLOCK, uppercase) ->
    # internal verdict string (lowercase, hyphenated -- matches
    # _GOAL_STALL_LOCKED_VERDICTS and the payload's `stall_verdict` field).
    # A word outside this map is an unparseable response (see _judge_stall).
    _GOAL_STALL_VERDICT_WORDS: ClassVar[dict[str, str]] = {
        "RESOLVABLE": "resolvable",
        "TIME-LOCKED": "time-locked",
        "STRUCTURE-LOCKED": "structure-locked",
        "HISTORY-LOCKED": "history-locked",
    }

    # One-line, user-facing explanation per locked verdict -- reused in the
    # one-shot escalation prompt sent back to the agent (see
    # _goal_stall_escalation_prompt) so the agent gets the judge's own
    # classification, not just the evaluator's raw reason.
    _GOAL_STALL_VERDICT_EXPLANATIONS: ClassVar[dict[str, str]] = {
        "time-locked": (
            "the condition requires elapsed wall-clock time that cannot "
            "pass within this session"
        ),
        "structure-locked": (
            "the condition applies a universal requirement to a set that "
            "contains a structurally-exempt member"
        ),
        "history-locked": (
            "the condition constrains the transcript's own past in a way "
            "this session cannot retroactively repair"
        ),
    }

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

    # Monotonic integer identifying the `orchestrator:goal_progress` payload's
    # field set (see _goal_progress_payload). Bump by 1 whenever a field is
    # added, removed, or renamed. Always an explicit int on every emitted
    # event -- never null -- so a consumer can tell "this event carries a
    # versioned contract" (key present) from "this predates versioning
    # entirely" (key absent), rather than confusing an absent key with a
    # null value (the exact confusion that broke consumers of the `metadata`
    # field, which shipped null on 100% of measured events).
    #
    # Version 1 (this change): adds `condition` (the fully-expanded goal text
    # the evaluator actually judged) and this `schema_version` field itself;
    # removes the always-null `metadata` slot (zero readers found in either
    # this repo or amplifier-app-cli). The three prior wire shapes --
    # baseline; +reasons/summary/stall_detail/continuations; +distinct_
    # blockers -- all shipped with no version key at all, so an absent key
    # unambiguously means "one of those three", never a specific one of them.
    _GOAL_PROGRESS_SCHEMA_VERSION: ClassVar[int] = 1

    # DEFECT 4 fix: shared cap for the evaluator/stall-judge/summary calls'
    # `max_output_tokens`. Each of these three calls has a contract of
    # "respond with EXACTLY two lines" (or, for the summary, one ~120-char
    # sentence) -- there is no legitimate reason for their output to approach
    # a full response budget. Real session e97e192b measured the evaluator
    # inheriting the provider's session default of 64000 (Haiku's
    # `max_output_tokens` capability default) purely because no
    # `max_output_tokens` was set on the request. 512 is a generous multiple
    # of the ~2-3 sentences these calls ever produce, while remaining far
    # below the threshold that (combined with a stray thinking budget) trips
    # a provider's "must stream for long operations" guard.
    _GOAL_INTERNAL_CALL_MAX_TOKENS: ClassVar[int] = 512

    # Default for the `goal_provider_preferences` config knob (see
    # `__init__` and `_resolve_goal_model`): consulted ONLY when role
    # routing (`model_role_resolver`) doesn't produce a usable, mounted
    # provider -- e.g. no routing bundle is installed at all. Without this,
    # that case falls all the way through to the session's DEFAULT
    # (expensive, conversational) provider/model for every evaluator call --
    # one per turn -- which is the cost regression this default closes.
    #
    # Mirrors the routing matrix's own `fast`-role membership (see
    # amplifier-bundle-routing-matrix's balanced matrix) using GLOB
    # patterns -- deliberately never pinned concrete versions -- so a new
    # model release (e.g. the next Haiku point release) is picked up
    # automatically the moment a provider lists it, with no code change
    # here. Ranking among glob matches is handled by
    # `_resolve_goal_pref_glob`/`_goal_pref_version_sort_key` above, NOT
    # left to bare lexicographic sort (see that section's comment for why).
    _DEFAULT_GOAL_PROVIDER_PREFERENCES: ClassVar[list[dict[str, Any]]] = [
        {"provider": "anthropic", "model": "claude-haiku-*"},
        {"provider": "openai", "model": "gpt-?.?-luna*"},
        {"provider": "openai", "model": "gpt-?.?-mini*"},
        {"provider": "gemini", "model": "gemini-*-flash-preview"},
        {"provider": "github-copilot", "model": "claude-haiku-4.5"},
        {"provider": "github-copilot", "model": "gpt-5.4-mini"},
        {"provider": "ollama", "model": "*"},
    ]

    def __init__(self, config: dict[str, Any]):
        self.config = config
        # -1 means unlimited iterations (default)
        max_iter_config = config.get("max_iterations", -1)
        self.max_iterations = int(max_iter_config) if max_iter_config != -1 else -1
        # /goal stall detection: consecutive continuation turns with zero tool
        # calls required before the stall judge is even consulted (see
        # execute()'s stall-detection block and _judge_stall).
        self.goal_stall_threshold = int(config.get("goal_stall_threshold", 3))
        # Model role (routing-matrix name) requested for the goal-loop's
        # internal LLM calls (evaluator, stall judge, run summary) via the
        # `model_role_resolver` coordinator capability -- see
        # `_resolve_goal_model`. Default "fast" matches these calls' actual
        # shape (cheap, tool-less, two-line/one-sentence verdicts).
        self.goal_model_role = config.get("goal_model_role", "fast")
        # Ordered fallback list of {"provider", "model", "config"?} dicts,
        # consulted by `_resolve_goal_model` ONLY when the role resolver
        # above didn't yield a usable, mounted provider (no routing bundle
        # installed, resolver returned no candidates, or the resolved
        # provider isn't mounted). Mirrors the `model_role` +
        # `provider_preferences` precedence already established by agent
        # frontmatter (see foundation/agents/explorer.md): role wins when it
        # resolves, this declared list is the fallback for when it doesn't.
        # Defaults to `_DEFAULT_GOAL_PROVIDER_PREFERENCES`.
        self.goal_provider_preferences: list[dict[str, Any]] = config.get(
            "goal_provider_preferences", self._DEFAULT_GOAL_PROVIDER_PREFERENCES
        )
        # Evaluator input hygiene (see _flatten_message_for_evaluator,
        # _evaluate_goal). Promoted from a bare class constant to a config
        # knob, same override pattern as `goal_stall_threshold` above.
        #
        # Message-count window: how many of the most recent messages are
        # even considered before the char-based bounds below apply.
        self.goal_max_transcript_messages = int(
            config.get(
                "goal_max_transcript_messages", self._GOAL_MAX_TRANSCRIPT_MESSAGES
            )
        )
        # Per-tool-result / tool-call-argument clip, in characters (see
        # `_clip_head_tail`). `<= 0` disables clipping.
        self.goal_tool_content_clip_chars = int(
            config.get(
                "goal_tool_content_clip_chars", _GOAL_DEFAULT_TOOL_CONTENT_CLIP_CHARS
            )
        )
        # Total transcript character budget -- a backstop applied
        # NEWEST-FIRST above the per-message clip and the message-count
        # window (see `_evaluate_goal`). `<= 0` disables the budget.
        self.goal_transcript_char_budget = int(
            config.get(
                "goal_transcript_char_budget",
                self._GOAL_DEFAULT_TRANSCRIPT_CHAR_BUDGET,
            )
        )
        # Busy-stall trigger (b) -- see `_busy_stall_pretrip` and
        # execute()'s stall-detection block. Independent of
        # `goal_stall_threshold`/`no_tool_turns` above, which can never
        # observe a stall where the agent keeps making tool calls every
        # turn (the dominant real-world failure mode -- see
        # GOAL-HARDENING-DESIGN.md sec 1.2; real sessions a3126f2f,
        # 6e64b3db). Number of most recent evaluator reasons (regardless of
        # tool activity that turn) inspected by the cheap, free,
        # every-turn token-overlap pre-filter. `< 2` disables trigger (b)
        # entirely (a single reason can't show recurrence).
        self.goal_busy_stall_window = int(config.get("goal_busy_stall_window", 3))
        # Minimum token-set (Jaccard) overlap ratio between the oldest
        # reason in that window and each subsequent one, required for the
        # pre-filter to trip. Only a trip pays for the (rare) `_judge_stall`
        # call -- this knob controls how readily that happens, not whether
        # a stall is ultimately declared (the judge still must confirm).
        self.goal_busy_stall_min_overlap = float(
            config.get("goal_busy_stall_min_overlap", 0.5)
        )
        # Cap on how many of `goal["reasons"]` are shipped to the summary
        # model (see `_summarize_goal_run`/`_cap_reasons_for_summary`).
        # `goal["reasons"]` grows one entry per turn with no cap of its
        # own (by design -- `_judge_stall` and the CLI's dedupe both need
        # the full history); the summary call only needs the CURRENT
        # state, which the tail of a long run already establishes. `<= 0`
        # disables the cap (send the whole list, pre-existing behavior).
        self.goal_summary_max_reasons = int(
            config.get("goal_summary_max_reasons", 20)
        )
        # Per-execute()-call cache for `_resolve_goal_model`'s result (see
        # that method's CRITICAL PERF note) -- reset to None at the top of
        # execute() so each run re-resolves once, not on every turn.
        self._goal_model_cache: tuple[str, Any, str | None, dict[str, Any]] | None = (
            None
        )
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
        # Reset the per-run model-role resolution cache (see
        # `_resolve_goal_model`'s CRITICAL PERF note) so each execute() call
        # re-resolves the goal-loop model role exactly once, not once per
        # turn.
        self._goal_model_cache = None

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
            stall_verdict: str | None = None
            stall_trigger: str | None = None
            if is_continuation_turn:
                if self._tool_calls_this_turn == 0:
                    goal["no_tool_turns"] += 1
                else:
                    goal["no_tool_turns"] = 0

                # Trigger (a) -- idle: mechanical absence-of-action streak
                # (original trigger).
                idle_trip = goal["no_tool_turns"] >= self.goal_stall_threshold
                # Trigger (b) -- busy: tool-activity-INDEPENDENT pre-filter
                # (see `_busy_stall_pretrip`). `idle_trip` can only ever
                # observe a stall during turns with ZERO tool calls -- it
                # never fires for the dominant real-world failure mode: the
                # agent stays busy (tool calls every turn) while the goal
                # has already become unsatisfiable (see
                # GOAL-HARDENING-DESIGN.md sec 1.2; real sessions
                # a3126f2f r8, 6e64b3db r1/r2 -- 9/15/8 turns wasted past
                # the point the evaluator itself named the lock). Skipped
                # once `idle_trip` already holds, so a turn never pays for
                # two judge calls; idle framing wins on the rare turn where
                # both mechanical conditions happen to hold at once (a
                # busy-then-suddenly-idle transition).
                busy_trip = (not idle_trip) and self._busy_stall_pretrip(goal)

                if idle_trip or busy_trip:
                    stall_trigger = "idle" if idle_trip else "busy"
                    # Condition (a)/(b) above -- absence of action, or a
                    # recurring blocker despite activity -- holds. Only now
                    # do we pay for the (rare) stall-judge call to check
                    # the other half of the dual condition: is this a
                    # static, unresolved blocker, or does it just look
                    # repetitive while genuinely progressing? Both the
                    # mechanical pre-filter AND the judge are required to
                    # ever trip -- text-similarity/repetition alone is
                    # never enough, since legitimate agent work (e.g.
                    # re-running a test after a fix) can look repetitive
                    # too. See TestDualConditionStallTrip.
                    try:
                        is_stalled, stall_detail, stall_verdict = (
                            await self._judge_stall(
                                goal,
                                providers,
                                hooks,
                                coordinator,
                                trigger=stall_trigger,
                            )
                        )
                    except Exception as e:
                        # Fail open: a flaky judge call must never itself
                        # manufacture a false stall.
                        logger.warning(
                            f"/goal: stall judge failed, continuing normally: {e}"
                        )
                        is_stalled, stall_detail, stall_verdict = False, None, None

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
                        stall_verdict=stall_verdict,
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
                        goal,
                        state="continuing",
                        reason=reason,
                        stall_verdict=stall_verdict,
                    ),
                )
                goal["continuations"] += 1
                stall_prompt = self._goal_stall_escalation_prompt(
                    goal,
                    reason,
                    trigger=stall_trigger or "idle",
                    verdict=stall_verdict,
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

    _REASON_TOKEN_RE: ClassVar[re.Pattern[str]] = re.compile(r"[a-z0-9]+")

    @classmethod
    def _reason_token_set(cls, text: str) -> set[str]:
        """Lowercase, alnum-run tokenization for the busy-stall pre-filter
        (see `_busy_stall_pretrip`). Deliberately crude -- this is a cheap,
        free, every-turn heuristic whose only job is deciding whether to
        PAY for a judge call, not deciding the stall itself (the judge
        always confirms; see TestDualConditionStallTrip).
        """
        return set(cls._REASON_TOKEN_RE.findall(text.lower()))

    @staticmethod
    def _token_overlap_ratio(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two token sets. Two empty sets are
        treated as fully overlapping (both say "nothing"); one empty and
        one non-empty are treated as fully disjoint.
        """
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _busy_stall_pretrip(self, goal: dict[str, Any]) -> bool:
        """Trigger (b)'s cheap, deterministic, tool-activity-INDEPENDENT
        pre-filter (see execute()'s stall-detection block).

        Runs every continuation turn at zero LLM cost -- only a trip here
        pays for a `_judge_stall` call, mirroring how trigger (a)
        (`no_tool_turns >= goal_stall_threshold`) already gates the
        original judge call. Trips when the most recent
        `goal_busy_stall_window` evaluator reasons (regardless of whether
        the assistant made tool calls on those turns) all share at least
        `goal_busy_stall_min_overlap` token-set overlap with the OLDEST
        reason in that window -- i.e. the evaluator has been reporting
        essentially the same blocker for `goal_busy_stall_window` turns
        straight.

        This is the trigger the real long stalls needed and trigger (a)
        structurally cannot provide: `no_tool_turns` can only ever
        increment on a turn with ZERO tool calls, so an agent that stays
        busy (tool calls every turn) while the goal has already become
        unsatisfiable never trips it at all (see GOAL-HARDENING-DESIGN.md
        sec 1.2; real sessions a3126f2f r8, 6e64b3db r1/r2 ran 9/15/8 turns
        past the point the evaluator itself named the lock).
        """
        window = self.goal_busy_stall_window
        if window < 2:
            return False
        reasons = goal.get("reasons") or []
        if len(reasons) < window:
            return False
        recent = reasons[-window:]
        anchor = self._reason_token_set(recent[0])
        threshold = self.goal_busy_stall_min_overlap
        return all(
            self._token_overlap_ratio(anchor, self._reason_token_set(r)) >= threshold
            for r in recent[1:]
        )

    def _goal_stall_escalation_prompt(
        self,
        goal: dict[str, Any],
        reason: str,
        *,
        trigger: str,
        verdict: str | None,
    ) -> str:
        """Build the one-shot rescue prompt sent back to the agent on a
        first stall trip (see execute()'s escalation branch -- reused
        unchanged, and unchanged in structure, for both triggers).

        Framing must match the REAL situation for whichever trigger fired:
        telling a busy agent "you haven't taken any actions" would hand it
        as false a premise as the stall judge's own prompt would get
        without the `_GOAL_STALL_SYSTEM_PROMPT_IDLE`/`_BUSY` split (see
        those constants for the matching concern on the judge side).
        """
        if trigger == "busy":
            activity_clause = (
                f"Across the last {self.goal_busy_stall_window} turns the "
                "evaluator keeps reporting the same kind of blocker "
                f"regardless of what tool activity did or didn't happen "
                f"those turns: {reason}"
            )
        else:
            activity_clause = (
                f"For the last {goal['no_tool_turns']} turns you haven't "
                "taken any actions (no tool calls), and the evaluator "
                f"keeps reporting the same kind of blocker: {reason}"
            )

        verdict_clause = ""
        explanation = self._GOAL_STALL_VERDICT_EXPLANATIONS.get(verdict or "")
        if explanation:
            verdict_clause = (
                f" A reviewing judge classified this as {verdict}: "
                f"{explanation}."
            )

        return (
            f"You've been asked to work toward this goal: "
            f"{goal['condition']}\n\n"
            f"{activity_clause}{verdict_clause}\n\n"
            "You appear stuck. Either try a genuinely different approach "
            "to make progress, or, if you believe this goal cannot be "
            "achieved as it's currently defined, say so plainly and "
            "explain specifically why -- don't just repeat what you've "
            "already said."
        )

    async def _resolve_goal_provider_preferences(
        self,
        providers: dict[str, Any],
    ) -> tuple[str, Any, str | None, dict[str, Any]] | None:
        """Walk ``self.goal_provider_preferences`` in order, returning the
        first entry whose provider is mounted for this session AND whose
        model (glob or exact) resolves against that provider.

        This is the fallback layer ``_resolve_goal_model`` consults ONLY
        when role routing (the ``model_role_resolver`` capability) didn't
        produce a usable result -- see that method's docstring for the
        overall precedence.

        A preference entry whose provider isn't mounted, or whose glob
        matches nothing (or whose ``list_models()`` raises), is skipped in
        favor of the NEXT entry -- it never aborts the whole fallback and
        never sends the raw, unresolved glob to a provider.

        Returns ``None`` when no entry resolves, meaning the caller must
        fall further back to the session default.
        """
        for entry in self.goal_provider_preferences:
            provider_key = entry.get("provider")
            model_pattern = entry.get("model")
            if not provider_key or not model_pattern:
                continue

            match: tuple[str, Any] | None = None
            for name, provider in providers.items():
                if provider_key in (
                    name,
                    name.replace("provider-", ""),
                    f"provider-{provider_key}",
                ):
                    match = (name, provider)
                    break
            if match is None:
                continue

            resolved_name, resolved_provider = match
            if _goal_pref_is_glob(model_pattern):
                resolved_model = await _resolve_goal_pref_glob(
                    model_pattern, resolved_provider
                )
                if resolved_model is None:
                    continue
            else:
                resolved_model = model_pattern

            logger.info(
                "/goal: goal_provider_preferences entry provider=%r "
                "model=%r resolved to %r (used because model_role routing "
                "did not yield a usable provider).",
                provider_key,
                model_pattern,
                resolved_model,
            )
            return (
                resolved_name,
                resolved_provider,
                resolved_model,
                dict(entry.get("config") or {}),
            )

        return None

    async def _resolve_goal_model(
        self,
        providers: dict[str, Any],
        coordinator: ModuleCoordinator | None,
    ) -> tuple[str, Any, str | None, dict[str, Any]]:
        """Resolve the provider/model/config to use for the goal-loop's
        internal LLM calls (evaluator, stall judge, run summary).

        Resolution order:

        1. ``model_role_resolver`` coordinator capability, honoring
           ``self.goal_model_role`` (default ``"fast"``) -- wins whenever it
           yields a usable, mounted provider.
        2. ``self.goal_provider_preferences`` (see
           ``_resolve_goal_provider_preferences``) -- consulted ONLY when
           (1) doesn't produce a usable result (no resolver registered, it
           resolved to no candidates, or the resolved provider isn't
           mounted). This mirrors the ``model_role`` + ``provider_preferences``
           precedence already established by agent frontmatter (see
           foundation/agents/explorer.md): role wins when it resolves, the
           declared preference list is the fallback for when it doesn't.
        3. The session's default (first-listed) provider with no model
           override, logged as a WARNING -- making it obvious the run is
           NOT using the configured ``goal_model_role`` or any configured
           preference.

        Returns ``(provider_name, provider, model, config)``:

        - ``provider_name``/``provider``: the resolved provider INSTANCE to
          call. This may be a *different* provider than the session default
          -- e.g. the matrix's ``fast`` role points at an installed OpenAI
          provider while the session's main conversation runs on Anthropic.
          Calling the WRONG provider instance with a model name from a
          DIFFERENT provider is a real failure mode this must avoid.
        - ``model``: a concrete model name (globs are already resolved --
          never re-glob here), or ``None`` when falling back to the
          provider's own default model.
        - ``config``: the resolved layer's per-provider config dict (e.g.
          ``{"reasoning_effort": "high"}``) to forward as ``complete()``
          kwargs. Callers must apply ``extended_thinking=False`` AFTER
          spreading this so it always wins (see DEFECT 4).

        CRITICAL TIMING: the ``model_role_resolver`` capability is
        registered by a hooks module that mounts *after* the orchestrator,
        so it must be looked up here (lazily, at call time) and never in
        ``mount()``/``__init__``.

        CRITICAL PERF: resolution is cached for the lifetime of one
        ``execute()`` call (see ``self._goal_model_cache``, reset at the
        top of ``execute()``) -- both the role resolver's ``resolve()`` and
        the preference list's glob resolution can perform a live
        ``provider.list_models()`` network round-trip, and the evaluator
        runs every turn.

        No silent fallbacks: when neither the role resolver nor the
        preference list produces a usable, mounted provider, this logs a
        WARNING naming the specific cause(s) and falls back to the
        session's default (first-listed) provider with no model override.
        """
        if self._goal_model_cache is not None:
            return self._goal_model_cache

        if not providers:
            raise RuntimeError("no provider mounted for goal-loop model resolution")
        default_name, default_provider = next(iter(providers.items()))

        def _fallback(cause: str) -> tuple[str, Any, str | None, dict[str, Any]]:
            logger.warning(
                "/goal: model_role '%s' %s -- falling back to the session "
                "default provider/model for the evaluator/stall-judge/"
                "summary calls, which is NOT the configured fast role.",
                self.goal_model_role,
                cause,
            )
            return (default_name, default_provider, None, {})

        async def _via_role_resolver() -> tuple[
            tuple[str, Any, str | None, dict[str, Any]] | None, str | None
        ]:
            """Try the ``model_role_resolver`` capability. Returns
            ``(result, None)`` on success or ``(None, cause)`` on failure --
            the cause is the exact WARNING-message fragment used by the
            pre-existing tests, unchanged."""
            resolver = (
                coordinator.get_capability("model_role_resolver")
                if coordinator and hasattr(coordinator, "get_capability")
                else None
            )
            if resolver is None:
                return None, (
                    "specified but no model_role_resolver capability is "
                    "registered (install a routing bundle)"
                )

            resolved = await resolver.resolve(self.goal_model_role)
            if not resolved:
                return None, (
                    f"resolved to no candidates against installed providers "
                    f"(resolver={getattr(resolver, 'name', type(resolver).__name__)})"
                )

            pref = resolved[0]
            match: tuple[str, Any] | None = None
            for name, provider in providers.items():
                if pref.provider in (
                    name,
                    name.replace("provider-", ""),
                    f"provider-{pref.provider}",
                ):
                    match = (name, provider)
                    break

            if match is None:
                return None, (
                    f"resolved to provider '{pref.provider}', which is not "
                    "mounted/installed for this session"
                )

            resolved_name, resolved_provider = match
            return (
                resolved_name,
                resolved_provider,
                pref.model,
                dict(pref.config or {}),
            ), None

        role_result, role_cause = await _via_role_resolver()
        if role_result is not None:
            self._goal_model_cache = role_result
            return role_result

        # Role routing didn't yield a usable provider -- try the declared
        # goal_provider_preferences fallback list before giving up to the
        # (expensive) session default. Logged at INFO, not WARNING: the
        # preference list may well rescue this, so it isn't yet the
        # alarming case -- that's reserved for reaching the final fallback.
        logger.info(
            "/goal: model_role '%s' %s -- trying goal_provider_preferences "
            "before the session default.",
            self.goal_model_role,
            role_cause,
        )
        pref_result = await self._resolve_goal_provider_preferences(providers)
        if pref_result is not None:
            self._goal_model_cache = pref_result
            return pref_result

        result = _fallback(
            f"{role_cause}, and no goal_provider_preferences entry resolved either"
        )
        self._goal_model_cache = result
        return result

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
        stall_verdict: str | None = None,
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
            # distinguishes "hit a wall" (1) from "flailing" (N). Left in
            # place unchanged -- see `stall_verdict` below for the fix to
            # the actual downstream bug (a rephrased-every-turn blocker
            # reads as "flailing" here even when it's really a wall).
            "distinct_blockers": self._distinct_blocker_count(reasons),
            # The taxonomy verdict from `_judge_stall` (see
            # _GOAL_STALL_TAXONOMY_BLOCK): "resolvable", "time-locked",
            # "structure-locked", "history-locked", or None (no stall
            # judge call was made this turn -- the common case). This is
            # the SEMANTIC signal `distinct_blockers` was being asked to
            # stand in for; consumers (see amplifier-app-cli's
            # goal_progress_hook.py) should prefer this for wall-vs-
            # flailing wording over re-deriving it from `reasons`.
            "stall_verdict": stall_verdict,
            # The fully-expanded goal condition -- @mentions already resolved
            # at /goal set-time (see amplifier-app-cli's process_runtime_
            # mentions call sites), i.e. the exact text re-sent to the
            # evaluator every turn. Deliberately the expanded form, not the
            # raw user-typed text: a reader of this event should see exactly
            # what the evaluator judged. Previously reconstructable only via
            # a fragile 4-step procedure (split runs on turn reset, find the
            # nearest preceding Prompt node, then distinguish the real user
            # prompt from this loop's own auto-injected continuation
            # prompts -- indistinguishable by node type alone).
            "condition": goal.get("condition"),
            # See _GOAL_PROGRESS_SCHEMA_VERSION for the versioning contract.
            # `metadata` (previously a hardcoded-None slot here) is removed:
            # it measured null on 100% of 328 sampled events across three
            # graph endpoints, and a repo-wide grep of both this repo and
            # amplifier-app-cli found zero readers of it on this event.
            "schema_version": self._GOAL_PROGRESS_SCHEMA_VERSION,
        }

    async def _judge_stall(
        self,
        goal: dict[str, Any],
        providers: dict[str, Any],
        hooks: HookRegistry,
        coordinator: ModuleCoordinator | None = None,
        *,
        trigger: str = "idle",
    ) -> tuple[bool, str | None, str | None]:
        """Ask a cheap, tool-less model to classify the recent run of
        evaluator reasons: is the goal durably stuck (and, if so, which
        taxonomy of lock), or is this genuine incremental progress that
        merely looks repetitive?

        ``trigger`` selects which of the two mechanical pre-filters called
        this (see execute()'s stall-detection block) and therefore which
        framing is TRUE for this call -- passing the wrong one would hand
        the judge a false premise, exactly the defect
        _GOAL_STALL_SYSTEM_PROMPT_IDLE/_BUSY exist to avoid:

        - ``"idle"`` (default): trigger (a), ``no_tool_turns >=
          goal_stall_threshold`` already holds -- the assistant took NO
          tool actions for that whole streak. Recent-reasons window is
          ``goal["reasons"][-goal["no_tool_turns"]:]``.
        - ``"busy"``: trigger (b), ``_busy_stall_pretrip`` already tripped
          -- the assistant HAS been taking tool actions, but the same
          blocker keeps recurring regardless. Recent-reasons window is
          ``goal["reasons"][-goal_busy_stall_window:]``.

        Only called once the relevant mechanical pre-filter already holds
        -- that's what keeps this rare and cheap. Returns
        ``(is_stalled, detail, verdict)`` where ``verdict`` is one of
        ``"resolvable"``, ``"time-locked"``, ``"structure-locked"``,
        ``"history-locked"`` (or ``None`` when there was nothing to judge).
        ``is_stalled`` is ``verdict != "resolvable"``. Raises on failure;
        the caller treats a raised exception as "not stalled" (fail open on
        the judge -- a flaky judge call must never itself manufacture a
        false stall; the mechanical pre-filter is what keeps stall
        detection safe).
        """
        if trigger == "busy":
            window = self.goal_busy_stall_window
            recent_reasons = goal["reasons"][-window:] if goal.get("reasons") else []
            system_prompt = self._GOAL_STALL_SYSTEM_PROMPT_BUSY
            activity_clause = (
                "the same kind of blocker kept recurring regardless of "
                "whatever tool activity did or didn't happen those turns"
            )
        else:
            recent_reasons = (
                goal["reasons"][-goal["no_tool_turns"] :]
                if goal.get("reasons")
                else []
            )
            system_prompt = self._GOAL_STALL_SYSTEM_PROMPT_IDLE
            activity_clause = "the assistant took no tool actions at all"

        if not recent_reasons:
            return False, None, None

        if not providers:
            raise RuntimeError("no provider mounted for stall judgment")
        (
            provider_name,
            provider,
            model_override,
            role_config,
        ) = await self._resolve_goal_model(providers, coordinator)

        history_text = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(recent_reasons))
        user_prompt = (
            f"GOAL CONDITION:\n{goal['condition']}\n\n"
            f"EVALUATOR REASONS across the last {len(recent_reasons)} turns, "
            f"during which {activity_clause}:\n"
            f"{history_text}\n\n"
            "Classify this using the required two-line format."
        )

        judge_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        # DEFECT 4 fix (real session e97e192b): this internal utility call was
        # built with none of the "don't leak into the foreground" flags that
        # amplifier-foundation's hooks-session-naming module already
        # establishes as precedent for exactly this situation (see its
        # __init__.py's `_generate_name`, ~line 519-549). Without
        # metadata={"stream": False}, providers stream by default
        # (config.get("use_streaming", True) -- see
        # docs/provider-streaming-contract.md), which emits the same
        # llm:stream_block_start/delta/end events a real user turn does --
        # indistinguishable to hooks-streaming-ui, so this judge's output
        # (and, if thinking is enabled, its entire chain-of-thought) paints
        # into the user-facing overlay. max_output_tokens is capped for the
        # same reason session-naming caps its own call: without an explicit
        # value the request inherits the provider's session default (64000
        # for Haiku), which combined with an uncapped thinking budget can
        # trip a provider's "must stream for long operations" guard.
        chat_request = ChatRequest(
            messages=judge_messages,
            tools=None,
            model=model_override,
            metadata={"stream": False},
            max_output_tokens=self._GOAL_INTERNAL_CALL_MAX_TOKENS,
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
        #
        # DEFECT 4 fix: `extended_thinking=False` is passed as an explicit
        # kwarg, not left to "just not setting it". The installed Anthropic
        # provider's thinking-enablement precedence is
        # kwargs["extended_thinking"] > request.reasoning_effort >
        # config["reasoning_effort"]/config["effort"] -- so when the
        # *session's* provider config carries a reasoning_effort/effort
        # setting (as it did in e97e192b, sized for the main conversational
        # model), every call through that same provider instance -- this one
        # included -- inherits it unless explicitly overridden. Relying on
        # this orchestrator never setting `reasoning_effort` itself is not
        # sufficient; the provider-level default applies regardless. Only an
        # explicit `extended_thinking=False` kwarg reliably opts out (mirrors
        # hooks-session-naming's identical opt-out, same file/line cited
        # above).
        #
        # `role_config` (from the resolved model role's `ProviderPreference`,
        # e.g. `{"reasoning_effort": "high"}`) is forwarded as complete()
        # kwargs first -- `extended_thinking=False` is applied AFTER so it
        # always wins regardless of what the routing matrix's per-role
        # config carries.
        complete_kwargs: dict[str, Any] = dict(role_config)
        if model_override:
            complete_kwargs["model"] = model_override
        complete_kwargs["extended_thinking"] = False
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

        verdict_word = lines[0].strip().upper()
        verdict = self._GOAL_STALL_VERDICT_WORDS.get(verdict_word)
        if verdict is None:
            raise ValueError(f"unparseable stall judge verdict: {lines[0]!r}")

        detail = lines[1] if len(lines) > 1 else "(judge gave no reason)"
        is_stalled = verdict in self._GOAL_STALL_LOCKED_VERDICTS
        return is_stalled, detail, verdict

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

    def _cap_reasons_for_summary(
        self, reasons: list[str]
    ) -> tuple[list[str], int]:
        """Bound how many of ``goal["reasons"]`` are shipped to the summary
        model (see ``_summarize_goal_run``).

        ``goal["reasons"]`` grows one entry per turn with no cap of its own
        by design -- ``_judge_stall`` and the CLI's dedupe both need the
        full history -- so a long stalled/cap_hit run (54 turns in the
        worst recorded case, real session a3126f2f) previously shipped the
        ENTIRE list, whole, to the summary call every time. The summary
        only ever needs to explain the CURRENT state, which the tail of
        the run already establishes, so only the most recent
        ``goal_summary_max_reasons`` entries are kept. Returns
        ``(kept_reasons, omitted_count)`` -- ``omitted_count`` is 0 when
        nothing was cut (including when the cap is disabled via ``<= 0``).
        """
        cap = self.goal_summary_max_reasons
        if cap <= 0 or len(reasons) <= cap:
            return list(reasons), 0
        return list(reasons[-cap:]), len(reasons) - cap

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
        rather than silently going missing.

        Returns the model's text in full, with no length cap -- storage and
        display are different concerns (previously this truncated to
        ``_GOAL_SUMMARY_MAX_CHARS`` before storage, so a stalled/cap_hit
        run's stored summary could end mid-clause and the full model text
        was never retained anywhere). Truncation for one-line terminal
        rendering is now applied only at display time, in amplifier-app-cli's
        ``goal_progress_hook.py``. The per-state system prompts above still
        ask the model for "no more than about 120 characters" to keep the
        text itself short, but that is a request to the model, not an
        enforced ceiling on what gets stored.
        """
        try:
            if not providers:
                raise RuntimeError("no provider mounted for summary")
            (
                provider_name,
                provider,
                model_override,
                role_config,
            ) = await self._resolve_goal_model(providers, coordinator)

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
                kept_reasons, omitted_count = self._cap_reasons_for_summary(reasons)
                reasons_lines = [
                    f"{i + 1}. {r}" for i, r in enumerate(kept_reasons)
                ]
                if omitted_count:
                    reasons_lines.insert(
                        0, f"(earliest {omitted_count} reasons omitted)"
                    )
                reasons_text = (
                    "\n".join(reasons_lines)
                    if reasons_lines
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
            # DEFECT 4 fix: see the matching comment in _judge_stall --
            # metadata={"stream": False} keeps this background call off the
            # streaming branch (and off hooks-streaming-ui's overlay), and
            # max_output_tokens caps the response to what a one-sentence
            # summary actually needs.
            chat_request = ChatRequest(
                messages=summary_messages,
                tools=None,
                model=model_override,
                metadata={"stream": False},
                max_output_tokens=self._GOAL_INTERNAL_CALL_MAX_TOKENS,
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
            #
            # DEFECT 4 fix: extended_thinking=False is likewise passed
            # explicitly -- see the matching comment in _judge_stall for why
            # simply not setting it is not sufficient (a session-level
            # provider config can force thinking on regardless).
            #
            # `role_config` forwarded first, extended_thinking=False applied
            # after so it always wins (see the matching comment in
            # _judge_stall / _resolve_goal_model).
            complete_kwargs: dict[str, Any] = dict(role_config)
            if model_override:
                complete_kwargs["model"] = model_override
            complete_kwargs["extended_thinking"] = False
            response = await provider.complete(chat_request, **complete_kwargs)
            summary_text = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    summary_text += getattr(block, "text", "")
            summary_text = summary_text.strip()
            if not summary_text:
                return self._goal_summary_fallback(goal, final_state)
            return summary_text
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
        if len(messages) > self.goal_max_transcript_messages:
            messages = messages[-self.goal_max_transcript_messages :]
            truncated = True

        flattened = [
            t
            for t in (
                _flatten_message_for_evaluator(m, self.goal_tool_content_clip_chars)
                for m in messages
            )
            if t
        ]

        # Total transcript character budget: a backstop ABOVE the
        # per-message clip and the message-count window above. Walked
        # NEWEST-FIRST so that when the budget binds, the most recent
        # (most verdict-relevant) turns survive and OLDER messages within
        # the window are what gets dropped -- then restored to
        # chronological order for the final prompt. The newest message is
        # always kept even if it alone exceeds the budget, so the
        # evaluator is never handed an empty transcript.
        budget = self.goal_transcript_char_budget
        if budget > 0 and flattened:
            kept_reversed: list[str] = []
            used = 0
            for text in reversed(flattened):
                # "\n\n".join cost: 2 extra chars per joiner once >1 kept.
                cost = len(text) + (2 if kept_reversed else 0)
                if used + cost > budget and kept_reversed:
                    truncated = True
                    break
                kept_reversed.append(text)
                used += cost
            flattened = list(reversed(kept_reversed))

        transcript_text = "\n\n".join(flattened)

        if not providers:
            raise RuntimeError("no provider mounted for evaluation")
        (
            provider_name,
            provider,
            model_override,
            role_config,
        ) = await self._resolve_goal_model(providers, coordinator)

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
        # DEFECT 4 fix: see the matching comment in _judge_stall --
        # metadata={"stream": False} keeps this background call off the
        # streaming branch (and off hooks-streaming-ui's overlay), and
        # max_output_tokens caps the response to what a two-line verdict
        # actually needs. This is the call confirmed responsible for real
        # session e97e192b's thinking-leak: 46 lines of the evaluator's
        # chain-of-thought streamed to the user because this request took
        # the same streaming code path as a normal user turn.
        chat_request = ChatRequest(
            messages=eval_messages,
            tools=None,
            model=model_override,
            metadata={"stream": False},
            max_output_tokens=self._GOAL_INTERNAL_CALL_MAX_TOKENS,
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
        #
        # DEFECT 4 fix: extended_thinking=False is likewise passed
        # explicitly -- see the matching comment in _judge_stall for why
        # simply not setting it is not sufficient (a session-level provider
        # config can force thinking on regardless). This is the exact call
        # confirmed (via real-session telemetry) to have run with thinking
        # enabled and a 32000-token budget in session e97e192b.
        #
        # `role_config` forwarded first, extended_thinking=False applied
        # after so it always wins (see the matching comment in
        # _judge_stall / _resolve_goal_model).
        complete_kwargs: dict[str, Any] = dict(role_config)
        if model_override:
            complete_kwargs["model"] = model_override
        complete_kwargs["extended_thinking"] = False
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
                        # metadata.ephemeral marks this as regenerated-per-turn content
                        # so the provider never places a prompt-cache breakpoint on it
                        # (see amplifier_module_provider_anthropic._count_trailing_ephemeral_messages).
                        message_dicts.append(
                            {
                                "role": result.context_injection_role,
                                "content": result.context_injection,
                                "metadata": {"ephemeral": True},
                            }
                        )
                        logger.debug(
                            f"Last message role is '{last_msg.get('role')}', not 'tool' - "
                            "created new message for injection"
                        )
                else:
                    # Default behavior: append as new message
                    # metadata.ephemeral marks this as regenerated-per-turn content
                    # so the provider never places a prompt-cache breakpoint on it
                    # (see amplifier_module_provider_anthropic._count_trailing_ephemeral_messages).
                    message_dicts.append(
                        {
                            "role": result.context_injection_role,
                            "content": result.context_injection,
                            "metadata": {"ephemeral": True},
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
                            # metadata.ephemeral marks this as regenerated-per-turn
                            # content so the provider never places a prompt-cache
                            # breakpoint on it (see
                            # amplifier_module_provider_anthropic._count_trailing_ephemeral_messages).
                            message_dicts.append(
                                {
                                    "role": injection["role"],
                                    "content": injection["content"],
                                    "metadata": {"ephemeral": True},
                                }
                            )
                            logger.debug(
                                "Last message not a tool result, created new message for injection"
                            )
                    else:
                        # metadata.ephemeral marks this as regenerated-per-turn
                        # content so the provider never places a prompt-cache
                        # breakpoint on it (see
                        # amplifier_module_provider_anthropic._count_trailing_ephemeral_messages).
                        message_dicts.append(
                            {
                                "role": injection["role"],
                                "content": injection["content"],
                                "metadata": {"ephemeral": True},
                            }
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
                tools_list = [_build_tool_spec(t) for t in tools.values()]

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
                    tools_list = [_build_tool_spec(t) for t in tools.values()]

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
