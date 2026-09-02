"""Regression tests: a cancelled tool must not destroy its siblings' results.

Defect (model_performance-g2e), as it stood before this file existed:

    __init__.py:3827-3832  tool_tasks = [self._execute_tool_only(tc, ...) for tc in tool_calls]
    __init__.py:3835       tool_results = await asyncio.gather(*tool_tasks)   # no return_exceptions
    __init__.py:4250       _execute_tool_only docstring: "Never raises - errors become error messages."
    __init__.py:4397       ...but its safety net is `except Exception`, and asyncio.CancelledError
                           is a BaseException (PEP 3110 / Python 3.8+), so it is NOT caught and
                           escapes the task.
    __init__.py:3836-3850  gather propagates it, and the handler then wrote
                           {"error": "Tool execution was cancelled by user", "cancelled": true, ...}
                           as the tool result for EVERY tool_call in the batch -- including the
                           siblings that had ALREADY COMPLETED SUCCESSFULLY. Their output was not
                           merely dropped, it was OVERWRITTEN.

The turn-closing assistant message written on that same path (:3881) says
"Results from completed tools have been preserved." It was not true. These
tests make it true and keep it true.

Real-world shape: `tool-delegate` re-raises CancelledError by design
(amplifier-foundation tool-delegate/__init__.py :1144-1153, :1334-1344), so ONE
cancelled delegate discarded the whole batch's completed work.

INVARIANT UNDER TEST: a completed sibling's result is never overwritten by
another task's cancellation. Only the cancelled tool's own result is reported
as cancelled.

NON-GOAL: these tests deliberately do NOT assert that cancellation is
swallowed. CancelledError must still propagate out of execute() (see
tests/test_steering.py::test_immediate_cancel_clears_pending_steers, which
pins that). Swallowing cancellation inside _execute_tool_only would defeat
cooperative cancellation; the fix is at the gather site, in what gets written
to context before the re-raise.
"""

from __future__ import annotations

import asyncio

import pytest

CANCEL_MARK = "Tool execution was cancelled by user"


# ---------------------------------------------------------------------------
# Minimal stubs (mirrors tests/test_steering.py -- kept local and explicit so
# this file reads standalone).
# ---------------------------------------------------------------------------


class MockHookResult:
    action = "pass"
    reason = None
    ephemeral = False
    context_injection = None
    context_injection_role = "user"
    append_to_last_tool_result = False
    data = None


class MockHooks:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(
        self, event_name: str, payload: dict | None = None
    ) -> MockHookResult:
        self.emitted.append((event_name, payload or {}))
        return MockHookResult()


class MockContext:
    def __init__(self) -> None:
        self._messages: list[dict] = []

    async def add_message(self, msg: dict) -> None:
        self._messages.append(msg)

    async def get_messages_for_request(self, provider=None) -> list[dict]:  # noqa: ANN001
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

    def register_contributor(self, name: str, source: str, fn) -> None:  # noqa: ANN001
        pass

    async def mount(self, name: str, obj) -> None:  # noqa: ANN001
        self._mounts[name] = obj

    def register_capability(self, name: str, value) -> None:  # noqa: ANN001
        self._capabilities[name] = value

    def get_capability(self, name: str):  # noqa: ANN201
        return self._capabilities.get(name)

    async def process_hook_result(self, result, *args, **kwargs):  # noqa: ANN001,ANN201
        return result


class ToolCall:
    def __init__(self, call_id: str, name: str) -> None:
        self.id = call_id
        self.name = name
        self.arguments: dict = {}


class MockResponse:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.content = None
        self.content_blocks = None
        self.usage = None
        self.metadata = None


# ---------------------------------------------------------------------------
# The batch under test: two tools that COMPLETE, one that raises CancelledError
# only after both siblings have finished.
# ---------------------------------------------------------------------------


def _build_batch(
    third_tool: str = "raises",
) -> tuple[list[ToolCall], dict, asyncio.Event, asyncio.Event]:
    """Build a 3-tool batch: two that complete, one that does not.

    ``third_tool="raises"`` -> the third tool raises CancelledError itself
    once both siblings are done (the tool-delegate shape: INNER cancellation).

    ``third_tool="hangs"``  -> the third tool never returns, so the test can
    cancel the ENCLOSING task instead (the second-Ctrl+C shape: OUTER
    cancellation). Both paths land in the same handler.
    """
    from amplifier_core import ToolResult

    done_a = asyncio.Event()
    done_b = asyncio.Event()

    class CompletingTool:
        """Completes successfully and records a unique sentinel."""

        def __init__(self, name: str, sentinel: str, flag: asyncio.Event) -> None:
            self.name = name
            self.description = "completes successfully"
            self.input_schema: dict = {"type": "object", "properties": {}}
            self._sentinel = sentinel
            self._flag = flag

        async def execute(self, arguments):  # noqa: ANN001,ANN201
            # Yield once so the batch is genuinely concurrent rather than
            # completing synchronously before the canceller is even scheduled.
            await asyncio.sleep(0)
            self._flag.set()
            return ToolResult(success=True, output=self._sentinel)

    class CancellingTool:
        """Raises CancelledError -- but only once both siblings are DONE.

        This is the exact shape that matters: at the instant cancellation
        escapes, the sibling results already exist. Anything that loses them
        is losing completed work, not in-flight work.
        """

        name = "canceller"
        description = "raises asyncio.CancelledError mid-batch"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, arguments):  # noqa: ANN001,ANN201
            await done_a.wait()
            await done_b.wait()
            raise asyncio.CancelledError("simulated mid-batch cancellation")

    class HangingTool:
        """Never returns -- the straggler the user gives up on."""

        name = "canceller"
        description = "never completes"
        input_schema: dict = {"type": "object", "properties": {}}

        async def execute(self, arguments):  # noqa: ANN001,ANN201
            await asyncio.Event().wait()  # pragma: no cover -- never resolves

    tool_calls = [
        ToolCall("tc-a", "completes_a"),
        ToolCall("tc-b", "completes_b"),
        ToolCall("tc-c", "canceller"),
    ]
    tools = {
        "completes_a": CompletingTool("completes_a", "SENTINEL_A_COMPLETED", done_a),
        "completes_b": CompletingTool("completes_b", "SENTINEL_B_COMPLETED", done_b),
        "canceller": CancellingTool() if third_tool == "raises" else HangingTool(),
    }
    return tool_calls, tools, done_a, done_b


class _ProviderWithBatch:
    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self._tool_calls = tool_calls

    async def complete(self, chat_request, **kwargs):  # noqa: ANN001,ANN201
        return MockResponse(text="running three tools in parallel")

    def parse_tool_calls(self, response):  # noqa: ANN001,ANN201
        return list(self._tool_calls)


async def _run_cancelled_batch() -> MockContext:
    """Drive one parallel batch to the cancellation path; return the context."""
    from amplifier_module_loop_streaming import StreamingOrchestrator

    orch = StreamingOrchestrator({})
    ctx = MockContext()
    hooks = MockHooks()
    coordinator = MockCoordinator()
    tool_calls, tools, _done_a, _done_b = _build_batch("raises")

    with pytest.raises(asyncio.CancelledError):
        await orch.execute(
            prompt="run the batch",
            context=ctx,
            providers={"main": _ProviderWithBatch(tool_calls)},
            tools=tools,
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )
    return ctx


async def _run_outer_cancelled_batch() -> MockContext:
    """Cancel the ENCLOSING task mid-batch (the second-Ctrl+C shape).

    Two tools complete; the third never returns. Once both siblings are done
    we cancel the task running execute(), which is exactly what an immediate
    cancel does to the orchestrator.
    """
    from amplifier_module_loop_streaming import StreamingOrchestrator

    orch = StreamingOrchestrator({})
    ctx = MockContext()
    hooks = MockHooks()
    coordinator = MockCoordinator()
    tool_calls, tools, done_a, done_b = _build_batch("hangs")

    task = asyncio.ensure_future(
        orch.execute(
            prompt="run the batch",
            context=ctx,
            providers={"main": _ProviderWithBatch(tool_calls)},
            tools=tools,
            hooks=hooks,  # type: ignore[arg-type]
            coordinator=coordinator,  # type: ignore[arg-type]
        )
    )
    await asyncio.wait_for(done_a.wait(), timeout=5)
    await asyncio.wait_for(done_b.wait(), timeout=5)
    # Let the two completed tool tasks actually finish before cancelling.
    for _ in range(5):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return ctx


def _tool_messages(ctx: MockContext) -> dict[str, str]:
    return {
        m["tool_call_id"]: str(m.get("content", ""))
        for m in ctx._messages
        if m.get("role") == "tool"
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCompletedSiblingsSurviveACancelledPeer:
    async def test_completed_sibling_results_are_not_overwritten(self) -> None:
        """THE regression test.

        Before the fix, tc-a and tc-b both carried
        {"error": "Tool execution was cancelled by user", ...} despite having
        returned real output. Their work was overwritten, not merely dropped.
        """
        ctx = await _run_cancelled_batch()
        results = _tool_messages(ctx)

        assert results.get("tc-a") == "SENTINEL_A_COMPLETED", (
            f"completed sibling tc-a lost its own result; got: {results.get('tc-a')!r}"
        )
        assert results.get("tc-b") == "SENTINEL_B_COMPLETED", (
            f"completed sibling tc-b lost its own result; got: {results.get('tc-b')!r}"
        )

    async def test_only_the_cancelled_tool_is_reported_cancelled(self) -> None:
        """Exactly one cancelled marker in the batch -- the canceller's own."""
        ctx = await _run_cancelled_batch()
        results = _tool_messages(ctx)

        cancelled = [cid for cid, content in results.items() if CANCEL_MARK in content]
        assert cancelled == ["tc-c"], (
            "exactly the cancelled tool (tc-c) must carry the cancelled marker; "
            f"marker found on: {cancelled}"
        )

    async def test_zero_completed_sibling_results_overwritten(self) -> None:
        """The invariant, stated as a count: zero overwritten completed results."""
        ctx = await _run_cancelled_batch()
        results = _tool_messages(ctx)

        overwritten = [
            cid
            for cid, content in results.items()
            if cid in ("tc-a", "tc-b") and CANCEL_MARK in content
        ]
        assert len(overwritten) == 0, (
            f"{len(overwritten)} completed sibling result(s) overwritten by another "
            f"task's cancellation: {overwritten}"
        )

    async def test_tool_call_result_pairing_is_still_one_to_one(self) -> None:
        """Guard on the property the original blanket-overwrite protected.

        Every tool_call in the batch must still get exactly one tool result, in
        the original order -- otherwise the transcript has orphaned tool_calls
        and providers (Anthropic, OpenAI) reject the next request.
        """
        ctx = await _run_cancelled_batch()
        ordered = [m["tool_call_id"] for m in ctx._messages if m.get("role") == "tool"]
        assert ordered == ["tc-a", "tc-b", "tc-c"], (
            f"expected one result per tool_call in original order; got {ordered}"
        )

    async def test_turn_is_still_closed_with_an_assistant_message(self) -> None:
        """The synthetic closing assistant message (FM3 guard) still lands --
        and is now truthful about preserving completed results."""
        ctx = await _run_cancelled_batch()
        assistant_tail = [
            m
            for m in ctx._messages
            if m.get("role") == "assistant" and not m.get("tool_calls")
        ]
        assert assistant_tail, "cancellation must still close the turn (FM3 guard)"
        assert "cancelled" in str(assistant_tail[-1].get("content", "")).lower()


@pytest.mark.asyncio
class TestOuterCancellationAlsoPreservesCompletedSiblings:
    """The second-Ctrl+C shape: the ENCLOSING task is cancelled mid-batch.

    gather cancels the children and re-raises; the children that had already
    finished still hold real results, and those must survive. This is the path
    whose own closing message claims "Results from completed tools have been
    preserved" -- these tests are what make that sentence true.
    """

    async def test_completed_siblings_survive_an_outer_cancel(self) -> None:
        ctx = await _run_outer_cancelled_batch()
        results = _tool_messages(ctx)

        assert results.get("tc-a") == "SENTINEL_A_COMPLETED"
        assert results.get("tc-b") == "SENTINEL_B_COMPLETED"

    async def test_only_the_unfinished_tool_is_reported_cancelled(self) -> None:
        ctx = await _run_outer_cancelled_batch()
        results = _tool_messages(ctx)

        cancelled = [cid for cid, content in results.items() if CANCEL_MARK in content]
        assert cancelled == ["tc-c"], (
            "only the tool that never finished may be reported cancelled; "
            f"marker found on: {cancelled}"
        )

    async def test_pairing_preserved_under_outer_cancel(self) -> None:
        ctx = await _run_outer_cancelled_batch()
        ordered = [m["tool_call_id"] for m in ctx._messages if m.get("role") == "tool"]
        assert ordered == ["tc-a", "tc-b", "tc-c"]


@pytest.mark.asyncio
class TestExecuteToolOnlyCancellationContract:
    """Pins the deliberate choice NOT to swallow cancellation in the worker.

    _execute_tool_only's safety net is `except Exception` on purpose:
    asyncio.CancelledError is a BaseException and must keep propagating so
    cooperative cancellation still works. The docstring was corrected to say
    so rather than the catch being widened to BaseException -- widening it
    would also swallow KeyboardInterrupt/SystemExit and would make
    task.cancel() ineffective.
    """

    async def test_cancellederror_propagates_out_of_execute_tool_only(self) -> None:
        from amplifier_module_loop_streaming import StreamingOrchestrator

        orch = StreamingOrchestrator({})
        hooks = MockHooks()
        coordinator = MockCoordinator()

        class Canceller:
            name = "canceller"
            description = "raises"
            input_schema: dict = {"type": "object", "properties": {}}

            async def execute(self, arguments):  # noqa: ANN001,ANN201
                raise asyncio.CancelledError("from the tool itself")

        with pytest.raises(asyncio.CancelledError):
            await orch._execute_tool_only(
                ToolCall("tc-x", "canceller"),
                {"canceller": Canceller()},
                hooks,  # type: ignore[arg-type]
                "group-1",
                coordinator,  # type: ignore[arg-type]
            )


class TestExecuteToolOnlyDocstring:
    """Sync-only: the docstring is a contract, and it was wrong."""

    def test_docstring_does_not_claim_it_never_raises(self) -> None:
        """The docstring said 'Never raises'. It could raise BaseException.

        A docstring that is wrong about exactly the case that costs you data
        is worse than no docstring -- it is what made the gather site above
        look safe.
        """
        from amplifier_module_loop_streaming import StreamingOrchestrator

        doc = StreamingOrchestrator._execute_tool_only.__doc__ or ""
        assert "CancelledError" in doc, (
            "_execute_tool_only's docstring must name the BaseException it can "
            "propagate"
        )
        assert "Never raises - errors become error messages." not in doc, (
            "the unqualified 'Never raises' claim is false for BaseException"
        )
