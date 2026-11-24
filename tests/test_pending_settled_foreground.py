"""Only a current, host-prepared ordinary request may pass an in-flight summary."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from amplifier_core import ContextLengthError

from amplifier_module_loop_streaming import ExecutionSuspended, StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import NRoundToolProvider, OneShotTool, ScriptedHooks
from tests.test_execution_suspension import Owner
from tests.test_request_retention import RetainingContext, coordinator_for, injection


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "held", "tools", "declined", "no_ticket", "no_host_projection", "changed_source",
    "changed_provider", "missing_validator", "provider_refusal", "stop", "suspend",
    "stream_in_hook", "budget_in_hook", "stream_after_fit", "budget_after_fit",
])
async def test_pending_view_is_borrowed_only_for_the_admitted_dispatch(case, tmp_path):
    context = RetainingContext()
    coordinator = coordinator_for(context)
    suspension = Owner(tmp_path / "frame.json", context)
    coordinator.register_capability("execution.suspension", suspension)
    coordinator.register_capability("execution.resume", suspension.claim)
    events = []
    providers = {}
    bound = None
    paid_summary = asyncio.Event()
    if case != "held":
        paid_summary.set()

    async def wait():
        assert bound is None
        events.append("join")
        await paid_summary.wait()

    class Provider(NRoundToolProvider):
        def get_info(self):
            return SimpleNamespace(id="main", defaults={"model": "unchanged"})

        @asynccontextmanager
        async def prepared_foreground(self, logical, *, request_options):
            nonlocal bound
            assert bound is None
            assert "ordinary prompt" in str(logical.messages)
            assert "CURRENT-REMINDER" in str(logical.messages)
            assert logical.tools[0].name == "mock_tool"
            assert not request_options
            bound = logical
            events.append("prepare")
            try:
                yield logical
            finally:
                bound = None
                events.append("release")

        async def complete(self, request, **options):
            events.append("dispatch")
            if bound is not None:
                assert request is bound
            if case == "provider_refusal":
                raise ContextLengthError("actual provider refusal")
            return await super().complete(request, **options)

    provider = Provider(n_tool_rounds=1 if case == "tools" else 0)
    providers["main"] = provider
    if case == "no_host_projection":
        provider.prepared_foreground = None

    async def regular(**kwargs):
        events.append("ordinary retention")
        return await context.retaining_view(**kwargs)

    def begin(*, provider):
        assert provider is providers["main"]
        assert not context.add_message_calls
        events.append("begin")
        return None if case == "no_ticket" else pending

    @asynccontextmanager
    async def pending(*, retain_contents, project_view, request_options):
        rows = await context.get_messages()
        request = project_view(rows)
        async with provider.prepared_foreground(request, request_options=request_options):
            if case != "declined":
                def validate():
                    events.append("validate")
                    if case == "changed_source":
                        raise RuntimeError("source changed")

                yield {"base_view": rows, "dispatch": request,
                       "validate": None if case == "missing_validator" else validate}
                return
        await wait()
        yield None

    coordinator.register_capability("context.begin_settled_foreground", begin)
    coordinator.register_capability("context.request_retention", regular)
    coordinator.register_capability("context.wait_for_settled_checkpoint", wait)
    loop = StreamingOrchestrator({"ephemeral_injection_mode": "tail", "reminder_placement": "tail"})

    def change_capability():
        if case.startswith("stream"):
            async def stream(request, **kwargs):
                events.append("unexpected stream")
                yield {"content": "unexpected transport"}
            provider.stream = stream
        else:
            def request_budget(request):
                events.append("unexpected budget")
                raise AssertionError("newly counted route needs ordinary replanning")
            provider.request_budget = request_budget

    class Hooks(ScriptedHooks):
        async def emit(self, event, data):
            if event == "provider:request" and case in {"stream_in_hook", "budget_in_hook"}:
                change_capability()
            return await super().emit(event, data)

    async def delay(hooks, iteration):
        if case == "suspend":
            suspension.hold = True
        elif case == "stop":
            coordinator.cancellation.is_cancelled = True
        elif case == "changed_provider":
            providers["main"] = object()
        elif case in {"stream_after_fit", "budget_after_fit"}:
            change_capability()
    loop._apply_rate_limit_delay = delay
    hooks = Hooks({"provider:request": injection("CURRENT-REMINDER")})
    run = loop.execute("ordinary prompt", context, providers, {"mock_tool": OneShotTool()}, hooks, coordinator)
    if case == "suspend":
        with pytest.raises(ExecutionSuspended) as stopped:
            await run
        assert events == ["begin", "prepare", "release", "join", "ordinary retention"]
        frame = stopped.value.continuation
        assert frame["phase"] == "prepared_request"
        assert "CURRENT-REMINDER" in str(frame["prepared_request"]["request"])
        assert "validate" not in frame["prepared_request"]
        suspension.hold = False
        result = await StreamingOrchestrator(loop.config).resume(
            frame, context, providers, {"mock_tool": OneShotTool()}, hooks, coordinator
        )
        assert result and events[-1] == "dispatch" and events.count("prepare") == 1
    elif case in {"stream_in_hook", "budget_in_hook", "stream_after_fit", "budget_after_fit"}:
        with pytest.raises(RuntimeError, match="provider.*changed"):
            await run
        assert events == (["begin"] if case.endswith("_in_hook") else ["begin", "prepare", "release"])
    elif case in {"changed_source", "changed_provider", "missing_validator", "provider_refusal"}:
        with pytest.raises((RuntimeError, ContextLengthError, TypeError)):
            await run
        assert events.count("dispatch") == int(case == "provider_refusal")
        assert events[-1] == "release"
    elif case == "stop":
        await run
        assert events == ["begin", "prepare", "release"]
    else:
        assert await asyncio.wait_for(run, timeout=1)
        if case == "held":
            assert not paid_summary.is_set()
            paid_summary.set()
        if case in {"held", "tools"}:
            assert events[:5] == ["begin", "prepare", "validate", "dispatch", "release"]
            assert "join" not in events
            assert events.count("dispatch") == (2 if case == "tools" else 1)
        else:
            assert events[-3:] == ["join", "ordinary retention", "dispatch"]
    assert bound is None
