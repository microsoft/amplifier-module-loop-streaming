"""Contract tests against the real coordinator and ordinary engine hook path."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from amplifier_core import AmplifierSession, HookResult, ToolResult
from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall


pytestmark = pytest.mark.asyncio


class Echo:
    name = "read"
    description = "Fixture read"
    input_schema = {"type": "object", "properties": {}}

    def __init__(self):
        self.calls = 0
        self.active = self.peak = 0
        self.started = asyncio.Event()
        self.release = None

    async def execute(self, arguments):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            if self.release:
                await self.release.wait()
            return ToolResult(
                success=True, output={"value": arguments.get("value", 42)}
            )
        finally:
            self.active -= 1


class Client:
    name = "tool_exec"
    description = "Fixture orchestration client"
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, coordinator, handler=None):
        self.coordinator, self.handler = coordinator, handler
        self.lease = None

    async def execute(self, arguments):
        self.lease = self.coordinator.get_capability("tools.dispatch").bind()
        result = await (
            self.handler(self.lease)
            if self.handler
            else self.lease.call("read", arguments, request_id="1")
        )
        return ToolResult(success=True, output=result)


@pytest_asyncio.fixture
async def environment():
    class Approval:
        def __init__(self):
            self.calls = []
            self.waiting = asyncio.Event()
            self.release = None
            self.answer = "deny"

        async def request_approval(self, prompt, options, timeout, default):
            self.calls.append(prompt)
            self.waiting.set()
            if self.release:
                await self.release.wait()
            return self.answer

    approval = Approval()
    session = AmplifierSession(
        {
            "session": {
                "orchestrator": {
                    "module": "loop-streaming",
                    "config": {
                        "min_delay_between_calls_ms": 0,
                        "programmatic_dispatch": True,
                    },
                },
                "context": {"module": "context-simple"},
            },
            "providers": [],
        },
        approval_system=approval,
    )
    await session.initialize()
    coordinator = session.coordinator
    coordinator._test_approval = approval
    loop = coordinator.get("orchestrator")
    echo = Echo()
    await coordinator.mount("tools", echo, name="read")
    try:
        yield session, coordinator, loop, echo
    finally:
        await session.cleanup()


async def call_client(environment, client=None):
    _, coordinator, loop, _ = environment
    client = client or Client(coordinator)
    await coordinator.mount("tools", client, name="tool_exec")
    with loop.approved_dispatch.run_scope(coordinator):
        _, _, result = await loop._execute_tool_only(
            ToolCall(id="parent", name="tool_exec", arguments={}),
            coordinator.get("tools"),
            coordinator.hooks,
            "group",
            coordinator,
        )
    return json.loads(result)["output"], client


async def test_nested_uses_standard_hooks_once_and_keeps_identity(environment):
    session, coordinator, loop, echo = environment
    events = []

    async def capture(event, data):
        events.append((event, dict(data)))
        return HookResult()

    coordinator.hooks.register("tool:pre", capture)
    coordinator.hooks.register("tool:post", capture)
    result, client = await call_client(environment)
    assert result["status"] == "completed"
    assert result["success"] is True
    assert result["output"] == {"value": 42}
    assert echo.calls == 1
    nested = [(event, data) for event, data in events if data["tool_name"] == "read"]
    assert [event for event, _ in nested] == ["tool:pre", "tool:post"]
    source = result["source"]
    assert source["session_id"] == session.session_id
    assert source["parent_tool_call_id"] == "parent"
    assert source["request_id"] == "1"
    assert source["tool_call_id"] != "parent"
    assert all(data["dispatch_source"] == source for _, data in nested)
    assert loop._tool_calls_this_turn == 2
    with pytest.raises(RuntimeError):
        await client.lease.call("read", {})


async def test_hook_denial_never_executes_nested_tool(environment):
    _, coordinator, _, echo = environment

    async def deny(event, data):
        return (
            HookResult(action="deny", reason="Policy rejected")
            if data["tool_name"] == "read"
            else HookResult()
        )

    coordinator.hooks.register("tool:pre", deny)
    result, _ = await call_client(environment)
    assert result["status"] == "denied"
    assert result["success"] is False
    assert echo.calls == 0


async def test_coordinator_approval_rejection_is_preserved(environment):
    _, coordinator, _, echo = environment

    async def policy(event, data):
        if data["tool_name"] == "read":
            return HookResult(
                action="ask_user",
                approval_prompt="Allow nested read?",
                approval_options=["allow", "deny"],
                approval_default="deny",
            )
        return HookResult()

    coordinator.hooks.register("tool:pre", policy)
    result, _ = await call_client(environment)
    assert coordinator._test_approval.calls == ["Allow nested read?"]
    assert result["status"] == "denied"
    assert echo.calls == 0


async def test_post_hook_modified_output_and_unknown_outcome_are_preserved(environment):
    _, coordinator, _, _ = environment

    async def change(event, data):
        if data["tool_name"] == "read":
            return HookResult(
                action="modify", data={**data, "result": {"output": "redacted"}}
            )
        return HookResult()

    coordinator.hooks.register("tool:post", change)
    result, _ = await call_client(environment)
    assert result["status"] == "unknown"
    assert result["success"] is None
    assert result["output"] == {"output": "redacted"}


async def test_bind_rejects_idle_and_wrong_parent(environment):
    _, coordinator, loop, echo = environment
    dispatch = coordinator.get_capability("tools.dispatch")
    with pytest.raises(RuntimeError):
        dispatch.bind()
    with dispatch.run_scope(coordinator):
        with dispatch.call_scope(
            ToolCall(id="ordinary", name="read", arguments={}),
            {"read": echo},
            coordinator.hooks,
            None,
        ):
            with pytest.raises(RuntimeError):
                dispatch.bind()


async def test_lease_cannot_cross_run_session_or_parent_scope(environment):
    _, coordinator, loop, echo = environment
    dispatch = loop.approved_dispatch
    parent = ToolCall(id="parent", name="tool_exec", arguments={})
    with dispatch.run_scope(coordinator):
        with dispatch.call_scope(parent, {"read": echo}, coordinator.hooks, None):
            lease = dispatch.bind()
            with dispatch.run_scope(SimpleNamespace(session_id="other")):
                with pytest.raises(RuntimeError):
                    await lease.call("read", {})
            with dispatch.call_scope(
                ToolCall(id="other-parent", name="tool_exec", arguments={}),
                {"read": echo},
                coordinator.hooks,
                None,
            ):
                with pytest.raises(RuntimeError):
                    await lease.call("read", {})
            lease.close()
            with pytest.raises(RuntimeError):
                await lease.call("read", {})
    assert echo.calls == 0


async def test_nested_cancellation_emits_unknown_outcome_with_identity(environment):
    _, coordinator, _, echo = environment
    echo.release = asyncio.Event()
    seen = []

    async def capture(event, data):
        seen.append(data)
        return HookResult()

    coordinator.hooks.register("tool:dispatch_cancelled", capture)
    task = asyncio.create_task(call_client(environment))
    await asyncio.wait_for(echo.started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert echo.active == 0
    assert seen[0]["outcome"] == "unknown"
    assert seen[0]["effects"] == "not_rolled_back"
    assert seen[0]["dispatch_source"]["parent_tool_call_id"] == "parent"


async def test_pending_approval_holds_no_tool_execution_lock(environment):
    _, coordinator, _, echo = environment
    approval = coordinator._test_approval
    approval.release = asyncio.Event()
    approval.answer = "allow"

    async def policy(event, data):
        if data["tool_name"] == "read" and data["tool_input"].get("approval"):
            return HookResult(
                action="ask_user",
                approval_prompt="Allow slow read?",
                approval_options=["allow", "deny"],
                approval_default="deny",
            )
        return HookResult()

    coordinator.hooks.register("tool:pre", policy)

    async def program(lease):
        pending = asyncio.create_task(lease.call("read", {"approval": True}))
        await asyncio.wait_for(approval.waiting.wait(), 1)
        immediate = await asyncio.wait_for(
            lease.call("read", {"value": "unrelated"}), 1
        )
        assert immediate["success"] is True
        approval.release.set()
        await pending
        return immediate

    result, _ = await call_client(environment, Client(coordinator, program))
    assert result["success"] is True
    assert echo.calls == 2


async def test_same_tool_executes_serially_in_parallel_nested_calls(environment):
    _, coordinator, _, echo = environment
    echo.release = asyncio.Event()

    async def program(lease):
        one = asyncio.create_task(lease.call("read", {}))
        await echo.started.wait()
        two = asyncio.create_task(lease.call("read", {}))
        await asyncio.sleep(0)
        assert echo.calls == 1
        echo.release.set()
        return await asyncio.gather(one, two)

    result, _ = await call_client(environment, Client(coordinator, program))
    assert all(row["success"] for row in result)
    assert echo.peak == 1


async def test_ordinary_tool_is_invoked_exactly_once(environment):
    _, coordinator, loop, echo = environment
    with loop.approved_dispatch.run_scope(coordinator):
        result = await loop._execute_tool_only(
            ToolCall(id="ordinary", name="read", arguments={}),
            coordinator.get("tools"),
            coordinator.hooks,
            None,
            coordinator,
        )
    assert json.loads(result[2])["output"] == {"value": 42}
    assert echo.calls == 1


async def test_real_finite_session_enters_and_expires_lease_scope(environment):
    session, coordinator, _, echo = environment
    client = Client(coordinator)
    await coordinator.mount("tools", client, name="tool_exec")

    class Provider:
        name = "fixture"
        config = {}
        count = 0

        def get_info(self):
            return SimpleNamespace(default_model="fixture")

        async def complete(self, request, **kwargs):
            self.count += 1
            if self.count == 1:
                return ChatResponse(
                    content=[],
                    tool_calls=[
                        ToolCall(id="actual-parent", name="tool_exec", arguments={})
                    ],
                )
            assert "actual-parent" in str(request.messages)
            return ChatResponse(content=[TextBlock(text="done")])

        def parse_tool_calls(self, response):
            return response.tool_calls or []

    await coordinator.mount("providers", Provider(), name="fixture")
    assert await session.execute("Use the programmatic tool") == "done"
    assert echo.calls == 1
    with pytest.raises(RuntimeError):
        await client.lease.call("read", {})


async def test_revoked_lease_cannot_execute_after_delayed_approval(environment):
    _, coordinator, loop, echo = environment
    approval = coordinator._test_approval
    approval.release = asyncio.Event()
    approval.answer = "allow"

    async def policy(event, data):
        return HookResult(
            action="ask_user",
            approval_prompt="Delayed approval",
            approval_options=["allow", "deny"],
        )

    coordinator.hooks.register("tool:pre", policy)
    dispatch = loop.approved_dispatch
    with dispatch.run_scope(coordinator):
        with dispatch.call_scope(
            ToolCall(id="parent", name="tool_exec", arguments={}),
            {"read": echo},
            coordinator.hooks,
            None,
        ):
            lease = dispatch.bind()
            task = asyncio.create_task(lease.call("read", {}))
            await asyncio.wait_for(approval.waiting.wait(), 1)
            lease.close()
            approval.release.set()
            with pytest.raises(RuntimeError):
                await task
    assert echo.calls == 0


async def test_disabled_dispatch_preserves_concurrent_ordinary_calls(environment):
    _, coordinator, loop, echo = environment
    loop.config.pop("programmatic_dispatch")
    echo.release = asyncio.Event()
    with loop.approved_dispatch.run_scope(coordinator):
        with pytest.raises(RuntimeError):
            loop.approved_dispatch.bind()
        tasks = [
            asyncio.create_task(
                loop._execute_tool_only(
                    ToolCall(id=f"ordinary-{i}", name="read", arguments={}),
                    coordinator.get("tools"),
                    coordinator.hooks,
                    "ordinary",
                    coordinator,
                )
            )
            for i in range(2)
        ]
        for _ in range(20):
            if echo.calls == 2:
                break
            await asyncio.sleep(0.01)
        assert echo.calls == 2
        assert echo.peak == 2
        echo.release.set()
        await asyncio.gather(*tasks)


async def test_lease_wait_observes_real_coordinator_immediate_cancellation(environment):
    _, coordinator, loop, echo = environment
    dispatch = loop.approved_dispatch
    with dispatch.run_scope(coordinator):
        with dispatch.call_scope(
            ToolCall(id="parent", name="tool_exec", arguments={}),
            {"read": echo},
            coordinator.hooks,
            None,
        ):
            lease = dispatch.bind()
            waiting = asyncio.create_task(lease.wait_cancelled())
            await coordinator.request_cancel(immediate=True)
            assert await asyncio.wait_for(waiting, 1) == "host_cancel_requested"
            with pytest.raises(RuntimeError):
                await lease.call("read", {})
    assert echo.calls == 0


async def test_delegation_aliases_are_excluded_and_keep_ordinary_concurrency(
    environment,
):
    _, coordinator, loop, echo = environment
    echo.name = "delegate"
    tools = {"alias": echo}
    dispatch = loop.approved_dispatch
    with dispatch.run_scope(coordinator):
        with dispatch.call_scope(
            ToolCall(id="parent", name="tool_exec", arguments={}),
            tools,
            coordinator.hooks,
            None,
        ):
            lease = dispatch.bind()
            assert lease.tools == ()
            with pytest.raises(ValueError):
                await lease.call("alias", {})
    assert dispatch.wrap_tools(tools)["alias"] is echo


async def test_portable_registration_requires_no_orchestrator_configuration(
    environment,
):
    _, coordinator, loop, echo = environment
    loop.config.pop("programmatic_dispatch")
    coordinator.register_capability(
        "tools.dispatch.registrations", {"ui_use": {"tools": ["read"]}}
    )
    client = Client(coordinator)
    client.name = "ui_use"
    await coordinator.mount("tools", client, name="ui_use")
    with loop.approved_dispatch.run_scope(coordinator):
        _, _, raw = await loop._execute_tool_only(
            ToolCall(id="ui", name="ui_use", arguments={}),
            coordinator.get("tools"),
            coordinator.hooks,
            "ui",
            coordinator,
        )
    assert json.loads(raw)["output"]["status"] == "completed" and echo.calls == 1
    assert client.lease.tools == ("read",)
    loop.config["programmatic_dispatch"] = False
    client.lease = None
    with loop.approved_dispatch.run_scope(coordinator):
        await loop._execute_tool_only(
            ToolCall(id="off", name="ui_use", arguments={}),
            coordinator.get("tools"),
            coordinator.hooks,
            "off",
            coordinator,
        )
    assert client.lease is None and echo.calls == 1


async def test_nested_inference_reuses_selection_counts_and_enforces_policy(
    environment,
):
    from amplifier_core.message_models import ChatRequest, Message

    _, coordinator, loop, _ = environment
    seen, events = [], []

    class Provider:
        name = "ordinary"

        def get_info(self):
            return SimpleNamespace(defaults={"model": "selected-model"})

        def request_budget(self, request, *, context_estimate, request_options):
            assert context_estimate > 0 and request_options["model"] == "selected-model"
            return {"estimated_input_tokens": 10, "input_limit_tokens": 100}

        async def complete(self, request, **kwargs):
            seen.append((request, kwargs))
            return ChatResponse(content=[TextBlock(text="chosen")])

    await coordinator.mount("providers", Provider(), name="ordinary")
    request = ChatRequest(
        messages=[Message(role="user", content="Pick a candidate")], model="unselected"
    )

    async def capture(event, data):
        events.append((event, data))
        return HookResult()

    coordinator.hooks.register("provider:request", capture)
    coordinator.hooks.register("model:complete", capture)
    loop.config["programmatic_max_model_calls"] = 1

    async def program(lease):
        response = await coordinator.get_capability("model.complete")(request)
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await coordinator.get_capability("model.complete")(request)
        return {"text": response.content[0].text}

    result, _ = await call_client(environment, Client(coordinator, program))
    assert isinstance(result, dict), result
    assert result["text"] == "chosen" and loop._llm_calls_this_turn == 1
    assert (
        seen[0][0].model == "selected-model" and seen[0][0].metadata["stream"] is False
    )
    assert request.model == "unselected" and request.metadata is None
    assert [event for event, _ in events] == ["provider:request", "model:complete"]
    assert all(data["parent_tool_call_id"] == "parent" for _, data in events)
    with pytest.raises(RuntimeError, match="authorized"):
        await coordinator.get_capability("model.complete")(request)


async def test_nested_inference_denial_does_not_call_provider(environment):
    from amplifier_core.message_models import ChatRequest, Message

    _, coordinator, loop, _ = environment

    class Provider:
        name = "ordinary"

        def get_info(self):
            return SimpleNamespace(defaults={"model": "selected"})

        async def complete(self, request, **kwargs):
            raise AssertionError("Policy denied inference")

    await coordinator.mount("providers", Provider(), name="ordinary")

    async def deny(event, data):
        return HookResult(action="deny", reason="test policy")

    coordinator.hooks.register("provider:request", deny)

    async def program(lease):
        with pytest.raises(RuntimeError, match="denied"):
            await coordinator.get_capability("model.complete")(
                ChatRequest(messages=[Message(role="user", content="private")])
            )
        return {"denied": True}

    result, _ = await call_client(environment, Client(coordinator, program))
    assert result["denied"] and loop._llm_calls_this_turn == 0


async def test_nested_dispatch_restores_parent_task_identity(environment):
    _, coordinator, _, _ = environment

    async def program(lease):
        before = dict(coordinator._tool_dispatch_contexts[asyncio.current_task()])
        result = await lease.call("read", {})
        assert coordinator._tool_dispatch_contexts[asyncio.current_task()] == before
        return result

    result, _ = await call_client(environment, Client(coordinator, program))
    assert result["status"] == "completed"


async def test_nested_inference_observes_host_cancel_and_cancels_provider(environment):
    from amplifier_core.message_models import ChatRequest, Message

    _, coordinator, _, _ = environment
    started, cancelled = asyncio.Event(), asyncio.Event()

    class Provider:
        name = "ordinary"

        def get_info(self):
            return SimpleNamespace(defaults={"model": "selected"})

        async def complete(self, request, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    await coordinator.mount("providers", Provider(), name="ordinary")

    async def program(lease):
        return await coordinator.get_capability("model.complete")(
            ChatRequest(messages=[Message(role="user", content="private")])
        )

    pending = asyncio.create_task(
        call_client(environment, Client(coordinator, program))
    )
    await asyncio.wait_for(started.wait(), 1)
    await coordinator.request_cancel(immediate=True)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 1)
    assert cancelled.is_set()


async def test_host_validity_callback_revokes_without_live_dependency(environment):
    _, coordinator, loop, echo = environment
    async def program(lease):
        loop.approved_dispatch.validate_context = lambda actual: False
        with pytest.raises(RuntimeError, match='no longer valid'):
            await lease.call('read', {})
        return {'revoked': True}
    result, _ = await call_client(environment, Client(coordinator, program))
    assert result['revoked'] and echo.calls == 0
