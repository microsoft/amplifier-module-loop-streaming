"""Active-call dispatch leases for portable programmatic orchestration.

The ordinary streaming engine remains the execution and hook authority. This
adapter adds scope, nested attribution, and structured outcomes, never a second
unchecked tool execution path.
"""

import asyncio
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4

from amplifier_core.message_models import ToolCall
from .suspension import NestedExecutionHeld, ToolExecutionSuspended

_RUN = ContextVar("approved_dispatch_run", default=None)
_CALL = ContextVar("approved_dispatch_call", default=None)
FORBIDDEN = frozenset({"tool_exec", "delegate", "task", "live_job"})


@dataclass
class Run:
    owner: object
    coordinator: object
    identity: str
    active: bool = True
    model_calls: int = 0


@dataclass
class CallScope:
    run: Run
    call: object
    tools: dict
    hooks: object
    parallel_group_id: str | None
    active: bool = True


class GuardedTool:
    """Serialize actual execution, after the engine has processed approvals."""

    def __init__(self, tool, lock, observation=None, authority=None):
        self.tool, self.lock, self.observation = tool, lock, observation
        self.authority = authority

    def __getattr__(self, name):
        return getattr(self.tool, name)

    async def execute(self, arguments):
        async with self.lock:
            if self.authority:
                self.authority()
            if self.observation is not None:
                self.observation["started"] = True
            result = await self.tool.execute(arguments)
            if self.observation is not None:
                self.observation["raw_success"] = getattr(result, "success", None)
            return result


class AttributedHooks:
    def __init__(self, hooks, source, observation):
        self.hooks, self.source, self.observation = hooks, source, observation

    async def emit(self, event, data):
        enriched = {
            **data,
            "parent_tool_call_id": self.source["parent_tool_call_id"],
            "dispatch_source": dict(self.source),
        }
        result = await self.hooks.emit(event, enriched)
        if event == "tool:post":
            self.observation["post_seen"] = True
            returned = result.data.get("result") if result and result.data else None
            self.observation["post_result"] = (
                returned if returned is not None else data.get("result")
            )
        elif event == "tool:error":
            self.observation["engine_error"] = True
        return result

    def __getattr__(self, name):
        return getattr(self.hooks, name)


class AttributedCoordinator:
    def __init__(self, coordinator, observation):
        object.__setattr__(self, "coordinator", coordinator)
        object.__setattr__(self, "observation", observation)

    def __getattr__(self, name):
        return getattr(self.coordinator, name)

    def __setattr__(self, name, value):
        setattr(self.coordinator, name, value)

    async def process_hook_result(self, result, event, name):
        result = await self.coordinator.process_hook_result(result, event, name)
        if result.action == "deny":
            self.observation["denied"] = True
        return result


class DispatchLease:
    def __init__(self, dispatcher, scope):
        self.dispatcher, self.scope = dispatcher, scope
        self.closed = False
        allowlist = dispatcher.allowed_tools(scope.call.name)
        self.tools = tuple(
            name
            for name in scope.tools
            if name not in FORBIDDEN
            and name not in dispatcher.callers
            and getattr(scope.tools[name], "name", name) not in FORBIDDEN
            and getattr(scope.tools[name], "name", name) not in dispatcher.callers
            and (allowlist is None or name in allowlist)
        )
        self.context = {
            "session_id": str(scope.run.coordinator.session_id),
            "run_id": scope.run.identity,
            "parent_tool_call_id": scope.call.id,
        }

    def close(self):
        self.closed = True

    def _valid(self, expected_scope=None):
        run = _RUN.get()
        scope = _CALL.get()
        if (
            self.closed
            or not self.dispatcher.enabled
            or self.scope.call.name not in self.dispatcher.callers
            or not self.scope.active
            or not self.scope.run.active
            or run is not self.scope.run
            or scope is not (expected_scope or self.scope)
            or run.owner is not self.dispatcher
            or run.coordinator is not self.scope.run.coordinator
        ):
            raise RuntimeError(
                "Dispatch lease is not active in this session/run/parent call"
            )
        if run.coordinator.cancellation.is_immediate:
            raise RuntimeError("Immediate host cancellation is active")
        validate = self.dispatcher.validate_context
        if validate is not None and validate(run.coordinator) is not True:
            raise RuntimeError("Host activation is no longer valid")

    async def wait_cancelled(self):
        """Observe the real host token even across an async foreign-runtime bridge."""
        while True:
            if self.scope.run.coordinator.cancellation.is_immediate:
                return "host_cancel_requested"
            try:
                self._valid()
            except RuntimeError:
                return "lease_revoked"
            await asyncio.sleep(0.05)

    async def call(self, name, arguments, *, request_id=None):
        self._valid()
        allowed = self.dispatcher.allowed_tools(self.scope.call.name)
        if (
            name not in self.tools
            or name in FORBIDDEN
            or (allowed is not None and name not in allowed)
        ):
            raise ValueError("Tool is not available through this lease")
        if not isinstance(arguments, dict) or arguments.get("async") is True:
            raise ValueError("Background tool dispatch is unsupported")

        def still_authorized(child_scope):
            self._valid(child_scope)
            current = self.dispatcher.allowed_tools(self.scope.call.name)
            if current is not None and name not in current:
                raise RuntimeError("Nested tool authorization was revoked")

        call = ToolCall(id=str(uuid4()), name=name, arguments=arguments)
        source = {
            **self.context,
            "tool_call_id": call.id,
            "tool_name": name,
            "request_id": request_id,
        }
        observation = {}
        tools = self.dispatcher.wrap_tools(
            self.scope.tools, observed_name=name, observation=observation
        )
        hooks = AttributedHooks(self.scope.hooks, source, observation)
        coordinator = AttributedCoordinator(self.scope.run.coordinator, observation)
        # Always take the existing ordinary engine path. In particular, never
        # call a mounted tool directly when hooks or authority are unavailable.
        try:
            with self.dispatcher.call_scope(
                call, tools, hooks, self.scope.parallel_group_id
            ) as child_scope:
                tools = self.dispatcher.wrap_tools(
                    tools,
                    observed_name=name,
                    observation=observation,
                    authority=lambda: still_authorized(child_scope),
                )
                _, _, content = await self.dispatcher.owner._execute_tool_unscoped(
                    call,
                    tools,
                    hooks,
                    self.scope.parallel_group_id or self.scope.call.id,
                    coordinator,
                )
        except ToolExecutionSuspended as suspended:
            # The nested invocation is not the enclosing controller. Propagating
            # its "not_started" frame would replay a controller that may already
            # have performed effects. Controllers have no portable stack here.
            try:
                await self.scope.hooks.emit("tool:dispatch_suspended", {
                    "source": source, "outcome": "held", "effects": "not_rolled_back",
                })
            except Exception as error:
                logging.getLogger(__name__).warning(
                    "Nested suspension observer failed: %s", type(error).__name__)
            raise NestedExecutionHeld(
                source, "Nested tool suspended; the enclosing controller already started. "
                "Inspect retained effects; do not replay the controller."
            ) from suspended
        except asyncio.CancelledError:
            await self.scope.hooks.emit(
                "tool:dispatch_cancelled",
                {
                    "tool_name": name,
                    "tool_call_id": call.id,
                    "parent_tool_call_id": self.scope.call.id,
                    "dispatch_source": source,
                    "outcome": "unknown",
                    "effects": "not_rolled_back",
                },
            )
            raise
        self._valid()
        if observation.get("denied"):
            status, success, output = (
                "denied",
                False,
                "Tool execution denied by host policy",
            )
        else:
            post = observation.get("post_result")
            confirmed = post.get("success") if isinstance(post, dict) else None
            if observation.get("engine_error"):
                status, success = "failed", False
            elif observation.get("post_seen") and confirmed is True:
                status, success = "completed", True
            elif observation.get("post_seen") and confirmed is False:
                status, success = "failed", False
            else:
                status, success = "unknown", None
            try:
                output = json.loads(content)
                # The core hook transport may copy result dictionaries, making
                # the ordinary engine serialize the full post-hook ToolResult.
                # Unwrap only when it is exactly the captured confirmed envelope.
                if (
                    isinstance(post, dict)
                    and isinstance(post.get("success"), bool)
                    and output == post
                ):
                    output = post.get("output")
            except (ValueError, TypeError):
                output = content
        return {
            "status": status,
            "success": success,
            "output": output,
            "source": source,
        }


class ApprovedDispatch:
    version = 1

    def __init__(self, owner):
        self.owner = owner
        self.locks = {}
        self.validate_context = None

    @property
    def enabled(self):
        explicit = self.owner.config.get("programmatic_dispatch")
        if explicit is not None:
            return explicit is True
        return bool(self.registrations)

    @property
    def registrations(self):
        run = _RUN.get()
        if not run or run.owner is not self or not run.coordinator:
            return {}
        if not callable(getattr(run.coordinator, "get_capability", None)):
            return {}
        rows = run.coordinator.get_capability("tools.dispatch.registrations")
        if not isinstance(rows, dict):
            return {}
        return {
            name: row
            for name, row in rows.items()
            if isinstance(name, str)
            and name not in FORBIDDEN
            and isinstance(row, dict)
            and isinstance(row.get("tools"), list)
            and all(isinstance(tool, str) for tool in row["tools"])
        }

    @property
    def callers(self):
        """Admit declared controllers, or the explicitly enabled legacy caller."""
        names = self.owner.config.get("programmatic_callers")
        if names is None:
            names = list(self.registrations)
            if self.owner.config.get("programmatic_dispatch") is True:
                names.append("tool_exec")
        if not isinstance(names, list) or any(
            not isinstance(name, str) for name in names
        ):
            return frozenset()
        return frozenset(
            name for name in names if name == "tool_exec" or name not in FORBIDDEN
        )

    def allowed_tools(self, caller):
        host = self.owner.config.get("programmatic_allowed_tools")
        declared = self.registrations.get(caller, {}).get("tools")
        if declared is None:
            return host
        return set(declared) if host is None else set(declared).intersection(host)

    def reset_model_budget(self):
        run = _RUN.get()
        if run and run.owner is self:
            run.model_calls = 0

    def _budget_run(self):
        run = _RUN.get()
        if run is None or run.owner is not self or not run.active:
            raise RuntimeError("Nested inference budget requires an active dispatch run")
        return run

    def model_budget_used(self):
        return self._budget_run().model_calls

    def restore_model_budget(self, used):
        if type(used) is not int or used < 0:
            raise ValueError("Invalid nested inference continuation budget")
        self._budget_run().model_calls = used

    @contextmanager
    def run_scope(self, coordinator):
        existing = _RUN.get()
        if (
            existing
            and existing.owner is self
            and existing.coordinator is coordinator
            and existing.active
        ):
            # loop-live surrounds its finite streaming turns with a longer run.
            yield
            return
        run = Run(self, coordinator, str(uuid4()))
        token = _RUN.set(run)
        try:
            yield
        finally:
            run.active = False
            _RUN.reset(token)

    @contextmanager
    def call_scope(self, call, tools, hooks, parallel_group_id):
        run = _RUN.get()
        scope = (
            CallScope(run, call, tools, hooks, parallel_group_id)
            if run and run.owner is self
            else None
        )
        token = _CALL.set(scope)
        try:
            yield scope
        finally:
            if scope:
                scope.active = False
            _CALL.reset(token)

    def bind(self):
        scope = _CALL.get()
        if (
            not self.enabled
            or scope is None
            or scope.call.name not in self.callers
            or not scope.active
            or not scope.run.active
            or scope.run.owner is not self
            or scope.run.coordinator is None
        ):
            raise RuntimeError("No active authorized programmatic caller")
        lease = DispatchLease(self, scope)
        lease._valid()
        return lease

    async def complete(self, request):
        """A bounded, attributed ordinary inference call for an active caller.

        This is a stateless utility request, not another foreground turn: it
        never mutates conversation history or its context occupancy meter.
        Provider-owned usage/cost hooks remain active, with streaming disabled.
        """
        from amplifier_core.message_models import ChatRequest

        lease = self.bind()
        scope = lease.scope
        coordinator = scope.run.coordinator
        try:
            if not isinstance(request, ChatRequest) or request.tools:
                raise ValueError("Nested inference requires a tool-free ChatRequest")
            model_limit = self.owner.config.get("programmatic_max_model_calls", 24)
            if type(model_limit) is not int or model_limit < 0:
                raise ValueError("Invalid nested inference budget")
            # Serialize admission so concurrent callers cannot overspend this turn.
            lock = self.locks.setdefault("model.complete", asyncio.Lock())
            async with lock:
                lease._valid()
                if scope.run.model_calls >= model_limit:
                    raise RuntimeError("Nested inference budget exhausted")
                provider = self.owner._select_provider(coordinator.get("providers"))
                if provider is None:
                    raise RuntimeError("No ordinary provider is configured")
                model = self.owner._provider_default_model(provider)
                # Reuse the current provider/account/model. The consumer cannot
                # secretly override model routing or turn on provider streaming.
                request = request.model_copy(
                    update={
                        "model": model,
                        "metadata": {**(request.metadata or {}), "stream": False},
                        "max_output_tokens": min(
                            request.max_output_tokens or 2048, 4096
                        ),
                    },
                    deep=True,
                )
                source = {
                    **lease.context,
                    "caller": scope.call.name,
                    "purpose": "nested_inference",
                }
                result = await scope.hooks.emit(
                    "provider:request",
                    {
                        "provider": getattr(provider, "name", "unknown"),
                        "model": model,
                        "iteration": 0,
                        **source,
                    },
                )
                result = await coordinator.process_hook_result(
                    result, "provider:request", scope.call.name
                )
                if result.action == "deny":
                    raise RuntimeError("Nested inference denied by host policy")
                lease._valid()
                options = {"extended_thinking": False}
                if model:
                    options["model"] = model
                budget = getattr(provider, "request_budget", None)
                if callable(budget):
                    import inspect

                    from ._text_estimate import estimate_messages

                    kwargs = {
                        "context_estimate": estimate_messages(
                            [
                                message.model_dump(exclude_none=True)
                                for message in request.messages
                            ]
                        ).tokens
                    }
                    if "request_options" in inspect.signature(budget).parameters:
                        kwargs["request_options"] = options
                    measured = budget(request, **kwargs)
                    if inspect.isawaitable(measured):
                        measured = await measured
                    if isinstance(measured, dict):
                        count = measured.get("estimated_input_tokens")
                        limit = measured.get("input_limit_tokens")
                        if type(count) is int and type(limit) is int and count > limit:
                            raise RuntimeError(
                                "Nested inference exceeds provider input budget"
                            )
                lease._valid()
                suspension = coordinator.get_capability("execution.suspension")
                if suspension is not None and suspension.requested():
                    raise NestedExecutionHeld(source, "Nested inference is held for execution suspension")
                scope.run.model_calls += 1
                self.owner._llm_calls_this_turn += 1
                pending = asyncio.create_task(provider.complete(request, **options))
                try:
                    while not pending.done():
                        await asyncio.wait({pending}, timeout=0.05)
                        if coordinator.cancellation.is_cancelled:
                            raise asyncio.CancelledError
                        lease._valid()
                    response = await pending
                finally:
                    if not pending.done():
                        pending.cancel()
                        await asyncio.gather(pending, return_exceptions=True)
                lease._valid()
                usage = getattr(response, "usage", None)
                await scope.hooks.emit(
                    "model:complete",
                    {
                        **source,
                        "provider": getattr(provider, "name", "unknown"),
                        "model": model,
                        "usage": usage.model_dump()
                        if hasattr(usage, "model_dump")
                        else usage,
                        "nested_model_calls": scope.run.model_calls,
                    },
                )
                return response
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await scope.hooks.emit(
                "provider:error",
                {
                    **lease.context,
                    "purpose": "nested_inference",
                    "error": {
                        "type": type(exc).__name__,
                        "msg": "Nested inference failed",
                    },
                },
            )
            raise
        finally:
            lease.close()

    def wrap_tools(self, tools, observed_name=None, observation=None, authority=None):
        wrapped = {}
        for name, tool in tools.items():
            raw = tool.tool if isinstance(tool, GuardedTool) else tool
            allowlist = self.owner.config.get("programmatic_allowed_tools")
            if name != "tool_exec" and (
                name in FORBIDDEN
                or getattr(raw, "name", name) in FORBIDDEN
                or (allowlist is not None and name not in allowlist)
            ):
                wrapped[name] = raw
                continue
            lock = self.locks.setdefault(id(raw), asyncio.Lock())
            wrapped[name] = GuardedTool(
                raw,
                lock,
                observation if name == observed_name else None,
                authority if name == observed_name else None,
            )
        return wrapped
