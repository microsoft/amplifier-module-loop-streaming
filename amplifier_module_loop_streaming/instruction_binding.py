"""Optional structural binding to ``context.instructions.v1``.

This module deliberately knows only the context capability's public structural
surface.  It neither imports context-simple nor keeps state beyond one
``StreamingOrchestrator.execute()`` invocation.
"""

from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_CAPABILITY = "context.instructions.v1"
_EXECUTION_INPUT_CAPABILITY = "execution.input.v1"
_REQUIRED_ASSEMBLY_METHODS = (
    "input_scope",
    "turn",
    "request",
    "accept_response",
    "register",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionInput:
    """One host-provided input boundary for the imminent outer execution."""

    input_id: str
    origin: str

    @classmethod
    def consume(cls, coordinator: Any) -> ExecutionInput | None:
        """Read and clear the optional host input binding exactly once.

        The application owns this capability and must re-provide it before
        every outer ``execute()``.  Clearing through the same public
        capability API prevents a later execution from inheriting a stale
        origin when the first execution errors or is cancelled.
        """
        get_capability = getattr(coordinator, "get_capability", None)
        register_capability = getattr(coordinator, "register_capability", None)
        if not callable(get_capability) or not callable(register_capability):
            return None
        supplied = get_capability(_EXECUTION_INPUT_CAPABILITY)
        if supplied is None:
            return None
        try:
            binding = copy.deepcopy(supplied)
            if (
                type(binding) is not dict
                or set(binding) != {"version", "input_id", "origin"}
                or type(binding["version"]) is not int
                or binding["version"] != 1
                or not isinstance(binding["input_id"], str)
                or not binding["input_id"]
                or binding["origin"] not in {"human", "delegation"}
            ):
                raise RuntimeError(
                    "execution.input.v1 must be exactly "
                    "{'version': 1, 'input_id': <non-empty string>, "
                    "'origin': 'human' | 'delegation'}"
                )
            return cls(input_id=binding["input_id"], origin=binding["origin"])
        finally:
            register_capability(_EXECUTION_INPUT_CAPABILITY, None)

    def anchor(self) -> dict[str, str]:
        """Return the exact input identity shared by turn and request scopes."""
        return {
            "input_id": self.input_id,
            "message_id": self.input_id,
            "origin": self.origin,
        }


@dataclass
class InstructionRequest:
    """One active context request scope, closed after response admission."""

    assembly: Any
    manager: Any
    request_id: str
    on_response_accepted: Callable[[], None] | None = None
    on_close: Callable[[str], None] | None = None
    closed: bool = False

    async def accept_response(self, response_message: dict[str, Any]) -> None:
        """Admit the canonical assistant response, then close this request.

        Context records a post-ingress observer failure as ``stored`` and
        explicitly permits retrying the *same* response.  Retry that bounded,
        idempotent operation once before surfacing a persistent failure; no
        provider call or second history append is involved.
        """
        try:
            try:
                await self.assembly.accept_response(self.request_id, response_message)
            except Exception:  # noqa: BLE001 - public capability has no shared exception type.
                await self.assembly.accept_response(self.request_id, response_message)
            if self.on_response_accepted is not None:
                self.on_response_accepted()
        finally:
            await self.close()

    async def abandon(self) -> None:
        """Close an unaccepted request so context marks prepared work abandoned."""
        await self.close()

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                await self.manager.__aexit__(None, None, None)
            finally:
                if self.on_close is not None:
                    self.on_close(self.request_id)


@dataclass
class InstructionBinding:
    """Per-outer-execute IDs and structural state for v1 request scopes."""

    assembly: Any
    selected_provider: Any
    execution_input: ExecutionInput
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_anchor: dict[str, str] | None = None
    completed_batches: list[dict[str, Any]] = field(default_factory=list)
    tail_anchor: dict[str, Any] | None = None
    staging_lease: Any | None = None
    staged_requests: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pending_staged: list[dict[str, Any]] = field(default_factory=list)
    _staging_index: int = 0
    _assistant_role_warning_emitted: bool = False

    @classmethod
    def context_supported(cls, coordinator: Any, context: Any) -> Any | None:
        """Return the complete optional assembly capability, else ``None``."""
        get_capability = getattr(coordinator, "get_capability", None)
        if not callable(get_capability):
            return None
        assembly = get_capability(_CAPABILITY)
        if assembly is None:
            return None
        if not all(callable(getattr(assembly, method, None)) for method in _REQUIRED_ASSEMBLY_METHODS):
            return None
        if not callable(getattr(context, "add_message", None)) or not callable(
            getattr(context, "get_messages_for_request", None)
        ):
            return None
        return assembly

    @classmethod
    def for_execution(
        cls,
        coordinator: Any,
        context: Any,
        selected_provider: Any,
        execution_input: ExecutionInput,
    ) -> InstructionBinding | None:
        """Return a binding only when every optional v1 boundary is usable.

        Activation requires the authority extension on both the context
        assembly and selected provider. Old coordinators, contexts, and
        providers remain on their unchanged path. Required filter readiness is
        checked by the context while preparing its v1 view; failure there fails
        that request rather than silently lowering it to legacy.
        """
        assembly = cls.context_supported(coordinator, context)
        if assembly is None:
            return None
        if getattr(assembly, "instruction_layout_authority_v1", None) is not True:
            return None
        if (
            getattr(selected_provider, "instruction_layout_version", None) != 1
            or getattr(selected_provider, "instruction_layout_authority_v1", None) is not True
        ):
            return None
        binding = cls(
            assembly=assembly,
            selected_provider=selected_provider,
            execution_input=execution_input,
            input_anchor=execution_input.anchor(),
        )
        binding.staging_lease = assembly.register(
            f"loop-streaming:{binding.turn_id}",
            binding._snapshot_staged_instructions,
        )
        return binding

    def ensure_selected_provider(self, provider: Any) -> None:
        """Reject a provider switch that cannot lower the active v1 route."""
        if (
            getattr(provider, "instruction_layout_version", None) != 1
            or getattr(provider, "instruction_layout_authority_v1", None) is not True
        ):
            raise RuntimeError(
                "the selected provider changed after v1 activation but does not declare "
                "instruction_layout_version == 1 and instruction_layout_authority_v1 "
                "is True; refusing to silently lower v1 instructions to the legacy path"
            )
        self.selected_provider = provider

    @staticmethod
    async def has_marked_fixed_state(context: Any, assembly: Any | None = None) -> bool:
        """Whether retained or pending fixed v1 state is unsafe for legacy."""
        pending_state = getattr(assembly, "has_marked_fixed_state", None)
        if callable(pending_state) and pending_state():
            return True
        get_messages = getattr(context, "get_messages", None)
        if not callable(get_messages):
            return False
        messages = await get_messages()
        return any(
            isinstance(message, dict)
            and isinstance(message.get("metadata"), dict)
            and "amplifier:instruction" in message["metadata"]
            for message in messages
        )

    def new_message_id(self) -> str:
        return str(uuid.uuid4())

    async def add_execution_input(self, context: Any, content: str) -> None:
        """Attach the host-provided provenance to the outer input boundary."""
        with self.assembly.input_scope(
            self.execution_input.origin, self.execution_input.input_id
        ):
            await context.add_message({"role": "user", "content": content})

    async def add_human_input(self, context: Any, content: str) -> None:
        """Attach fresh provenance only to explicit human steering."""
        input_id = self.new_message_id()
        with self.assembly.input_scope("human", input_id):
            await context.add_message({"role": "user", "content": content})
        self.input_anchor = {
            "input_id": input_id,
            "message_id": input_id,
            "origin": "human",
        }

    def response_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Give every provider response a public stable message identity."""
        response = copy.deepcopy(message)
        metadata = dict(response.get("metadata") or {})
        metadata.setdefault("message_id", self.new_message_id())
        response["metadata"] = metadata
        return response

    def stage_legacy_injection(
        self,
        *,
        content: str,
        role: Any,
        request: InstructionRequest | None = None,
    ) -> None:
        """Stage one legacy ephemeral injection for exactly one v1 request.

        Canonical instruction records remain system-role.  The legacy carrier
        role is retained as explicit authority: system remains authoritative;
        user and assistant are advisory.  Assistant carriers were never a
        supported instruction surface, so retain them as advisory while making
        the migration visible once per execution.
        """
        if role == "system":
            authority = "authoritative"
        elif role == "user":
            authority = "advisory"
        elif role == "assistant":
            authority = "advisory"
            if not self._assistant_role_warning_emitted:
                self._assistant_role_warning_emitted = True
                logger.warning(
                    "Legacy ephemeral assistant-role context injection is staged as "
                    "an advisory v1 instruction; migrate the producer to an "
                    "explicit instruction source."
                )
        else:
            raise RuntimeError(
                "legacy context_injection_role must be 'system', 'user', or 'assistant'"
            )

        self._staging_index += 1
        entry = {
            "key": f"legacy-{self._staging_index}",
            "content": content,
            "placement": "tail",
            "authority": authority,
        }
        if request is None:
            self.pending_staged.append(entry)
        else:
            self.staged_requests.setdefault(request.request_id, []).append(entry)

    def stage_advisory_reminder(
        self, content: str, *, request: InstructionRequest | None = None
    ) -> None:
        """Stage one orchestrator-authored advisory tail reminder."""
        self.stage_legacy_injection(content=content, role="user", request=request)

    def _snapshot_staged_instructions(self, scope: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a detached request-local source snapshot."""
        request_id = scope.get("request_id")
        if not isinstance(request_id, str):
            return []
        staged = copy.deepcopy(self.staged_requests.get(request_id, []))
        tail_anchor = scope.get("tail_anchor")
        after_message_id = (
            tail_anchor.get("after_message_id") if isinstance(tail_anchor, dict) else None
        )
        if not isinstance(after_message_id, str) or not after_message_id:
            raise RuntimeError("staged legacy instruction has no validated tail anchor")
        for entry in staged:
            entry["after"] = {"after_message_id": after_message_id}
        return staged

    def _clear_staged_request(self, request_id: str) -> None:
        self.staged_requests.pop(request_id, None)

    def close(self) -> None:
        """Discard unsent request staging and unregister the local source."""
        self.pending_staged.clear()
        self.staged_requests.clear()
        if self.staging_lease is not None:
            self.staging_lease.close()
            self.staging_lease = None

    def tool_message(
        self, *, name: str, tool_call_id: str, content: str
    ) -> tuple[dict[str, Any], str]:
        """Build one causal tool result record with a public identity."""
        message_id = self.new_message_id()
        return (
            {
                "role": "tool",
                "name": name,
                "tool_call_id": tool_call_id,
                "content": content,
                "metadata": {"message_id": message_id},
            },
            message_id,
        )

    async def begin_request(self, provider: Any) -> InstructionRequest:
        """Open one request before its ordinary request hooks are emitted."""
        request_id = self.new_message_id()
        completed_batches = copy.deepcopy(self.completed_batches)
        tail_anchor = copy.deepcopy(self.tail_anchor)
        if tail_anchor is None and self.input_anchor is not None:
            tail_anchor = {"after_message_id": self.input_anchor["message_id"]}
        request_scope = {
            "turn_id": self.turn_id,
            "request_id": request_id,
            "llm_step_id": self.new_message_id(),
            "input_anchor": copy.deepcopy(self.input_anchor),
            "completed_batches": completed_batches,
            "tail_anchor": tail_anchor,
        }
        manager = self.assembly.request(request_scope, provider)
        await manager.__aenter__()
        self.staged_requests[request_id] = self.pending_staged
        self.pending_staged = []
        return InstructionRequest(
            self.assembly,
            manager,
            request_id,
            on_response_accepted=self.completed_batches.clear if completed_batches else None,
            on_close=self._clear_staged_request,
        )

    async def prepare_request(
        self, context: Any, request: InstructionRequest, provider: Any
    ) -> list[dict[str, Any]]:
        """Freeze the request view after ordinary request hooks complete."""
        try:
            messages = await context.get_messages_for_request(provider=provider)
        except BaseException:
            await request.abandon()
            raise
        return list(messages)

    def record_tool_batch(
        self,
        *,
        assistant_message_id: str,
        batch_id: str,
        tool_calls: list[Any],
        result_message_ids: list[str],
    ) -> None:
        """Make a completed batch available to only the next request."""
        call_ids = [call.id for call in tool_calls]
        batch = {
            "turn_id": self.turn_id,
            "batch_id": batch_id,
            "assistant_message_id": assistant_message_id,
            "call_ids": call_ids,
            "result_message_ids": list(result_message_ids),
        }
        self.completed_batches.append(batch)
        if result_message_ids:
            self.tail_anchor = {
                "turn_id": self.turn_id,
                "step_id": self.new_message_id(),
                "after_message_id": result_message_ids[-1],
                "batch_id": batch_id,
            }