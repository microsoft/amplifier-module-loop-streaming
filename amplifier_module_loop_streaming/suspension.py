"""Public, versioned continuation at settled execution boundaries.

The host owns durable storage and admission. The engine owns this small program
counter. A continuation never contains a Python frame, provider connection, or
an already-running tool invocation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


class ExecutionSuspended(BaseException):
    """Execution has a host-acknowledged checkpoint, not a completed result."""

    def __init__(self, continuation: dict[str, Any]):
        super().__init__("Execution suspended at a durable boundary")
        self.continuation = continuation


class ToolExecutionSuspended(BaseException):
    """A cooperating tool stopped with an explicit public continuation.

    Only the tool's ``resume_execution`` method can continue this operation.
    Calling ``execute`` again is never a valid fallback.
    """

    def __init__(self, continuation: dict[str, Any]):
        super().__init__("Tool execution suspended")
        self.continuation = canonical_copy(continuation)


class ResumingTool:
    execution_resumption = True

    def __init__(self, tool, continuation):
        self.tool, self.continuation = tool, continuation
        if not callable(getattr(tool, "resume_execution", None)):
            raise ValueError("Tool does not support explicit execution resumption")

    def __getattr__(self, name):
        return getattr(self.tool, name)

    async def execute(self, arguments):
        return await self.tool.resume_execution(self.continuation)


def canonical_copy(value: Any) -> Any:
    """Fail rather than silently discard nonportable continuation state."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate(continuation: dict[str, Any], messages: list, config: dict) -> dict:
    frame = canonical_copy(continuation)
    if frame.get("version") != 1 or frame.get("engine") != "loop-streaming":
        raise ValueError("Unsupported execution continuation")
    if frame.get("phase") not in {"provider", "prepared_request", "tools", "settling_tools", "goal", "prepared_finalization"}:
        raise ValueError("Unsupported execution continuation phase")
    if frame.get("history") != fingerprint(messages):
        raise ValueError("Execution continuation history changed; inspect before resuming")
    if frame.get("configuration") != fingerprint(config):
        raise ValueError("Execution configuration changed; inspect before resuming")
    if not isinstance(frame.get("iteration"), int) or frame["iteration"] < 0:
        raise ValueError("Invalid execution continuation iteration")
    if frame["phase"] in {"tools", "settling_tools"} and not frame.get("tool_calls"):
        raise ValueError("Tool continuation has no pending calls")
    if frame["phase"] == "settling_tools" and (
        len(frame.get("settled_tools", [])) != len(frame["tool_calls"])
    ):
        raise ValueError("Tool continuation outcomes do not match calls")
    if frame["phase"] in {"prepared_request", "prepared_finalization"} and not isinstance(frame.get("prepared_request"), dict):
        raise ValueError("Prepared continuation has no request")
    if frame["phase"] == "goal" and (not isinstance(frame.get("goal_loop"), dict) or
            frame["goal_loop"].get("phase") not in {"evaluate", "evaluated", "judge", "judged", "finish", "next_turn"}):
        raise ValueError("Unknown goal continuation phase")
    return frame
