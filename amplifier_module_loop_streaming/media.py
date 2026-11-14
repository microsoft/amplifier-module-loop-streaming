"""Refs-only tool settlement. Storage and request hydration belong to the host.

Only canonical ToolResult content and the exact reserved output member are
interpreted. In particular, JSON-looking text is never a media carrier.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from amplifier_core import HookResult, ToolResult

NAMESPACE = "experimental.media.v1"
UNAVAILABLE = (
    "[Image unavailable to this session: compatible media handling is not "
    "installed; no pixels were sent.]"
)


@dataclass
class SettledToolResult:
    tool_call_id: str
    tool_name: str
    content: str
    metadata: dict = field(default_factory=dict)
    result: Any = None
    success: bool | None = None

    def __post_init__(self):
        if (not isinstance(self.tool_call_id, str) or not isinstance(self.tool_name, str)
                or not isinstance(self.content, str) or not isinstance(self.metadata, dict)
                or (self.success is not None and type(self.success) is not bool)):
            raise ValueError("Invalid settled tool result")

    def model_dump(self, **kwargs) -> dict:
        # Deliberately fail on nonportable data rather than stringifying pixels,
        # handles, prepared requests or arbitrary runtime objects.
        return json.loads(json.dumps({"version": 1, **self.__dict__}, allow_nan=False))

    @classmethod
    def from_legacy(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            value = deepcopy(dict(value))
            version = value.pop("version", 1)
            if type(version) is not int or version != 1:
                raise ValueError("Unsupported settled tool result version")
            return cls(**value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return cls(*value)
        raise ValueError("Unsupported settled tool result")

    def message(self) -> dict:
        value = {
            "role": "tool", "name": self.tool_name,
            "tool_call_id": self.tool_call_id, "content": self.content,
        }
        if self.metadata:
            value["metadata"] = deepcopy(self.metadata)
        return value

    # Read compatibility for existing third-party text consumers. Internal
    # persistence MUST use model_dump, never list(result).
    def __iter__(self):
        return iter((self.tool_call_id, self.tool_name, self.content))

    def __getitem__(self, index):
        return (self.tool_call_id, self.tool_name, self.content)[index]


def capability(coordinator):
    getter = getattr(coordinator, "get_capability", None)
    cap = getter(NAMESPACE) if callable(getter) else None
    if type(getattr(cap, "version", None)) is not int or cap.version != 1:
        return None
    return cap if all(callable(getattr(cap, name, None)) for name in (
        "ingest", "publish", "validate", "resolve", "prepare_request", "report"
    )) else None


def tool_post_boundary(coordinator):
    getter = getattr(coordinator, "get_capability", None)
    boundary = getter("session.media.tool_post") if callable(getter) else None
    return boundary if callable(boundary) else None


@dataclass
class ToolPostResult:
    hook_result: Any
    result: Any
    complete: bool


def validate_prepared_media(request, coordinator):
    if capability(coordinator) is None and any(
        NAMESPACE in (message.metadata or {}) for message in request.messages
    ):
        # This candidate was already fitted before suspension. Rewriting it
        # now would invalidate its count; only an authorized refit can replace
        # it. A local refusal is safe and never replays settled tools.
        raise RuntimeError("Prepared media request lost compatible handling; refit required")


def origin_for(coordinator, call):
    session_id = getattr(coordinator, "session_id", None)
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Media requires the actual session identity")
    return {"kind": "tool", "session_id": session_id, "id": call.id}


def text_only_messages(messages, *, reason=UNAVAILABLE, images=False):
    """Request-local unavailable projection at the original message position.

    Used on paths with no installed media handler, or by explicitly text-only
    utilities. Canonical context is not changed and no locator is resolved.
    """
    projected = []
    for message in messages:
        metadata = message.get("metadata")
        has_refs = isinstance(metadata, Mapping) and NAMESPACE in metadata
        content = message.get("content")
        has_images = images and isinstance(content, list) and any(
            isinstance(block, Mapping) and block.get("type") in {"image", "image_url", "input_image"}
            for block in content
        )
        if not has_refs and not has_images:
            projected.append(message)
            continue
        message = deepcopy(message)
        if has_refs:
            message["metadata"] = {k: v for k, v in metadata.items() if k != NAMESPACE}
        if isinstance(content, list):
            message["content"] = [
                {"type": "text", "text": reason}
                if images and isinstance(block, Mapping) and block.get("type") in {"image", "image_url", "input_image"}
                else block for block in content
            ]
            if has_refs:
                message["content"].append({"type": "text", "text": reason})
        else:
            message["content"] = (content or "") + "\n" + reason
        projected.append(message)
    return projected


def text_only_request(request, coordinator, *, reason):
    """Validate known refs, then freeze a utility's no-pixels candidate."""
    from amplifier_core.message_models import Message
    cap = capability(coordinator)
    if cap is not None:
        for message in request.messages:
            metadata = message.metadata or {}
            if NAMESPACE not in metadata:
                continue
            media = metadata[NAMESPACE]
            if (not isinstance(media, dict) or set(media) != {"refs"}
                    or not isinstance(media["refs"], list)):
                raise ValueError("Invalid utility media envelope")
            cap.validate(media["refs"], purpose="reference")
    request = request.model_copy(deep=True)
    request.messages = [Message(**message) for message in text_only_messages(
        [message.model_dump(exclude_none=True) for message in request.messages],
        reason=reason, images=True,
    )]
    return request


async def _await(value):
    return await value if inspect.isawaitable(value) else value


async def checkpoint_settled_results(coordinator, *, has_media=False):
    """Await the existing host-owned, zero-argument settled-append checkpoint.

    This is not a media commit receipt. A media-enabled host must verify its
    exact canonical prefix/source acknowledgment before returning and before
    its provider adapter can prepare these refs. No report grants acceptance.
    """
    checkpoint = coordinator.get_capability("session.durable_checkpoint") if coordinator else None
    if checkpoint is None:
        if has_media:
            raise RuntimeError("Media tool results require a durable session checkpoint")
        return
    try:
        if not callable(checkpoint):
            raise TypeError("session.durable_checkpoint is not callable")
        result = await _await(checkpoint())
        if result is False:
            raise RuntimeError("session.durable_checkpoint returned False")
    except Exception as exc:
        raise RuntimeError(
            "Durable session checkpoint failed; refusing further provider dispatch"
        ) from exc


async def report_consumer_ready(coordinator):
    cap = capability(coordinator)
    if cap is not None and tool_post_boundary(coordinator) is not None:
        await _await(cap.report(
            "loop-streaming",
            event="consumer-ready", outcome="ready", details={"contract": 1, "role": "loop"},
        ))


def effective_result(hook_result, current):
    """Presence, not identity or truthiness, decides the effective result."""
    action = getattr(hook_result, "action", None)
    if action == "deny":
        return None
    data = getattr(hook_result, "data", None)
    if isinstance(data, Mapping):
        # Core returns final chained data with action=continue even when an
        # earlier modifying handler removed result. Missing is authoritative.
        return data.get("result")
    return None if action == "modify" else current


def _refs(value):
    if not isinstance(value, Mapping):
        return []
    output = value.get("output")
    if not isinstance(output, Mapping) or NAMESPACE not in output:
        return []
    envelope = output[NAMESPACE]
    if (not isinstance(envelope, Mapping) or set(envelope) != {"refs"}
            or not isinstance(envelope["refs"], list)):
        raise ValueError("Invalid media output envelope")
    return envelope["refs"]


def without_images(value, *, strip_refs=False):
    """Sanitize only known canonical positions, never recursively parse text."""
    if not isinstance(value, Mapping):
        return value
    value = deepcopy(dict(value))
    unavailable = False
    if isinstance(value.get("content"), list):
        content = []
        for block in value["content"]:
            if isinstance(block, Mapping) and block.get("type") == "image":
                content.append({"type": "text", "text": UNAVAILABLE})
                unavailable = True
            else:
                content.append(block)
        value["content"] = content
    output = value.get("output")
    if strip_refs and isinstance(output, Mapping) and NAMESPACE in output:
        value["output"] = {k: v for k, v in output.items() if k != NAMESPACE}
        unavailable = True
    if unavailable:
        # Rich text alone is not consumed by legacy get_serialized_output.
        if value.get("output") is None:
            texts = [block.get("text", "") for block in value.get("content", [])
                     if isinstance(block, Mapping) and block.get("type") == "text"]
            value["output"] = "\n".join(texts) or UNAVAILABLE
        elif isinstance(value["output"], str):
            value["output"] += "\n" + UNAVAILABLE
        else:
            value["output"] = {"output": value["output"], "image_status": UNAVAILABLE}
    return value


async def stage_result(result, coordinator, call):
    cap = capability(coordinator)
    if cap is None:
        if isinstance(result, ToolResult):
            data = result.model_dump()
            safe = without_images(data, strip_refs=True)
            if safe != data:
                result = ToolResult.model_validate(safe)
        return result, None, []
    try:
        receipt = await _await(cap.ingest(
            result, source_format="core-tool-result",
            origin=origin_for(coordinator, call), policy=None,
        ))
        normalized = receipt["value"]
        if not isinstance(normalized, ToolResult):
            raise TypeError("Media ingest must return a Core ToolResult")
        if any(getattr(block, "type", None) == "image" for block in normalized.content or []):
            raise ValueError("Media ingest left inline pixels")
        return normalized, cap, receipt["staged_refs"]
    except Exception:
        raise RuntimeError("Media staging failed; tool effects were not rolled back") from None


async def post_result(hooks, coordinator, call, data, cap, staged):
    try:
        boundary = tool_post_boundary(coordinator) if cap is not None else None
        if boundary is not None:
            receipt = await _await(boundary(
                data, hooks=hooks, coordinator=coordinator
            ))
            if (not isinstance(receipt, Mapping) or receipt.get("complete") is not True
                    or "result" not in receipt
                    or not isinstance(receipt.get("hook_result"), HookResult)):
                raise ValueError("Media tool post boundary did not attest the complete chain")
            post = receipt["hook_result"]
            value = None if post.action == "deny" else receipt["result"]
            return ToolPostResult(post, value, True)
        post = await hooks.emit("tool:post", data)
        if coordinator:
            post = await coordinator.process_hook_result(post, "tool:post", call.name)
        return ToolPostResult(post, effective_result(post, data["result"]), False)
    except BaseException as error:
        if cap is not None and staged:
            # Disposal failure leaves a host-owned staging orphan, never
            # accepted canonical reachability.
            try:
                await _await(cap.publish(
                    staged, origin=origin_for(coordinator, call), disposition="drop"
                ))
            except Exception:
                pass
        if cap is not None and isinstance(error, Exception):
            raise RuntimeError("Media tool post boundary failed; no media was accepted") from None
        raise


async def settle_result(call, result, current, original, post, cap, staged, coordinator):
    value = post.result
    unchanged_text = value is current and value == original
    # Hooks may redact/replace, but may not introduce inline pixels into durable
    # state. Only the pre-hook canonical ingest can stage image content.
    value = without_images(value, strip_refs=cap is None or not post.complete)
    refs = _refs(value)
    metadata = {}
    if cap is not None:
        origin = origin_for(coordinator, call)
        try:
            if refs:
                receipt = await _await(cap.publish(refs, origin=origin, disposition="retain"))
                refs = receipt["refs"]
                metadata[NAMESPACE] = {"refs": deepcopy(refs)}
                value["output"][NAMESPACE] = {"refs": deepcopy(refs)}
            elif staged:
                await _await(cap.publish(staged, origin=origin, disposition="drop"))
        except Exception:
            raise RuntimeError("Media publication refused; tool effects were not rolled back") from None
    if unchanged_text and value == original:
        content = result.get_serialized_output()
    elif isinstance(value, (dict, list)) or value is None:
        content = json.dumps(value)
    else:
        content = str(value)
    success = value.get("success") if isinstance(value, Mapping) else None
    return SettledToolResult(
        call.id, call.name, content, metadata, deepcopy(value),
        success if isinstance(success, bool) else None,
    )