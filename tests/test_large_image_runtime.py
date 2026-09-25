"""Real context + loop regression; no credentials or paid provider calls."""

import base64
import random
import struct
import zlib
from types import SimpleNamespace

import pytest
from amplifier_core import ContextLengthError

pytest.importorskip("amplifier_module_context_simple")
from amplifier_module_context_simple import SimpleContextManager

from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import (
    MockCoordinator,
    MockResponse,
    ScriptedHooks,
)


@pytest.fixture(scope="module")
def image_data():
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    width = 1450
    pixels = random.Random(17).randbytes(width * width * 3)
    rows = b"".join(
        b"\0" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(width)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, width, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    assert 6_000_000 < len(png) < 8_000_000
    return base64.b64encode(png).decode()


class ImageProvider:
    def __init__(self, mode, streaming):
        self.mode = mode
        self.requests = []
        self.estimates = []
        if mode == "absent":
            self.request_budget = None
        if streaming:
            self.stream = self._stream

    def get_info(self):
        return SimpleNamespace(
            capabilities=["request_budget:provider_count"], defaults={}
        )

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.estimates.append(context_estimate)
        if self.mode == "unavailable":
            return None
        if self.mode == "failed":
            raise RuntimeError("counter failed")
        count = 150_000 if self.mode == "oversized" else 1200
        return {
            "estimated_input_tokens": count,
            "input_limit_tokens": 100_000,
            "context_token_budget": context_estimate if count < 100_000 else 1,
            "measurement": {
                "kind": "provider_count",
                "source": "fixture",
                "input_tokens": count,
            },
        }

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        return MockResponse(text="image received")

    async def _stream(self, request, *, tools):
        self.requests.append(request)
        yield {"content": "image received"}

    def parse_tool_calls(self, response):
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("meter", ["estimate", "actual"])
@pytest.mark.parametrize(
    "mode", ["absent", "unavailable", "exact", "oversized", "failed"]
)
async def test_large_image_preserved_and_native_limits_remain_authoritative(
    image_data, mode, streaming, meter
):
    context = SimpleContextManager(
        max_tokens=100_000, compaction_notice_enabled=False, token_meter=meter
    )
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": image_data},
    }
    await context.add_message({"role": "user", "content": [image]})
    coordinator = MockCoordinator()
    coordinator.register_capability(
        "context.request_retention", context.get_messages_for_request_retaining
    )
    if meter == "actual":
        coordinator.register_capability(
            "context.measured_request_view", context.get_measured_request_view
        )
    provider = ImageProvider(mode, streaming)
    operation = StreamingOrchestrator({}).execute(
        "Describe this image",
        context,
        {"test": provider},
        {},
        ScriptedHooks({}),
        coordinator,
    )
    if mode in {"oversized", "failed"}:
        with pytest.raises(ContextLengthError if mode == "oversized" else RuntimeError):
            await operation
        assert not provider.requests
    else:
        await operation
        assert len(provider.requests) == 1
        images = [
            block
            for msg in provider.requests[0].messages
            if isinstance(msg.content, list)
            for block in msg.content
            if getattr(block, "type", None) == "image"
        ]
        assert len(images) == 1
        assert images[0].source["data"] == image_data
    assert all(estimate < 100 for estimate in provider.estimates)
    assert (await context.get_messages())[0]["content"][0]["source"][
        "data"
    ] == image_data
