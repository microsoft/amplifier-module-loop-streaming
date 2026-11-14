"""Configured, offline-only transport for the explicit media integration profile.

Loaded by Core's real ModuleLoader, not mounted over a provider after sealing.
This sink performs no network or inference. Snapshots are test-local private
transport evidence; canonical messages and continuations must never contain them.
"""

import copy
from types import SimpleNamespace

from amplifier_core.message_models import ChatResponse, TextBlock, ToolCall
from amplifier_foundation.session.media_provider import MediaPreparedProvider


class OfflineTransport:
    name = "media-profile"

    def __init__(self, config):
        self.config = config
        self.counted, self.sent = [], []
        self.counted_views, self.sent_views = [], []
        self.count_options, self.send_options = [], []
        self.responses = []
        self.image_call = True
        self.canonical_counts, self.releases = [], []
        self.live_count_options = []

    def get_info(self):
        return SimpleNamespace(
            defaults={"model": "fixture"},
            capabilities=["request_budget:provider_count"],
        )

    def get_media_request_support(self, *, model, request_options=None):
        return {"v": 1, "positions": ["tool"], "mime_types": [self.config["mime"]]}

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.counted.append(request)
        self.counted_views.append(request.model_dump(mode="json"))
        self.count_options.append(dict(request_options or {}))
        return {
            "estimated_input_tokens": 20, "input_limit_tokens": 10000,
            "context_token_budget": 10000,
            "measurement": {"kind": "provider_count", "source": "offline-profile", "input_tokens": 20},
        }

    async def complete(self, request, **options):
        # Goal utilities are deliberately tool-less direct complete calls; the
        # loop does not request budget measurement for them. Main/final calls
        # must reuse the exact object observed by the previous real count.
        if request.metadata != {"stream": False}:
            assert self.counted[-1] is request
        self.sent.append(request)
        self.sent_views.append(request.model_dump(mode="json"))
        self.send_options.append(dict(options))
        text = self.responses.pop(0) if self.responses else "done"
        return ChatResponse(content=[TextBlock(text=text)])

    def parse_tool_calls(self, response):
        return ([ToolCall(id="engine-image", name="image", arguments={})]
                if self.image_call and len(self.sent) == 1 else [])


class ObservedMediaProvider(MediaPreparedProvider):
    """Observe only public adapter methods; retain real Foundation semantics."""

    def release_media_request(self, canonical, *, request_options=None):
        released = super().release_media_request(canonical, request_options=request_options)
        self.evidence.releases.append((canonical, copy.deepcopy(request_options), released))
        return released


async def mount(coordinator, config):
    sink = OfflineTransport(config)
    provider = ObservedMediaProvider(
        sink, coordinator.get_capability("experimental.media.v1"),
        provider_identity={"provider": sink.name, "account": "offline", "generation": 1},
        model="fixture", request_scope=coordinator.get_capability("session.media.tool_post").request_scope,
    )
    provider.evidence = sink
    budget = provider.request_budget

    async def count(canonical, *, context_estimate, request_options=None):
        sink.canonical_counts.append((canonical, copy.deepcopy(request_options)))
        sink.live_count_options.append(request_options)
        return await budget(canonical, context_estimate=context_estimate, request_options=request_options)

    provider.request_budget = count
    await coordinator.mount("providers", provider, name=sink.name)
    coordinator.register_capability("profile.offline_transport", sink)
    return provider.close