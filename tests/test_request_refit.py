"""Actual orchestrator dispatch with an optional local-refusal provider seam."""
import asyncio

import pytest
from amplifier_core import ProviderInfo

from amplifier_module_loop_streaming import StreamingOrchestrator
from tests.test_ephemeral_cache_persist_mode import MockResponse, RequestCapturingProvider, ScriptedHooks
from tests.test_provider_budget_guard import HardFitBudgetContext, _retaining_coordinator


class LocalRefusal(ValueError):
    pass


class Provider(RequestCapturingProvider):
    def __init__(self, behavior='recover'):
        super().__init__()
        self.behavior = behavior
        self.error = LocalRefusal('synthetic before-transport refusal')
        self.budgets, self.refits, self.options = [], [], []

    def get_info(self):
        return ProviderInfo(id='fixture', display_name='Fixture', capabilities=['tools'], defaults={'model': 'fixture'})

    def request_budget(self, request, *, context_estimate, request_options=None):
        self.budgets.append(request)
        return {'estimated_input_tokens': 1, 'input_limit_tokens': 10,
                'context_token_budget': 0, 'max_output_tokens': 128}

    async def refit_request(self, request, *, error, context_estimate, request_options=None):
        self.refits.append((request, error, request_options))
        assert error is self.error and request is self.requests[-1]
        if self.behavior == 'cancel':
            raise asyncio.CancelledError
        if self.behavior == 'decline':
            return None
        if self.behavior == 'malformed':
            return {'estimated_input_tokens': True}
        decision = self.request_budget(request, context_estimate=context_estimate)
        if self.behavior == 'oversized':
            decision['estimated_input_tokens'] = 11
        return decision

    async def complete(self, request, **kwargs):
        self.requests.append(request)
        self.options.append(kwargs)
        if len(self.requests) == 1 or self.behavior == 'repeat':
            raise self.error
        return MockResponse(text='refitted')


class StreamProvider(Provider):
    async def stream(self, request, **kwargs):
        self.requests.append(request)
        self.options.append(kwargs)
        if self.behavior == 'partial':
            yield {'content': 'already visible'}
            raise self.error
        if len(self.requests) == 1 or self.behavior == 'repeat':
            raise self.error
        yield {'content': 'refitted'}


async def run(provider):
    context, hooks = HardFitBudgetContext(), ScriptedHooks({})
    loop = StreamingOrchestrator({'extended_thinking': True, 'min_delay_between_calls_ms': 0})
    result = await loop.execute(
        'work', context, {'main': provider}, {}, hooks, _retaining_coordinator(context))
    return result, context, hooks, loop


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
async def test_refit_reuses_one_prepared_request_without_replaying_hooks(streaming):
    provider = (StreamProvider if streaming else Provider)()
    result, context, hooks, loop = await run(provider)
    assert result == 'refitted'
    assert len(provider.requests) == len(provider.budgets) == 2
    assert provider.requests[0] is provider.requests[1] is provider.budgets[-1]
    assert len(provider.refits) == 1
    assert provider.options[0] == provider.options[1]
    assert context.hard_fit_calls == [False]
    assert loop._llm_calls_this_turn == 1
    assert sum(name == 'provider:request' for name, _ in hooks.emitted) == 1
    assert [payload for name, payload in hooks.emitted if name == 'orchestrator:request_refit'] == [
        {'attempt': 1, 'phase': 'before_transport', 'result': 'refitted'}]


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('behavior', ['decline', 'malformed', 'oversized', 'repeat'])
async def test_refit_is_bounded_and_requires_a_valid_fitting_measurement(streaming, behavior):
    provider = (StreamProvider if streaming else Provider)(behavior)
    with pytest.raises(LocalRefusal):
        await run(provider)
    assert len(provider.refits) == 1
    assert len(provider.requests) == (2 if behavior == 'repeat' else 1)


@pytest.mark.asyncio
async def test_no_refit_after_any_stream_event():
    provider = StreamProvider('partial')
    with pytest.raises(LocalRefusal):
        await run(provider)
    assert len(provider.requests) == 1 and provider.refits == []


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
async def test_cancellation_during_refit_propagates_without_transport(streaming):
    provider = (StreamProvider if streaming else Provider)('cancel')
    with pytest.raises(asyncio.CancelledError):
        await run(provider)
    assert len(provider.requests) == 1 and len(provider.refits) == 1
