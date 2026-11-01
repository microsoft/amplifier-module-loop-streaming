"""One provider-owned, proven-before-transport refit per prepared dispatch.

The provider must identify its exact local refusal. This layer never recognizes
exception text or assumes an unreturned request was unsent.
"""


class RefitProvider:
    def __init__(self, provider, refit):
        self.provider, self.refit = provider, refit

    async def complete(self, request, **kwargs):
        try:
            return await self.provider.complete(request, **kwargs)
        except Exception as error:
            if not await self.refit(request, error, kwargs, streaming=False):
                raise
        # Deliberately outside the catch: a repeated refusal is terminal.
        return await self.provider.complete(request, **kwargs)

    def __getattr__(self, name):
        method = getattr(self.provider, name)
        if name != 'stream' or not callable(method):
            return method

        async def stream(request, **kwargs):
            for attempt in range(2):
                emitted = False
                iterator = None
                try:
                    iterator = aiter(method(request, **kwargs))
                    async for event in iterator:
                        emitted = True
                        yield event
                    return
                except Exception as error:
                    if emitted or attempt or not await self.refit(request, error, kwargs, streaming=True):
                        raise
                finally:
                    if callable(getattr(iterator, 'aclose', None)):
                        await iterator.aclose()
        return stream
