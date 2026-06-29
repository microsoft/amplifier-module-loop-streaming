"""Bounded, fail-loud steering queue for the streaming orchestrator.

Mid-turn steering: the host enqueues user messages while a turn is running;
the orchestrator drains them at the top-of-iteration boundary (after the prior
tool round, before the next provider call) and injects each as a user-role
message. FIFO. Fail loud — never silently drop.
"""

from __future__ import annotations

import asyncio


class SteeringQueueFull(RuntimeError):
    """Raised when steer() is called on a full bounded queue (fail loud)."""


class SteeringQueue:
    """FIFO queue for mid-turn steering messages.

    Bounded to surface misuse loudly rather than grow without limit.
    """

    DEFAULT_MAXSIZE = 100

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=maxsize)

    def steer(self, message: str) -> None:
        """Non-blocking enqueue of one steering message.

        Raises ValueError on empty/whitespace-only input, SteeringQueueFull
        when the bound is reached. Never blocks, never silently drops.
        """
        if message is None or not message.strip():
            raise ValueError("steering message must be non-empty")
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull as exc:
            raise SteeringQueueFull("steering queue is full; message rejected") from exc

    def drain(self) -> list[str]:
        """Dequeue all pending messages in FIFO order (possibly empty)."""
        messages: list[str] = []
        while not self._queue.empty():
            try:
                messages.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def clear(self) -> int:
        """Discard all pending steers (e.g. on cancellation). Returns count discarded."""
        return len(self.drain())

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()
