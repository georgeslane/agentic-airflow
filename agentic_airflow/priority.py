"""Per-model priority queues (spec §5).

Each model has a :class:`ModelQueue` with ``concurrency`` slots. Waiters are
ordered by ``(priority, order_key, arrival)``:

* **priority 0** - live user requests (preempt at the next atomic boundary);
* **priority 1** - DAG tasks; ``order_key`` is the DAG run's creation time so
  the longest-waiting run is served first (aging FIFO).

Agentic tasks acquire a slot for **one inference step** and release it
afterwards, which is what makes user requests able to cut in between steps.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

USER_PRIORITY = 0
DAG_PRIORITY = 1


class ModelQueue:
    def __init__(self, model: str, concurrency: int = 1):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.model = model
        self.concurrency = concurrency
        self._active = 0
        self._waiters: list[tuple[int, float, int, asyncio.Future]] = []
        self._counter = itertools.count()
        self.completed_steps = 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return sum(1 for w in self._waiters if not w[3].done())

    def has_waiting(self, priority: int | None = None) -> bool:
        return any(not w[3].done() and (priority is None or w[0] == priority) for w in self._waiters)

    async def acquire(self, priority: int = DAG_PRIORITY, order_key: float = 0.0) -> None:
        loop = asyncio.get_running_loop()
        if self._active < self.concurrency and not self._has_higher_or_equal_waiter(priority, order_key):
            self._active += 1
            return
        fut: asyncio.Future = loop.create_future()
        heapq.heappush(self._waiters, (priority, order_key, next(self._counter), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if not fut.done():
                fut.cancel()
            else:
                # We were granted the slot right as we were cancelled: give it back.
                self._active -= 1
                self._wake_next()
            raise

    def _has_higher_or_equal_waiter(self, priority: int, order_key: float) -> bool:
        for w in self._waiters:
            if w[3].done():
                continue
            if (w[0], w[1]) <= (priority, order_key):
                return True
        return False

    def release(self) -> None:
        if self._active <= 0:
            raise RuntimeError("release() called more times than acquire()")
        self._active -= 1
        self.completed_steps += 1
        self._wake_next()

    def _wake_next(self) -> None:
        while self._waiters and self._active < self.concurrency:
            _, _, _, fut = heapq.heappop(self._waiters)
            if fut.done():
                continue
            self._active += 1
            fut.set_result(None)

    @asynccontextmanager
    async def slot(self, priority: int = DAG_PRIORITY, order_key: float = 0.0) -> AsyncIterator[None]:
        await self.acquire(priority, order_key)
        try:
            yield
        finally:
            self.release()

    def stats(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "concurrency": self.concurrency,
            "active": self._active,
            "waiting": self.waiting,
            "waiting_user": sum(1 for w in self._waiters if not w[3].done() and w[0] == USER_PRIORITY),
            "completed_steps": self.completed_steps,
        }


class ModelQueueSet:
    """Lazily creates one :class:`ModelQueue` per model id."""

    def __init__(self, concurrency: dict[str, int] | None = None, default_concurrency: int = 1):
        self._concurrency = dict(concurrency or {})
        self._default = default_concurrency
        self._queues: dict[str, ModelQueue] = {}

    def get(self, model: str) -> ModelQueue:
        q = self._queues.get(model)
        if q is None:
            q = ModelQueue(model, self._concurrency.get(model, self._default))
            self._queues[model] = q
        return q

    def stats(self) -> list[dict[str, Any]]:
        return [q.stats() for q in self._queues.values()]
