"""Bounded, single-worker prefetch for a known, ordered set of experts.

Research prototype. The caller computes the router first and provides the
exact chosen IDs. The implementation does not select, prune or reorder them.
The budget reserves current, next, and one previous consumer payload during
iterator handoff. Consumers must not retain additional payloads. Loader
temporaries, RSS and the OS page cache require separate accounting.
"""

from concurrent.futures import ThreadPoolExecutor


class SelectedExpertPrefetch:
    def __init__(self, ids, loader, max_expert_bytes, max_buffer_bytes):
        self.ids = tuple(ids)
        if (
            not callable(loader)
            or type(max_expert_bytes) is not int
            or max_expert_bytes <= 0
            or type(max_buffer_bytes) is not int
            or max_buffer_bytes <= 0
        ):
            raise ValueError("invalid loader or expert cap")
        if max_buffer_bytes < 3 * max_expert_bytes:
            raise ValueError(
                "budget must hold current, next and consumer handoff payload"
            )
        self.loader = loader
        self.max_expert_bytes = max_expert_bytes
        self.executor = None
        self.iterator = None
        self.closed = False
        self.started = False

    def __enter__(self):
        if self.started or self.closed:
            raise RuntimeError("prefetch cannot be reused")
        self.started = True
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="expert-prefetch"
        )
        return self

    def _load(self, expert):
        data = self.loader(expert)
        if not isinstance(data, bytes) or len(data) > self.max_expert_bytes:
            raise ValueError("expert loader returned an invalid or oversized payload")
        return data

    def __iter__(self):
        if self.executor is None or self.iterator is not None or self.closed:
            raise RuntimeError("use once inside the context manager")
        self.iterator = self._iterate()
        return self.iterator

    def _iterate(self):
        pending = None
        try:
            for i, expert in enumerate(self.ids):
                if pending is None:
                    pending = self.executor.submit(self._load, expert)
                payload = pending.result()
                pending = None
                if i + 1 < len(self.ids):
                    pending = self.executor.submit(self._load, self.ids[i + 1])
                yield expert, payload
                del payload
        finally:
            if pending is not None:
                pending.cancel()

    def __exit__(self, exc_type, exc, tb):
        self.closed = True
        if self.iterator is not None:
            self.iterator.close()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        return False
