"""Bounded byte cache for explicitly selected model tensors.

An admission predicate can keep a measured hot set across sequential scans.
The byte budget covers retained payloads only, not process RSS or page cache.
"""

from collections import OrderedDict
from dataclasses import dataclass
import os
from typing import Callable, Mapping

MAX_PAYLOAD_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class ExpertKey:
    # Include the immutable revision and a fresh fstat-derived file identity.
    revision: str
    file_identity: tuple[int, int, int, int, int]
    layer: int
    expert: int
    metadata: tuple = ()


def file_identity(fd: int) -> tuple[int, int, int, int, int]:
    stat = os.fstat(fd)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def read_tensors(
    fd: int, key: ExpertKey, base: int, offsets: Mapping[str, tuple[int, int]]
) -> dict[str, bytes]:
    """Read explicitly selected safetensors ranges after validating the shard.

    Offsets are relative to the tensor data start, exactly as in the header.
    The caller must validate and bind the original header to the model revision.
    This function is intentionally standalone, not wired into the live runner.
    """
    if file_identity(fd) != key.file_identity or base < 0:
        raise ValueError("shard identity changed")
    file_size = key.file_identity[2]
    result = {}
    for name, (begin, end) in offsets.items():
        if begin < 0 or end < begin or base + end > file_size:
            raise ValueError("invalid safetensors offset")
        data = os.pread(fd, end - begin, base + begin)
        if len(data) != end - begin:
            raise EOFError(name)
        result[name] = data
    if file_identity(fd) != key.file_identity:
        raise ValueError("shard changed while reading")
    return result


class ExpertCache:
    def __init__(
        self,
        budget: int,
        admit: Callable[[ExpertKey], bool] = lambda _: True,
        max_per_layer: int | None = None,
    ):
        if budget < 0 or budget > MAX_PAYLOAD_BYTES:
            raise ValueError("cache budget must be between zero and 8 GiB")
        if max_per_layer is not None and max_per_layer < 1:
            raise ValueError("max_per_layer must be positive")
        self.budget = budget
        self.admit = admit
        self.max_per_layer = max_per_layer
        self.entries: OrderedDict[ExpertKey, dict[str, bytes]] = OrderedDict()
        self.retained_bytes = 0
        self.hits = 0
        self.misses = 0

    def grow_to(self, budget: int) -> None:
        """Raise the payload cap gradually. Caller must monitor total RSS."""
        if budget < self.budget or budget > MAX_PAYLOAD_BYTES:
            raise ValueError("budget must grow and stay at or below 8 GiB")
        self.budget = budget

    def admits(self, key: ExpertKey, expected: int) -> bool:
        return expected <= self.budget and self.admit(key)

    def get(
        self, key: ExpertKey, lengths: Mapping[str, int], loader: Callable[[str], bytes]
    ) -> dict[str, bytes]:
        """Read complete tensors, validating sizes before admission.

        In a real adapter, loader reads the named tensor at its exact safetensors
        offset. Revalidate the file identity when opening each shard.
        """
        if not lengths or any(
            not isinstance(n, int) or n < 0 for n in lengths.values()
        ):
            raise ValueError("invalid tensor lengths")
        expected = sum(lengths.values())
        if key in self.entries:
            entry = self.entries[key]
            if {name: len(data) for name, data in entry.items()} != dict(lengths):
                raise ValueError("metadata changed for cached expert")
            self.entries.move_to_end(key)
            self.hits += 1
            return entry.copy()

        self.misses += 1
        keep = self.admits(key, expected)
        if keep:
            if self.max_per_layer is not None:
                layer_keys = [k for k in self.entries if k.layer == key.layer]
                if len(layer_keys) >= self.max_per_layer:
                    victim = self.entries.pop(layer_keys[0])
                    self.retained_bytes -= sum(map(len, victim.values()))
            # Evict before allocation, so a new group fits in the payload budget.
            while self.retained_bytes + expected > self.budget:
                _, victim = self.entries.popitem(last=False)
                self.retained_bytes -= sum(map(len, victim.values()))
        entry = {}
        for name, length in lengths.items():
            data = loader(name)
            if not isinstance(data, bytes) or len(data) != length:
                raise ValueError("tensor payload does not match metadata")
            entry[name] = data
        if keep:
            self.entries[key] = entry
            self.retained_bytes += expected
        return entry.copy()


def slice_tensor(
    entry: Mapping[str, bytes], name: str, start: int = 0, size: int | None = None
) -> bytes:
    data = entry[name]
    if size is None:
        size = len(data) - start
    if start < 0 or size < 0 or start + size > len(data):
        raise ValueError("tensor read outside metadata range")
    return data[start : start + size]
