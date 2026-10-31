"""Process-local bounded safetensors metadata cache; never caches weights.

Use MetadataCache.open(ReaderClass, shard_path, prefix) in a new process.
The original Reader class and its weight read/compute methods are unchanged.
The budget accounts for retained Python objects, not allocator overhead/RSS;
callers still need a process/system memory watchdog.
"""

from collections import OrderedDict, namedtuple
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import MappingProxyType


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate safetensors metadata key")
        result[key] = value
    return result


class FrozenSequence(tuple):
    """Immutable JSON array retaining the Reader's comparisons with lists."""

    def __eq__(self, other):
        if isinstance(other, (tuple, list)):
            return tuple.__eq__(self, tuple(other))
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = tuple.__hash__


def freeze(value):
    """Return immutable JSON value and conservative retained object bytes."""
    if isinstance(value, dict):
        data = {}
        sizes = 0
        for key, item in value.items():
            frozen, size = freeze(item)
            data[key] = frozen
            sizes += sys.getsizeof(key) + size
        proxy = MappingProxyType(data)
        return proxy, sys.getsizeof(data) + sys.getsizeof(proxy) + sizes
    if isinstance(value, list):
        pairs = [freeze(item) for item in value]
        result = FrozenSequence(item for item, _ in pairs)
        return result, sys.getsizeof(result) + sum(size for _, size in pairs)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value, sys.getsizeof(value)
    raise TypeError("not a JSON value")


def identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


Entry = namedtuple("Entry", "base tensors retained_bytes header_bytes")


class MetadataCache:
    def __init__(
        self, max_bytes=192 * 1024**2, max_entries=64, max_header_bytes=8 * 1024**2
    ):
        if min(max_bytes, max_entries, max_header_bytes) <= 0:
            raise ValueError("positive cache limits required")
        if max_bytes < sys.getsizeof(OrderedDict()):
            raise ValueError("budget smaller than empty cache container")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.max_header_bytes = max_header_bytes
        self._entries = OrderedDict()
        self._payload_bytes = 0
        self._lock = threading.RLock()
        self.hits = self.misses = self.evictions = self.bypasses = 0
        self.parsed_header_bytes = 0
        self.parse_seconds = 0.0

    @property
    def retained_bytes(self):
        return self._payload_bytes + sys.getsizeof(self._entries)

    def stats(self):
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "bypasses": self.bypasses,
                "entries": len(self._entries),
                "retained_object_bytes": self.retained_bytes,
                "max_retained_object_bytes": self.max_bytes,
                "parsed_header_bytes": self.parsed_header_bytes,
                "parse_seconds": self.parse_seconds,
            }

    def _load(self, file, prefix):
        before = identity(os.fstat(file.fileno()))
        key = before + (prefix,)
        with self._lock:
            if key in self._entries:
                self.hits += 1
                entry = self._entries.pop(key)
                self._entries[key] = entry
                return entry
            self.misses += 1
            started = time.perf_counter()
            first = os.pread(file.fileno(), 8, 0)
            if len(first) != 8:
                raise EOFError("short safetensors header length")
            size = int.from_bytes(first, "little")
            if size <= 0 or size > self.max_header_bytes or size + 8 > before[2]:
                raise ValueError("invalid/oversized safetensors header")
            raw = os.pread(file.fileno(), size, 8)
            if len(raw) != size:
                raise EOFError("short safetensors header")
            header = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
            if not isinstance(header, dict):
                raise ValueError("header must be an object")
            selected = {
                name[len(prefix) :]: meta
                for name, meta in header.items()
                if name != "__metadata__" and name.startswith(prefix)
            }
            frozen, retained = freeze(selected)
            if identity(os.fstat(file.fileno())) != before:
                raise RuntimeError("source changed while reading metadata")
            # Counting shared scalars repeatedly overestimates the retained size.
            key_bytes = sys.getsizeof(key) + sum(sys.getsizeof(v) for v in key)
            overhead = sys.getsizeof(Entry(0, None, 0, 0)) + 3 * sys.getsizeof(0)
            entry = Entry(8 + size, frozen, retained + key_bytes + overhead, size)
            self.parsed_header_bytes += size
            self.parse_seconds += time.perf_counter() - started
            if entry.retained_bytes + sys.getsizeof(OrderedDict()) > self.max_bytes:
                self.bypasses += 1
                return entry
            self._entries[key] = entry
            self._payload_bytes += entry.retained_bytes
            while (
                len(self._entries) > self.max_entries
                or self.retained_bytes > self.max_bytes
            ):
                _, removed = self._entries.popitem(last=False)
                self._payload_bytes -= removed.retained_bytes
                self.evictions += 1
            return entry

    def open(self, reader_class, path, prefix="", **reader_options):
        if not isinstance(prefix, str):
            raise TypeError("prefix must be text")
        if not callable(getattr(reader_class, "from_metadata", None)):
            raise TypeError(
                "reader class must implement from_metadata(file, base, tensors, **options)"
            )
        file = Path(path).open("rb")
        try:
            expected_identity = identity(os.fstat(file.fileno()))
            entry = self._load(file, prefix)
            return reader_class.from_metadata(
                file,
                entry.base,
                dict(entry.tensors),
                expected_identity=expected_identity,
                **reader_options,
            )
        except BaseException:
            file.close()
            raise
