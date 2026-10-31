"""Bounded LRU of native BF16 MLX blocks, without a retained host weight copy.

The budget bounds retained payload and backing buffers, not total RSS. Loading
a block and expanding it for FP32 matmul require additional transient memory.
No physical residency, separate VRAM, or zero-copy transfer is assumed.
"""

from collections import OrderedDict
from dataclasses import dataclass
import os
from typing import BinaryIO, Protocol

import mlx.core as mx
import numpy as np

from .bf16 import (
    TensorReaderLike,
    _check_input,
    _matrix_metadata,
    native_bf16_from_bytes,
)


class FileTensorReaderLike(TensorReaderLike, Protocol):
    file: BinaryIO
    base: int


def _identity(fd: int) -> tuple[int, ...]:
    stat = os.fstat(fd)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class _BlockKey:
    revision: str
    identity: tuple[int, ...]
    base: int
    name: str
    offsets: tuple[int, int]
    shape: tuple[int, int]
    first_row: int
    rows: int

    @property
    def nbytes(self) -> int:
        return self.rows * self.shape[1] * 2


class DeviceBF16Cache:
    """Single-threaded cache. Keep one cache across readers sharing a revision.

    Do not keep a second model-sized host weight cache alongside this cache.
    Entries are evaluated MLX BF16 arrays only. `clear()` drops those references;
    MLX's allocator may still retain free buffers until its own cache is cleared.
    """

    def __init__(self, max_bytes: int):
        if type(max_bytes) is not int or not 0 < max_bytes <= 5 * 1024**3:
            raise ValueError("max_bytes must be between 1 and 5 GiB")
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._buffer_sizes = {}
        self.retained_native_bytes = 0
        self.retained_buffer_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.source_read_bytes = self.source_read_calls = 0

    def _evict(self):
        key, array = self._entries.popitem(last=False)
        self.retained_native_bytes -= key.nbytes
        self.retained_buffer_bytes -= self._buffer_sizes.pop(key)
        self.evictions += 1
        del array

    def _get(self, key: _BlockKey, loader):
        if key.nbytes > min(self.max_bytes, 16 * 1024**2):
            raise ValueError("native BF16 block exceeds the configured payload budget")
        if key in self._entries:
            array = self._entries[key]
            if (
                array.dtype != mx.bfloat16
                or array.shape != (key.rows, key.shape[1])
                or array.nbytes != key.nbytes
            ):
                raise RuntimeError("cached native BF16 entry changed")
            self._entries.move_to_end(key)
            self.hits += 1
            return array
        self.misses += 1
        while self.retained_native_bytes + key.nbytes > self.max_bytes:
            self._evict()
        raw = loader()
        array = native_bf16_from_bytes(raw, (key.rows, key.shape[1]))
        del raw
        allocated = int(mx.get_array_buffer_size(array))
        if allocated > self.max_bytes:
            raise ValueError("backing buffer exceeds the configured cache budget")
        while self.retained_buffer_bytes + allocated > self.max_bytes:
            self._evict()
        self._entries[key] = array
        self._buffer_sizes[key] = allocated
        self.retained_native_bytes += key.nbytes
        self.retained_buffer_bytes += allocated
        return array

    def clear(self) -> None:
        self._entries.clear()
        self._buffer_sizes.clear()
        self.retained_native_bytes = self.retained_buffer_bytes = 0

    def snapshot(self) -> dict:
        return {
            "max_bytes": self.max_bytes,
            "retained_native_bytes": self.retained_native_bytes,
            "retained_backing_buffer_bytes": int(
                mx.get_array_buffer_size(list(self._entries.values()))
            ),
            "explicit_retained_host_weight_bytes": 0,
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "source_read_calls": self.source_read_calls,
            "source_read_bytes": self.source_read_bytes,
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_allocator_cache_bytes": int(mx.get_cache_memory()),
            "mlx_peak_active_bytes": int(mx.get_peak_memory()),
        }


def cached_dense_bf16(
    reader: FileTensorReaderLike,
    name: str,
    x: np.ndarray,
    cache: DeviceBF16Cache,
    *,
    revision: str,
    block_rows: int = 512,
) -> np.ndarray:
    """Dense FP32 matmul reusing native BF16 blocks from a file-backed reader.

    Revision must identify the caller's model manifest. Every projection also
    checks file identity, dtype, tensor offsets and shape. These checks are not
    a cryptographic verification of all model bytes. Logical reader.bytes_read
    includes hits; cache.source_read_bytes counts actual requested file bytes.
    """
    if (
        not isinstance(revision, str)
        or not revision
        or type(block_rows) is not int
        or not 1 <= block_rows <= 512
    ):
        raise ValueError("explicit nonempty revision and block_rows in1..512 required")
    rows, cols = _matrix_metadata(reader, name)
    _check_input(x, cols)
    fd = reader.file.fileno()
    identity = _identity(fd)
    offsets = tuple(reader.tensors[name]["data_offsets"])
    if (
        type(reader.base) is not int
        or reader.base < 0
        or reader.base + offsets[1] > identity[2]
    ):
        raise ValueError("tensor extends outside its backing file")
    if min(rows, block_rows) * cols * 2 > min(cache.max_bytes, 16 * 1024**2):
        raise ValueError("block does not fit the native cache budget")
    output = np.empty((len(x), rows), dtype=np.float32)
    vector = mx.array(x)
    for first in range(0, rows, block_rows):
        count = min(block_rows, rows - first)
        key = _BlockKey(
            revision, identity, reader.base, name, offsets, (rows, cols), first, count
        )
        if _identity(fd) != identity:
            raise RuntimeError("source file changed during dense projection")

        def load():
            raw = os.pread(fd, key.nbytes, reader.base + offsets[0] + first * cols * 2)
            cache.source_read_calls += 1
            cache.source_read_bytes += key.nbytes
            if len(raw) != key.nbytes or _identity(fd) != identity:
                raise RuntimeError("incomplete or changed source block")
            return raw

        block = cache._get(key, load)
        result = vector @ block.astype(mx.float32).T
        mx.eval(result)
        output[:, first : first + count] = np.asarray(result)
        reader.bytes_read += key.nbytes
        del block, result
    if _identity(fd) != identity or not np.isfinite(output).all():
        raise RuntimeError("source changed or dense result is nonfinite")
    return output
