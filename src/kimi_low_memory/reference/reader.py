"""Bounded CPU reads for native Kimi BF16 and packed INT4 safetensors.

No model-repository Python code is imported. Bounds cover individual reads and
weight blocks, not process RSS, BLAS temporaries or a caller's retained arrays.
"""

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
import struct
from types import MappingProxyType

import numpy as np

_ITEM_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def bf16(raw):
    """Interpret little-endian BF16 bits as FP32; no floating-point rounding."""
    if len(raw) % 2:
        raise ValueError("BF16 byte count must be even")
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate safetensors JSON key")
        result[key] = value
    return result


def file_identity(stat):
    """Open-file identity used to bind parsed metadata to an unchanged source."""
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class TensorReader:
    def __init__(
        self,
        path,
        prefix="",
        *,
        block_rows=64,
        max_header_bytes=8 * 1024**2,
        max_read_bytes=128 * 1024**2,
        max_small_bytes=8 * 1024**2,
    ):
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a string")
        file = Path(path).open("rb")
        try:
            before = file_identity(os.fstat(file.fileno()))
            self._validate_options(
                block_rows, max_header_bytes, max_read_bytes, max_small_bytes
            )
            raw_size = file.read(8)
            if len(raw_size) != 8:
                raise ValueError("truncated safetensors prefix")
            size = int.from_bytes(raw_size, "little")
            if not 0 < size <= min(max_header_bytes, before[2] - 8):
                raise ValueError("invalid safetensors header length")
            header = json.loads(file.read(size), object_pairs_hook=_unique_object)
            if not isinstance(header, dict):
                raise ValueError("safetensors header must be an object")
            selected = {
                name[len(prefix) :]: meta
                for name, meta in header.items()
                if name != "__metadata__" and name.startswith(prefix)
            }
            self._initialize(
                file,
                8 + size,
                selected,
                expected_identity=before,
                block_rows=block_rows,
                max_header_bytes=max_header_bytes,
                max_read_bytes=max_read_bytes,
                max_small_bytes=max_small_bytes,
            )
        except BaseException:
            file.close()
            raise

    @classmethod
    def from_metadata(
        cls,
        file,
        base,
        tensors,
        *,
        expected_identity,
        block_rows=64,
        max_header_bytes=8 * 1024**2,
        max_read_bytes=128 * 1024**2,
        max_small_bytes=8 * 1024**2,
    ):
        """Build from selected metadata with all normal reader validations.

        expected_identity must be the five-field file_identity() captured before
        obtaining metadata. Immutable Mapping/tuple metadata can be shared.
        The factory takes ownership of file, including closing it on failure.
        Only the eight-byte header-length prefix is read; weight bytes are not.
        """
        instance = cls.__new__(cls)
        try:
            instance._initialize(
                file,
                base,
                tensors,
                expected_identity=expected_identity,
                block_rows=block_rows,
                max_header_bytes=max_header_bytes,
                max_read_bytes=max_read_bytes,
                max_small_bytes=max_small_bytes,
            )
            return instance
        except BaseException:
            file.close()
            raise

    @staticmethod
    def _validate_options(
        block_rows, max_header_bytes, max_read_bytes, max_small_bytes
    ):
        if type(block_rows) is not int or not 1 <= block_rows <= 512:
            raise ValueError("block_rows must be 1..512")
        if any(
            type(n) is not int or n <= 0
            for n in (max_header_bytes, max_read_bytes, max_small_bytes)
        ):
            raise ValueError("byte limits must be positive integers")

    def _initialize(
        self,
        file,
        base,
        tensors,
        *,
        expected_identity,
        block_rows,
        max_header_bytes,
        max_read_bytes,
        max_small_bytes,
    ):
        self._validate_options(
            block_rows, max_header_bytes, max_read_bytes, max_small_bytes
        )
        before = file_identity(os.fstat(file.fileno()))
        if (
            not isinstance(expected_identity, (tuple, list))
            or len(expected_identity) != 5
            or any(type(n) is not int for n in expected_identity)
            or tuple(expected_identity) != before
        ):
            raise ValueError("metadata source file identity mismatch")
        if type(base) is not int or not 8 < base <= min(
            8 + max_header_bytes, before[2]
        ):
            raise ValueError("invalid cached safetensors payload base")
        prefix = os.pread(file.fileno(), 8, 0)
        if len(prefix) != 8 or int.from_bytes(prefix, "little") != base - 8:
            raise ValueError("payload base differs from file header length")
        if not isinstance(tensors, Mapping):
            raise ValueError("tensor metadata must be a mapping")
        checked, occupied_ranges = {}, []
        for name, meta in tensors.items():
            if not isinstance(name, str) or not name or name == "__metadata__":
                raise ValueError("expected selected tensor names without __metadata__")
            if not isinstance(meta, Mapping) or meta.get("dtype") not in _ITEM_BYTES:
                raise ValueError("unsupported tensor metadata")
            shape, offsets = meta.get("shape"), meta.get("data_offsets")
            if (
                not isinstance(shape, (list, tuple))
                or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, (list, tuple))
                or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
            ):
                raise ValueError("invalid tensor shape/offsets")
            begin, end = offsets
            if not 0 <= begin <= end <= before[2] - base:
                raise ValueError("tensor outside safetensors payload")
            if end - begin != math.prod(shape) * _ITEM_BYTES[meta["dtype"]]:
                raise ValueError("tensor byte count differs from shape/dtype")
            if begin != end:
                occupied_ranges.append((begin, end))
            if (
                isinstance(meta, MappingProxyType)
                and isinstance(shape, tuple)
                and isinstance(offsets, tuple)
            ):
                checked[name] = meta
            else:
                checked[name] = MappingProxyType(
                    {
                        "dtype": meta["dtype"],
                        "shape": tuple(shape),
                        "data_offsets": tuple(offsets),
                    }
                )
        occupied_ranges.sort()
        for previous, current in zip(occupied_ranges, occupied_ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError("overlapping tensor data offsets")
        if file_identity(os.fstat(file.fileno())) != before:
            raise ValueError("source changed during reader initialization")
        self.file, self.path, self.base = file, Path(file.name), base
        self.file_identity, self.block_rows = before, block_rows
        self.max_read_bytes, self.max_small_bytes = max_read_bytes, max_small_bytes
        self.tensors, self.bytes_read = checked, 0

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read(self, name, start=0, size=None):
        if file_identity(os.fstat(self.file.fileno())) != self.file_identity:
            raise ValueError("source file changed after reader initialization")
        begin, end = self.tensors[name]["data_offsets"]
        size = end - begin - start if size is None else size
        if (
            type(start) is not int
            or type(size) is not int
            or start < 0
            or size < 0
            or start + size > end - begin
            or size > self.max_read_bytes
        ):
            raise ValueError("tensor read outside range or byte budget")
        raw = os.pread(self.file.fileno(), size, self.base + begin + start)
        if len(raw) != size:
            raise EOFError(name)
        if file_identity(os.fstat(self.file.fileno())) != self.file_identity:
            raise ValueError("source file changed during tensor read")
        self.bytes_read += size
        return raw

    def small(self, name):
        meta = self.tensors[name]
        begin, end = meta["data_offsets"]
        if end - begin > self.max_small_bytes:
            raise ValueError("small tensor exceeds byte budget")
        raw = self.read(name)
        if meta["dtype"] == "BF16":
            return bf16(raw).reshape(meta["shape"])
        if meta["dtype"] == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(meta["shape"]).copy()
        raise ValueError("small() supports BF16 or F32")

    @staticmethod
    def _input(x, columns):
        if (
            type(x) is not np.ndarray
            or x.dtype != np.float32
            or x.ndim != 2
            or x.shape[1] != columns
            or not np.isfinite(x).all()
        ):
            raise ValueError("expected finite FP32 [tokens, columns] input")

    def dense(self, name, x):
        meta = self.tensors[name]
        if meta["dtype"] != "BF16" or len(meta["shape"]) != 2:
            raise ValueError("dense() requires a BF16 matrix")
        rows, columns = meta["shape"]
        self._input(x, columns)
        out = np.empty((len(x), rows), dtype=np.float32)
        for lo in range(0, rows, self.block_rows):
            count = min(self.block_rows, rows - lo)
            weights = bf16(
                self.read(name, lo * columns * 2, count * columns * 2)
            ).reshape(count, columns)
            out[:, lo : lo + count] = x @ weights.T
        return out

    def quant_linear(self, prefix, x):
        packed_name, scale_name, shape_name = (
            prefix + suffix
            for suffix in (".weight_packed", ".weight_scale", ".weight_shape")
        )
        if self.tensors[shape_name]["dtype"] != "I32" or tuple(
            self.tensors[shape_name]["shape"]
        ) != (2,):
            raise ValueError("packed matrix requires I32 [rows, columns] metadata")
        rows, columns = struct.unpack("<ii", self.read(shape_name))
        if rows <= 0 or columns <= 0 or columns % 32:
            raise ValueError("invalid INT4 matrix dimensions")
        self._input(x, columns)
        packed_meta, scale_meta = self.tensors[packed_name], self.tensors[scale_name]
        if (
            packed_meta["dtype"] != "I32"
            or tuple(packed_meta["shape"]) != (rows, columns // 8)
            or scale_meta["dtype"] != "BF16"
            or tuple(scale_meta["shape"]) != (rows, columns // 32)
        ):
            raise ValueError("packed or scale matrix layout mismatch")
        out = np.empty((len(x), rows), dtype=np.float32)
        shifts = np.arange(8, dtype=np.uint32) * np.uint32(4)
        for lo in range(0, rows, self.block_rows):
            count = min(self.block_rows, rows - lo)
            packed = np.frombuffer(
                self.read(packed_name, lo * columns // 2, count * columns // 2),
                dtype="<u4",
            ).reshape(count, columns // 8)
            scale = bf16(
                self.read(scale_name, lo * columns // 32 * 2, count * columns // 32 * 2)
            ).reshape(count, columns // 32)
            integers = ((packed[:, :, None] >> shifts) & np.uint32(15)).reshape(
                count, columns
            ).astype(np.int16) - 8
            weights = integers.astype(np.float32) * np.repeat(scale, 32, axis=1)
            out[:, lo : lo + count] = x @ weights.T
        return out


Reader = TensorReader
