"""Optional MLX BF16 operations with explicit setup and bounded weight blocks.

The experimental fused GEMV passed small synthetic checks only. It changes
FP32 reduction order and has not been validated for a complete model response.
"""

from collections.abc import Mapping
import sys
from typing import Any, Protocol

import numpy as np

try:
    import mlx.core as mx
except ImportError as exc:
    raise ImportError(
        "The optional MLX backend requires the mlx extra on a supported Mac."
    ) from exc


class TensorReaderLike(Protocol):
    tensors: Mapping[str, Mapping[str, Any]]
    bytes_read: int

    def read(self, name: str, start: int = 0, size: int | None = None) -> bytes: ...


def configure(
    *, memory_limit_bytes: int, cache_limit_bytes: int, device: str = "gpu"
) -> None:
    """Explicit process-global setup. Operation functions never reset these limits.

    MLX's memory limit is advisory, not an RSS/system-memory watchdog. Retained
    GPU arrays use unified RAM. This function does not set wired-memory limits.
    """
    if (
        type(memory_limit_bytes) is not int
        or memory_limit_bytes <= 0
        or type(cache_limit_bytes) is not int
        or not 0 <= cache_limit_bytes <= memory_limit_bytes
        or device not in ("gpu", "cpu")
    ):
        raise ValueError("invalid explicit MLX memory/device configuration")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    mx.set_memory_limit(memory_limit_bytes)
    mx.set_cache_limit(cache_limit_bytes)


def _check_input(x: np.ndarray, cols: int) -> None:
    if (
        not isinstance(x, np.ndarray)
        or x.dtype != np.float32
        or x.ndim != 2
        or not len(x)
        or x.shape[1] != cols
        or not np.isfinite(x).all()
    ):
        raise ValueError("expected finite FP32 activations with shape (batch, columns)")


def _matrix_metadata(reader: TensorReaderLike, name: str) -> tuple[int, int]:
    meta = reader.tensors[name]
    shape = tuple(meta["shape"])
    if (
        meta["dtype"] != "BF16"
        or len(shape) != 2
        or not all(type(v) is int and v > 0 for v in shape)
    ):
        raise ValueError("expected a positive two-dimensional BF16 tensor")
    if "data_offsets" in meta:
        start, end = meta["data_offsets"]
        if start < 0 or end - start != shape[0] * shape[1] * 2:
            raise ValueError("BF16 tensor size does not match its metadata")
    return shape


def native_bf16_from_bytes(raw: bytes | memoryview, shape: tuple[int, int]):
    """Create an evaluated, independently owned native BF16 MLX block."""
    if (
        sys.byteorder != "little"
        or len(shape) != 2
        or not all(type(v) is int and v > 0 for v in shape)
        or len(raw) != shape[0] * shape[1] * 2
    ):
        raise ValueError("unexpected BF16 block shape, size or host byte order")
    bits = np.frombuffer(raw, dtype="<u2").reshape(shape)
    native = mx.asarray(mx.array(bits).view(mx.bfloat16), copy=True)
    mx.eval(native)
    return native


def dense_bf16(
    reader: TensorReaderLike, name: str, x: np.ndarray, *, block_rows: int = 512
) -> np.ndarray:
    """Stream BF16 blocks and multiply in FP32; no cache or global setup changes."""
    if type(block_rows) is not int or not 1 <= block_rows <= 512:
        raise ValueError("block_rows must be between 1 and 512")
    rows, cols = _matrix_metadata(reader, name)
    _check_input(x, cols)
    if block_rows * cols * 2 > 16 * 1024**2:
        raise ValueError("BF16 block exceeds the 16 MiB payload bound")
    result = np.empty((len(x), rows), dtype=np.float32)
    vector = mx.array(x)
    for first in range(0, rows, block_rows):
        count = min(block_rows, rows - first)
        raw = reader.read(name, first * cols * 2, count * cols * 2)
        if len(raw) != count * cols * 2:
            raise EOFError("incomplete BF16 block")
        weights = (
            mx.array(np.frombuffer(raw, dtype="<u2").reshape(count, cols))
            .view(mx.bfloat16)
            .astype(mx.float32)
        )
        output = vector @ weights.T
        mx.eval(output)
        result[:, first : first + count] = np.asarray(output)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite dense result")
    return result


def cached_bf16_matmul(native_bf16, vector):
    """Reference sequence for retained BF16 weights: FP32 expansion then matmul."""
    _gemv_shape(native_bf16.shape, vector.shape)
    if native_bf16.dtype != mx.bfloat16 or vector.dtype != mx.float32:
        raise ValueError("expected native BF16 weights and FP32 input")
    return vector @ native_bf16.astype(mx.float32).T


def _gemv_shape(weight_shape, vector_shape):
    if (
        len(weight_shape) != 2
        or len(vector_shape) != 2
        or not all(type(v) is int and 0 < v <= 8192 for v in weight_shape)
        or vector_shape != (1, weight_shape[1])
    ):
        raise ValueError(
            "experimental GEMV requires matrix <=8192x8192 and input (1,cols)"
        )
    return weight_shape


_FUSED_SOURCE = r"""
    const uint row = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    float acc = 0.0f;
    for (uint col = lane; col < COLS; col += 32) {
        const float weight = as_type<float>(uint(weight_bits[row * COLS + col]) << 16);
        acc = metal::fma(weight, vector[col], acc);
    }
    const float total = metal::simd_sum(acc);
    if (lane == 0) output[row] = total;
"""
_fused_kernel = None


def fused_bf16_gemv(weight_bits, vector):
    """Experimental batch-one GEMV with exact BF16 decoding inside the kernel.

    Inputs are uint16 BF16 bits and FP32 activations. Accumulation is FP32;
    output bit parity with ordinary matmul is not expected. No full FP32 weight
    matrix is materialized. Noncontiguous native inputs may be copied by MLX.
    This is an implementation experiment, not a claim of algorithmic novelty.
    """
    global _fused_kernel
    rows, cols = _gemv_shape(weight_bits.shape, vector.shape)
    if weight_bits.dtype != mx.uint16 or vector.dtype != mx.float32:
        raise ValueError("expected uint16 BF16 bits and FP32 input")
    if _fused_kernel is None:
        _fused_kernel = mx.fast.metal_kernel(
            name="kimi_low_memory_fused_bf16_v1",
            input_names=["weight_bits", "vector"],
            output_names=["output"],
            source=_FUSED_SOURCE,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    return _fused_kernel(
        inputs=[weight_bits, vector],
        template=[("COLS", cols)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, rows)],
        output_dtypes=[mx.float32],
    )[0]
