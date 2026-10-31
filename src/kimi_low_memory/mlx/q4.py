"""Native signed-nibble INT4 projections through MLX quantized_matmul."""

import struct

import mlx.core as mx
import numpy as np

from .bf16 import TensorReaderLike, _check_input


def _quantized_device(
    reader: TensorReaderLike, prefix: str, x, *, block_rows: int, deferred: bool
):
    if type(block_rows) is not int or not 1 <= block_rows <= 2048:
        raise ValueError("block_rows must be between 1 and 2048")
    shape_bytes = reader.read(prefix + ".weight_shape")
    if len(shape_bytes) != 8:
        raise ValueError("expected an INT32 pair for quantized shape")
    rows, cols = struct.unpack("<ii", shape_bytes)
    if (
        rows <= 0
        or cols <= 0
        or cols % 32
        or x.ndim != 2
        or x.shape[1] != cols
        or x.dtype != mx.float32
    ):
        raise ValueError("invalid INT4 projection shape or activation dtype")
    packed_meta = reader.tensors[prefix + ".weight_packed"]
    scale_meta = reader.tensors[prefix + ".weight_scale"]
    if (
        packed_meta["dtype"] != "I32"
        or scale_meta["dtype"] != "BF16"
        or tuple(packed_meta["shape"]) != (rows, cols // 8)
        or tuple(scale_meta["shape"]) != (rows, cols // 32)
    ):
        raise ValueError("unexpected packed INT4 or BF16 scale metadata")
    payload_bytes = rows * (cols // 2 + cols // 32 * 2)
    if deferred and payload_bytes > 16 * 1024**2:
        raise ValueError("deferred projection exceeds 16 MiB native payload")
    if min(rows, block_rows) * (cols // 2 + cols // 32 * 2) > 16 * 1024**2:
        raise ValueError("INT4 block exceeds 16 MiB native payload")
    blocks = []
    for first in range(0, rows, block_rows):
        count = min(block_rows, rows - first)
        packed_raw = reader.read(
            prefix + ".weight_packed", first * cols // 2, count * cols // 2
        )
        scales_raw = reader.read(
            prefix + ".weight_scale", first * cols // 32 * 2, count * cols // 32 * 2
        )
        if (
            len(packed_raw) != count * cols // 2
            or len(scales_raw) != count * cols // 32 * 2
        ):
            raise EOFError("incomplete packed weight or scale block")
        packed = np.frombuffer(packed_raw, dtype="<u4").reshape(count, cols // 8)
        scales = (
            (np.frombuffer(scales_raw, dtype="<u2").astype(np.uint32) << 16)
            .view(np.float32)
            .reshape(count, cols // 32)
        )
        weights, scale = mx.array(packed), mx.array(scales)
        # Affine bias expresses the checkpoint convention: (unsigned nibble - 8) * scale.
        output = mx.quantized_matmul(
            x,
            weights,
            scale,
            -8 * scale,
            transpose=True,
            group_size=32,
            bits=4,
            mode="affine",
        )
        if not deferred:
            mx.eval(output)
        blocks.append(output)
    result = mx.concatenate(blocks, axis=1)
    mx.eval(result)
    return result


def quantized_linear(
    reader: TensorReaderLike, prefix: str, x: np.ndarray, *, block_rows: int = 512
) -> np.ndarray:
    """One bounded projection; returns FP32 NumPy activations."""
    _check_input(x, x.shape[1] if isinstance(x, np.ndarray) and x.ndim == 2 else -1)
    result = np.array(
        _quantized_device(
            reader, prefix, mx.array(x), block_rows=block_rows, deferred=False
        )
    )
    if not np.isfinite(result).all():
        raise ValueError("nonfinite INT4 projection result")
    return result


def routed_expert(
    reader: TensorReaderLike, prefix: str, x: np.ndarray, *, block_rows: int = 512
) -> np.ndarray:
    """Three bounded INT4 projections, keeping SiLU and intermediates on GPU.

    Defers blocks within one projection, capped at 16 MiB native payload.
    Routing and cross-expert accumulation remain the caller's responsibility.
    """
    _check_input(x, x.shape[1] if isinstance(x, np.ndarray) and x.ndim == 2 else -1)
    vector = mx.array(x)
    gate = _quantized_device(
        reader, prefix + ".gate_proj", vector, block_rows=block_rows, deferred=True
    )
    up = _quantized_device(
        reader, prefix + ".up_proj", vector, block_rows=block_rows, deferred=True
    )
    mixed = (gate / (1 + mx.exp(-gate))) * up
    mx.eval(mixed)
    output = _quantized_device(
        reader, prefix + ".down_proj", mixed, block_rows=block_rows, deferred=True
    )
    result = np.array(output)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite expert result")
    return result
