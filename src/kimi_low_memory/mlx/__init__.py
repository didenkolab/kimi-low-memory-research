"""Optional MLX backend. Importing the core package does not require MLX.

Call configure() explicitly once before GPU operations, and monitor process
and system memory independently. No function changes wired-memory settings.
"""

from .bf16 import (
    cached_bf16_matmul,
    configure,
    dense_bf16,
    fused_bf16_gemv,
    native_bf16_from_bytes,
)
from .device_cache import DeviceBF16Cache, cached_dense_bf16
from .q4 import quantized_linear, routed_expert

__all__ = [
    "configure",
    "dense_bf16",
    "native_bf16_from_bytes",
    "cached_bf16_matmul",
    "fused_bf16_gemv",
    "DeviceBF16Cache",
    "cached_dense_bf16",
    "quantized_linear",
    "routed_expert",
]
