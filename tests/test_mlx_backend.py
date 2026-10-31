"""Opt-in tiny GPU checks; never reads model weights.

Set KIMI_MLX_TESTS=1 only inside a suitable external resource guard. Default
test discovery skips this module's GPU tests. Arrays are well below 32 MiB.
"""

import importlib.util
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

ENABLED = (
    os.environ.get("KIMI_MLX_TESTS") == "1"
    and importlib.util.find_spec("mlx") is not None
)


class ByteReader:
    def __init__(self, entries):
        self.data = {name: raw for name, (_, _, raw) in entries.items()}
        self.tensors = {
            name: {"dtype": dtype, "shape": shape}
            for name, (dtype, shape, _) in entries.items()
        }
        self.bytes_read = 0

    def read(self, name, start=0, size=None):
        raw = self.data[name]
        size = len(raw) - start if size is None else size
        if start < 0 or size < 0 or start + size > len(raw):
            raise ValueError("invalid synthetic read")
        self.bytes_read += size
        return raw[start : start + size]


@unittest.skipUnless(ENABLED, "optional GPU tests require KIMI_MLX_TESTS=1 and MLX")
class MlxBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import kimi_low_memory.mlx as backend

        cls.mx, cls.backend = mx, backend
        backend.configure(memory_limit_bytes=64 * 1024**2, cache_limit_bytes=1024**2)

    def tearDown(self):
        self.mx.synchronize()
        self.mx.clear_cache()
        self.assertLess(self.mx.get_peak_memory(), 16 * 1024**2)

    def test_bf16_stream_and_experimental_fused(self):
        rng = np.random.default_rng(73)
        for rows, cols in ((5, 33), (3, 7168)):
            values = rng.normal(0, 0.1, (rows, cols)).astype(np.float32)
            bits = (values.view(np.uint32) >> 16).astype(np.uint16)
            expected_weights = (bits.astype(np.uint32) << 16).view(np.float32)
            x = rng.normal(0, 0.1, (1, cols)).astype(np.float32)
            reader = ByteReader({"weight": ("BF16", [rows, cols], bits.tobytes())})
            streamed = self.backend.dense_bf16(reader, "weight", x, block_rows=2)
            np.testing.assert_allclose(
                streamed, x @ expected_weights.T, atol=1e-4, rtol=1e-3
            )
            native = self.backend.native_bf16_from_bytes(bits.tobytes(), (rows, cols))
            vector = self.mx.array(x)
            baseline = self.backend.cached_bf16_matmul(native, vector)
            fused = self.backend.fused_bf16_gemv(native.view(self.mx.uint16), vector)
            self.mx.eval(baseline, fused)
            np.testing.assert_allclose(
                np.array(fused), np.array(baseline), atol=1e-4, rtol=1e-3
            )

    def test_q4_signed_nibble_and_gpu_expert_chain(self):
        rows = cols = 32
        rng = np.random.default_rng(84)
        signed = rng.integers(-8, 8, (rows, cols), dtype=np.int32)
        unsigned = (signed + 8).astype(np.uint32).reshape(rows, cols // 8, 8)
        packed = np.bitwise_or.reduce(
            unsigned << (np.arange(8, dtype=np.uint32) * 4), axis=-1
        )
        scale = np.full((rows, cols // 32), 0.125, dtype=np.float32)
        scale_bits = (scale.view(np.uint32) >> 16).astype(np.uint16)
        entries = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = "expert." + projection
            entries[prefix + ".weight_shape"] = (
                "I32",
                [2],
                struct.pack("<ii", rows, cols),
            )
            entries[prefix + ".weight_packed"] = (
                "I32",
                [rows, cols // 8],
                packed.tobytes(),
            )
            entries[prefix + ".weight_scale"] = (
                "BF16",
                [rows, cols // 32],
                scale_bits.tobytes(),
            )
        reader = ByteReader(entries)
        x = rng.normal(0, 0.05, (1, cols)).astype(np.float32)
        weight = signed.astype(np.float32) * np.repeat(scale, 32, axis=1)
        actual = self.backend.quantized_linear(
            reader, "expert.gate_proj", x, block_rows=7
        )
        gate = x @ weight.T
        np.testing.assert_allclose(actual, gate, atol=1e-5, rtol=1e-4)
        expected = ((gate / (1 + np.exp(-gate))) * gate) @ weight.T
        chain = self.backend.routed_expert(reader, "expert", x, block_rows=7)
        np.testing.assert_allclose(chain, expected, atol=1e-5, rtol=1e-4)

    def test_device_cache_reuse_identity_and_release(self):
        bits = np.full((16, 32), 0x3F80, dtype=np.uint16)
        x = np.full((1, 32), 0.125, dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix="native-bf16-test-") as directory:
            path = Path(directory) / "synthetic-data.bin"
            path.write_bytes(bits.tobytes())
            reader = ByteReader({"weight": ("BF16", [16, 32], bits.tobytes())})
            reader.base = 0
            reader.tensors["weight"]["data_offsets"] = [0, bits.nbytes]
            reader.file = path.open("rb")
            try:
                cache = self.backend.DeviceBF16Cache(4096)
                first = self.backend.cached_dense_bf16(
                    reader, "weight", x, cache, revision="synthetic-v1", block_rows=8
                )
                reads = cache.source_read_bytes
                second = self.backend.cached_dense_bf16(
                    reader, "weight", x, cache, revision="synthetic-v1", block_rows=8
                )
                np.testing.assert_array_equal(first, second)
                self.assertEqual(cache.source_read_bytes, reads)
                self.assertEqual(
                    cache.snapshot()["explicit_retained_host_weight_bytes"], 0
                )
                self.assertEqual(cache.retained_native_bytes, bits.nbytes)
                stat = path.stat()
                path.write_bytes(bytes(bits.nbytes))
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                third = self.backend.cached_dense_bf16(
                    reader, "weight", x, cache, revision="synthetic-v1", block_rows=8
                )
                self.assertFalse(np.any(third))
                self.assertGreater(cache.source_read_bytes, reads)
                cache.clear()
                self.assertEqual(cache.snapshot()["retained_backing_buffer_bytes"], 0)
            finally:
                reader.file.close()

    def test_reject_invalid_configuration_and_batch(self):
        with self.assertRaises(ValueError):
            self.backend.DeviceBF16Cache(5 * 1024**3 + 1)
        with self.assertRaises(ValueError):
            self.backend.configure(memory_limit_bytes=1024, cache_limit_bytes=2048)
        with self.assertRaises(ValueError):
            self.backend.fused_bf16_gemv(
                self.mx.array([[0]], dtype=self.mx.uint16),
                self.mx.zeros((2, 1), dtype=self.mx.float32),
            )


if __name__ == "__main__":
    unittest.main()
