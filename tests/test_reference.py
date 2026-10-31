"""Synthetic reference tests. No downloaded model, GPU, network or real weights."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kimi_low_memory.reference.model import (
    ExperimentalKimi,
    EXPECTED_CONFIG,
    EXPECTED_ROPE,
    validate_config,
)
from kimi_low_memory.reference.reader import TensorReader, file_identity
from kimi_low_memory.metadata import freeze
from kimi_low_memory.reference.safe_tokenizer import read_bpe, load_tokenizer


def bf16_bytes(values):
    return (
        (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16)
        .astype("<u2")
        .tobytes()
    )


def write_tensors(path, tensors):
    header, payload = {}, bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def synthetic_config(directory):
    config = dict(EXPECTED_CONFIG, rope_scaling=dict(EXPECTED_ROPE))
    (directory / "config.json").write_text(json.dumps(config))
    names = [f"model.layers.{i}.input_layernorm.weight" for i in range(61)]
    names += ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "synthetic.safetensors" for name in names}})
    )
    return config


class SyntheticReader:
    """Deterministic projection outputs at actual architecture shapes; no weights.

    Used to isolate attention, routing, residuals and cache mechanics. This is
    not a learned model and cannot establish real Kimi numerical correctness.
    """

    def __init__(self, path=None, prefix="", block_rows=64):
        self.file, self.bytes_read = self, 0
        self.tensors = {
            "model.embed_tokens.weight": {"dtype": "BF16", "shape": [163840, 7168]}
        }

    def close(self):
        pass

    def small(self, name):
        if name == "mlp.gate.e_score_correction_bias":
            return np.arange(384, dtype=np.float32) * np.float32(0.00001)
        width = (
            1536
            if name == "self_attn.q_a_layernorm.weight"
            else 512
            if name == "self_attn.kv_a_layernorm.weight"
            else 7168
        )
        return np.ones(width, dtype=np.float32)

    def read(self, name, start=0, size=None):
        if name != "model.embed_tokens.weight" or size != 7168 * 2:
            raise ValueError("unexpected synthetic read")
        token = start // (7168 * 2)
        return bf16_bytes(
            (np.arange(7168) % 7 + token % 5).astype(np.float32) * np.float32(0.001)
        )

    @staticmethod
    def projection(name, x, rows):
        base = np.mean(x, axis=1, keepdims=True) * np.float32(0.03125)
        pattern = (np.arange(rows) % 17 - 8).astype(np.float32) * np.float32(0.0005)
        phase = np.float32(sum(name.encode()) % 23) * np.float32(0.0001)
        return np.asarray(base + pattern[None, :] + phase, dtype=np.float32)

    def dense(self, name, x):
        dimensions = {
            "self_attn.q_a_proj.weight": 1536,
            "self_attn.q_b_proj.weight": 64 * 192,
            "self_attn.kv_a_proj_with_mqa.weight": 576,
            "self_attn.kv_b_proj.weight": 64 * 256,
            "self_attn.o_proj.weight": 7168,
            "mlp.gate.weight": 384,
            "mlp.gate_proj.weight": 18432,
            "mlp.up_proj.weight": 18432,
            "mlp.down_proj.weight": 7168,
            "mlp.shared_experts.gate_proj.weight": 2048,
            "mlp.shared_experts.up_proj.weight": 2048,
            "mlp.shared_experts.down_proj.weight": 7168,
            "lm_head.weight": 163840,
        }
        return self.projection(name, x, dimensions[name])

    def quant_linear(self, prefix, x):
        return self.projection(
            prefix, x, 7168 if prefix.endswith("down_proj") else 2048
        )


def synthetic_inputs():
    rng = np.random.default_rng(20261031)
    return [
        rng.standard_normal((n, 7168)).astype(np.float32) * np.float32(0.1)
        for n in (2, 1, 1)
    ]


def sample_result(hidden, cache, info):
    return {
        "hidden_first8": hidden.ravel()[:8].tolist(),
        "hidden_sum_fp64": float(np.sum(hidden, dtype=np.float64)),
        "kv_first8": [a.ravel()[:8].tolist() for a in cache],
        "kv_sum_fp64": [float(np.sum(a, dtype=np.float64)) for a in cache],
        "expert_ids": info["expert_ids"],
    }


class ReaderTests(unittest.TestCase):
    def test_overlap_rejected_by_both_constructors_but_empty_ranges_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ranges.safetensors"
            metadata = {
                "a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
                "b": {"dtype": "BF16", "shape": [2], "data_offsets": [2, 6]},
            }
            raw = json.dumps(metadata).encode()
            path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(6))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                TensorReader(path)
            file = path.open("rb")
            expected = file_identity(os.fstat(file.fileno()))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                TensorReader.from_metadata(
                    file, 8 + len(raw), metadata, expected_identity=expected
                )
            self.assertTrue(file.closed)
            metadata["b"]["data_offsets"] = [4, 8]
            metadata["empty"] = {"dtype": "BF16", "shape": [0], "data_offsets": [2, 2]}
            raw = json.dumps(metadata).encode()
            path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(8))
            with TensorReader(path) as reader:
                self.assertEqual(reader.read("empty"), b"")
            file = path.open("rb")
            with TensorReader.from_metadata(
                file,
                8 + len(raw),
                metadata,
                expected_identity=file_identity(os.fstat(file.fileno())),
            ) as reader:
                self.assertEqual(reader.read("b"), bytes(4))

    def test_validated_cached_factory_initializes_limits_without_weight_read(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cached.safetensors"
            values = np.array([[1, 2], [3, 4], [5, 6]], np.float32)
            write_tensors(path, [("p.w", "BF16", values.shape, bf16_bytes(values))])
            with TensorReader(path, "p.") as parsed:
                base, expected = parsed.base, parsed.file_identity
                immutable, _ = freeze(
                    {
                        "w": {
                            "dtype": "BF16",
                            "shape": list(values.shape),
                            "data_offsets": [0, values.nbytes // 2],
                        }
                    }
                )
            actual_pread = os.pread
            reads = []

            def measured_pread(fd, size, offset):
                reads.append((size, offset))
                return actual_pread(fd, size, offset)

            with patch(
                "kimi_low_memory.reference.reader.os.pread", side_effect=measured_pread
            ):
                reader = TensorReader.from_metadata(
                    path.open("rb"),
                    base,
                    immutable,
                    expected_identity=expected,
                    block_rows=2,
                    max_read_bytes=8,
                    max_small_bytes=4,
                )
            self.assertEqual(reads, [(8, 0)])
            with reader:
                self.assertEqual(
                    (reader.block_rows, reader.max_read_bytes, reader.max_small_bytes),
                    (2, 8, 4),
                )
                self.assertIs(reader.tensors["w"], immutable["w"])
                np.testing.assert_array_equal(
                    reader.dense("w", np.ones((1, 2), np.float32)), [[3, 7, 11]]
                )
                with self.assertRaises(ValueError):
                    reader.small("w")
                with self.assertRaises(ValueError):
                    reader.read("w")

    def test_cached_factory_rejects_bad_metadata_base_and_identity(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cached.safetensors"
            write_tensors(path, [("w", "BF16", (2,), bf16_bytes([1, 2]))])
            with TensorReader(path) as parsed:
                base, expected = parsed.base, parsed.file_identity
            good = {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
            cases = [
                (
                    {"w": {"dtype": "UNKNOWN", "shape": [2], "data_offsets": [0, 4]}},
                    base,
                    expected,
                ),
                (
                    {"w": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]}},
                    base,
                    expected,
                ),
                (
                    {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [-1, 3]}},
                    base,
                    expected,
                ),
                ({"__metadata__": {}}, base, expected),
                (good, base + 1, expected),
                (good, base, (*expected[:-1], expected[-1] - 1)),
            ]
            for tensors, bad_base, identity in cases:
                with self.subTest(tensors=tensors, base=bad_base):
                    file = path.open("rb")
                    with self.assertRaises(ValueError):
                        TensorReader.from_metadata(
                            file, bad_base, tensors, expected_identity=identity
                        )
                    self.assertTrue(file.closed)
            with TensorReader(path) as reader:
                with path.open("r+b") as changed:
                    changed.seek(-1, 2)
                    changed.write(b"\x01")
                with self.assertRaisesRegex(ValueError, "changed"):
                    reader.read("w")

    def test_bf16_dense_and_budgeted_reads(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "matrix.safetensors"
            weights = np.array([[1, -2, 3], [4, 5, -0.5], [-6, 0, 8]], dtype=np.float32)
            write_tensors(
                path, [("layer.w", "BF16", weights.shape, bf16_bytes(weights))]
            )
            x = np.array([[0.5, -1, 2], [1, 2, 3]], dtype=np.float32)
            with TensorReader(
                path, "layer.", block_rows=2, max_read_bytes=12
            ) as reader:
                np.testing.assert_array_equal(reader.dense("w", x), x @ weights.T)
                self.assertEqual(reader.bytes_read, 18)
                with self.assertRaises(ValueError):
                    reader.read("w")
                with self.assertRaises(ValueError):
                    reader.read("w", -1, 2)
            self.assertTrue(reader.file.closed)

    def test_native_int4_groups_and_partial_final_block(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "q4.safetensors"
            rows, columns = 3, 64
            signed = (
                (np.arange(rows * columns).reshape(rows, columns) * 7) % 16 - 8
            ).astype(np.int32)
            nibble = (signed + 8).astype(np.uint32).reshape(rows, columns // 8, 8)
            packed = np.zeros((rows, columns // 8), dtype="<u4")
            for i in range(8):
                packed |= nibble[:, :, i] << (4 * i)
            scales = np.array([[0.5, 1], [2, 0.25], [4, 0.125]], np.float32)
            write_tensors(
                path,
                [
                    ("q.weight_packed", "I32", packed.shape, packed.tobytes()),
                    ("q.weight_scale", "BF16", scales.shape, bf16_bytes(scales)),
                    (
                        "q.weight_shape",
                        "I32",
                        (2,),
                        np.array([rows, columns], dtype="<i4").tobytes(),
                    ),
                ],
            )
            x = (np.arange(2 * columns).reshape(2, columns) % 9).astype(np.float32)
            with TensorReader(path, block_rows=2) as reader:
                np.testing.assert_array_equal(
                    reader.quant_linear("q", x),
                    x @ (signed * np.repeat(scales, 32, axis=1)).T,
                )

    def test_rejects_truncated_and_out_of_bounds_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.safetensors"
            path.write_bytes((100).to_bytes(8, "little") + b"{}")
            with self.assertRaises(ValueError):
                TensorReader(path)
            encoded = json.dumps(
                {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
            ).encode()
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\x00\x00")
            with self.assertRaises(ValueError):
                TensorReader(path)


class TokenizerTests(unittest.TestCase):
    def test_local_bpe_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiktoken.model"
            path.write_bytes(b"YQ== 0\nYg== 1\n")
            self.assertEqual(read_bpe(path), {b"a": 0, b"b": 1})
            path.write_bytes(b"YQ== 0\nYQ== 1\n")
            with self.assertRaises(ValueError):
                read_bpe(path)
            path.write_bytes(b"YQ== 0\n")
            (Path(td) / "tokenizer_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "base vocabulary"):
                load_tokenizer(td)


class ModelTests(unittest.TestCase):
    def test_unsupported_activation_bias_and_moe_frequency_rejected(self):
        for name, value in (
            ("hidden_act", "gelu"),
            ("attention_bias", True),
            ("moe_layer_freq", 2),
        ):
            config = dict(EXPECTED_CONFIG, rope_scaling=dict(EXPECTED_ROPE))
            config[name] = value
            with self.assertRaisesRegex(ValueError, name):
                validate_config(config)

    def test_architecture_and_safe_index_paths(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            config = synthetic_config(directory)
            validate_config(config)
            bad = dict(config, num_attention_heads=32)
            with self.assertRaisesRegex(ValueError, "num_attention_heads"):
                validate_config(bad)
            path = directory / "model.safetensors.index.json"
            index = json.loads(path.read_text())
            index["weight_map"]["model.layers.0.input_layernorm.weight"] = (
                "../outside.safetensors"
            )
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "escapes"):
                ExperimentalKimi(directory)

    def test_layer_synthetic_golden_and_incremental_cache(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/reference-synthetic.json").read_text()
        )
        with tempfile.TemporaryDirectory() as td:
            synthetic_config(Path(td))
            model = ExperimentalKimi(td, reader_factory=SyntheticReader)
            for layer in (0, 1):
                cache = None
                for step, x in enumerate(synthetic_inputs()):
                    hidden, cache, info = model.forward_layer(layer, x, cache)
                    sample = sample_result(hidden, cache, info)
                    expected = fixture["layers"][str(layer)][step]
                    self.assertEqual(sample["expert_ids"], expected["expert_ids"])
                    for name in (
                        "hidden_first8",
                        "hidden_sum_fp64",
                        "kv_first8",
                        "kv_sum_fp64",
                    ):
                        np.testing.assert_allclose(
                            sample[name], expected[name], rtol=2e-6, atol=2e-6
                        )
                    self.assertEqual(cache[0].shape, (2 + step, 64, 128))
                    self.assertTrue(np.isfinite(hidden).all())

    def test_public_embeddings_and_head_contracts(self):
        with tempfile.TemporaryDirectory() as td:
            synthetic_config(Path(td))
            model = ExperimentalKimi(td, reader_factory=SyntheticReader)
            hidden, _ = model.embeddings([0, 1])
            self.assertEqual(hidden.shape, (2, 7168))
            token, _ = model.logits_last(hidden)
            self.assertIs(type(token), int)
            self.assertTrue(0 <= token < 163840)
            with self.assertRaises(ValueError):
                model.embeddings([True])
            with self.assertRaises(ValueError):
                model.forward_layer(1, np.zeros((1, 7168), np.float64))
            with self.assertRaises(ValueError):
                model.forward_layer(
                    1,
                    np.zeros((1, 7168), np.float32),
                    (np.zeros((1, 1), np.float32),) * 3,
                )


if __name__ == "__main__":
    unittest.main()
