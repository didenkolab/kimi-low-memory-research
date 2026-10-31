"""Experimental CPU FP32 Kimi layer API, extracted from the local prototype.

This is not an official Kimi runtime and has not been validated for full-model
inference after refactoring. No generation loop, download or repository Python
execution is provided. A caller must supply local native packed INT4/BF16 files
and an external resource guard before running real layers.
"""

import json
import math
from pathlib import Path

import numpy as np

from .reader import TensorReader, bf16

# Verified against the inspected Kimi-K2-Thinking config; other layouts fail closed.
EXPECTED_CONFIG = {
    "model_type": "kimi_k2",
    "hidden_size": 7168,
    "vocab_size": 163840,
    "num_hidden_layers": 61,
    "num_attention_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 384,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "intermediate_size": 18432,
    "first_k_dense_replace": 1,
    "routed_scaling_factor": 2.827,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "rms_norm_eps": 1e-5,
    "rope_theta": 50000.0,
    "hidden_act": "silu",
    "attention_bias": False,
    "moe_layer_freq": 1,
}
EXPECTED_ROPE = {
    "beta_fast": 1.0,
    "beta_slow": 1.0,
    "factor": 64.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn",
}


def validate_config(config):
    mismatches = [
        name
        for name, expected in EXPECTED_CONFIG.items()
        if config.get(name) != expected
    ]
    rope = config.get("rope_scaling", {})
    mismatches.extend(
        "rope_scaling." + name
        for name, expected in EXPECTED_ROPE.items()
        if rope.get(name) != expected
    )
    if mismatches:
        raise ValueError(
            "unsupported Kimi architecture fields: " + ", ".join(mismatches)
        )


def rms(x, w, epsilon):
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon)) * w


def silu(x):
    return x / (1 + np.exp(-x))


def rotate_yarn(pe, positions):
    dim, base, factor, original = 64, 50000.0, 64.0, 4096.0
    correction = dim * math.log(original / (2 * math.pi)) / (2 * math.log(base))
    low, high = max(math.floor(correction), 0), min(math.ceil(correction), dim - 1)
    if low == high:
        high += 0.001
    frequency = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    inv_freq = frequency / factor * ramp + frequency * (1 - ramp)
    phase = np.asarray(positions, dtype=np.float32)[:, None] * inv_freq[None, :]
    phase = np.concatenate([phase, phase], axis=-1)
    cos, sin = np.cos(phase)[:, None, :], np.sin(phase)[:, None, :]
    rearranged = (
        pe.reshape(len(positions), pe.shape[1], dim // 2, 2)
        .transpose(0, 1, 3, 2)
        .reshape(len(positions), pe.shape[1], dim)
    )
    rotated = np.concatenate(
        [-rearranged[..., dim // 2 :], rearranged[..., : dim // 2]], axis=-1
    )
    return rearranged * cos + rotated * sin


class ExperimentalKimi:
    """Layer, embedding and final-logit operations for one explicit Kimi layout.

    Expanded KV is (k[T,64,128], kpe[T,1,64], v[T,64,128]), FP32.
    context_limit defaults to 64; raising it changes a bound, not validation of
    longer inference. Latent RMS epsilon defaults to the historical prototype's
    1e-6, while layer RMS uses config 1e-5. This choice is explicit because it is
    not evidence of equivalence to the publisher's BF16 implementation.

    reader_factory(path, prefix=..., block_rows=...) permits explicit adapters;
    no GPU backend is selected or installed automatically.
    """

    hidden_size, vocab_size, layers = 7168, 163840, 61

    def __init__(
        self,
        weights_dir,
        *,
        context_limit=64,
        block_rows=64,
        latent_norm_epsilon=1e-6,
        reader_factory=TensorReader,
    ):
        self.weights_dir = Path(weights_dir).resolve(strict=True)
        if not self.weights_dir.is_dir():
            raise ValueError("weights_dir must be a local directory")
        if type(context_limit) is not int or not 1 <= context_limit <= 512:
            raise ValueError("experimental context bound must be 1..512")
        if type(block_rows) is not int or not 1 <= block_rows <= 512:
            raise ValueError("block_rows must be 1..512")
        if not math.isfinite(latent_norm_epsilon) or latent_norm_epsilon <= 0:
            raise ValueError("positive finite latent RMS epsilon required")
        if not callable(reader_factory):
            raise ValueError("reader_factory must be callable")
        self.context_limit, self.block_rows = context_limit, block_rows
        self.latent_norm_epsilon, self.reader_factory = (
            float(latent_norm_epsilon),
            reader_factory,
        )

        def local_json(name, limit):
            path = self.weights_dir / name
            if not path.is_file() or path.stat().st_size > limit:
                raise ValueError("missing or oversized local model metadata: " + name)
            return json.loads(path.read_text(encoding="utf-8"))

        self.config = local_json("config.json", 1024**2)
        validate_config(self.config)
        index = local_json("model.safetensors.index.json", 64 * 1024**2)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight_map missing from safetensors index")
        paths = {}
        for name, relative in weight_map.items():
            if (
                not isinstance(name, str)
                or not isinstance(relative, str)
                or Path(relative).is_absolute()
            ):
                raise ValueError("invalid tensor index entry")
            path = (self.weights_dir / relative).resolve()
            if (
                not path.is_relative_to(self.weights_dir)
                or path.suffix != ".safetensors"
            ):
                raise ValueError(
                    "shard path escapes weights_dir or has wrong extension"
                )
            paths[name] = path
        self._layer_paths = []
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            shard = paths.get(prefix + "input_layernorm.weight")
            members = {path for name, path in paths.items() if name.startswith(prefix)}
            if shard is None or members != {shard}:
                raise ValueError("this reference requires each layer within one shard")
            self._layer_paths.append(shard)
        shared_names = (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        )
        shared_paths = {paths.get(name) for name in shared_names}
        if None in shared_paths or len(shared_paths) != 1:
            raise ValueError("embedding, output norm and head must share one shard")
        self._head_path = next(iter(shared_paths))

    def open_reader(self, layer=None):
        if layer is None:
            path, prefix = self._head_path, ""
        else:
            if type(layer) is not int or not 0 <= layer < self.layers:
                raise ValueError("invalid layer index")
            path, prefix = self._layer_paths[layer], f"model.layers.{layer}."
        return self.reader_factory(path, prefix=prefix, block_rows=self.block_rows)

    def forward_layer(self, layer, x, cache=None):
        """x=[new_tokens,7168]; cache=(k_nope,k_pe,values), sequence-major."""
        if type(layer) is not int or not 0 <= layer < self.layers:
            raise ValueError("invalid layer index")
        if (
            type(x) is not np.ndarray
            or x.dtype != np.float32
            or x.ndim != 2
            or x.shape[1] != self.hidden_size
            or not len(x)
            or not np.isfinite(x).all()
        ):
            raise ValueError("expected finite FP32 hidden input")
        if cache is None:
            prev = 0
        else:
            if len(cache) != 3 or len(set(len(a) for a in cache)) != 1:
                raise ValueError("bad cache")
            prev = len(cache[0])
            for array, shape in zip(
                cache, ((prev, 64, 128), (prev, 1, 64), (prev, 64, 128))
            ):
                if (
                    type(array) is not np.ndarray
                    or array.dtype != np.float32
                    or array.shape != shape
                    or not np.isfinite(array).all()
                ):
                    raise ValueError("invalid expanded FP32 KV layout")
        if prev + len(x) > self.context_limit:
            raise ValueError("max context exceeded")
        positions = np.arange(prev, prev + len(x))
        r = self.open_reader(layer)
        try:
            h = rms(x, r.small("input_layernorm.weight"), 1e-5)
            qa = r.dense("self_attn.q_a_proj.weight", h)
            qa = rms(
                qa, r.small("self_attn.q_a_layernorm.weight"), self.latent_norm_epsilon
            )
            q = r.dense("self_attn.q_b_proj.weight", qa).reshape(len(x), 64, 192)
            kva = r.dense("self_attn.kv_a_proj_with_mqa.weight", h)
            knew_pe = rotate_yarn(kva[:, 512:].reshape(len(x), 1, 64), positions)
            qpe = rotate_yarn(q[:, :, 128:], positions)
            latent = rms(
                kva[:, :512],
                r.small("self_attn.kv_a_layernorm.weight"),
                self.latent_norm_epsilon,
            )
            kv = r.dense("self_attn.kv_b_proj.weight", latent).reshape(len(x), 64, 256)
            knew, vnew = kv[:, :, :128].copy(), kv[:, :, 128:].copy()
            if cache is None:
                k, kp, v = knew, knew_pe, vnew
            else:
                k, kp, v = [
                    np.concatenate((old, new), axis=0)
                    for old, new in zip(cache, (knew, knew_pe, vnew))
                ]
            scale = (192**-0.5) * (1 + 0.1 * math.log(64)) ** 2
            attn = np.empty((len(x), 64, 128), dtype=np.float32)
            for t in range(len(x)):
                limit = prev + t + 1
                scores = (
                    np.einsum("hd,thd->ht", q[t, :, :128], k[:limit])
                    + np.einsum(
                        "hd,thd->ht",
                        qpe[t],
                        np.broadcast_to(kp[:limit], (limit, 64, 64)),
                    )
                ) * scale
                scores -= scores.max(axis=-1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(axis=-1, keepdims=True)
                attn[t] = np.einsum("ht,thd->hd", probs, v[:limit])
            a = r.dense("self_attn.o_proj.weight", attn.reshape(len(x), 8192))
            residual = x + a
            h = rms(residual, r.small("post_attention_layernorm.weight"), 1e-5)
            if layer == 0:
                gate = r.dense("mlp.gate_proj.weight", h)
                up = r.dense("mlp.up_proj.weight", h)
                feedforward = r.dense("mlp.down_proj.weight", silu(gate) * up)
                ids = np.empty((len(x), 0), dtype=np.int64)
            else:
                scores = 1 / (1 + np.exp(-r.dense("mlp.gate.weight", h)))
                choices = scores + r.small("mlp.gate.e_score_correction_bias")
                ids = np.argsort(-choices, axis=1)[:, :8]
                weights = np.take_along_axis(scores, ids, axis=1)
                weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-20) * 2.827
                routed = np.zeros((len(x), self.hidden_size), dtype=np.float32)
                for expert in np.unique(ids):
                    tokens, slots = np.where(ids == expert)
                    inputs = h[tokens]
                    prefix = f"mlp.experts.{int(expert)}"
                    gate = r.quant_linear(prefix + ".gate_proj", inputs)
                    up = r.quant_linear(prefix + ".up_proj", inputs)
                    out = r.quant_linear(prefix + ".down_proj", silu(gate) * up)
                    routed[tokens] += out * weights[tokens, slots, None]
                gate = r.dense("mlp.shared_experts.gate_proj.weight", h)
                up = r.dense("mlp.shared_experts.up_proj.weight", h)
                shared = r.dense("mlp.shared_experts.down_proj.weight", silu(gate) * up)
                feedforward = routed + shared
            out = residual + feedforward
            if not np.isfinite(out).all():
                raise ValueError(f"nonfinite layer {layer}")
            return (
                out,
                (k, kp, v),
                {"bytes_read": r.bytes_read, "expert_ids": ids.tolist()},
            )
        finally:
            r.close()

    def embeddings(self, ids):
        if (
            not isinstance(ids, (list, tuple))
            or not ids
            or any(type(i) is not int or not 0 <= i < self.vocab_size for i in ids)
        ):
            raise ValueError("invalid input token IDs")
        if len(ids) > self.context_limit:
            raise ValueError("embedding batch exceeds context bound")
        reader = self.open_reader()
        try:
            meta = reader.tensors["model.embed_tokens.weight"]
            if meta["dtype"] != "BF16" or tuple(meta["shape"]) != (
                self.vocab_size,
                self.hidden_size,
            ):
                raise ValueError("unexpected embedding layout")
            values = np.stack(
                [
                    bf16(
                        reader.read(
                            "model.embed_tokens.weight",
                            i * self.hidden_size * 2,
                            self.hidden_size * 2,
                        )
                    )
                    for i in ids
                ]
            )
            if not np.isfinite(values).all():
                raise ValueError("nonfinite embedding")
            return values, reader.bytes_read
        finally:
            reader.close()

    def logits_last(self, x):
        if (
            type(x) is not np.ndarray
            or x.dtype != np.float32
            or x.ndim != 2
            or x.shape[1] != self.hidden_size
            or not len(x)
            or not np.isfinite(x).all()
        ):
            raise ValueError("expected finite FP32 hidden input")
        reader = self.open_reader()
        try:
            hidden = rms(x[-1:], reader.small("model.norm.weight"), 1e-5)
            scores = reader.dense("lm_head.weight", hidden)[0]
            if len(scores) != self.vocab_size or not np.isfinite(scores).all():
                raise ValueError("bad logits")
            return int(np.argmax(scores)), reader.bytes_read
        finally:
            reader.close()
