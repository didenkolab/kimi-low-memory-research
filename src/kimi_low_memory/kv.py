"""Bounded expanded FP32 KV ownership for one sequential Kimi generation.

No model, disk, MLX or checkpoint I/O. This storage is independent of the model execution loop. The caller transfers ownership of arrays returned by its builder and
must not retain/mutate those arrays or borrowed old KV. Read-only NumPy flags
prevent accidental writes, not malicious aliases. Budgets cover retained array
backing allocations and old+new handoff, not arbitrary builder temporaries,
Python overhead, GPU buffers or process RSS. An external process guard is still
required. Failure invalidates the mixed generation; no rollback is attempted.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


class KVStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class KVLayout:
    heads: int = 64
    width: int = 128
    rope_heads: int = 1
    rope_width: int = 64

    def __post_init__(self):
        if any(
            type(x) is not int or x <= 0
            for x in (self.heads, self.width, self.rope_heads, self.rope_width)
        ):
            raise ValueError("layout dimensions must be positive integers")

    def shapes(self, positions):
        return (
            (positions, self.heads, self.width),
            (positions, self.rope_heads, self.rope_width),
            (positions, self.heads, self.width),
        )

    @property
    def bytes_per_position(self):
        return (2 * self.heads * self.width + self.rope_heads * self.rope_width) * 4


def _owners(arrays):
    """Count each complete owned allocation once, including oversized bases."""
    result = {}
    for array in arrays:
        if type(array) is not np.ndarray:
            raise ValueError("KV must contain plain NumPy arrays")
        owner = array
        while type(owner.base) is np.ndarray:
            owner = owner.base
        if not owner.flags.owndata or owner.base is not None:
            raise ValueError("foreign buffers and memory-mapped arrays are unsupported")
        result[id(owner)] = owner
    return result


class ResidentKV:
    """One committed position and, at most, one partially updated generation.

    begin_step() reserves a single layer handoff before calling any builder.
    advance_layer() visits layers strictly 0..layers-1. Its callback receives
    None for prefill or the old read-only (k, kp, v), and returns the new tuple.
    finish_step() succeeds only after every layer is replaced. Checkpointing may
    borrow completed layers via committed_layer(); no second KV copy is made.
    """

    def __init__(
        self,
        *,
        max_retained_bytes: int,
        max_transient_bytes: int,
        layers: int = 61,
        max_context: int = 64,
        layout: KVLayout = KVLayout(),
    ):
        for name, value in (
            ("layers", layers),
            ("max_context", max_context),
            ("max_retained_bytes", max_retained_bytes),
            ("max_transient_bytes", max_transient_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(layout, KVLayout):
            raise ValueError("layout must be KVLayout")
        self.layers, self.max_context, self.layout = layers, max_context, layout
        self.max_retained_bytes = max_retained_bytes
        self.max_transient_bytes = max_transient_bytes
        self._cache = [None] * layers
        self._position = 0
        self._target = None
        self._next_layer = 0
        self._invalid_reason = None
        self._building = False
        self.peak_transient_bytes = 0
        self.peak_retained_bytes = 0
        self.completed_steps = 0

    @property
    def valid(self):
        return self._invalid_reason is None

    @property
    def position(self):
        self._require_valid()
        return self._position

    def _require_valid(self):
        if not self.valid:
            raise KVStateError(f"KV store unusable: {self._invalid_reason}")

    def _fail(self, reason):
        if self._invalid_reason is None:
            self._invalid_reason = str(reason)
        raise KVStateError(str(reason))

    def _arrays(self):
        return [a for item in self._cache if item is not None for a in item]

    @property
    def retained_payload_bytes(self):
        """Unique owned backing bytes, not physical resident memory."""
        return sum(a.nbytes for a in _owners(self._arrays()).values())

    def begin_step(self, *, position: int, new_positions: int = 1):
        self._require_valid()
        if self._target is not None:
            self._fail("a step is already active")
        if type(position) is not int or position != self._position:
            self._fail("starting position mismatch")
        if type(new_positions) is not int or new_positions <= 0:
            self._fail("new_positions must be a positive integer")
        if position and new_positions != 1:
            self._fail("decode updates exactly one position")
        target = position + new_positions
        if target > self.max_context:
            self._fail("context limit exceeded")
        projected = self.layers * target * self.layout.bytes_per_position
        if projected > self.max_retained_bytes:
            self._fail("retained payload budget exceeded before step")
        # The final layer handoff has the largest store+new payload. This is a
        # lower bound on arbitrary builder storage, not a process memory cap.
        projected_transient = projected + position * self.layout.bytes_per_position
        if projected_transient > self.max_transient_bytes:
            self._fail("transient old+new budget exceeded before step")
        self._target, self._next_layer = target, 0

    def advance_layer(self, layer: int, builder: Callable):
        self._require_valid()
        if self._building:
            self._fail("reentrant builder is unsupported")
        if self._target is None:
            self._fail("no active step")
        if type(layer) is not int or layer != self._next_layer or layer >= self.layers:
            self._fail("layer sequence mismatch")
        if not callable(builder):
            self._fail("builder must be callable")
        minimum_new = self._target * self.layout.bytes_per_position
        if self.retained_payload_bytes + minimum_new > self.max_transient_bytes:
            self._fail("transient budget exceeded before builder")
        old = self._cache[layer]
        self._building = True
        try:
            new = builder(old)
            self._require_valid()
            if not isinstance(new, (tuple, list)) or len(new) != 3:
                raise ValueError("builder must return (k, kp, v)")
            new = tuple(new)
            for arr, shape in zip(new, self.layout.shapes(self._target)):
                if type(arr) is not np.ndarray or arr.dtype != np.dtype("float32"):
                    raise ValueError("KV dtype must be native float32")
                if arr.shape != shape or not arr.flags.c_contiguous:
                    raise ValueError("KV shape or C-contiguity mismatch")
            owners = _owners(self._arrays() + list(new))
            transient = sum(a.nbytes for a in owners.values())
            self.peak_transient_bytes = max(self.peak_transient_bytes, transient)
            if transient > self.max_transient_bytes:
                raise ValueError("actual transient backing allocation budget exceeded")
            for arr in new:
                if not np.isfinite(arr).all():
                    raise ValueError("nonfinite KV")
            if old is not None:
                for before, after in zip(old, new):
                    if not np.array_equal(before, after[: self._position]):
                        raise ValueError("old KV prefix changed")
            others = [
                a
                for i, item in enumerate(self._cache)
                if i != layer and item is not None
                for a in item
            ]
            retained = sum(a.nbytes for a in _owners(others + list(new)).values())
            if retained > self.max_retained_bytes:
                raise ValueError("actual retained backing allocation budget exceeded")
            for owner in _owners(new).values():
                owner.flags.writeable = False
            for arr in new:
                arr.flags.writeable = False
            self._cache[layer] = new
            self._next_layer += 1
            self.peak_retained_bytes = max(self.peak_retained_bytes, retained)
        except BaseException as exc:
            self._invalid_reason = f"layer {layer}: {type(exc).__name__}: {exc}"
            raise
        finally:
            self._building = False
            # Does not revoke references deliberately retained by caller code.
            del old

    def finish_step(self):
        self._require_valid()
        if self._target is None or self._next_layer != self.layers:
            self._fail("cannot finish an incomplete step")
        self._position, self._target = self._target, None
        self.completed_steps += 1

    def committed_layer(self, layer: int):
        """Borrow for checkpoint output; caller must promptly release references."""
        self._require_valid()
        if self._target is not None or not self._position:
            self._fail("only a completed generation can be borrowed")
        if type(layer) is not int or not 0 <= layer < self.layers:
            self._fail("invalid layer index")
        return self._cache[layer]

    def abort(self, reason="caller aborted"):
        """Invalidate on failures outside the callback, including logits failure."""
        if self._invalid_reason is None:
            self._invalid_reason = str(reason)

    def close(self):
        """Release owned references and permanently invalidate this store."""
        self.abort("closed")
        self._cache = [None] * self.layers

    def stats(self):
        return {
            "valid": self.valid,
            "invalid_reason": self._invalid_reason,
            "committed_position": self._position,
            "target_position": self._target,
            "next_layer": self._next_layer,
            "completed_steps": self.completed_steps,
            "retained_payload_bytes": self.retained_payload_bytes,
            "peak_retained_payload_bytes": self.peak_retained_bytes,
            "peak_transient_payload_bytes": self.peak_transient_bytes,
            "max_retained_bytes": self.max_retained_bytes,
            "max_transient_bytes": self.max_transient_bytes,
        }
