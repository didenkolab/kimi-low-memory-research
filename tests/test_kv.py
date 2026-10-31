"""Small synthetic storage tests. No weight files, MLX, GPU or model imports."""

import gc
import unittest
import weakref

import numpy as np

from kimi_low_memory.kv import KVLayout, KVStateError, ResidentKV

TINY = KVLayout(heads=4, width=8, rope_heads=1, rope_width=4)
MIB = 1024**2


def arrays(layout, positions, value):
    return tuple(
        np.full(shape, value + i, dtype=np.float32)
        for i, shape in enumerate(layout.shapes(positions))
    )


def append(old, layout, count, value):
    extra = arrays(layout, count, value)
    return (
        extra
        if old is None
        else tuple(np.concatenate((a, b)) for a, b in zip(old, extra))
    )


def store(**kwargs):
    params = dict(
        layout=TINY,
        layers=3,
        max_context=4,
        max_retained_bytes=100_000,
        max_transient_bytes=110_000,
    )
    params.update(kwargs)
    return ResidentKV(**params)


def prefill(s, positions=1):
    s.begin_step(position=0, new_positions=positions)
    for layer in range(s.layers):
        s.advance_layer(layer, lambda old: arrays(s.layout, positions, layer))
    s.finish_step()


class Tests(unittest.TestCase):
    def test_kimi_layout_budget(self):
        layout = KVLayout()
        self.assertEqual(layout.shapes(22), ((22, 64, 128), (22, 1, 64), (22, 64, 128)))
        self.assertEqual(61 * layout.bytes_per_position, 4_013_312)
        self.assertEqual(61 * 64 * layout.bytes_per_position, 256_851_968)

    def test_61_layer_sequential_generation_matches_list_reference(self):
        s = store(layers=61)
        reference = [None] * 61
        # <128 KiB/store; reference plus transient allocations remain <1 MiB.
        for position, count in ((0, 2), (2, 1), (3, 1)):
            s.begin_step(position=position, new_positions=count)
            for layer in range(61):
                value = 100 * position + layer
                reference[layer] = append(reference[layer], TINY, count, value)
                s.advance_layer(layer, lambda old: append(old, TINY, count, value))
            s.finish_step()
            self.assertEqual(s.position, position + count)
            for layer in range(61):
                for actual, expected in zip(s.committed_layer(layer), reference[layer]):
                    np.testing.assert_array_equal(actual, expected)
                    self.assertFalse(actual.flags.writeable)
            self.assertEqual(
                s.retained_payload_bytes,
                61 * (position + count) * TINY.bytes_per_position,
            )
        self.assertEqual(s.completed_steps, 3)
        self.assertEqual(s.peak_transient_bytes, (61 * 4 + 3) * TINY.bytes_per_position)

    def test_actual_shapes_one_layer(self):
        s = store(
            layers=1, layout=KVLayout(), max_retained_bytes=MIB, max_transient_bytes=MIB
        )
        prefill(s, 2)
        self.assertEqual(s.retained_payload_bytes, 131_584)

    def test_old_layer_released_after_replacement(self):
        s = store(layers=1)
        prefill(s)
        old_ref = weakref.ref(s.committed_layer(0)[0])
        s.begin_step(position=1)
        s.advance_layer(0, lambda old: append(old, TINY, 1, 9))
        gc.collect()
        self.assertIsNone(old_ref())
        s.finish_step()

    def test_duplicate_backing_owner_is_counted_once(self):
        s = store(layers=1)
        s.begin_step(position=0)

        def make(_):
            base = np.zeros(TINY.bytes_per_position // 4, np.float32)
            n = TINY.heads * TINY.width
            r = TINY.rope_heads * TINY.rope_width
            # k/v share one owner and overlap intentionally; neither counted twice.
            k = base[:n].reshape(1, TINY.heads, TINY.width)
            kp = base[n : n + r].reshape(1, TINY.rope_heads, TINY.rope_width)
            return k, kp, k

        s.advance_layer(0, make)
        s.finish_step()
        self.assertEqual(s.retained_payload_bytes, TINY.bytes_per_position)

    def test_wrong_layer_and_position_invalidate(self):
        s = store()
        s.begin_step(position=0)
        with self.assertRaisesRegex(KVStateError, "sequence"):
            s.advance_layer(1, lambda _: arrays(TINY, 1, 0))
        self.assertFalse(s.valid)
        with self.assertRaises(KVStateError):
            s.begin_step(position=0)
        s = store()
        with self.assertRaisesRegex(KVStateError, "position"):
            s.begin_step(position=1)
        self.assertFalse(s.valid)

    def test_dtype_shape_nan_and_prefix_rejected(self):
        for kind in ("dtype", "shape", "nan", "prefix"):
            with self.subTest(kind=kind):
                s = store(layers=1)
                prefill(s)
                s.begin_step(position=1)

                def broken(old):
                    new = list(append(old, TINY, 1, 2))
                    if kind == "dtype":
                        new[0] = new[0].astype(np.float64)
                    if kind == "shape":
                        new[0] = new[0][:1]
                    if kind == "nan":
                        new[0][1, 0, 0] = np.nan
                    if kind == "prefix":
                        new[0][0, 0, 0] = 123
                    return new

                with self.assertRaises(ValueError):
                    s.advance_layer(0, broken)
                self.assertFalse(s.valid)
                with self.assertRaises(KVStateError):
                    s.finish_step()

    def test_partial_failure_cannot_continue_and_has_no_rollback(self):
        s = store()
        prefill(s)
        s.begin_step(position=1)
        s.advance_layer(0, lambda old: append(old, TINY, 1, 5))

        def fail(_):
            raise ArithmeticError("synthetic failure")

        with self.assertRaises(ArithmeticError):
            s.advance_layer(1, fail)
        self.assertFalse(s.valid)
        self.assertEqual(s.stats()["next_layer"], 1)
        self.assertEqual(s.retained_payload_bytes, 4 * TINY.bytes_per_position)
        with self.assertRaises(KVStateError):
            s.committed_layer(0)
        s.close()
        self.assertEqual(s.retained_payload_bytes, 0)

    def test_incomplete_step_and_context_limits(self):
        s = store()
        s.begin_step(position=0)
        with self.assertRaises(KVStateError):
            s.finish_step()
        s = store(layers=1, max_context=2)
        prefill(s, 2)
        with self.assertRaisesRegex(KVStateError, "context"):
            s.begin_step(position=2)
        s = store(layers=1)
        prefill(s)
        with self.assertRaisesRegex(KVStateError, "one position"):
            s.begin_step(position=1, new_positions=2)

    def test_transient_budget_is_distinct_and_checked_before_builder(self):
        per = TINY.bytes_per_position
        s = store(layers=2, max_retained_bytes=4 * per, max_transient_bytes=4 * per)
        prefill(s)
        with self.assertRaisesRegex(KVStateError, "transient"):
            s.begin_step(position=1)  # retained=4*per; old+new handoff=5*per
        self.assertEqual(s.retained_payload_bytes, 2 * per)

    def test_oversized_base_and_nonowned_buffers_rejected(self):
        for kind in ("oversized", "foreign"):
            s = store(layers=1, max_transient_bytes=2048)
            s.begin_step(position=0)

            def bad(_):
                new = list(arrays(TINY, 1, 0))
                if kind == "oversized":
                    base = np.zeros(1024, np.float32)
                    new[0] = base[:32].reshape(1, 4, 8)
                else:
                    new[0] = np.frombuffer(bytes(128), np.float32).reshape(1, 4, 8)
                return new

            with self.assertRaises(ValueError):
                s.advance_layer(0, bad)
            self.assertFalse(s.valid)

    def test_builder_abort_and_reentry_fail_closed(self):
        for kind in ("abort", "reentry"):
            s = store(layers=1)
            s.begin_step(position=0)

            def bad(_):
                if kind == "abort":
                    s.abort("external failure")
                else:
                    try:
                        s.advance_layer(0, lambda _: arrays(TINY, 1, 1))
                    except KVStateError:
                        pass
                return arrays(TINY, 1, 0)

            with self.assertRaises(KVStateError):
                s.advance_layer(0, bad)
            self.assertFalse(s.valid)


if __name__ == "__main__":
    unittest.main()
