import unittest
import tempfile

from kimi_low_memory.caches import (
    ExpertCache,
    ExpertKey,
    file_identity,
    read_tensors,
    slice_tensor,
)


def key(layer, expert, revision="r1", identity=(1, 2, 100, 10)):
    return ExpertKey(revision, identity, layer, expert)


class CacheTests(unittest.TestCase):
    def test_hit_exact_slices_and_size(self):
        cache = ExpertCache(6)
        source = {"packed": b"abcd", "scale": b"ef"}
        first = cache.get(
            key(1, 0), {k: len(v) for k, v in source.items()}, source.__getitem__
        )
        self.assertEqual(slice_tensor(first, "packed", 1, 2), b"bc")
        self.assertEqual(slice_tensor(first, "scale", 1), b"f")
        with self.assertRaises(ValueError):
            slice_tensor(first, "packed", 3, 2)
        first.clear()  # A caller cannot mutate the cache's mapping.
        cache.get(
            key(1, 0),
            {k: len(v) for k, v in source.items()},
            lambda _: self.fail("unexpected load"),
        )
        self.assertEqual((cache.hits, cache.misses, cache.retained_bytes), (1, 1, 6))

    def test_lru_eviction_and_identity(self):
        cache = ExpertCache(2)
        read = lambda _: b"x"
        for k in (key(1, 0), key(2, 0), key(1, 0), key(3, 0)):
            cache.get(k, {"x": 1}, read)
        self.assertNotIn(key(2, 0), cache.entries)
        cache.get(key(1, 0, identity=(1, 2, 100, 11)), {"x": 1}, read)
        self.assertEqual(cache.misses, 4)
        self.assertLessEqual(cache.retained_bytes, cache.budget)

    def test_invalid_loader_and_metadata(self):
        cache = ExpertCache(4)
        with self.assertRaises(ValueError):
            cache.get(key(0, 0), {"a": 2, "b": 2}, lambda name: b"x")
        self.assertEqual(cache.retained_bytes, 0)
        self.assertFalse(cache.entries)
        cache.get(key(0, 0), {"a": 2}, lambda _: b"xx")
        with self.assertRaises(ValueError):
            cache.get(key(0, 0), {"a": 1}, lambda _: b"x")

    def test_oversize_is_not_retained(self):
        with self.assertRaises(ValueError):
            ExpertCache(8 * 1024**3 + 1)
        cache = ExpertCache(1)
        cache.get(key(0, 0), {"x": 2}, lambda _: b"xx")
        self.assertEqual(cache.retained_bytes, 0)
        self.assertFalse(cache.entries)

    def test_scan_pollution_and_selective_admission(self):
        route = [key(layer, expert) for layer in range(3) for expert in range(2)]
        lru = ExpertCache(3)
        selective = ExpertCache(3, admit=lambda k: k.expert == 0)
        for _ in range(2):
            for k in route:
                for cache in (lru, selective):
                    cache.get(k, {"x": 1}, lambda _: b"x")
        self.assertEqual(lru.hits, 0)
        self.assertEqual(selective.hits, 3)
        self.assertEqual(selective.retained_bytes, 3)

    def test_exact_offsets_and_shard_identity(self):
        with tempfile.TemporaryFile() as f:
            f.write(b"header0123456789")
            f.flush()
            k = key(0, 0, identity=file_identity(f.fileno()))
            tensors = read_tensors(
                f.fileno(), k, 6, {"packed": (1, 5), "scale": (7, 10)}
            )
            self.assertEqual(tensors, {"packed": b"1234", "scale": b"789"})
            with self.assertRaises(ValueError):
                read_tensors(f.fileno(), k, 6, {"bad": (0, 11)})
            bad_key = key(0, 0, identity=(1, 2, 100, 11))
            with self.assertRaises(ValueError):
                read_tensors(f.fileno(), bad_key, 6, {"packed": (1, 5)})


if __name__ == "__main__":
    unittest.main()
