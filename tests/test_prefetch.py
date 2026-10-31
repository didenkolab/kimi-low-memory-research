import hashlib
import os
from pathlib import Path
import tempfile
import threading
import unittest

from kimi_low_memory.prefetch import SelectedExpertPrefetch


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw = {n: bytes([n]) * 65536 for n in (1, 2, 3, 4)}
        for key, data in self.raw.items():
            (self.root / str(key)).write_bytes(data)

    def load(self, n):
        with (self.root / str(n)).open("rb") as f:
            data = os.pread(f.fileno(), 65536, 0)
        if len(data) != 65536:
            raise EOFError(n)
        return data

    def test_exact_selected_order_and_values(self):
        ids = [3, 1, 4, 2]
        with SelectedExpertPrefetch(ids, self.load, 65536, 196608) as reader:
            out = [(key, hashlib.sha256(data).digest()) for key, data in reader]
        self.assertEqual(out, [(i, hashlib.sha256(self.raw[i]).digest()) for i in ids])

    def test_prefetch_runs_while_first_item_is_in_use(self):
        ready = threading.Event()

        def load(n):
            data = self.load(n)
            if n == 2:
                ready.set()
            return data

        with SelectedExpertPrefetch([1, 2], load, 65536, 196608) as reader:
            items = iter(reader)
            self.assertEqual(next(items), (1, self.raw[1]))
            self.assertTrue(ready.wait(timeout=2), "next read did not overlap consumer")
            self.assertEqual(next(items), (2, self.raw[2]))
            with self.assertRaises(StopIteration):
                next(items)

    def test_loader_error_stops_future_work(self):
        seen = []

        def load(n):
            seen.append(n)
            if n == 2:
                raise OSError("injected read error")
            return self.load(n)

        with self.assertRaises(OSError):
            with SelectedExpertPrefetch([1, 2, 3], load, 65536, 196608) as reader:
                for _ in reader:
                    pass
        self.assertEqual(seen, [1, 2])

    def test_early_exit_releases_executor_without_starting_rest(self):
        seen = []

        def load(n):
            seen.append(n)
            return self.load(n)

        with SelectedExpertPrefetch([1, 2, 3], load, 65536, 196608) as reader:
            for _ in reader:
                break
        self.assertNotIn(3, seen)
        self.assertIsNone(reader.executor)

    def test_noninteger_and_nan_limits_rejected(self):
        for expert, total in (
            (float("nan"), 1),
            (1, float("nan")),
            (True, 3),
            (1, 3.0),
        ):
            with self.assertRaises(ValueError):
                SelectedExpertPrefetch([1], lambda _: bytes(16), expert, total)

    def test_bounds_and_context(self):
        with self.assertRaises(ValueError):
            SelectedExpertPrefetch([1], self.load, 65536, 196607)
        with self.assertRaises(ValueError):
            with SelectedExpertPrefetch([1], self.load, 10, 30) as reader:
                list(reader)
        reader = SelectedExpertPrefetch([], self.load, 1, 3)
        with self.assertRaises(RuntimeError):
            iter(reader)
        with reader:
            self.assertEqual(list(reader), [])
        with self.assertRaises(RuntimeError):
            reader.__enter__()


if __name__ == "__main__":
    unittest.main()
