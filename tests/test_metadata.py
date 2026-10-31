import json
import os
from pathlib import Path
import tempfile
import unittest
from kimi_low_memory.metadata import MetadataCache


from kimi_low_memory.reference.reader import TensorReader as Reader


def write(path, payload=b"abcd"):
    data = {
        "model.layers.1.w": {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [0, len(payload)],
        }
    }
    raw = json.dumps(data).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)


class CacheTests(unittest.TestCase):
    def test_cached_reader_preserves_limits_and_rejects_malformed_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bounded"
            write(path)
            cache = MetadataCache()
            with cache.open(Reader, path, max_read_bytes=2) as reader:
                self.assertEqual(reader.read("model.layers.1.w", 0, 2), b"ab")
                with self.assertRaises(ValueError):
                    reader.read("model.layers.1.w")
            header = b'{"x":{"dtype":"U8","shape":[5],"data_offsets":[0,4]}}'
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"abcd")
            with self.assertRaises(ValueError):
                cache.open(Reader, path)

    def test_duplicate_header_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate"
            tensor = b'{"dtype":"U8","shape":[4],"data_offsets":[0,4]}'
            header = b'{"x":' + tensor + b',"x":' + tensor + b"}"
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"abcd")
            with self.assertRaises(ValueError):
                MetadataCache().open(Reader, path)

    def test_hit_preserves_bytes_and_nested_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a"
            write(p)
            c = MetadataCache()
            a = c.open(Reader, p, "model.layers.1.")
            b = c.open(Reader, p, "model.layers.1.")
            try:
                self.assertEqual(a.read("w"), b"abcd")
                self.assertEqual(b.read("w"), b"abcd")
                self.assertEqual(a.tensors["w"]["shape"], [4])
                self.assertFalse(a.tensors["w"]["shape"] != [4])
                with self.assertRaises(TypeError):
                    a.tensors["w"]["shape"][0] = 2
                with self.assertRaises(TypeError):
                    a.tensors["w"]["dtype"] = "F32"
                m = dict(a.tensors["w"])
                m["shape"] = [2]
                a.tensors["w"] = m
                self.assertEqual(b.tensors["w"]["shape"], [4])
                self.assertEqual(c.stats()["hits"], 1)
            finally:
                a.file.close()
                b.file.close()

    def test_changed_and_replaced_files_invalidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a"
            write(p)
            c = MetadataCache()
            a = c.open(Reader, p)
            a.file.close()
            old = p.stat()
            write(p, b"uvwxyz")
            os.utime(p, ns=(old.st_atime_ns, old.st_mtime_ns + 1000000000))
            a = c.open(Reader, p)
            self.assertEqual(a.read("model.layers.1.w"), b"uvwxyz")
            a.file.close()
            q = Path(tmp) / "b"
            write(q, b"1234")
            os.replace(q, p)
            a = c.open(Reader, p)
            self.assertEqual(a.read("model.layers.1.w"), b"1234")
            a.file.close()
            self.assertEqual(c.stats()["misses"], 3)

    def test_lru_eviction_and_small_budget_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = MetadataCache(max_entries=1)
            for name in ("a", "b", "a"):
                p = Path(tmp) / name
                if not p.exists():
                    write(p)
                a = c.open(Reader, p)
                a.file.close()
            self.assertEqual(c.stats()["misses"], 3)
            self.assertEqual(c.stats()["evictions"], 2)
            c = MetadataCache(max_bytes=512)
            a = c.open(Reader, p)
            self.assertEqual(a.read("model.layers.1.w"), b"abcd")
            a.file.close()
            self.assertEqual(c.stats()["entries"], 0)
            self.assertEqual(c.stats()["bypasses"], 1)
            self.assertLessEqual(c.stats()["retained_object_bytes"], 512)

    def test_prefix_is_part_of_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a"
            write(p)
            c = MetadataCache()
            a = c.open(Reader, p, "model.layers.1.")
            b = c.open(Reader, p, "")
            try:
                self.assertEqual(set(a.tensors), {"w"})
                self.assertEqual(set(b.tensors), {"model.layers.1.w"})
                self.assertEqual(c.stats()["misses"], 2)
            finally:
                a.file.close()
                b.file.close()

    def test_truncated_oversized_and_escaped_read_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a"
            p.write_bytes(b"bad")
            with self.assertRaises(EOFError):
                MetadataCache().open(Reader, p)
            p.write_bytes((1024**3).to_bytes(8, "little"))
            with self.assertRaises(ValueError):
                MetadataCache().open(Reader, p)
            write(p)
            a = MetadataCache().open(Reader, p)
            try:
                with self.assertRaises(ValueError):
                    a.read("model.layers.1.w", 3, 2)
            finally:
                a.file.close()


if __name__ == "__main__":
    unittest.main()
