from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from kimi_low_memory.cli import inspect_shard, main


class CLITests(unittest.TestCase):
    def test_inspect_header_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.safetensors"
            header = json.dumps(
                {"x": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
            ).encode()
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"abcd")
            report = inspect_shard(path)
            self.assertEqual(report["tensor_count"], 1)
            self.assertEqual(report["declared_payload_bytes"], 4)
            self.assertFalse(report["weight_payload_read"])

    def test_inspect_rejects_header_and_payload_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            path.write_bytes((2**32).to_bytes(8, "little"))
            with self.assertRaises(ValueError):
                inspect_shard(path)
            header = json.dumps(
                {"x": {"dtype": "U8", "shape": [5], "data_offsets": [0, 5]}}
            ).encode()
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"abcd")
            with self.assertRaises(ValueError):
                inspect_shard(path)

    def test_inspect_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            header = json.dumps(
                {
                    name: {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}
                    for name in ("x", "y")
                }
            ).encode()
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"abcd")
            with self.assertRaises(ValueError):
                inspect_shard(path)

    def test_kv_budget(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["kv-budget", "--context", "512"]), 0)
        self.assertEqual(json.loads(output.getvalue())["kv_payload_bytes"], 2054815744)

    def test_negative_context_fails(self):
        with self.assertRaises(SystemExit):
            main(["kv-budget", "--context", "-1"])
