"""Read-only model inspection and explicit KV payload calculations."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from .kv import KVLayout

MAX_HEADER_BYTES = 8 * 1024**2


def inspect_shard(path):
    """Validate a bounded safetensors header without reading tensor payloads."""
    from .reference.reader import TensorReader

    with TensorReader(path, max_header_bytes=MAX_HEADER_BYTES) as reader:
        counts = Counter(meta["dtype"] for meta in reader.tensors.values())
        payload = sum(
            meta["data_offsets"][1] - meta["data_offsets"][0]
            for meta in reader.tensors.values()
        )
        return {
            "file_bytes": os.fstat(reader.file.fileno()).st_size,
            "header_bytes": reader.base - 8,
            "tensor_count": sum(counts.values()),
            "tensor_dtypes": dict(counts),
            "declared_payload_bytes": payload,
            "weight_payload_read": False,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="inspect one safetensors header")
    inspect.add_argument("path", type=Path)
    budget = commands.add_parser(
        "kv-budget", help="calculate expanded FP32 Kimi KV payload"
    )
    budget.add_argument("--context", type=int, required=True)
    commands.add_parser("memory", help="sample this process's macOS Mach counters")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_shard(args.path)
        elif args.command == "kv-budget":
            if args.context <= 0:
                raise ValueError("context must be positive")
            payload = 61 * args.context * KVLayout().bytes_per_position
            result = {
                "context": args.context,
                "layers": 61,
                "kv_payload_bytes": payload,
                "kv_payload_gib": payload / 1024**3,
                "scope": "Expanded FP32 KV payload only; excludes weights, temporaries and runtime.",
            }
        else:
            from .memory import sample_self

            result = sample_self()
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0
