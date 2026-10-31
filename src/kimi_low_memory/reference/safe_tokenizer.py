"""Local Kimi-K2 tiktoken vocabulary loader; no downloads or model code.

This exposes raw encode/decode. It does not execute a repository chat template
or assume that arbitrary prompts have the model's expected chat formatting.
"""

import base64
import json
from pathlib import Path

PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


def read_bpe(path, *, max_bytes=64 * 1024**2):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("missing or oversized local BPE vocabulary")
    ranks = {}
    seen_ranks = set()
    with path.open("rb") as source:
        for line in source:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError("invalid BPE vocabulary line")
            token, rank = base64.b64decode(fields[0], validate=True), int(fields[1])
            if not token or token in ranks or rank < 0 or rank in seen_ranks:
                raise ValueError("duplicate/invalid BPE token or rank")
            ranks[token] = rank
            seen_ranks.add(rank)
    if seen_ranks != set(range(len(ranks))):
        raise ValueError("BPE ranks must be contiguous")
    return ranks


def load_tokenizer(weights_dir):
    root = Path(weights_dir)
    config = root / "tokenizer_config.json"
    if not config.is_file() or config.stat().st_size > 8 * 1024**2:
        raise ValueError("missing or oversized tokenizer config")
    cfg = json.loads(config.read_text(encoding="utf-8"))
    ranks = read_bpe(root / "tiktoken.model")
    if len(ranks) != 163584:
        raise ValueError("unexpected Kimi base vocabulary size")
    added = {int(i): v["content"] for i, v in cfg["added_tokens_decoder"].items()}
    if any(not isinstance(content, str) or not content for content in added.values()):
        raise ValueError("invalid special-token text")
    special = {
        added.get(i, f"<|reserved_token_{i}|>"): i for i in range(163584, 163840)
    }
    if (
        len(special) != 256
        or special.get("[BOS]") != 163584
        or special.get("<|im_end|>") != 163586
    ):
        raise ValueError("unexpected Kimi special-token mapping")
    import tiktoken

    encoding = tiktoken.Encoding(
        name="kimi_k2_local",
        pat_str=PATTERN,
        mergeable_ranks=ranks,
        special_tokens=special,
    )
    if encoding.n_vocab != 163840:
        raise ValueError("unexpected Kimi vocabulary size")
    return encoding
