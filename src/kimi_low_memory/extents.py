"""Bounded physical read planning; callers restore logical result order."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Extent:
    label: str
    start: int
    end: int


def plan_extents(extents, *, max_read_bytes, max_amplification, max_gap_bytes):
    """Greedy bounded plan; not a claim of optimal I/O scheduling.

    Real callers must validate source identity and EOF, bound resident/in-flight
    buffers, and restore original computation order. No I/O occurs here.
    """
    if (
        type(max_read_bytes) is not int
        or max_read_bytes <= 0
        or type(max_gap_bytes) is not int
        or max_gap_bytes < 0
        or not isinstance(max_amplification, (int, float))
        or not math.isfinite(max_amplification)
        or max_amplification < 1
    ):
        raise ValueError("invalid planner limits")
    extents = tuple(extents)
    if any(
        not isinstance(x, Extent)
        or not isinstance(x.label, str)
        or not x.label
        or type(x.start) is not int
        or type(x.end) is not int
        for x in extents
    ):
        raise ValueError("invalid extent type, label or offsets")
    ordered = sorted(extents, key=lambda x: x.start)
    if len({x.label for x in ordered}) != len(ordered):
        raise ValueError("duplicate labels")
    groups = []
    previous_end = 0
    for x in ordered:
        if (
            type(x.start) is not int
            or type(x.end) is not int
            or x.start < previous_end
            or x.end <= x.start
            or x.end - x.start > max_read_bytes
        ):
            raise ValueError("invalid, overlapping or oversized extent")
        previous_end = x.end
        size = x.end - x.start
        if groups:
            old = groups[-1]
            span = x.end - old["start"]
            useful = old["useful_bytes"] + size
            if (
                x.start - old["end"] <= max_gap_bytes
                and span <= max_read_bytes
                and span <= max_amplification * useful
            ):
                old["end"], old["useful_bytes"] = x.end, useful
                old["extents"].append(x)
                continue
        groups.append(
            {"start": x.start, "end": x.end, "useful_bytes": size, "extents": [x]}
        )
    return groups
