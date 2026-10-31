import unittest

from kimi_low_memory.extents import Extent, plan_extents


class ExtentTests(unittest.TestCase):
    def plan(self, extents, **kwargs):
        limits = dict(max_read_bytes=65536, max_amplification=1.03, max_gap_bytes=256)
        return plan_extents(extents, **(limits | kwargs))

    def test_coalescing_preserves_payloads_and_logical_order(self):
        requests = [Extent(str(i), i * 8320, i * 8320 + 8192) for i in range(24)][::-1]
        source = bytes(i % 251 for i in range(200000))
        groups = self.plan(requests)
        restored = {}
        for group in groups:
            block = source[group["start"] : group["end"]]
            for request in group["extents"]:
                restored[request.label] = block[
                    request.start - group["start"] : request.end - group["start"]
                ]
        self.assertEqual(len(groups), 4)
        for request in requests:
            self.assertEqual(
                restored[request.label], source[request.start : request.end]
            )
        requested = sum(group["end"] - group["start"] for group in groups)
        self.assertLessEqual(requested, 1.03 * sum(x.end - x.start for x in requests))

    def test_large_gaps_are_not_read(self):
        self.assertEqual(
            len(self.plan([Extent("a", 0, 8), Extent("b", 10000, 10008)])), 2
        )

    def test_overlap_and_duplicate_labels_rejected(self):
        for requests in (
            [Extent("a", 0, 8), Extent("b", 4, 12)],
            [Extent("a", 0, 8), Extent("a", 8, 16)],
        ):
            with self.assertRaises(ValueError):
                self.plan(requests)

    def test_invalid_limits_and_ranges_rejected(self):
        for args in (
            {"max_amplification": float("nan")},
            {"max_gap_bytes": -1},
            {"max_read_bytes": True},
            {"max_amplification": float("inf")},
        ):
            with self.assertRaises(ValueError):
                self.plan([], **args)
        for request in (Extent("a", -1, 2), Extent("a", 0, 65537), Extent("a", 1, 1)):
            with self.assertRaises(ValueError):
                self.plan([request])

    def test_empty_plan(self):
        self.assertEqual(self.plan([]), [])
