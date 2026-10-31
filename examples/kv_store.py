"""Two tiny synthetic steps; no model weights, disk checkpoints or GPU."""

import json
import numpy as np

from kimi_low_memory import KVLayout, ResidentKV

layout = KVLayout(heads=2, width=4, rope_heads=1, rope_width=2)
store = ResidentKV(
    layers=3,
    max_context=8,
    layout=layout,
    max_retained_bytes=65536,
    max_transient_bytes=131072,
)
for position in range(2):
    store.begin_step(position=position)
    for layer in range(3):

        def append(old, layer=layer):
            new = tuple(
                np.full(shape, layer, dtype=np.float32) for shape in layout.shapes(1)
            )
            return (
                new
                if old is None
                else tuple(np.concatenate((a, b)) for a, b in zip(old, new))
            )

        store.advance_layer(layer, append)
    store.finish_step()
print(json.dumps(store.stats(), indent=2))
