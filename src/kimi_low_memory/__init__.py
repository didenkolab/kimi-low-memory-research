"""Components for bounded-memory inference experiments.

Importing the package does not load MLX, model weights, or GPU resources.
"""

from .caches import ExpertCache, ExpertKey
from .kv import KVLayout, KVStateError, ResidentKV
from .metadata import MetadataCache
from .prefetch import SelectedExpertPrefetch

__version__ = "0.1.0"
__all__ = [
    "ExpertCache",
    "ExpertKey",
    "KVLayout",
    "KVStateError",
    "ResidentKV",
    "MetadataCache",
    "SelectedExpertPrefetch",
]
