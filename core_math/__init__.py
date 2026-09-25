"""core_math: pure, stateless, causal compute primitives for algo-trading engines.

Two halves, kept in lockstep by parity tests:
  - the Python reference (this package): readable ground truth.
  - the C++ mirror (``core_math/cpp/``): the fast path, bit-locked to the reference.

Nothing here loads data or reaches for a database. Feed it arrays/DataFrames; it
returns arrays/DataFrames. Every function is causal: the value at time t uses only
data timestamped at or before t, never the future.
"""
from __future__ import annotations

from . import bars_math, calculos_l2, labeling, meta_model, micro_math

__all__ = ["bars_math", "calculos_l2", "labeling", "meta_model", "micro_math"]
