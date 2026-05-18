"""
full_spectrum_utils.py
======================

Phase 1 utilities for the full-spectrum splitting-function PLI example.

Provides:
    BlockIndex          — frozen dataclass for a (s, t) spectral block.
    enumerate_blocks    — list all non-empty (s, t) blocks for even s <= s_max.
    block_data_split    — split a NormalModeDataRegistry by BlockIndex.
"""

import dataclasses
from typing import List, Dict

from normal_mode_kernel_utils import NormalModeDataRegistry


@dataclasses.dataclass(frozen=True)
class BlockIndex:
    """
    A spectral block identified by splitting degree *s* and component index *t*.

    Attributes
    ----------
    s : int
        Splitting degree (must be even, $0 \\leq s \\leq s_{\\mathrm{max}}$).
    t : int
        Real-harmonic component index ($0 \\leq t \\leq 2s$).
    """

    s: int
    t: int


def enumerate_blocks(reg: NormalModeDataRegistry, s_max: int) -> List[BlockIndex]:
    """
    Return the sorted list of non-empty ``BlockIndex`` objects.

    A ``BlockIndex(s, t)`` is included if and only if:

    - *s* is even and $0 \\leq s \\leq s_{\\mathrm{max}}$,
    - $0 \\leq t \\leq 2s$ (real-harmonic component index),
    - there is at least one observation in *reg* with this $(s, t)$.

    Ordering
    --------
    Blocks are sorted by $(s, t)$, so for each *s* the order is
    $t = 0, 1, 2, \\ldots, 2s$ (real-harmonic component index convention).

    Parameters
    ----------
    reg : NormalModeDataRegistry
        The data registry to query.
    s_max : int
        Maximum splitting degree to consider (inclusive).

    Returns
    -------
    list of BlockIndex
        Sorted non-empty block indices.
    """
    # Collect all (s, t) pairs that appear in the registry and pass the filter.
    # Note: t is a real-component index (0 ≤ t ≤ 2s), not standard azimuthal order.
    covered: set = set()
    for _mode_id, s, t in reg.observations:
        if s % 2 == 0 and 0 <= s <= s_max:
            covered.add((s, t))

    # Sort by (s, t) for canonical ordering (t is a non-negative real-harmonic index).
    sorted_pairs = sorted(covered, key=lambda st: (st[0], st[1]))
    return [BlockIndex(s=s, t=t) for s, t in sorted_pairs]


def block_data_split(
    reg: NormalModeDataRegistry,
    blocks: List[BlockIndex],
) -> Dict[BlockIndex, NormalModeDataRegistry]:
    """
    Split *reg* into per-block sub-registries.

    Parameters
    ----------
    reg : NormalModeDataRegistry
        The full data registry to partition.
    blocks : list of BlockIndex
        Block indices (typically from :func:`enumerate_blocks`).

    Returns
    -------
    dict mapping BlockIndex to NormalModeDataRegistry
        Each value is the sub-registry returned by
        ``reg.filter_by_st(s=b.s, t=b.t)``.  The union of all sub-registry
        observations equals the set of observations in *reg* whose $(s, t)$
        pair appears in *blocks* — with no duplicates.
    """
    return {b: reg.filter_by_st(s=b.s, t=b.t) for b in blocks}
