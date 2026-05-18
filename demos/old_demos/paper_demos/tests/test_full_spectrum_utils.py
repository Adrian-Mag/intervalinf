"""Tests for full_spectrum_utils Phase 1: block indexing."""

import sys
import os

# Ensure paper_demos is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from full_spectrum_utils import BlockIndex, enumerate_blocks, block_data_split
from normal_mode_kernel_utils import NormalModeDataRegistry, NormalModeKernelCatalog

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "normal-mode-data")
KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "normal-mode-kernels",
    "kernels-all_PREM-layers_Adrian"
)
S_MAX = 4


def make_registry():
    catalog = NormalModeKernelCatalog(KERNEL_DIR)
    return NormalModeDataRegistry(DATA_DIR, mode_filter=catalog.list_modes())


def test_enumerate_blocks_respects_smax_and_parity():
    """Only even s with 0 <= s <= S_MAX appear in the block list."""
    reg = make_registry()
    blocks = enumerate_blocks(reg, s_max=S_MAX)
    assert len(blocks) > 0, "Expected at least one block"
    for b in blocks:
        assert b.s % 2 == 0, f"Non-even s={b.s} found"
        assert 0 <= b.s <= S_MAX, f"s={b.s} out of range"
        assert 0 <= b.t <= 2 * b.s, f"t={b.t} out of range [0, 2s={2*b.s}]"


def test_enumerate_blocks_skips_empty_blocks():
    """(s, t) pairs with zero observations are excluded."""
    reg = make_registry()
    blocks = enumerate_blocks(reg, s_max=S_MAX)
    block_set = set(blocks)
    # Verify no block has an empty sub-registry
    for b in blocks:
        sub = reg.filter_by_st(s=b.s, t=b.t)
        assert len(sub) > 0, f"Block {b} has zero observations"
    # Verify every covered (s,t) with even s <= S_MAX IS in blocks
    covered = set()
    for mode_id, s, t in reg.observations:
        if s % 2 == 0 and 0 <= s <= S_MAX:
            covered.add((s, t))
    for s, t in covered:
        assert BlockIndex(s=s, t=t) in block_set, f"Block (s={s},t={t}) missing"


def test_block_data_split_partitions_registry():
    """Union of per-block data equals the full covered subset; blocks are non-overlapping."""
    reg = make_registry()
    blocks = enumerate_blocks(reg, s_max=S_MAX)
    split = block_data_split(reg, blocks)
    assert set(split.keys()) == set(blocks), "split keys don't match blocks"

    # Collect all (mode_id, s, t) tuples from all sub-registries
    all_obs = []
    for b, sub_reg in split.items():
        for obs in sub_reg.observations:
            all_obs.append(obs)

    # No duplicates (same (mode_id, s, t) should not appear in two blocks)
    assert len(all_obs) == len(set(all_obs)), "Duplicate observations across blocks"

    # Union equals the covered observations in full registry
    expected = set(
        (mode_id, s, t)
        for mode_id, s, t in reg.observations
        if s % 2 == 0 and 0 <= s <= S_MAX
    )
    assert set(all_obs) == expected, "Split does not cover expected observations"
