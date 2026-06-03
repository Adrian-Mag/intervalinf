"""
Smoke tests for full_spectrum_viz — Phase 7 (Visualisation and Reporting).

Four tests:
    test_plot_block_posterior_returns_figure
    test_plot_block_posterior_uses_posterior_sigma_1_std
    test_plot_cmb_map_returns_figure
    test_plot_equatorial_slice_gated_off_by_default
"""

import sys
import os

# Ensure paper_demos directory is on path
_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_TEST_DIR, "../utils"))
sys.path.insert(0, os.path.join(_TEST_DIR, "../visualization"))

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.figure

from full_spectrum_viz import (
    plot_block_posterior,
    plot_cmb_map,
    plot_equatorial_slice,
    plot_property_posterior_summary,
)

from full_spectrum_utils import (
    BlockIndex,
    RadialSpecs,
)

# ─── helpers for building tiny synthetic setups ──────────────────────────────

def _make_synthetic_posterior_for_block_plot(n_basis=5, *, sigma_var=100.0):
    """
    Build a self-contained (block, posterior, forward_dict, specs) without
    loading any external data files.

    Uses ``build_block_prior`` (which requires no external files) as the
    posterior.  The prior covariance operator has the same nested structure
    as a real posterior, so ``plot_block_posterior`` can exercise all code
    paths.
    """
    from full_spectrum_utils import (
        RadialSpecs, build_block_prior, build_shared_bessel_blocks,
    )
    from intervalinf import ParallelConfig

    block = BlockIndex(s=2, t=0)
    specs = RadialSpecs(
        n_basis=n_basis,
        parallel_cfg=ParallelConfig(enabled=False, n_jobs=1),
    )
    shared = build_shared_bessel_blocks(specs)
    prior = build_block_prior(block.s, block.t, shared, specs, sigma_var=sigma_var)

    # The prior's domain IS M_st — use it as a stand-in for the forward_dict
    # entry.  plot_block_posterior only reads M_st to get component subspaces.
    M_st = prior.domain
    forward_dict = {block: (None, None, M_st, None)}

    return block, prior, forward_dict, specs


def _make_synthetic_posterior_dict_for_cmb():
    """
    Build a minimal posterior_dict with the correct expectation nesting structure
    for testing plot_cmb_map, without loading any real data.
    """
    from pygeoinf import GaussianMeasure, EuclideanSpace

    block = BlockIndex(s=0, t=0)

    # Fake expectation: [[f_vp, [f_vs_IC, f_vs_M], f_rho], [sigma_0, sigma_1]]
    # We only need the sigma_1 value: expectation[1][1][0]
    sigma_0 = np.array([0.5])
    sigma_1 = np.array([1.3])

    # Minimal covariance: identity on R^1
    M_sigma_0 = EuclideanSpace(1)
    M_sigma_1 = EuclideanSpace(1)

    # Build a GaussianMeasure with a fake expectation that has the right nested structure.
    # We construct it with from_covariance_matrix on a 1D space and then patch .expectation
    # by wrapping in a tiny container object.
    from pygeoinf import HilbertSpaceDirectSum

    M_euclidean = HilbertSpaceDirectSum([M_sigma_0, M_sigma_1])

    fake_prior_euclidean = GaussianMeasure.from_covariance_matrix(
        M_sigma_1, np.array([[100.0]]), expectation=sigma_1,
    )

    class _FakePosterior:
        """Minimal fake posterior that exposes the nesting structure."""
        def __init__(self, sigma_0_val, sigma_1_val):
            # expectation[1][0] = sigma_0, expectation[1][1] = sigma_1
            self.expectation = [
                None,                     # placeholder for functions component
                [sigma_0_val, sigma_1_val],
            ]

    posterior_dict = {block: _FakePosterior(sigma_0, sigma_1)}
    return posterior_dict, block


# =============================================================================
# Tests
# =============================================================================

def test_plot_block_posterior_returns_figure():
    """plot_block_posterior returns a matplotlib Figure with at least 4 axes."""
    block, posterior, forward_dict, specs = _make_synthetic_posterior_for_block_plot(n_basis=5)

    fig = plot_block_posterior(block, posterior, forward_dict, specs, n_grid=30)

    assert isinstance(fig, matplotlib.figure.Figure), (
        f"Expected matplotlib.figure.Figure, got {type(fig)}"
    )
    assert len(fig.axes) >= 4, (
        f"Expected at least 4 axes panels, got {len(fig.axes)}"
    )
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_block_posterior_uses_posterior_sigma_1_std():
    """sigma_1 panel should report posterior, not fallback prior, uncertainty."""
    block, posterior, forward_dict, specs = _make_synthetic_posterior_for_block_plot(
        n_basis=5,
        sigma_var=9.0,
    )

    fig = plot_block_posterior(block, posterior, forward_dict, specs, n_grid=30)

    legend = fig.axes[4].get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert any("±σ_post = 3.00" in label for label in labels), labels

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_cmb_map_returns_figure():
    """plot_cmb_map returns a matplotlib Figure given minimal synthetic data."""
    posterior_dict, block = _make_synthetic_posterior_dict_for_cmb()
    blocks = [block]
    s_max = 2

    fig = plot_cmb_map(blocks, posterior_dict, s_max)

    assert isinstance(fig, matplotlib.figure.Figure), (
        f"Expected matplotlib.figure.Figure, got {type(fig)}"
    )
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_equatorial_slice_gated_off_by_default():
    """plot_equatorial_slice returns None when enabled=False."""
    result = plot_equatorial_slice([], {}, 2, enabled=False)
    assert result is None, f"Expected None when enabled=False, got {result}"
