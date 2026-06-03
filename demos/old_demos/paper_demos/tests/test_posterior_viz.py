"""
Smoke tests for posterior_viz.

Tests:
    test_compute_posterior_display_data_shapes
    test_posterior_viewer_caching
    test_render_posterior_figure_returns_figure
    test_matplotlib_posterior_viewer_compute_redraws
"""

import sys
import os

_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_TEST_DIR, "../utils"))
sys.path.insert(0, os.path.join(_TEST_DIR, "../visualization"))

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

from full_spectrum_utils import BlockIndex, RadialSpecs, build_shared_bessel_blocks
from intervalinf import ParallelConfig

from posterior_viz import (
    PosteriorViewer,
    _compute_posterior_display_data,
    _draw_posterior_axes,
    render_posterior_figure,
    MatplotlibPosteriorViewer,
)


# ---------------------------------------------------------------------------
# Shared fixture: synthetic posterior built from the block prior
# ---------------------------------------------------------------------------


class _ZeroForward:
    """Minimal forward operator stub for display-data smoke tests."""

    def __call__(self, _model):
        return np.zeros(0)


def _make_tiny_posterior_setup(
    n_basis: int = 5,
    n_grid: int = 30,
    *,
    sigma_var: float = 100.0,
):
    """
    Build (block, posterior, forward_dict, specs) without external data files.

    Uses build_block_prior as the posterior stand-in — the prior GaussianMeasure
    has the same nested model-space structure as a real posterior, so all
    posterior_viz code paths can be exercised.
    """
    from full_spectrum_utils import build_block_prior

    block = BlockIndex(s=2, t=0)
    specs = RadialSpecs(
        n_basis=n_basis,
        parallel_cfg=ParallelConfig(enabled=False, n_jobs=1),
    )
    shared = build_shared_bessel_blocks(specs)
    posterior = build_block_prior(block.s, block.t, shared, specs, sigma_var=sigma_var)

    M_st = posterior.domain
    forward_dict = {block: (_ZeroForward(), None, M_st, None)}

    return block, posterior, forward_dict, specs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_posterior_display_data_shapes():
    """_compute_posterior_display_data returns arrays with expected shapes."""
    block, posterior, forward_dict, specs = _make_tiny_posterior_setup(
        n_basis=5, n_grid=30
    )
    n_probes = 4

    data = _compute_posterior_display_data(
        block, posterior, forward_dict, specs,
        n_grid=30, n_probes=n_probes,
    )

    # Mean arrays have shape (n_grid,)
    for arr_name in ("r_vp", "r_vs_IC", "r_vs_M", "r_rho",
                     "mean_vp", "mean_vs_IC", "mean_vs_M", "mean_rho"):
        arr = getattr(data, arr_name)
        assert arr.shape == (30,), f"{arr_name}: expected (30,), got {arr.shape}"

    # Probe arrays have shape (n_probes,)
    for arr_name in ("probe_r_vp", "probe_r_vs_IC", "probe_r_vs_M", "probe_r_rho",
                     "std_vp", "std_vs_IC", "std_vs_M", "std_rho"):
        arr = getattr(data, arr_name)
        assert arr.shape == (n_probes,), (
            f"{arr_name}: expected ({n_probes},), got {arr.shape}"
        )

    # Std values must be non-negative
    for arr_name in ("std_vp", "std_vs_IC", "std_vs_M", "std_rho"):
        arr = getattr(data, arr_name)
        assert np.all(arr >= 0.0), f"{arr_name}: negative std values"


def test_compute_posterior_display_data_uses_posterior_sigma_1_std():
    """sigma_1_std should come from the posterior covariance, not a fallback prior scale."""
    block, posterior, forward_dict, specs = _make_tiny_posterior_setup(
        n_basis=5,
        sigma_var=9.0,
    )

    data = _compute_posterior_display_data(
        block, posterior, forward_dict, specs,
        n_grid=30, n_probes=4,
    )

    assert data.sigma_1_std == pytest.approx(3.0, rel=1e-6)


def test_posterior_viewer_caching():
    """PosteriorViewer caches computed blocks and avoids redundant computation."""
    block, posterior, forward_dict, specs = _make_tiny_posterior_setup(n_basis=5)
    posterior_dict = {block: posterior}

    viewer = PosteriorViewer(
        posterior_dict, forward_dict, specs, n_grid=20, n_probes=3
    )
    assert not viewer.is_computed(block)
    assert viewer.n_computed == 0

    data1 = viewer.compute_block(block)
    assert viewer.is_computed(block)
    assert viewer.n_computed == 1

    # Second call returns the same cached object (no recomputation)
    data2 = viewer.compute_block(block)
    assert data1 is data2


def test_render_posterior_figure_returns_figure():
    """render_posterior_figure returns a Figure with 4 axes."""
    block, posterior, forward_dict, specs = _make_tiny_posterior_setup(n_basis=5)
    posterior_dict = {block: posterior}

    viewer = PosteriorViewer(
        posterior_dict, forward_dict, specs, n_grid=20, n_probes=3
    )
    data = viewer.compute_block(block)

    fig = render_posterior_figure(data, block, specs, n_sigma=1.0)

    assert isinstance(fig, matplotlib.figure.Figure), (
        f"Expected Figure, got {type(fig)}"
    )
    assert len(fig.axes) == 4, f"Expected 4 axes, got {len(fig.axes)}"

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_matplotlib_posterior_viewer_compute_redraws():
    """MatplotlibPosteriorViewer.compute_current() populates axes after compute."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    block, posterior, forward_dict, specs = _make_tiny_posterior_setup(n_basis=5)
    posterior_dict = {block: posterior}

    viewer = PosteriorViewer(
        posterior_dict, forward_dict, specs, n_grid=20, n_probes=3
    )

    app = MatplotlibPosteriorViewer(
        viewer,
        [block],
        specs,
        plt,
        Button,
        Slider,
    )

    # Before compute: axes are hidden (placeholder shown)
    assert not app.axes[0].axison, "axes[0] should be off before compute"

    # Compute and redraw
    app.compute_current()

    assert app.axes[0].has_data(), "axes[0] should have data after compute"
    assert "Ready" in app.status_text.get_text(), (
        f"Expected 'Ready' in status, got: {app.status_text.get_text()!r}"
    )

    plt.close(app.fig)


def test_posterior_viewer_disk_cache_roundtrip(tmp_path):
    """Disk cache: data is saved on first compute and loaded on second viewer instance."""
    import numpy as np

    block, posterior, forward_dict, specs = _make_tiny_posterior_setup(n_basis=5)
    posterior_dict = {block: posterior}
    cache_key = "test1234abcd5678"

    # First viewer: computes and writes the .npz file.
    viewer1 = PosteriorViewer(
        posterior_dict, forward_dict, specs, n_grid=20, n_probes=3,
        cache_dir=tmp_path, cache_key=cache_key,
    )
    assert not viewer1.has_disk_cache(block), "cache file should not exist yet"
    data1 = viewer1.compute_block(block)
    assert viewer1.has_disk_cache(block), "cache file should exist after compute"

    # Second viewer: loads from disk without calling _compute_posterior_display_data.
    viewer2 = PosteriorViewer(
        posterior_dict, forward_dict, specs, n_grid=20, n_probes=3,
        cache_dir=tmp_path, cache_key=cache_key,
    )
    assert not viewer2.is_computed(block), "in-memory cache should be empty on fresh viewer"
    data2 = viewer2.compute_block(block)
    assert viewer2.is_computed(block)

    # The arrays should be numerically identical.
    np.testing.assert_allclose(data2.mean_vp, data1.mean_vp, rtol=1e-6)
    np.testing.assert_allclose(data2.std_vp,  data1.std_vp,  rtol=1e-6)
    assert data2.sigma_1_mean == pytest.approx(data1.sigma_1_mean, rel=1e-6)
    assert data2.sigma_1_std == pytest.approx(data1.sigma_1_std, rel=1e-6)


def test_matplotlib_posterior_viewer_autoloads_cached_block(tmp_path):
    """A fresh viewer should immediately draw the current block when it is disk-cached."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    block, posterior, forward_dict, specs = _make_tiny_posterior_setup(n_basis=5)
    posterior_dict = {block: posterior}
    cache_key = "test1234abcd5678"

    viewer1 = PosteriorViewer(
        posterior_dict,
        forward_dict,
        specs,
        n_grid=20,
        n_probes=3,
        cache_dir=tmp_path,
        cache_key=cache_key,
    )
    viewer1.compute_block(block)

    viewer2 = PosteriorViewer(
        {},
        {},
        specs,
        n_grid=20,
        n_probes=3,
        cache_dir=tmp_path,
        cache_key=cache_key,
        blocks=[block],
    )

    app = MatplotlibPosteriorViewer(
        viewer2,
        [block],
        specs,
        plt,
        Button,
        Slider,
    )

    assert app.axes[0].has_data(), "axes[0] should show cached data immediately"
    assert "Ready" in app.status_text.get_text(), (
        f"Expected 'Ready' in status, got: {app.status_text.get_text()!r}"
    )

    plt.close(app.fig)
