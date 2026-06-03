"""
Smoke tests for prior_viz.PriorViewer — Phase 8 (Prior Visualisation).

Four tests:
    test_prior_viewer_precompute_runs
    test_get_display_data_scales_with_tau
    test_get_display_data_samples_shape
    test_prior_viewer_resample_changes_samples
"""

import sys
import os

# Ensure paper_demos directory is on path
_TEST_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_TEST_DIR, "../utils"))
sys.path.insert(0, os.path.join(_TEST_DIR, "../visualization"))

import numpy as np
import pytest

from full_spectrum_utils import BlockIndex, RadialSpecs, build_shared_bessel_blocks
from intervalinf import ParallelConfig
from prior_viz import PriorViewer


# ─── shared fixture ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_setup():
    """Build shared_bessel + specs with n_basis=5 (fast, ~5 s)."""
    specs = RadialSpecs(
        n_basis=5,
        parallel_cfg=ParallelConfig(enabled=False, n_jobs=1),
    )
    shared = build_shared_bessel_blocks(specs)
    return shared, specs


# ─── tests ────────────────────────────────────────────────────────────────

def test_prior_viewer_precompute_runs(tiny_setup):
    """precompute() populates reference data with correct shapes and non-negative std."""
    shared, specs = tiny_setup
    viewer = PriorViewer(shared, specs, n_grid=20, n_samples=3)
    assert not viewer.is_precomputed

    viewer.precompute()

    assert viewer.is_precomputed
    for p in ("vp", "vs_IC", "vs_M", "rho"):
        assert p in viewer._ref_data, f"Missing reference data for '{p}'"
        data = viewer._ref_data[p]
        assert data.r_grid.shape == (20,), f"{p}: r_grid shape mismatch"
        assert data.ref_std.shape == (20,), f"{p}: ref_std shape mismatch"
        assert data.ref_samples.shape == (3, 20), f"{p}: ref_samples shape mismatch"
        assert np.all(data.ref_std >= 0), f"{p}: ref_std has negative values"


def test_get_display_data_scales_with_tau(tiny_setup):
    """Doubling tau_vp doubles std and samples for the vp component; others unchanged."""
    shared, specs = tiny_setup
    viewer = PriorViewer(shared, specs, n_grid=20, n_samples=3, seed=0)
    viewer.precompute()

    block = BlockIndex(s=2, t=0)
    d1 = viewer.get_display_data(block, tau_vp=1.0, tau_rho=1.5)
    d2 = viewer.get_display_data(block, tau_vp=2.0, tau_rho=1.5)

    np.testing.assert_allclose(d2["vp"]["std"], 2.0 * d1["vp"]["std"],
                                rtol=1e-12, err_msg="std not linearly scaled by tau_vp")
    np.testing.assert_allclose(d2["vp"]["samples"], 2.0 * d1["vp"]["samples"],
                                rtol=1e-12, err_msg="samples not linearly scaled by tau_vp")

    # Other components must be unaffected
    np.testing.assert_allclose(d2["rho"]["std"], d1["rho"]["std"], rtol=1e-12,
                                err_msg="rho std changed when only tau_vp was changed")


def test_get_display_data_samples_shape(tiny_setup):
    """get_display_data returns correct shapes and sigma_1 key."""
    shared, specs = tiny_setup
    viewer = PriorViewer(shared, specs, n_grid=20, n_samples=4)
    viewer.precompute()

    block = BlockIndex(s=0, t=0)
    data = viewer.get_display_data(block)

    for p in ("vp", "vs_IC", "vs_M", "rho"):
        assert "r" in data[p], f"{p}: missing 'r' key"
        assert "mean" in data[p], f"{p}: missing 'mean' key"
        assert "std" in data[p], f"{p}: missing 'std' key"
        assert "samples" in data[p], f"{p}: missing 'samples' key"
        assert data[p]["r"].shape == (20,), f"{p}: r shape mismatch"
        assert data[p]["std"].shape == (20,), f"{p}: std shape mismatch"
        assert data[p]["samples"].shape == (4, 20), f"{p}: samples shape mismatch"

    assert "sigma_1" in data, "Missing 'sigma_1' key"
    assert isinstance(data["sigma_1"]["std"], float), "sigma_1 std should be float"
    assert data["sigma_1"]["samples"].shape == (4,), "sigma_1 samples shape mismatch"


def test_prior_viewer_resample_changes_samples(tiny_setup):
    """resample() with a different seed produces different sample curves."""
    shared, specs = tiny_setup
    viewer = PriorViewer(shared, specs, n_grid=20, n_samples=3, seed=42)
    viewer.precompute()

    samples_before = viewer._ref_data["vp"].ref_samples.copy()
    std_before = viewer._ref_data["vp"].ref_std.copy()

    viewer.resample(seed=99)

    assert not np.allclose(viewer._ref_data["vp"].ref_samples, samples_before), (
        "resample() with a different seed produced identical samples"
    )
    # std must be unchanged (only random coefficients differ, not eigenpairs)
    data_after = viewer._ref_data["vp"]
    # ref_std is unchanged since it depends only on eigenvalues
    viewer_ref = PriorViewer(shared, specs, n_grid=20, n_samples=3, seed=42)
    viewer_ref.precompute()
    np.testing.assert_allclose(data_after.ref_std, viewer_ref._ref_data["vp"].ref_std,
                                rtol=1e-12, err_msg="resample() must not change ref_std")


# ─── Phase 2 tests ────────────────────────────────────────────────────────

def test_render_prior_figure_returns_figure(tiny_setup):
    """_render_prior_figure returns a Figure with 4 axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.figure
    from prior_viz import _render_prior_figure

    shared, specs = tiny_setup
    viewer = PriorViewer(shared, specs, n_grid=20, n_samples=3)
    viewer.precompute()

    block = BlockIndex(s=1, t=0)
    display_data = viewer.get_display_data(block)
    fig = _render_prior_figure(display_data, block, specs)

    assert isinstance(fig, matplotlib.figure.Figure), "Expected a matplotlib Figure"
    assert len(fig.axes) == 4, f"Expected 4 axes, got {len(fig.axes)}"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_make_prior_widget_returns_widget(tiny_setup):
    """make_prior_widget returns an ipywidgets widget."""
    import ipywidgets as widgets
    from prior_viz import make_prior_widget

    shared, specs = tiny_setup
    blocks = [BlockIndex(s=0, t=0), BlockIndex(s=1, t=0)]
    viewer = PriorViewer(shared, specs, n_grid=20, n_samples=3)
    # widget should not require precompute upfront (user presses button)
    widget = make_prior_widget(viewer, blocks, specs)

    assert isinstance(widget, widgets.Widget), "Expected an ipywidgets Widget"


def test_matplotlib_prior_viewer_precompute_redraws(tiny_setup):
    """MatplotlibPriorViewer precomputes and renders the selected block."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider
    from prior_viz import MatplotlibPriorViewer

    shared, specs = tiny_setup
    blocks = [BlockIndex(s=0, t=0), BlockIndex(s=2, t=1)]
    viewer = PriorViewer(shared, specs, n_grid=20, n_samples=3)
    panel = MatplotlibPriorViewer(viewer, blocks, specs, plt, Button, Slider)

    panel.precompute()

    assert viewer.is_precomputed, "Expected precompute() to populate the cache"
    assert panel.current_block == blocks[0], "Expected the initial block to be selected"
    assert panel.axes[0].has_data(), "Expected the first radial axis to contain plotted data"
    assert "Ready" in panel.status_text.get_text(), "Expected a ready status message after redraw"
    plt.close(panel.fig)
