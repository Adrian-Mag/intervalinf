"""
Smoke tests for target_kernel_viz.

Tests:
    test_target_kernel_viewer_construction
    test_geographic_map_shapes
    test_geographic_map_blocks_projection_differs_from_full
    test_radial_profile_bulk_target
    test_radial_profile_cmb_target_raises
    test_st_coefficients_returns_dict
    test_st_radial_kernels_bulk_target
    test_st_radial_kernels_cmb_target_raises
    test_draw_geographic_axes_bulk
    test_draw_geographic_axes_cmb
    test_draw_st_axes_bulk
    test_draw_st_axes_cmb
    test_matplotlib_viewer_construction
    test_matplotlib_viewer_mode_toggle
    test_matplotlib_viewer_geo_reconstruction_toggle
    test_matplotlib_viewer_target_step
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
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from full_spectrum_utils import BlockIndex, enumerate_blocks, RadialSpecs
from normal_mode_kernel_utils import (
    NormalModeDataRegistry,
    NormalModeKernelCatalog,
    EARTH_RADIUS_KM,
)
from intervalinf import ParallelConfig
from property_targets import CapBulkTarget, BoxcarBulkTarget, CapCMBTarget

from target_kernel_viz import (
    TargetKernelViewer,
    _draw_geographic_axes,
    _draw_st_axes,
    MatplotlibTargetKernelViewer,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "normal-mode-data"
)
KERNEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "normal-mode-kernels",
    "kernels-all_PREM-layers_Adrian",
)
S_MAX = 2
EARTH_R = EARTH_RADIUS_KM
ICB_R = 1221.0
CMB_R = 3480.0


def _make_blocks():
    catalog = NormalModeKernelCatalog(KERNEL_DIR)
    reg = NormalModeDataRegistry(DATA_DIR, mode_filter=catalog.list_modes())
    return enumerate_blocks(reg, s_max=S_MAX)


def _make_targets():
    return [
        CapBulkTarget("vp", 10.0, 20.0, 10.0, (CMB_R + EARTH_R) / 2, 400.0),
        BoxcarBulkTarget("vs", -15.0, 45.0, 8.0, CMB_R, CMB_R + 300.0),
        CapCMBTarget(0.0, 0.0, 10.0),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blocks():
    return _make_blocks()


@pytest.fixture(scope="module")
def targets():
    return _make_targets()


@pytest.fixture(scope="module")
def viewer(blocks, targets):
    return TargetKernelViewer(
        targets,
        blocks,
        S_MAX,
        n_radial=50,
        earth_radius_km=EARTH_R,
        icb_radius_km=ICB_R,
        cmb_radius_km=CMB_R,
    )


# ---------------------------------------------------------------------------
# TargetKernelViewer unit tests
# ---------------------------------------------------------------------------

def test_target_kernel_viewer_construction(viewer, blocks, targets):
    """Viewer should store targets and blocks and pre-compute coefficients."""
    assert viewer.n_targets == len(targets)
    assert len(viewer.blocks) == len(blocks)
    assert viewer.s_max == S_MAX


def test_geographic_map_shapes(viewer):
    """geographic_map should return consistent lat/lon/data arrays."""
    for idx in range(viewer.n_targets):
        for reconstruction in ("smax", "blocks"):
            lats, lons, data = viewer.geographic_map(
                idx,
                grid_lmax=8,
                reconstruction=reconstruction,
            )
            assert lats.ndim == 1
            assert lons.ndim == 1
            assert data.shape == (len(lats), len(lons))
            assert np.all(np.isfinite(data))


def test_geographic_map_blocks_projection_differs_from_full(viewer):
    """The blocks-only reconstruction should differ from the full s<=smax map."""
    lats_full, lons_full, data_full = viewer.geographic_map(
        0,
        grid_lmax=8,
        reconstruction="smax",
    )
    lats_blocks, lons_blocks, data_blocks = viewer.geographic_map(
        0,
        grid_lmax=8,
        reconstruction="blocks",
    )

    np.testing.assert_allclose(lats_full, lats_blocks)
    np.testing.assert_allclose(lons_full, lons_blocks)
    assert np.linalg.norm(data_full - data_blocks) > 1e-6


def test_radial_profile_bulk_target(viewer):
    """radial_profile returns a volume-normalized radial factor for CapBulkTarget."""
    r_km, a_r = viewer.radial_profile(0, n_r=50)
    assert r_km.shape == (50,)
    assert a_r.shape == (50,)
    volume_integral = float(np.trapezoid(a_r * (r_km ** 2), r_km))
    np.testing.assert_allclose(volume_integral, 1.0, rtol=5e-2)
    # Peak of C∞ bump should be near r0_km
    r0 = viewer.targets[0].r0_km
    peak_r = r_km[np.argmax(a_r)]
    np.testing.assert_allclose(peak_r, r0, rtol=0.05)


def test_radial_profile_cmb_target_raises(viewer):
    """radial_profile should raise TypeError for CMBTarget."""
    cmb_idx = next(
        i for i, t in enumerate(viewer.targets) if isinstance(t, CapCMBTarget)
    )
    with pytest.raises(TypeError):
        viewer.radial_profile(cmb_idx)


def test_st_coefficients_returns_dict(viewer, blocks):
    """st_coefficients should return a dict with one entry per block."""
    for idx in range(viewer.n_targets):
        coeffs = viewer.st_coefficients(idx)
        assert set(coeffs.keys()) == set(blocks)
        for val in coeffs.values():
            assert np.isfinite(val)


def test_st_radial_kernels_bulk_target(viewer, blocks):
    """st_radial_kernels should return (r_km, kernel) tuples per block."""
    kernels = viewer.st_radial_kernels(0, n_r=50)
    assert set(kernels.keys()) == set(blocks)
    for block, (r_km, kernel) in kernels.items():
        assert r_km.shape == (50,)
        assert kernel.shape == (50,)
        assert np.all(np.isfinite(kernel))


def test_st_radial_kernels_cmb_target_raises(viewer):
    """st_radial_kernels should raise TypeError for CMBTarget."""
    cmb_idx = next(
        i for i, t in enumerate(viewer.targets) if isinstance(t, CapCMBTarget)
    )
    with pytest.raises(TypeError):
        viewer.st_radial_kernels(cmb_idx)


# ---------------------------------------------------------------------------
# Drawing helper smoke tests
# ---------------------------------------------------------------------------

def test_draw_geographic_axes_bulk(viewer):
    """_draw_geographic_axes should not raise for BulkTarget."""
    fig, (ax_main, ax_side) = plt.subplots(1, 2)
    _draw_geographic_axes(
        ax_main,
        ax_side,
        viewer,
        0,
        reconstruction="blocks",
        plt=plt,
    )
    plt.close(fig)


def test_draw_geographic_axes_cmb(viewer):
    """_draw_geographic_axes should not raise for CMBTarget (no ax_radial)."""
    cmb_idx = next(
        i for i, t in enumerate(viewer.targets) if isinstance(t, CapCMBTarget)
    )
    fig, ax_main = plt.subplots(1, 1)
    _draw_geographic_axes(
        ax_main,
        None,
        viewer,
        cmb_idx,
        reconstruction="smax",
        plt=plt,
    )
    plt.close(fig)


def test_draw_st_axes_bulk(viewer):
    """_draw_st_axes should not raise for BulkTarget."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2)
    _draw_st_axes(ax_left, ax_right, viewer, 0, plt=plt)
    plt.close(fig)


def test_draw_st_axes_cmb(viewer):
    """_draw_st_axes should not raise for CMBTarget (no ax_bar)."""
    cmb_idx = next(
        i for i, t in enumerate(viewer.targets) if isinstance(t, CapCMBTarget)
    )
    fig, ax = plt.subplots(1, 1)
    _draw_st_axes(ax, None, viewer, cmb_idx, plt=plt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MatplotlibTargetKernelViewer smoke tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mpl_viewer(viewer):
    mv = MatplotlibTargetKernelViewer(viewer, plt, Button, Slider)
    yield mv
    plt.close(mv.fig)


def test_matplotlib_viewer_construction(mpl_viewer):
    """MatplotlibTargetKernelViewer should construct without error."""
    assert mpl_viewer.fig is not None
    assert mpl_viewer._mode == "geo"
    assert mpl_viewer._geo_reconstruction == "smax"
    assert mpl_viewer._current_idx == 0


def test_matplotlib_viewer_mode_toggle(mpl_viewer):
    """Mode toggle should switch between geo and st."""
    assert mpl_viewer._mode == "geo"
    mpl_viewer._on_mode_toggle(None)
    assert mpl_viewer._mode == "st"
    mpl_viewer._on_mode_toggle(None)
    assert mpl_viewer._mode == "geo"


def test_matplotlib_viewer_geo_reconstruction_toggle(mpl_viewer):
    """Geographic reconstruction toggle should switch between full and blocks."""
    assert mpl_viewer._geo_reconstruction == "smax"
    mpl_viewer._on_geo_reconstruction_toggle(None)
    assert mpl_viewer._geo_reconstruction == "blocks"
    mpl_viewer._on_geo_reconstruction_toggle(None)
    assert mpl_viewer._geo_reconstruction == "smax"


def test_matplotlib_viewer_target_step(mpl_viewer, viewer):
    """_step_target should wrap around and trigger a redraw."""
    n = viewer.n_targets
    mpl_viewer._step_target(1)
    assert mpl_viewer._current_idx == 1
    # Wrap around
    for _ in range(n):
        mpl_viewer._step_target(1)
    assert mpl_viewer._current_idx == 1  # back to 1 after n steps
