"""Tests for property_targets.py — Phase 5."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from property_targets import (
    angular_bump_coeffs,
    rotate_bump_coeffs,
    extract_st_coeff,
    radial_bump,
    BulkTarget,
    CMBTarget,
    build_block_property_coeffs,
)
from full_spectrum_utils import (
    BlockIndex,
    enumerate_blocks,
    build_property_operator,
    build_block_forward,
    block_data_split,
    RadialSpecs,
    build_shared_bessel_blocks,
    build_block_prior,
)
from normal_mode_kernel_utils import (
    NormalModeDataRegistry,
    NormalModeKernelCatalog,
    EARTH_RADIUS_KM,
)
from intervalinf import (
    IntervalDomain,
    LebesgueIntegrationConfig,
    IntegrationConfig,
    ParallelConfig,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "normal-mode-data")
KERNEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "normal-mode-kernels",
    "kernels-all_PREM-layers_Adrian",
)
S_MAX = 2


# ---------------------------------------------------------------------------
# Shared fixture helper
# ---------------------------------------------------------------------------

def _make_phase5_setup(s_max=S_MAX, n_basis=20):
    """Build forward_dict, prior_dict, blocks for Phase 5 tests."""
    catalog = NormalModeKernelCatalog(KERNEL_DIR)
    reg = NormalModeDataRegistry(DATA_DIR, mode_filter=catalog.list_modes())
    blocks = enumerate_blocks(reg, s_max=s_max)
    split = block_data_split(reg, blocks)
    specs = RadialSpecs(
        n_basis=n_basis,
        parallel_cfg=ParallelConfig(enabled=False, n_jobs=1),
    )
    forward_dict = {
        b: build_block_forward(b.s, b.t, split[b], catalog, specs)
        for b in blocks
    }
    return blocks, forward_dict, specs


# =============================================================================
# Test 1: angular_bump_coeffs — axisymmetric at pole
# =============================================================================

def test_angular_bump_coeffs_axisymmetric_at_pole():
    """Pole bump should have t=0 (m=0) components only; all m>0 near zero."""
    coeffs = angular_bump_coeffs(sigma_deg=10.0, s_max=4)
    for s in range(0, 5, 2):  # only even s appear in data
        for m in range(1, s + 1):
            # Sine components (index 1): should be zero at pole
            np.testing.assert_allclose(
                coeffs[1, s, m],
                0.0,
                atol=1e-5,
                err_msg=f"Sine component at s={s},m={m} should be zero for pole bump",
            )
            # Cosine components for m>0: should also be ~zero (axisymmetric)
            np.testing.assert_allclose(
                coeffs[0, s, m],
                0.0,
                atol=1e-5,
                err_msg=f"Cosine m={m} component at s={s} should be zero for pole bump",
            )


# =============================================================================
# Test 2: rotation preserves total SH power
# =============================================================================

def test_rotation_preserves_power():
    """Rotation preserves total SH power (Parseval identity)."""
    coeffs_pole = angular_bump_coeffs(sigma_deg=15.0, s_max=4)
    coeffs_rot = rotate_bump_coeffs(coeffs_pole, lat_deg=30.0, lon_deg=60.0)

    power_pole = np.sum(coeffs_pole ** 2)
    power_rot = np.sum(coeffs_rot ** 2)

    np.testing.assert_allclose(power_rot, power_pole, rtol=1e-6)


# =============================================================================
# Test 3: radial_bump peaks at r0
# =============================================================================

def test_radial_bump_peak_location():
    """Radial bump peaks at r0 with value ~1.0 and decays away from it."""
    r = np.linspace(0.0, EARTH_RADIUS_KM, 1000)
    r0, sigma_r = 5000.0, 200.0
    h = radial_bump(r, r0, sigma_r)

    # Maximum value must be close to 1.0 (exact 1.0 only if r0 is on the grid)
    assert h.max() == pytest.approx(1.0, abs=1e-3)

    # Peak must be nearest to r0
    peak_idx = np.argmax(h)
    nearest_idx = np.argmin(np.abs(r - r0))
    assert peak_idx == nearest_idx


# =============================================================================
# Test 4: property operator output shape
# =============================================================================

def test_property_operator_shape():
    """Each block's property operator maps M_st → R^{N_p}."""
    blocks, forward_dict, specs = _make_phase5_setup()

    targets = [
        BulkTarget(param="vp", lat_deg=10.0, lon_deg=210.0, sigma_ang_deg=20.0, r0_km=4500.0, sigma_r_km=300.0),
        CMBTarget(lat_deg=0.0, lon_deg=0.0, sigma_ang_deg=20.0),
    ]
    N_p = len(targets)

    property_op_dict = build_property_operator(
        targets, blocks, forward_dict, specs, s_max=S_MAX, n_radial=100,
    )

    assert set(property_op_dict.keys()) == set(blocks), "Missing blocks in property op dict"

    for block in blocks:
        T = property_op_dict[block]
        _, _, M_st, _ = forward_dict[block]
        m_zero = M_st.zero
        out = T(m_zero)
        assert out.shape == (N_p,), (
            f"Block {block}: expected shape ({N_p},), got {out.shape}"
        )


# =============================================================================
# Test 5: CMB target forward
# =============================================================================

def test_property_operator_forward_cmb_target():
    """CMB target: T(m)[i] = B_st[i] * sigma_1[0]."""
    blocks, forward_dict, specs = _make_phase5_setup()

    # Single CMB target at the north pole to keep B_st = coeffs[0,0,0] (zonal)
    targets = [
        CMBTarget(lat_deg=90.0, lon_deg=0.0, sigma_ang_deg=20.0),
    ]

    property_op_dict = build_property_operator(
        targets, blocks, forward_dict, specs, s_max=S_MAX, n_radial=100,
    )

    # Pick block (s=0, t=0) — it always exists
    block_00 = BlockIndex(s=0, t=0)
    assert block_00 in property_op_dict, "Block (0,0) missing"

    T = property_op_dict[block_00]
    _, _, M_st, _ = forward_dict[block_00]
    m_zero = M_st.zero

    # Build a model element with sigma_1 = [3.7], all functions zero
    from intervalinf.core.functions import Function as _IFunction
    m_test = [
        [
            _IFunction(IntervalDomain(0.0, specs.earth_radius_km), evaluate_callable=lambda r: np.zeros_like(r)),
            [
                _IFunction(IntervalDomain(0.0, specs.icb_radius_km), evaluate_callable=lambda r: np.zeros_like(r)),
                _IFunction(IntervalDomain(specs.cmb_radius_km, specs.earth_radius_km), evaluate_callable=lambda r: np.zeros_like(r)),
            ],
            _IFunction(IntervalDomain(0.0, specs.earth_radius_km), evaluate_callable=lambda r: np.zeros_like(r)),
        ],
        [np.zeros(1), np.array([3.7])],
    ]

    from property_targets import angular_bump_coeffs, rotate_bump_coeffs, extract_st_coeff
    coeffs_pole = angular_bump_coeffs(20.0, S_MAX)
    coeffs_rot = rotate_bump_coeffs(coeffs_pole, lat_deg=90.0, lon_deg=0.0)
    B_00 = extract_st_coeff(coeffs_rot, 0, 0)

    out = T(m_test)
    expected = B_00 * 3.7
    np.testing.assert_allclose(out[0], expected, rtol=1e-8)


# =============================================================================
# Test 6: adjoint consistency  <T(m), lam> == <m, T*(lam)>
# =============================================================================

def test_property_operator_adjoint_consistency():
    """<T(m), lambda>_{R^N_p} == <m, T*(lambda)>_{M_st} to numerical tolerance."""
    blocks, forward_dict, specs = _make_phase5_setup()

    np.random.seed(42)
    targets = [
        BulkTarget(param="vp", lat_deg=10.0, lon_deg=210.0, sigma_ang_deg=20.0, r0_km=4500.0, sigma_r_km=300.0),
        BulkTarget(param="rho", lat_deg=45.0, lon_deg=15.0, sigma_ang_deg=20.0, r0_km=4500.0, sigma_r_km=300.0),
        CMBTarget(lat_deg=0.0, lon_deg=0.0, sigma_ang_deg=20.0),
    ]
    N_p = len(targets)

    property_op_dict = build_property_operator(
        targets, blocks, forward_dict, specs, s_max=S_MAX, n_radial=200,
    )

    # Test on the first block
    block = blocks[0]
    T = property_op_dict[block]
    _, _, M_st, _ = forward_dict[block]

    # Build a non-trivial model element: constant functions + non-zero sigma_1
    from intervalinf.core.functions import Function as _IFunction
    c_vp, c_rho, sigma_1_val = 1.3, 0.7, 2.1
    m_test = [
        [
            _IFunction(IntervalDomain(0.0, specs.earth_radius_km), evaluate_callable=lambda r: np.full(np.asarray(r).shape, c_vp) if np.asarray(r).shape else float(c_vp)),
            [
                _IFunction(IntervalDomain(0.0, specs.icb_radius_km), evaluate_callable=lambda r: np.zeros_like(np.asarray(r, dtype=float))),
                _IFunction(IntervalDomain(specs.cmb_radius_km, specs.earth_radius_km), evaluate_callable=lambda r: np.zeros_like(np.asarray(r, dtype=float))),
            ],
            _IFunction(IntervalDomain(0.0, specs.earth_radius_km), evaluate_callable=lambda r: np.full(np.asarray(r).shape, c_rho) if np.asarray(r).shape else float(c_rho)),
        ],
        [np.zeros(1), np.array([sigma_1_val])],
    ]

    lam = np.array([0.5, -1.2, 2.0])

    # Forward: T(m)
    Tm = T(m_test)
    # Adjoint: T*(lam)
    Tstar_lam = T.adjoint(lam)

    # <T(m), lam>_{R^N_p}
    inner_data = float(np.dot(Tm, lam))

    # <m, T*(lam)>_{M_st}: sum over components
    # vp component: integral of c_vp * f_vp_vals over [0, R]
    r_check = np.linspace(0.0, specs.earth_radius_km, 500)
    r_IC_check = r_check[r_check <= specs.icb_radius_km]
    r_M_check = r_check[r_check >= specs.cmb_radius_km]

    ip_vp = float(np.trapezoid(c_vp * Tstar_lam[0][0](r_check), r_check))
    # vs components: all-zero in m_test → zero inner product
    ip_rho = float(np.trapezoid(c_rho * Tstar_lam[0][2](r_check), r_check))
    ip_sigma_1 = float(sigma_1_val * Tstar_lam[1][1][0])

    inner_model = ip_vp + ip_rho + ip_sigma_1

    np.testing.assert_allclose(
        inner_data, inner_model, rtol=1e-3,
        err_msg=(
            f"Adjoint inconsistency: <T(m),lam>={inner_data:.6g} "
            f"vs <m,T*(lam)>={inner_model:.6g}"
        ),
    )


# =============================================================================
# Test 7: vs BulkTarget forward and adjoint — exercises IC/M subdomain paths
# =============================================================================

def test_property_operator_vs_target():
    """BulkTarget(param='vs') integrates over IC and M subdomains separately."""
    blocks, forward_dict, specs = _make_phase5_setup()

    # Use r0 inside IC domain so the IC adjoint component is non-zero
    targets = [
        BulkTarget(param='vs', lat_deg=0.0, lon_deg=0.0, sigma_ang_deg=20.0,
                   r0_km=500.0, sigma_r_km=300.0),
    ]

    property_op_dict = build_property_operator(
        targets, blocks, forward_dict, specs, s_max=S_MAX, n_radial=200,
    )

    block = blocks[0]
    T = property_op_dict[block]
    _, _, M_st, _ = forward_dict[block]

    # Build model element with constant vs = 1.0 in both subdomains, rest zero
    from intervalinf.core.functions import Function as _IFunction
    const_fn = lambda domain: _IFunction(
        domain, evaluate_callable=lambda r: np.ones_like(np.asarray(r, dtype=float))
    )
    zero_fn = lambda domain: _IFunction(
        domain, evaluate_callable=lambda r: np.zeros_like(np.asarray(r, dtype=float))
    )
    domain_full = IntervalDomain(0.0, specs.earth_radius_km)
    domain_IC = IntervalDomain(0.0, specs.icb_radius_km)
    domain_M = IntervalDomain(specs.cmb_radius_km, specs.earth_radius_km)

    m_vs_only = [
        [
            zero_fn(domain_full),
            [const_fn(domain_IC), const_fn(domain_M)],
            zero_fn(domain_full),
        ],
        [np.zeros(1), np.zeros(1)],
    ]

    # Forward: T(m)[0] should be B_st * integral(h over IC + h over M)
    out = T(m_vs_only)
    assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"

    # Adjoint: T*(e_0) should produce non-zero f_vs_IC (r0=500 km is in IC),
    # vp and rho should be exactly zero
    lam = np.array([1.0])
    adj = T.adjoint(lam)
    r_full = np.linspace(0.0, specs.earth_radius_km, 300)
    np.testing.assert_allclose(adj[0][0](r_full), 0.0, atol=1e-12,
                               err_msg="vp adjoint component should be zero for vs-only target")
    np.testing.assert_allclose(adj[0][2](r_full), 0.0, atol=1e-12,
                               err_msg="rho adjoint component should be zero for vs-only target")
    # f_vs_IC should be non-zero: bump centered at 500 km, sigma 300 km, well inside IC
    r_IC = np.linspace(0.0, specs.icb_radius_km, 100)
    assert np.any(np.abs(adj[0][1][0](r_IC)) > 1e-3), \
        "vs_IC adjoint component should be non-zero for bump centered in IC"
