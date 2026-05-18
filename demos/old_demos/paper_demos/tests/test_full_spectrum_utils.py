"""Tests for full_spectrum_utils — Phase 1 (block indexing) and Phase 2 (forward operators)."""

import sys
import os

# Ensure paper_demos is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import dataclasses
from full_spectrum_utils import (
    BlockIndex, enumerate_blocks, block_data_split,
    RadialSpecs, make_radial_space, build_block_forward,
    build_shared_bessel_blocks, prior_power_spectrum_default, build_block_prior,
)
from normal_mode_kernel_utils import NormalModeDataRegistry, NormalModeKernelCatalog, EARTH_RADIUS_KM
from intervalinf import (
    Lebesgue, IntervalDomain, LebesgueIntegrationConfig,
    IntegrationConfig, ParallelConfig, Function,
)

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


# =============================================================================
# Phase 2 helpers
# =============================================================================

def make_specs():
    """Return a RadialSpecs with small n_basis for speed in tests."""
    return RadialSpecs(n_basis=30)


# =============================================================================
# Phase 2 Tests
# =============================================================================

def test_make_radial_space_unit_weight_matches_lebesgue():
    """make_radial_space with weight=1.0 produces a Lebesgue with equivalent inner product."""
    domain = IntervalDomain(0, EARTH_RADIUS_KM)
    cfg = IntegrationConfig(method='trapz', n_points=256)
    space = make_radial_space(30, domain, weight=1.0, basis='ND', integration_config=cfg)
    ref = Lebesgue(30, domain, basis='ND', integration_config=cfg)

    fn = lambda x: np.sin(np.pi * x / EARTH_RADIUS_KM)
    f = Function(space, evaluate_callable=fn)
    g = Function(ref, evaluate_callable=fn)

    np.testing.assert_allclose(
        space.inner_product(f, f),
        ref.inner_product(g, g),
        rtol=1e-6,
        err_msg="Inner product mismatch between make_radial_space and Lebesgue",
    )


def test_make_radial_space_nonunit_weight_raises():
    """make_radial_space raises NotImplementedError for weight != 1.0."""
    domain = IntervalDomain(0, EARTH_RADIUS_KM)
    with pytest.raises(NotImplementedError):
        make_radial_space(10, domain, weight=2.0)


def test_block_forward_shapes():
    """build_block_forward returns G with codomain dim == N_d and applies to zero."""
    catalog = NormalModeKernelCatalog(KERNEL_DIR)
    reg = NormalModeDataRegistry(DATA_DIR, mode_filter=catalog.list_modes())
    reg_st = reg.filter_by_st(s=2, t=0)
    N_d = len(reg_st)
    specs = make_specs()
    G_st, C_D_st, M_st, D_st = build_block_forward(2, 0, reg_st, catalog, specs)

    assert D_st.dim == N_d, f"Expected D_st.dim={N_d}, got {D_st.dim}"

    m_zero = M_st.zero
    result = G_st(m_zero)
    assert result.shape == (N_d,), f"Expected shape ({N_d},), got {result.shape}"


def test_block_forward_noise_covariance_diagonal():
    """C_D_st diagonal matches reg_st.covariance_diagonal."""
    catalog = NormalModeKernelCatalog(KERNEL_DIR)
    reg = NormalModeDataRegistry(DATA_DIR, mode_filter=catalog.list_modes())
    reg_st = reg.filter_by_st(s=2, t=0)
    specs = make_specs()
    G_st, C_D_st, M_st, D_st = build_block_forward(2, 0, reg_st, catalog, specs)

    expected = reg_st.covariance_diagonal
    C_D_matrix = C_D_st.covariance.matrix(dense=True)
    np.testing.assert_allclose(
        np.diag(C_D_matrix), expected, rtol=1e-10,
        err_msg="C_D_st diagonal doesn't match reg_st.covariance_diagonal",
    )


# =============================================================================
# Phase 3 Tests
# =============================================================================

def test_shared_bessel_is_built_once_per_parameter():
    """build_shared_bessel_blocks returns exactly one BesselSobolevInverse per parameter."""
    from intervalinf.operators import BesselSobolevInverse
    specs = make_specs()
    shared = build_shared_bessel_blocks(specs)
    assert set(shared.keys()) == {'vp', 'vs_IC', 'vs_M', 'rho'}, (
        f"Expected keys {{'vp','vs_IC','vs_M','rho'}}, got {set(shared.keys())}"
    )
    for p, op in shared.items():
        assert isinstance(op, BesselSobolevInverse), (
            f"Expected BesselSobolevInverse for '{p}', got {type(op)}"
        )


def test_block_prior_covariance_scales_with_tau():
    """Doubling tau quadruples the radial covariance output (tau² scaling)."""
    specs = make_specs()
    shared = build_shared_bessel_blocks(specs)

    prior_1 = build_block_prior(2, 0, shared, specs, tau_fn=lambda p, s: 1.0)
    prior_2 = build_block_prior(2, 0, shared, specs, tau_fn=lambda p, s: 2.0)

    R = specs.earth_radius_km
    ICB = specs.icb_radius_km
    CMB = specs.cmb_radius_km
    M_vp = Lebesgue(0, IntervalDomain(0, R), basis=None)
    M_vs_IC = Lebesgue(0, IntervalDomain(0, ICB), basis=None)
    M_vs_M = Lebesgue(0, IntervalDomain(CMB, R), basis=None)
    M_rho = Lebesgue(0, IntervalDomain(0, R), basis=None)

    const = lambda x: np.ones_like(np.asarray(x, dtype=float))
    zero_fn = lambda x: np.zeros_like(np.asarray(x, dtype=float))
    f_vp = Function(M_vp, evaluate_callable=const)
    f_zero_vsIC = Function(M_vs_IC, evaluate_callable=zero_fn)
    f_zero_vsM = Function(M_vs_M, evaluate_callable=zero_fn)
    f_zero_rho = Function(M_rho, evaluate_callable=zero_fn)

    x_input = [
        [f_vp, [f_zero_vsIC, f_zero_vsM], f_zero_rho],
        [np.ones(1), np.ones(1)],
    ]
    out_1 = prior_1.covariance(x_input)
    out_2 = prior_2.covariance(x_input)

    x_eval = R / 2.0
    # vp output: tau doubled → tau²=4× covariance
    np.testing.assert_allclose(
        out_2[0][0](x_eval), 4.0 * out_1[0][0](x_eval), rtol=1e-10,
        err_msg="Radial covariance should scale as tau² when tau is doubled",
    )
    # Sigma output: fixed sigma_var, unaffected by tau_fn
    np.testing.assert_allclose(
        out_1[1][0], out_2[1][0], rtol=1e-10,
        err_msg="Sigma covariance should be unchanged by tau_fn",
    )


def test_block_prior_is_block_diagonal():
    """Prior covariance is block-diagonal: nonzero vp input produces zero vs/rho output."""
    specs = make_specs()
    shared = build_shared_bessel_blocks(specs)
    prior = build_block_prior(2, 0, shared, specs)

    R = specs.earth_radius_km
    ICB = specs.icb_radius_km
    CMB = specs.cmb_radius_km
    M_vp = Lebesgue(0, IntervalDomain(0, R), basis=None)
    M_vs_IC = Lebesgue(0, IntervalDomain(0, ICB), basis=None)
    M_vs_M = Lebesgue(0, IntervalDomain(CMB, R), basis=None)
    M_rho = Lebesgue(0, IntervalDomain(0, R), basis=None)

    const = lambda x: np.ones_like(np.asarray(x, dtype=float))
    zero_fn = lambda x: np.zeros_like(np.asarray(x, dtype=float))
    f_vp = Function(M_vp, evaluate_callable=const)
    f_zero_vsIC = Function(M_vs_IC, evaluate_callable=zero_fn)
    f_zero_vsM = Function(M_vs_M, evaluate_callable=zero_fn)
    f_zero_rho = Function(M_rho, evaluate_callable=zero_fn)

    x_input = [
        [f_vp, [f_zero_vsIC, f_zero_vsM], f_zero_rho],
        [np.zeros(1), np.zeros(1)],
    ]
    out = prior.covariance(x_input)

    # Evaluate within each component's own subdomain
    x_vsIC = ICB / 2.0                # midpoint of [0, ICB]
    x_vsM = (CMB + R) / 2.0          # midpoint of [CMB, R]
    x_rho = R / 2.0                   # midpoint of [0, R]
    # vs_IC, vs_M, rho outputs must be zero (input is zero for these components)
    np.testing.assert_allclose(
        out[0][1][0](x_vsIC), 0.0, atol=1e-12,
        err_msg="vs_IC off-diagonal block should be zero",
    )
    np.testing.assert_allclose(
        out[0][1][1](x_vsM), 0.0, atol=1e-12,
        err_msg="vs_M off-diagonal block should be zero",
    )
    np.testing.assert_allclose(
        out[0][2](x_rho), 0.0, atol=1e-12,
        err_msg="rho off-diagonal block should be zero",
    )


# =============================================================================
# Phase 4 Tests
# =============================================================================

from full_spectrum_utils import solve_block, solve_all_blocks
from pygeoinf import GaussianMeasure


def _make_phase4_setup(s_max=2, n_basis=20):
    """Build forward_dict, prior_dict, split, blocks for Phase 4 tests."""
    catalog = NormalModeKernelCatalog(KERNEL_DIR)
    reg = NormalModeDataRegistry(DATA_DIR, mode_filter=catalog.list_modes())
    blocks = enumerate_blocks(reg, s_max=s_max)
    split = block_data_split(reg, blocks)
    specs = RadialSpecs(
        n_basis=n_basis,
        parallel_cfg=ParallelConfig(enabled=False, n_jobs=1),
    )
    shared = build_shared_bessel_blocks(specs)
    forward_dict = {
        b: build_block_forward(b.s, b.t, split[b], catalog, specs)
        for b in blocks
    }
    prior_dict = {
        b: build_block_prior(b.s, b.t, shared, specs)
        for b in blocks
    }
    return blocks, forward_dict, prior_dict, split


def test_solve_block_smoke_one_block():
    """solve_block returns a GaussianMeasure with an accessible expectation and covariance."""
    blocks, forward_dict, prior_dict, split = _make_phase4_setup(s_max=2, n_basis=20)
    block = blocks[0]
    G_st, C_D_st, M_st, D_st = forward_dict[block]
    prior_st = prior_dict[block]
    d_st = split[block].data_vector

    posterior = solve_block(block.s, block.t, G_st, C_D_st, prior_st, d_st)

    assert isinstance(posterior, GaussianMeasure), (
        f"Expected GaussianMeasure, got {type(posterior)}"
    )
    # Expectation must exist and have the right nesting structure (list of lists)
    exp = posterior.expectation
    assert exp is not None, "Posterior expectation should not be None"
    assert isinstance(exp, list), f"Expected list structure for expectation, got {type(exp)}"
    assert len(exp) == 2, f"Expected 2-element list (functions, euclidean), got len={len(exp)}"

    # Apply covariance to zero vector; must not raise and must return same structure
    m_zero = M_st.zero
    cov_out = posterior.covariance(m_zero)
    assert cov_out is not None, "Covariance applied to zero should return something"


def test_solve_block_posterior_reduces_uncertainty():
    """Posterior mean predicts data better than the zero prior mean."""
    blocks, forward_dict, prior_dict, split = _make_phase4_setup(s_max=2, n_basis=20)
    block = blocks[0]
    G_st, C_D_st, M_st, D_st = forward_dict[block]
    prior_st = prior_dict[block]
    d_st = split[block].data_vector

    posterior = solve_block(block.s, block.t, G_st, C_D_st, prior_st, d_st)

    # Prior mean is zero → G(0) = 0 → residual equals ||d_st||
    d_pred_prior = G_st(M_st.zero)
    residual_prior = np.linalg.norm(d_st - d_pred_prior)

    # Posterior mean should predict data better (or equal at worst)
    d_pred_post = G_st(posterior.expectation)
    residual_post = np.linalg.norm(d_st - d_pred_post)

    assert residual_post <= residual_prior + 1e-12, (
        f"Posterior residual {residual_post:.6g} should be <= prior residual {residual_prior:.6g}"
    )


def test_solve_all_blocks_parallel_matches_serial():
    """n_jobs=1 and n_jobs=2 produce identical posterior sigma_1 expectations."""
    blocks, forward_dict, prior_dict, split = _make_phase4_setup(s_max=2, n_basis=20)

    serial_results = solve_all_blocks(forward_dict, prior_dict, split, n_jobs=1)
    parallel_results = solve_all_blocks(forward_dict, prior_dict, split, n_jobs=2)

    assert set(serial_results.keys()) == set(parallel_results.keys()), (
        "Block keys differ between serial and parallel runs"
    )
    for b in blocks:
        # expectation structure: [[f_vp, [f_vs_IC, f_vs_M], f_rho], [sigma_0, sigma_1]]
        # sigma_1 is expectation[1][1], a numpy array of shape (1,)
        sigma_1_serial = serial_results[b].expectation[1][1]
        sigma_1_parallel = parallel_results[b].expectation[1][1]
        np.testing.assert_allclose(
            sigma_1_serial, sigma_1_parallel, atol=1e-10,
            err_msg=f"sigma_1 posterior mean differs for block (s={b.s}, t={b.t})",
        )
