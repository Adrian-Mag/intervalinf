"""
full_spectrum_utils.py
======================

Utilities for the full-spectrum splitting-function PLI example.

Provides:
    BlockIndex          — frozen dataclass for a (s, t) spectral block.
    enumerate_blocks    — list all non-empty (s, t) blocks for even s <= s_max.
    block_data_split    — split a NormalModeDataRegistry by BlockIndex.

    RadialSpecs         — dataclass holding shared configuration (Phase 2).
    make_radial_space   — construct a Lebesgue model space (Phase 2).
    build_block_forward — build model space + forward operator for one (s,t) block (Phase 2).

    prior_power_spectrum_default — default per-parameter spectral amplitude τ_{p,s} (Phase 3).
    build_shared_bessel_blocks   — build one BesselSobolevInverse per radial parameter (Phase 3).
    build_block_prior            — assemble block GaussianMeasure prior (Phase 3).
"""

import dataclasses
from typing import List, Dict, Optional, Tuple

import numpy as np

from intervalinf import (
    Lebesgue, IntervalDomain, LebesgueIntegrationConfig,
    IntegrationConfig, ParallelConfig, LebesgueSpaceDirectSum,
    KnownRegion, PartitionedLebesgueSpace, BoundaryConditions,
)
from intervalinf.operators import Laplacian, BesselSobolevInverse
from intervalinf.core.functions import Function as _IFunction
from normal_mode_kernel_utils import (
    NormalModeDataRegistry, NormalModeKernelCatalog,
    NormalModeSplittingKernelProvider,
    EARTH_RADIUS_KM, PREM_LAYER_CMB,
)
from pygeoinf import (
    EuclideanSpace, RowLinearOperator, GaussianMeasure,
    HilbertSpaceDirectSum, LinearOperator,
)


@dataclasses.dataclass(frozen=True, order=True)
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


# =============================================================================
# Phase 2 — RadialSpecs, make_radial_space, build_block_forward
# =============================================================================

@dataclasses.dataclass
class RadialSpecs:
    """
    Shared configuration for full-spectrum radial model spaces and forward operators.

    All fields have sensible defaults.  Pass a ``RadialSpecs`` instance to
    :func:`build_block_forward` instead of threading individual config objects
    through every call.

    Attributes
    ----------
    n_basis : int
        Number of basis functions per model component (vp, vs, rho).
        Default 100.
    earth_radius_km : float
        Earth radius in km defining the outer boundary of the radial domain.
        Default ``EARTH_RADIUS_KM`` (6371.0).
    icb_radius_km : float
        Inner Core Boundary radius in km.  Vs is assumed zero in the outer
        core $[\\text{icb\\_radius\\_km}, \\text{cmb\\_radius\\_km}]$.
        Default 1217.5 km.
    cmb_radius_km : float
        Core-Mantle Boundary radius in km.
        Default 3480.0 km.
    lebesgue_cfg : LebesgueIntegrationConfig
        Integration configuration used when constructing Lebesgue spaces.
        Defaults to trapz with 1024 points for all sub-systems.
    sola_cfg : IntegrationConfig
        Integration configuration passed to ``SOLAOperator``.
        Defaults to trapz with 2048 points.
    parallel_cfg : ParallelConfig
        Parallelisation configuration for space construction.
        Defaults to serial (``n_jobs=1``).
    """

    n_basis: int = 100
    earth_radius_km: float = EARTH_RADIUS_KM
    icb_radius_km: float = 1217.5
    cmb_radius_km: float = 3480.0
    lebesgue_cfg: Optional[LebesgueIntegrationConfig] = None
    sola_cfg: Optional[IntegrationConfig] = None
    parallel_cfg: Optional[ParallelConfig] = None

    def __post_init__(self):
        if self.lebesgue_cfg is None:
            self.lebesgue_cfg = LebesgueIntegrationConfig(
                inner_product=IntegrationConfig(method='trapz', n_points=1024),
                dual=IntegrationConfig(method='trapz', n_points=1024),
                general=IntegrationConfig(method='trapz', n_points=1024),
            )
        if self.sola_cfg is None:
            self.sola_cfg = IntegrationConfig(method='trapz', n_points=2048)
        if self.parallel_cfg is None:
            self.parallel_cfg = ParallelConfig(enabled=False, n_jobs=1)


def make_radial_space(
    n: int,
    domain: IntervalDomain,
    *,
    weight: float = 1.0,
    basis: str = 'ND',
    integration_config: Optional[IntegrationConfig] = None,
    parallel_config: Optional[ParallelConfig] = None,
) -> Lebesgue:
    """
    Construct a Lebesgue model space on *domain* with *n* basis functions.

    Currently only the unweighted ($w \\equiv 1$) case is supported.  A
    radially-weighted $r^2$ inner product is planned but not yet implemented
    in intervalinf.

    Parameters
    ----------
    n : int
        Number of basis functions.
    domain : IntervalDomain
        Radial domain (e.g. $[0, R_{\\oplus}]$).
    weight : float
        Inner-product weight.  Must be 1.0; any other value raises
        ``NotImplementedError``.
    basis : str
        Basis type string passed to :class:`Lebesgue` (default ``'ND'``).
    integration_config : IntegrationConfig, optional
        Integration configuration.  If *None*, the :class:`Lebesgue` default
        is used.
    parallel_config : ParallelConfig, optional
        Parallelisation configuration.

    Returns
    -------
    Lebesgue
        The constructed model space.

    Raises
    ------
    NotImplementedError
        If ``weight != 1.0``.
    """
    if weight != 1.0:
        raise NotImplementedError(
            "weight != 1.0 requires r^2-elliptic covariances not yet implemented in intervalinf"
        )
    return Lebesgue(
        n,
        domain,
        basis=basis,
        integration_config=integration_config,
        parallel_config=parallel_config,
    )


def build_block_forward(
    s: int,
    t: int,
    reg_st: NormalModeDataRegistry,
    catalog: NormalModeKernelCatalog,
    specs: RadialSpecs,
) -> Tuple:
    """
    Build the model space and forward operator for a single $(s, t)$ block.

    Follows the structure of the single-(s,t) inference in ``example_2.ipynb``:

    - **vp** and **rho** span the full radial domain $[0, R_{\\oplus}]$.
    - **vs** is zero in the outer core $[r_{\\text{ICB}}, r_{\\text{CMB}}]$;
      the active regions (inner core and mantle) are handled via
      :class:`PartitionedLebesgueSpace`.
    - Two scalar topography parameters $\\sigma_0$ (ICB, zero sensitivity) and
      $\\sigma_1$ (CMB) complete the model.

    The full model space is
    $$\\mathcal{M} = (M_{v_p} \\oplus M_{v_s} \\oplus M_{\\rho}) \\oplus (M_{\\sigma_0} \\oplus M_{\\sigma_1})$$

    and the forward operator is the corresponding block-row operator
    $G = [G_{v_p},\\, G_{v_s},\\, G_{\\rho},\\, G_{\\sigma_0},\\, G_{\\sigma_1}]$.

    Parameters
    ----------
    s : int
        Splitting degree of the block.
    t : int
        Real-harmonic component index.
    reg_st : NormalModeDataRegistry
        Sub-registry pre-filtered to observations for this $(s, t)$ block.
    catalog : NormalModeKernelCatalog
        Sensitivity kernel catalog providing radial kernels for each mode.
    specs : RadialSpecs
        Shared configuration (basis count, domain bounds, integration configs).

    Returns
    -------
    G_st : RowLinearOperator
        Full forward operator mapping $\\mathcal{M}$ to $\\mathcal{D}$.
    C_D_st : GaussianMeasure
        Data noise measure $\\mathcal{N}(0, C_D)$ with diagonal covariance
        derived from ``reg_st.covariance_diagonal``.
    M_st : HilbertSpaceDirectSum
        Full model space $\\mathcal{M}$.
    D_st : EuclideanSpace
        Data space $\\mathcal{D}$ of dimension $N_d = |\\text{reg\\_st}|$.
    """
    from intervalinf.operators import SOLAOperator

    N = specs.n_basis
    function_domain = IntervalDomain(0, specs.earth_radius_km)
    inner_product_cfg = specs.lebesgue_cfg.inner_product  # IntegrationConfig

    # ---- vp model space ----
    M_vp = Lebesgue(
        0, function_domain,
        basis=None,
        integration_config=inner_product_cfg,
        parallel_config=specs.parallel_cfg,
    )

    # ---- vs model space (zero in outer core) ----
    outer_core_interval = IntervalDomain(specs.icb_radius_km, specs.cmb_radius_km)
    outer_core = KnownRegion.zero(outer_core_interval)
    partitioned_vs = PartitionedLebesgueSpace(
        full_domain=function_domain,
        known_regions=[outer_core],
        dims=[0, 0],
        bases=[None, None],
        integration_config=inner_product_cfg,
        parallel_config=specs.parallel_cfg,
    )
    M_vs = partitioned_vs.model_space
    M_vs_IC = partitioned_vs.unknown_spaces[0]   # [0, ICB]
    M_vs_M = partitioned_vs.unknown_spaces[1]    # [CMB, R_earth]

    # ---- rho model space ----
    M_rho = Lebesgue(
        0, function_domain,
        basis=None,
        integration_config=inner_product_cfg,
        parallel_config=specs.parallel_cfg,
    )

    # ---- topography / discontinuity model spaces ----
    M_sigma_0 = EuclideanSpace(1)   # ICB (zero sensitivity)
    M_sigma_1 = EuclideanSpace(1)   # CMB

    # ---- combined model space ----
    M_functions = LebesgueSpaceDirectSum([M_vp, M_vs, M_rho])
    M_euclidean = HilbertSpaceDirectSum([M_sigma_0, M_sigma_1])
    M_st = HilbertSpaceDirectSum([M_functions, M_euclidean])

    # ---- data space ----
    N_d = len(reg_st)
    D_st = EuclideanSpace(N_d)

    # ---- kernel providers ----
    vp_provider = NormalModeSplittingKernelProvider(
        M_vp, catalog, registry=reg_st,
        interpolation_method='linear', kernel_type='vp',
    )
    rho_provider = NormalModeSplittingKernelProvider(
        M_rho, catalog, registry=reg_st,
        interpolation_method='linear', kernel_type='rho',
    )
    vs_full_space = Lebesgue(0, function_domain, basis=None)
    vs_full_provider = NormalModeSplittingKernelProvider(
        vs_full_space, catalog, registry=reg_st,
        interpolation_method='linear', kernel_type='vs',
    )
    vs_IC_provider = vs_full_provider.restrict(M_vs_IC)
    vs_M_provider = vs_full_provider.restrict(M_vs_M)

    # ---- topography sensitivity vectors ----
    K_sigma_1 = vp_provider.build_topo_matrix(prem_layer=PREM_LAYER_CMB)
    K_sigma_0 = np.zeros(N_d)

    # ---- SOLA forward operators ----
    G_vp = SOLAOperator(M_vp, D_st, vp_provider, integration_config=specs.sola_cfg)
    G_rho = SOLAOperator(M_rho, D_st, rho_provider, integration_config=specs.sola_cfg)
    G_vs_IC = SOLAOperator(M_vs_IC, D_st, vs_IC_provider, integration_config=specs.sola_cfg)
    G_vs_M = SOLAOperator(M_vs_M, D_st, vs_M_provider, integration_config=specs.sola_cfg)
    G_vs = RowLinearOperator([G_vs_IC, G_vs_M])

    G_sigma_0 = LinearOperator(M_sigma_0, D_st, lambda x: K_sigma_0 * x)
    G_sigma_1 = LinearOperator(M_sigma_1, D_st, lambda x: K_sigma_1 * x)
    G_functions = RowLinearOperator([G_vp, G_vs, G_rho])
    G_euclidean = RowLinearOperator([G_sigma_0, G_sigma_1])
    G_st = RowLinearOperator([G_functions, G_euclidean])

    # ---- data noise covariance ----
    C_D_matrix = np.diag(reg_st.covariance_diagonal)
    C_D_st = GaussianMeasure.from_covariance_matrix(
        D_st, C_D_matrix, expectation=np.zeros(N_d),
    )

    return G_st, C_D_st, M_st, D_st


# =============================================================================
# Phase 3 — Shared Bessel-Sobolev priors
# =============================================================================

# Per-parameter Bessel-Sobolev hyperparameters matching example_2.ipynb.
_BESSEL_PARAMS: Dict[str, dict] = {
    'vp':    {'s_order': 6.0, 'length': 20.0, 'overall_var': 10.0, 'bc': 'mixed_neumann_dirichlet'},
    'vs_IC': {'s_order': 4.0, 'length': 20.0, 'overall_var': 10.0, 'bc': 'neumann'},
    'vs_M':  {'s_order': 4.0, 'length': 20.0, 'overall_var': 10.0, 'bc': 'mixed_neumann_dirichlet'},
    'rho':   {'s_order': 5.0, 'length': 25.0, 'overall_var': 10.0, 'bc': 'mixed_neumann_dirichlet'},
}


def prior_power_spectrum_default(p: str, s: int) -> float:
    """
    Return the default per-parameter spectral amplitude $\\tau_{p,s}$ (flat in $s$).

    The covariance for parameter *p* in block *s* is $\\tau_{p,s}^2 C_p$,
    where $C_p$ is the normalised Bessel-Sobolev covariance operator.
    This default returns 1.0 for all parameters and all splitting degrees.

    Parameters
    ----------
    p : str
        Parameter name: one of ``'vp'``, ``'vs_IC'``, ``'vs_M'``, ``'rho'``.
    s : int
        Splitting degree (unused in this default).

    Returns
    -------
    float
        Amplitude $\\tau_{p,s} = 1.0$.
    """
    return 1.0


def build_shared_bessel_blocks(specs: RadialSpecs) -> Dict[str, 'BesselSobolevInverse']:
    """
    Build one :class:`BesselSobolevInverse` per radial parameter on reference spaces.

    Each operator is constructed once and can be shared across all $(s, t)$
    blocks via :func:`build_block_prior`, avoiding redundant eigendecompositions.

    The reference spaces mirror the component spaces created inside
    :func:`build_block_forward`:

    - ``'vp'``, ``'rho'``: full domain $[0, R_\\oplus]$, ``'ND'`` basis.
    - ``'vs_IC'``: inner-core domain $[0, r_\\mathrm{ICB}]$, ``'cosine'`` basis.
    - ``'vs_M'``: mantle domain $[r_\\mathrm{CMB}, R_\\oplus]$, ``'ND'`` basis.

    The Bessel-Sobolev hyperparameters are taken from ``_BESSEL_PARAMS``,
    matching ``example_2.ipynb``.

    Parameters
    ----------
    specs : RadialSpecs
        Shared configuration.  ``specs.n_basis`` sets the reference space
        dimension and the number of eigenfunctions in each operator.

    Returns
    -------
    dict mapping str to BesselSobolevInverse
        Keys: ``'vp'``, ``'vs_IC'``, ``'vs_M'``, ``'rho'``.
    """
    N = specs.n_basis
    cfg = specs.lebesgue_cfg.inner_product
    pcfg = specs.parallel_cfg
    domain = IntervalDomain(0, specs.earth_radius_km)
    ICB = specs.icb_radius_km
    CMB = specs.cmb_radius_km
    R = specs.earth_radius_km

    # Lebesgue reference spaces, one per parameter
    ref_spaces = {
        'vp':    Lebesgue(0, domain, basis=None, integration_config=cfg, parallel_config=pcfg),
        'vs_IC': Lebesgue(0, IntervalDomain(0, ICB), basis=None, integration_config=cfg, parallel_config=pcfg),
        'vs_M':  Lebesgue(0, IntervalDomain(CMB, R), basis=None, integration_config=cfg, parallel_config=pcfg),
        'rho':   Lebesgue(0, domain, basis=None, integration_config=cfg, parallel_config=pcfg),
    }

    # Integration config for BesselSobolevInverse (trapz is sufficient for tests)
    bessel_cfg = IntegrationConfig(method='trapz', n_points=max(2000, N * 40))
    n_samples = max(512, N * 16)

    result: Dict[str, 'BesselSobolevInverse'] = {}
    for p, M_ref in ref_spaces.items():
        prms = _BESSEL_PARAMS[p]
        k = np.power(prms['overall_var'], -0.5 / prms['s_order'])
        alpha = (prms['length'] ** 2) * (k ** 2)
        bcs = BoundaryConditions(bc_type=prms['bc'])
        L = Laplacian(
            M_ref, bcs, alpha,
            method='spectral', dofs=N, integration_config=cfg, n_samples=n_samples,
        )
        result[p] = BesselSobolevInverse(
            M_ref, M_ref, k, prms['s_order'], L,
            dofs=N, n_samples=n_samples, use_fast_transforms=True,
            integration_config=bessel_cfg,
        )
    return result


def build_block_prior(
    s: int,
    t: int,
    shared_bessel: Dict[str, 'BesselSobolevInverse'],
    specs: RadialSpecs,
    *,
    tau_fn=None,
    sigma_var: float = 100.0,
) -> GaussianMeasure:
    """
    Assemble a :class:`GaussianMeasure` prior for the $(s, t)$ block.

    The prior is block-diagonal over the six model components:

    $$Q = Q_{v_p} \\oplus Q_{v_s^\\mathrm{IC}} \\oplus Q_{v_s^\\mathrm{M}} \\oplus Q_\\rho
           \\oplus Q_{\\sigma_0} \\oplus Q_{\\sigma_1}$$

    Each radial component prior uses the shared Bessel-Sobolev covariance
    scaled by $\\tau_{p,s}^2$:

    $$C_{p,st} = \\tau_{p,s}^2 \\, C_p^\\mathrm{ref}$$

    where $\\tau_{p,s} = \\texttt{tau\\_fn}(p, s)$.  The topography priors
    are scalar Gaussian with variance ``sigma_var``.

    .. note::
        The $w(r) = r^2$ radial weighting needed for a geophysically correct
        inner product is **not yet implemented** in intervalinf.  The priors
        here use the flat ($w = 1$) inner product.

    The returned measure's covariance domain has the nested structure

    ``HilbertSpaceDirectSum([prior_functions, prior_euclidean])``

    which matches the ``M_st`` produced by :func:`build_block_forward`, so
    the two can be combined in :class:`~pygeoinf.linear_bayesian.LinearBayesianInversion`.

    Parameters
    ----------
    s : int
        Splitting degree.
    t : int
        Real-harmonic component index (unused beyond documenting the block).
    shared_bessel : dict
        Output of :func:`build_shared_bessel_blocks`.
    specs : RadialSpecs
        Shared configuration (domain bounds, basis count, integration configs).
    tau_fn : callable, optional
        ``tau_fn(p, s) -> float`` returning the amplitude $\\tau_{p,s}$.
        Defaults to :func:`prior_power_spectrum_default` (returns 1.0).
    sigma_var : float, optional
        Variance for the scalar topography priors $Q_{\\sigma_0}$, $Q_{\\sigma_1}$.
        Defaults to 100.0.

    Returns
    -------
    GaussianMeasure
        Block-diagonal prior measure on the full model space.
    """
    if tau_fn is None:
        tau_fn = prior_power_spectrum_default

    N = specs.n_basis
    cfg = specs.lebesgue_cfg.inner_product
    pcfg = specs.parallel_cfg
    domain = IntervalDomain(0, specs.earth_radius_km)
    ICB = specs.icb_radius_km
    CMB = specs.cmb_radius_km
    R = specs.earth_radius_km

    # Component model spaces (identical configuration to build_block_forward)
    M_vp = Lebesgue(0, domain, basis=None, integration_config=cfg, parallel_config=pcfg)
    M_vs_IC = Lebesgue(0, IntervalDomain(0, ICB), basis=None, integration_config=cfg, parallel_config=pcfg)
    M_vs_M = Lebesgue(0, IntervalDomain(CMB, R), basis=None, integration_config=cfg, parallel_config=pcfg)
    M_rho = Lebesgue(0, domain, basis=None, integration_config=cfg, parallel_config=pcfg)
    M_sigma_0 = EuclideanSpace(1)
    M_sigma_1 = EuclideanSpace(1)

    def _make_scaled_cov(p: str, M_block: Lebesgue) -> LinearOperator:
        """Create a scaled covariance LinearOperator: tau^2 * C_ref_p, reassociated to M_block."""
        tau_sq = tau_fn(p, s) ** 2
        shared_op = shared_bessel[p]

        def apply(f):
            # Reassociate f to shared_op's reference domain (same callable, different .space)
            f_ref = _IFunction(shared_op.domain, evaluate_callable=f.__call__)
            out_ref = shared_op(f_ref)
            # Scale by tau² and reassociate output to M_block
            return _IFunction(M_block, evaluate_callable=lambda x: tau_sq * out_ref(x))

        return LinearOperator(M_block, M_block, apply)

    # Per-component radial priors (expectation=None → zero mean)
    prior_vp = GaussianMeasure(covariance=_make_scaled_cov('vp', M_vp))
    prior_vs_IC = GaussianMeasure(covariance=_make_scaled_cov('vs_IC', M_vs_IC))
    prior_vs_M = GaussianMeasure(covariance=_make_scaled_cov('vs_M', M_vs_M))
    prior_rho = GaussianMeasure(covariance=_make_scaled_cov('rho', M_rho))

    # Scalar topography priors
    prior_sigma_0 = GaussianMeasure.from_covariance_matrix(
        M_sigma_0, np.array([[sigma_var]]), expectation=np.array([0.0]),
    )
    prior_sigma_1 = GaussianMeasure.from_covariance_matrix(
        M_sigma_1, np.array([[sigma_var]]), expectation=np.array([0.0]),
    )

    # Assemble with nested structure matching M_st from build_block_forward:
    #   M_st = HilbertSpaceDirectSum([M_functions, M_euclidean])
    #   M_functions = LebesgueSpaceDirectSum([M_vp, M_vs, M_rho])
    #   M_vs = LebesgueSpaceDirectSum([M_vs_IC, M_vs_M])
    prior_vs = GaussianMeasure.from_direct_sum([prior_vs_IC, prior_vs_M])
    prior_functions = GaussianMeasure.from_direct_sum([prior_vp, prior_vs, prior_rho])
    prior_euclidean = GaussianMeasure.from_direct_sum([prior_sigma_0, prior_sigma_1])
    return GaussianMeasure.from_direct_sum([prior_functions, prior_euclidean])
