# Plan: General-ell Radial Laplacian and Degree-Aware Weighted Priors

**Status:** COMPLETE — 2026-06-06
**Owner context:** Adrian (normal-mode full-spectrum PLI / weighted radial priors)
**Goal:** Implement `RadialLaplacian(ell > 0)` and use `ell=s` for weighted
full-spectrum `(s,t)` priors, so the radial covariance reflects the angular
degree of each spherical-harmonic block.

## Completion Notes

- Implemented general `ell > 0` radial spectral modes with SciPy spherical
  Bessel functions, robust bracketing, cached roots/modes, and numerical
  `r^2 dr` normalization.
- Preserved the specialized `ell=0` providers and restricted the radial
  Dirichlet DST fast path to `ell=0`.
- Wired weighted full-spectrum priors, prior calibration, prior maps, tuner,
  remote, and headless paths so weighted radial covariance blocks are keyed by
  `(component, degree)` and built with `RadialLaplacian(..., ell=s)`.
- Added focused operator/Bessel tests and paper-demo weighted-prior smoke
  tests. Targeted operator tests pass; targeted paper-demo tests pass. The
  broader intervalinf suite still has unrelated pre-existing SOLA/cache
  failures and one collection-time missing `pygeoinf.matrix_function` issue.

## Why

The current weighted radial prior uses `WeightedLebesgue(r^2)` and
`RadialLaplacian`, but the spectral radial Laplacian only implements `ell=0`.
The paper-demo weighted prior builder currently constructs one shared
`ell=0` Bessel-Sobolev covariance per physical component and reuses it across
all `(s,t)` blocks.

For a 3D isotropic Laplacian separated in spherical harmonics, the radial
operator for angular degree `ell` is

```math
L_\ell f
=
-f'' - \frac{2}{r}f' + \frac{\ell(\ell+1)}{r^2}f
=
-\frac{1}{r^2}(r^2 f')' + \frac{\ell(\ell+1)}{r^2}f .
```

Using `ell=0` for all degrees omits the angular stiffness term
`ell(ell+1)/r^2`, so high-degree blocks receive the wrong radial covariance
shape.

## Current State

- `intervalinf/operators/radial.py` accepts an `ell` argument on
  `RadialLaplacian`, `InverseRadialLaplacian`,
  `RadialLaplacianSpectrumProvider`, and
  `RadialLaplacianEigenvalueProvider`.
- `RadialLaplacianEigenvalueProvider._compute_eigenvalue()` raises
  `NotImplementedError` for `ell != 0`.
- `intervalinf/providers/radial.py` contains special `ell=0` eigenfunction
  providers for `(0,R)` and shell domains `(a,b)`.
- `demos/old_demos/paper_demos/utils/full_spectrum_utils.py`
  `build_shared_bessel_blocks()` builds one `BesselSobolevInverse` per
  component, not per `(component, degree)`.
- The tuner and remote helper rely on the same shared-Bessel abstraction.
- Prior-predictive tau calibration currently assumes one base radial covariance
  per component and only changes `tau_{p,s}` by degree.

## Mathematical Specification

Use the transformed equation with `u(r)=r f(r)`:

```math
-u'' + \frac{\ell(\ell+1)}{r^2}u = \lambda u,
\qquad \lambda=k^2 .
```

For `lambda > 0`, radial eigenfunctions are

```math
f(r)=A j_\ell(kr)+B y_\ell(kr),
```

where `j_ell` and `y_ell` are spherical Bessel functions.

For domains starting at zero, regularity forces `B=0`, so

```math
f(r)=A j_\ell(kr).
```

Normalize every eigenfunction numerically in the weighted inner product:

```math
\int_a^b \phi_n(r)^2 r^2\,dr = 1 .
```

### Boundary Conditions

Support the boundary cases already used by the current radial implementation.

For `(0,R)`:

- regularity at `r=0` plus outer `dirichlet`;
- regularity at `r=0` plus outer `neumann`.

For shell domains `(a,b)` with `a > 0`:

- `dirichlet`;
- `neumann`;
- `mixed_dirichlet_neumann`;
- `mixed_neumann_dirichlet`.

For `(0,R)` root equations:

- Dirichlet: `j_ell(kR)=0`.
- Neumann: `k j_ell'(kR)=0`.

For shell domains, build a 2x2 endpoint matrix and solve
`det M_ell(k)=0`.

Endpoint rows:

- Dirichlet at `r0`: `[j_ell(k r0), y_ell(k r0)]`.
- Neumann at `r0`: `[k j_ell'(k r0), k y_ell'(k r0)]`.

Use `scipy.special.spherical_jn`, `scipy.special.spherical_yn`, and
`scipy.optimize.brentq` or `root_scalar(method="brentq")`. Do not add new
dependencies.

## Implementation Phases

### Phase 1: General spectral providers

**Objective:** Add correct `ell > 0` eigenvalues and eigenfunctions for
`RadialLaplacian(method="spectral")`.

**Implementation notes:**

1. Keep the existing `ell=0` providers and formulas as the compatibility
   baseline.
2. Add a general provider path in `intervalinf/providers/radial.py` for
   `ell > 0`.
3. Cache roots, normalization constants, and constructed `Function` objects.
4. Use numerical normalization with the domain's integration settings or a
   conservative high-resolution trapezoidal/Simpson rule.
5. For shell roots, scan increasing `k` intervals and collect the first
   `dofs` positive roots. Use robust sign-change bracketing before Brent
   refinement.
6. For `ell > 0`, do not create a zero Neumann mode. The constant zero mode is
   only the `ell=0` pure-Neumann case.

**Acceptance criteria:**

- `RadialLaplacian(..., ell=1)` and `ell=2` return increasing positive
  eigenvalues for supported BCs.
- Returned eigenfunctions satisfy endpoint conditions numerically.
- Weighted Gram matrices of the first few modes are close to identity.
- Existing `ell=0` tests continue to pass unchanged.

### Phase 2: Bessel-Sobolev compatibility

**Objective:** Ensure `BesselSobolev` and `BesselSobolevInverse` work with
general-`ell` radial spectra.

**Implementation notes:**

1. The existing slow weighted projection path should work once
   `RadialLaplacian.get_eigenfunction()` and `.get_eigenvalue()` work.
2. Keep the current radial Dirichlet DST fast path restricted to `ell=0`.
3. Update `_radial_dirichlet_fast_eligible(L)` so it explicitly checks
   `L._ell == 0`.
4. Add tests showing `BesselSobolevInverse` scales `ell > 0` eigenfunctions by
   `(k^2 + lambda_n)^(-s_order/2)`.

**Acceptance criteria:**

- Weighted Bessel covariance remains self-adjoint and positive for `ell > 0`.
- No incorrect fast transform is used for `ell > 0`.

### Phase 3: Degree-aware weighted prior construction

**Objective:** Use `ell=s` for weighted full-spectrum priors.

**Implementation notes:**

1. In weighted mode, replace component-only shared Bessel blocks with
   degree-aware blocks keyed by `(component, degree)`.
2. In flat mode, keep the existing `{component: BesselSobolevInverse}` shape.
3. Add a small resolver helper so callers do not manually branch on the
   shared-Bessel key shape.
4. Update `build_block_prior(s, t, shared_bessel, specs, ...)` so weighted
   mode resolves the covariance for `(component, s)`.
5. Update the tuner helper and remote helper paths to use the same resolver.
6. Keep synchronized `k`, Sobolev order, and `alpha` across all blocks; only
   the radial Laplacian `ell=s` changes by block.

**Acceptance criteria:**

- Weighted `build_block_prior(s=2, ...)` uses `RadialLaplacian._ell == 2`.
- Weighted all-block posterior and inference paths build priors without domain
  mismatches.
- Flat-mode prior construction remains unchanged.

### Phase 4: Prior calibration update

**Objective:** Calibrate `tau_{p,s}` on top of degree-specific weighted radial
covariances.

**Implementation notes:**

1. In weighted mode, build radial synthesis matrices per `(component, degree)`.
2. In flat mode, keep one synthesis matrix per component.
3. In the Monte Carlo reconstruction loop, each block draws from its own
   degree-specific base covariance, then applies the current degree weight and
   calibrated `tau_{p,s}`.
4. Keep the calibration output schema unchanged:
   `tau_by_degree.csv`, `tau_st.csv`, `component_amplitude_summary.csv`,
   `calibration_samples.npz`.
5. Keep the interpretation of `tau_{p,s}` unchanged: it is a scalar amplitude
   on top of the base radial covariance for that component and degree.

**Acceptance criteria:**

- A small weighted calibration smoke run completes with `s_max=2`,
  `n_basis=8`, and `n_mc=4`.
- The calibrated 95% max-amplitude diagnostic still lands near the configured
  target in the smoke setup.

### Phase 5: Documentation and cleanup

**Objective:** Document the new operator behavior and keep future agents from
returning to the shared `ell=0` weighted prior.

**Implementation notes:**

1. Update the living intervalinf reference with the general-`ell`
   `RadialLaplacian` behavior.
2. Update the full-spectrum paper-demo notes to explain that weighted priors
   use `ell=s` radial covariance blocks.
3. Move this plan to completed plans when implementation and tests are done.

## Test Plan

Add or update focused tests rather than relying only on paper-demo smoke runs.

Operator tests:

- `(0,R)` Dirichlet and Neumann boundary residuals for `ell=1,2`.
- Shell DD/DN/ND/NN boundary residuals for `ell=1,2`.
- Weighted orthonormality of the first few modes with explicit `rtol`/`atol`.
- Monotone positive eigenvalues for `ell > 0`.
- Regression tests for existing `ell=0` eigenvalues and zero-mode behavior.

Bessel tests:

- `BesselSobolevInverse(phi_n)` equals
  `(k^2 + lambda_n)^(-s_order/2) phi_n` for `ell > 0`.
- Self-adjointness and positivity in `WeightedLebesgue(r^2)`.
- `_radial_dirichlet_fast` is true for `ell=0` Dirichlet and false for
  `ell > 0`.

Paper-demo tests:

- Weighted `build_block_prior()` chooses `ell=s`.
- Weighted prior calibration smoke test with tiny dimensions.
- Local tuner/remote parameter resolver handles degree-aware weighted priors.
- Flat prior tests still pass with the old component-only shared covariance.

Run targets:

```bash
cd intervalinf
/home/adrian/miniconda3/envs/inferences/bin/python -m pytest \
  tests/operators/test_weighted_bessel_radial.py \
  tests/operators/test_operators.py \
  tests/test_full_spectrum_utils.py \
  tests/test_prior_viz.py -q
```

