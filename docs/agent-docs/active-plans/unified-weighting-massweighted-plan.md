# Plan: Unify inner-product weighting via MassWeightedHilbertSpace

**Status:** COMPLETE (Phases 1–4) — 2026-06-02. Reduced-operator weighted
follow-up (noted in Phase 3) optional/outstanding.
**Owner context:** Adrian (PhD geophysical inversion monorepo)
**Design decision:** memory `weighting-via-massweighted-design.md`. Implement all
interval inner-product weighting (e.g. radial `w(r)=r²`) through pygeoinf's
`MassWeightedHilbertSpace` (mass = multiply-by-`w`), **remove** the legacy
`Lebesgue(weight=...)` mechanism entirely (no bloat), and make `SOLAOperator`
adjoints correct on weighted domains via `M⁻¹`.

## Why
- `Lebesgue(weight=w)` only weighted `inner_product`; its Riesz maps
  (`to_dual`/`from_dual`) ignored `w`, so any operator's generic `.adjoint` was
  inconsistent on a weighted Lebesgue.
- `MassWeightedHilbertSpace` carries the weight in the mass operator `M`, so
  Riesz maps are weight-consistent by construction and adjoints become correct.
- A multiplicative weight is the diagonal special case of a general SPD mass
  (Sobolev already uses the differential mass `(k²I+Δ)^s`).

## Key technical notes (read before continuing)
- **LinearFormKernel legacy coupling:** `forms.py` `LinearFormKernel.__init__`
  does `self._weight = getattr(domain, '_weight', None)` and its `.kernel`
  property returns `kernel * (1/w)`. This is the legacy `Lebesgue(weight=)`
  dual machinery. It *conflicts* with the mass approach (it divides our
  already-mass-folded kernel by `w`). `WeightedLebesgue` therefore deliberately
  stores the weight as `self._weight_fn` (NOT `_weight`) to avoid the pickup.
  Phase 2 removes this legacy `_weight` logic from `forms.py`.
- **r=0 singularity:** `w=r²` ⇒ `1/w` singular at 0; mass is only PSD there.
  Use `[ε,R]` domains or a regularized `inverse_weight`.
- **Bessel fast-path gate:** the DST/DCT fast path computes *unweighted*
  coefficients, so it must be disabled for any weighted/mass-weighted domain.
  Detect via `isinstance(domain, MassWeightedHilbertSpace)` (after Phase 2 the
  `Lebesgue.weight` attribute no longer exists).
- **Bessel slow-path projection:** must use `domain.inner_product(φ_k, f)` for
  weighted domains (generic; works for any mass) rather than the unweighted
  `(f*φ_k).integrate()`.
- **SOLA adjoint:** generic `dual_mapping`→`from_dual` path is already correct
  for a space with correct Riesz; only the FAST/cached shortcuts hard-code the
  identity Riesz. Fix per the three `TODO(mass-weighted)` at sola.py:753, 891,
  967 — build an "adjoint-kernel matrix" with rows `M⁻¹(kᵢ)` (for plain L²,
  `M⁻¹=I` so rows = kernels, unchanged).
- **Pre-existing failures:** 6 tests in `tests/operators/test_sola.py` fail on a
  clean tree (kernel eval-cache instrumentation), UNRELATED to this work.
  Verified by stashing. Do not be alarmed; do not "fix" by accident.

## Environment
- `conda activate inferences` (Python 3.12). Run tests from `intervalinf/`:
  `python -m pytest tests/operators/... -q`. NOTE: `intervalinf` is a NESTED git
  repo (`intervalinf/.git`); commit there, not the parent.
- Full-suite operator/space tests are fast (~20s). Paper-demo tests load real
  kernel data and are slow (~8–13 min) — avoid unless needed.

---

## Phase 1 — `WeightedLebesgue` factory  ✅ DONE
**Goal:** `L²([a,b];w)` as `MassWeightedHilbertSpace(plain Lebesgue, M=×w, M⁻¹=×1/w)`.
**Files:**
- `intervalinf/spaces/weighted_lebesgue.py` (NEW) — `WeightedLebesgue` subclass
  with Sobolev-style kernel-based `to_dual`/`from_dual` (kernel = `Mx`,
  from_dual = `M⁻¹ kernel`). Public `weight`, `function_domain`, `integration`,
  `parallel` accessors. Mass ops are self-adjoint multiply-by-`w` LinearOperators.
- `intervalinf/spaces/__init__.py`, `intervalinf/__init__.py` — exported.
**Tests:** `tests/spaces/test_weighted_lebesgue.py` — 8 tests, all GREEN.
**Notes:**
- Had to name the weight attr `_weight_fn` (see LinearFormKernel note above).
- `test_to_dual_kernel_is_weighted` confirms kernel = `w·x`; roundtrip confirms
  `from_dual∘to_dual = id` (i.e. `M⁻¹M=I`).

---

## Phase 2 — Remove legacy `Lebesgue(weight=)`; migrate all consumers  ✅ DONE
**Goal:** Delete the old weighting mechanism; everything weighted uses
`WeightedLebesgue`.
**Files / steps:**
1. `intervalinf/spaces/lebesgue.py`: remove `weight` ctor param, `self._weight`,
   the `weight` property (added in the earlier Bessel work), weight branch in
   `_continuous_l2_inner_product`, and `weight=self._weight` threading in
   subdomain/`PartitionedLebesgueSpace`/`LebesgueSpaceDirectSum` constructors
   (lines ~249, 297, 358, 689, 785, 944, 1328 pre-edit).
2. `intervalinf/spaces/forms.py`: remove the `_weight` capture (line ~115) and
   the `k/w` division in `.kernel` (lines ~197–205); `.kernel` returns the raw
   kernel. Also the `weight=self._weight` passes at ~157/174.
3. `intervalinf/operators/bessel.py`: change fast-path gate to
   `isinstance(domain, MassWeightedHilbertSpace)`; slow path uses
   `domain.inner_product` for weighted domains. Remove `getattr(domain,'weight')`.
4. `intervalinf/operators/spectral_helpers.py`: replace the `weight=` param of
   `compute_spectral_coefficients_slow` with an `inner_product=` callable
   (or pass the domain); when provided use `inner_product(φ_k, f)`.
5. `tests/operators/test_weighted_bessel_radial.py`: rewrite to build the radial
   space with `WeightedLebesgue` instead of `Lebesgue(weight=_r2)`.
6. **Notebooks (migrate, do NOT delete):**
   - `intervalinf/demos/3_lebesgue_space_demo.ipynb` (1 use, ~line 1340).
   - `intervalinf/demos/3.1_kernel_functionals_demo.ipynb` (3 uses, ~1512/1641/1844).
   Replace `Lebesgue(..., weight=w)` with `WeightedLebesgue(..., w)`; update
   surrounding prose to mention the MassWeighted mechanism.
**Tests:** existing space/operator suites stay green; `test_weighted_lebesgue.py`
and `test_weighted_bessel_radial.py` green. Grep for residual `weight=` Lebesgue
usage = none.
**Notes (done 2026-06-02):**
- `lebesgue.py`: removed `weight` ctor arg, `self._weight`, the `weight` property,
  the weight branch in `_continuous_l2_inner_product`, weight threading in
  `restrict`/`with_discontinuities`/`PartitionedLebesgueSpace`, and the now-unused
  `_evaluate_array_callable`. Module docstring points to `WeightedLebesgue`.
- `forms.py`: removed `self._weight = getattr(domain,'_weight',None)`, the
  `weight=` passes in `_mapping_impl`, and the `k/w` division in `.kernel`
  (now returns the raw kernel). This is THE key behavioural change: kernels are
  no longer divided by w; the weight lives in the `WeightedLebesgue` mass op.
- `bessel.py`: gate is now `not isinstance(domain, MassWeightedHilbertSpace)`;
  slow path passes `inner_product=domain.inner_product` for mass-weighted domains
  (stored as `self._projection_inner_product`). Imports `MassWeightedHilbertSpace`.
- `spectral_helpers.py`: `compute_spectral_coefficients_slow` now takes
  `inner_product=` callable (replaced `weight=`); uses it when provided else plain
  `(f*φ).integrate()`.
- Tests migrated: `test_weighted_bessel_radial.py` (uses `WeightedLebesgue`),
  `tests/core/test_materialization.py::test_inner_product_with_materialization_matches_original`
  (was using `Lebesgue(weight=)`).
- Notebooks migrated (json-level edits): `3_lebesgue_space_demo.ipynb` (import +
  1 ctor) and `3.1_kernel_functionals_demo.ipynb` (import + 3 ctors). Cell 24 +
  cell 16 markdown REWRITTEN — they previously taught the old `k_w=k/w`
  LinearFormKernel behaviour, now explain the mass-operator approach (`kernel=w·f`).
  Verified the weighted snippet runs and `phi_f(g)==inner_product==manual` (0.09948).
  `4_function_and_basis_providers_demo.ipynb` was a false positive (`fontweight=`).
- Suite state: `tests/spaces tests/operators tests/core tests/providers` → 518
  passed, only the 6 PRE-EXISTING `test_sola.py` instrumentation failures remain.

---

## Phase 3 — SOLAOperator correct adjoints in all cases  ✅ DONE
**Goal:** Adjoint/Gram/cross-Gram correct for plain L² AND MassWeighted domains.
**Files / steps:**
1. `intervalinf/operators/sola.py`:
   - Add an "adjoint-kernel matrix" with rows `M⁻¹(kᵢ)` on the shared mesh,
     cached once. For plain L² (identity Riesz) rows == kernels (no cost).
     Detect via `isinstance(self.domain, MassWeightedHilbertSpace)`.
   - Use it on the RHS in: cached adjoint (`sola.py:~758` `K_matᵀ@data`), Gram
     `_gram_matrix_fast` (`~897`), and cross-Gram (`~967`).
   - Confirm the generic `dual_mapping`→`from_dual` path already applies `M⁻¹`.
**Tests (NEW `tests/operators/test_sola_weighted_adjoint.py`):**
   - For a WeightedLebesgue domain: fast-path `G*y` == generic-path `G*y` ==
     `M⁻¹(Σ yᵢ kᵢ)`.
   - Adjoint identity `⟨Gf, y⟩_D == ⟨f, G*y⟩_w`.
   - `GG*` Gram matches pairwise quadrature with `M⁻¹` kernels.
   - Plain-L² regression: behaviour unchanged.
**Notes (done 2026-06-02):**
- KEY FINDING (TDD): the SOLA **adjoint application was ALREADY correct** on
  weighted domains. SOLA only sets `dual_mapping` (kernel = Σ yᵢ kᵢ); pygeoinf
  derives `.adjoint = from_dual ∘ dual_mapping`, and `WeightedLebesgue.from_dual`
  applies M⁻¹. Verified: adjoint identity `⟨Gf,y⟩=⟨f,G*y⟩_w` and
  `G*y = (Σ yᵢ kᵢ)/w` both pass. The forward is the plain data functional
  `∫ kᵢ f dx` (inner-product-independent) — also verified.
- The ACTUAL bug was only in the **Gram / cross-Gram assembly** (`(GG*)_{ij}`
  computed `∫ kᵢ kⱼ dx` instead of `∫ kᵢ M⁻¹(kⱼ) dx`). Fixed all four methods:
  `compute_gram_matrix_fast`, `compute_gram_matrix` (slow), `compute_cross_gram_matrix`
  (fast), `_compute_cross_gram_matrix_slow`.
- Added helpers `SOLAOperator._is_mass_weighted_domain`, `_adjoint_kernel(i)`
  (= `domain.inverse_mass_operator(k_i)`, or `k_i` for plain L²), and
  `_build_adjoint_kernel_matrix(xs)`. Gram now contracts the kernel table against
  the adjoint-kernel table on one side. Plain L² ⇒ M⁻¹=I ⇒ identical to before.
  Imported `MassWeightedHilbertSpace` from `pygeoinf.hilbert_space`.
- Tests: `tests/operators/test_sola_weighted_adjoint.py` (6 tests, green).
- Suite: 524 passed; only the same 6 PRE-EXISTING `test_sola.py` instrumentation
  failures remain (unchanged).
- ⚠ NOT verified: the reduced operators in `operators/reduced.py`
  (`ReducedGramOperator`, `compute_reduced_covariance`, etc.) may assemble
  Gram-like products directly and could need the same M⁻¹ adjoint-kernel
  treatment on weighted domains. Their existing (plain-L²) tests pass; weighted
  behaviour is untested. Consider as a follow-up if reduced/DLI paths are used
  with weighted domains.

---

## Phase 4 (OPTIONAL) — radial Dirichlet DST fast path  ✅ DONE
**Goal:** Speed up radial Bessel/Laplacian for the Dirichlet case.
**Math:** `(0,R)` Dirichlet radial eigenfns `y_n=√(2/R) sin(nπr/R)/r`,
`λ_n=(nπ/R)²`. Weighted projection `∫ f y_n r² dr = √(2/R)∫(r f)sin(nπr/R)dr`
= **DST of `u=r·f`**. Apply: `u=r·f` → DST → scale by `(k²+λ_n)^{∓s/2}` →
inverse DST → divide by `r`. Same `O(N log N)` as flat. Neumann/Robin radial =
transcendental spectra → no FFT → slow path (already correct).
**Status:** DONE 2026-06-02.
**Notes:**
- Implemented in `bessel.py`: module helpers `_radial_dirichlet_fast_eligible(L)`
  (True iff `L.__class__.__name__=='RadialLaplacian'`, spectral method, bc
  type=='dirichlet') and `_apply_radial_fast_impl(op, f, scale_func)`.
- Insight: the weighted radial projection of `f` equals the **flat sine
  (Dirichlet) projection of `u=r·f`** (same `c_n=√(2/L)`, same `λ_n=(nπ/L)²`),
  so reuse the existing flat DST machinery (`create_uniform_samples`,
  `fast_spectral_coefficients(u_samples,'dirichlet',L,dofs)`) on `u`, then
  reconstruct with the *radial* eigenfunctions via
  `compute_spectral_coefficients_fast(L, f, coeffs, scale_func)`. No separate
  divide-by-r needed — the radial eigenfunctions carry the 1/r.
- Both `BesselSobolev` and `BesselSobolevInverse` get `self._radial_dirichlet_fast`
  in `__init__` and check it first in `_apply` (before the generic fast/slow gate;
  the generic fast path stays disabled on the mass-weighted domain).
- Only fires for Dirichlet radial (incl. a=0 regularity-Dirichlet and a>0 DD).
  Neumann/mixed/Robin radial → slow path (correct, transcendental spectra).
- Tests: `test_radial_dirichlet_fast_path_enabled`, `test_radial_fast_matches_slow`
  in `test_weighted_bessel_radial.py`; AND the existing eigenfunction-scaling /
  self-adjoint / positivity tests now run THROUGH the fast path (default
  `use_fast_transforms=True`) and pass. Suite: 526 passed, 6 pre-existing
  SOLA failures unchanged.

---

## Quick regression command
```
conda activate inferences && cd intervalinf && \
python -m pytest tests/spaces tests/operators tests/core tests/providers -q
# expect: only the 6 pre-existing test_sola.py failures (instrumentation), plus
# any NEW intended changes. Everything else green.
```
