# Plan: intervalinf convex_analysis branch update

Bring `intervalinf/convex_analysis` in sync with `pygeoinf/convex_analysis`. This
covers fixing stale imports left over from when intervalinf was a submodule of pygeoinf,
bumping the pygeoinf version pin, adding Sobolev-aware convex analysis, and verifying
the full test suite at every step.

**Context:**
- `intervalinf` was previously a branch/submodule of `pygeoinf` (under `pygeoinf/interval/`).
  Several imports of the form `from pygeoinf.interval.operators import ...` remain and are
  now broken — those classes live in `intervalinf.operators.*` instead.
- pygeoinf is editable-installed from the local `convex_analysis` checkout, so intervalinf
  already sees all pygeoinf convex_analysis code without a version bump.
- pygeoinf `convex_analysis` adds: `SupportFunction` hierarchy (`convex_analysis.py`),
  `HalfSpace`/`PolyhedralSet`/`HyperPlane` (`subsets.py`), `SubgradientDescent`
  (`convex_optimisation.py`), `SubspaceSlicePlotter` (`visualization.py`),
  `get_tangent_basis()` on `AffineSubspace`, and `subgradient` on `NonLinearForm`.
- intervalinf baseline: **270 tests, all passing** (as of 2026-02-18).

---

## Phase 1: Fix stale pygeoinf.interval.operators imports

**Status:** ✅ Complete (2026-02-18)

**Objective:** Replace the 4 `try/except ImportError` blocks in `intervalinf/spaces/sobolev.py`
that import from the non-existent `pygeoinf.interval.operators` with correct imports from
`intervalinf.operators.bessel` and `intervalinf.operators.laplacian`.

**Files:**
- `intervalinf/intervalinf/spaces/sobolev.py` — 4 broken import sites:
  - `mass_operator_factor` (line ~270) — needs `BesselSobolev`
  - `inverse_mass_operator_factor` (line ~286) — needs `BesselSobolevInverse`
  - `restrict` (line ~338) — needs `BesselSobolev`, `BesselSobolevInverse`
  - `create_subspace_decomposition` (line ~479) — needs `Laplacian`
- New: `tests/spaces/test_sobolev_operators.py`

**Tests to write:**
- `test_mass_operator_factor_returns_bessel_sobolev`
- `test_inverse_mass_operator_factor_returns_bessel_sobolev_inverse`
- `test_restrict_updates_mass_operators`
- `test_mass_operator_factor_inverse_roundtrip` (optional numerical correctness)

**Outcome:** All 270 existing + 9 new tests pass (279 total); methods no longer raise `NotImplementedError`.

**Changes made:**
- `intervalinf/intervalinf/spaces/sobolev.py`: Removed 4 `try/except ImportError` blocks that imported from `pygeoinf.interval.operators`. Replaced with direct imports from `intervalinf.operators.bessel` (`BesselSobolev`, `BesselSobolevInverse`) and `intervalinf.operators.laplacian` (`Laplacian`).
- New: `tests/spaces/test_sobolev_operators.py` — 9 tests covering type assertions, NotImplementedError absence, roundtrip numerical correctness, restrict, and with_discontinuities.

---

## Phase 2: Bump pygeoinf version constraint

**Status:** ✅ Complete (2026-02-19)

**Objective:** Update `pygeoinf>=1.3.3` → `>=1.4.2` in `pyproject.toml`.

**Files:** `intervalinf/pyproject.toml`

**Outcome:** Constraint updated. pygeoinf 1.4.2 is in use. Both packages are path-resolved (not pip-installed), so the constraint is enforced at install time for downstream users. 279/279 tests pass.

---

## Phase 3: Integration tests — pygeoinf convex analysis with intervalinf spaces

**Status:** ⬜ Not started

**Rationale:** `pygeoinf.convex_analysis` (`BallSupportFunction`, `EllipsoidSupportFunction`,
`HalfSpaceSupportFunction`) and `pygeoinf.subsets` (`HalfSpace`, `PolyhedralSet`, `Ball`,
`Ellipsoid`) already work against the abstract `HilbertSpace` interface. Since `Lebesgue`
and `Sobolev` both implement `HilbertSpace`, no new `intervalinf/convex_analysis.py` is
needed — the pygeoinf classes automatically use whichever inner product the space provides
(including the Sobolev mass-weighted inner product).

**Objective:** Write integration tests confirming that:
1. `BallSupportFunction`, `EllipsoidSupportFunction`, `HalfSpaceSupportFunction` evaluate
   correctly when instantiated with a `Lebesgue` or `Sobolev` primal domain.
2. `SubspaceSlicePlotter` (from `pygeoinf.visualization`) works with intervalinf spaces
   and a `Ball` or `HalfSpace` subset.
3. Export the most-used pygeoinf convex analysis symbols from `intervalinf/__init__.py`
   as a convenience (so users don't need to `from pygeoinf.convex_analysis import ...`).

**Files:**
- New: `tests/test_convex_analysis_integration.py`
- Modified: `intervalinf/intervalinf/__init__.py` (re-export convenience)

---

## Phase 4: Update reference files ⬜

**Status:** ⬜ Not started

**Objective:** Update `intervalinf/plans/intervalinf-reference.md` to document
the fixed sobolev methods and the new `convex_analysis.py` module.
Update `pygeoinf/plans/pygeoinf-reference.md` for any corrections found during this work.

---

## Change Log

| Date | Phase | Action |
|------|-------|--------|
| 2026-02-18 | Planning | Initial plan created |
| 2026-02-18 | Phase 1 | Fixed 4 stale `pygeoinf.interval.operators` import blocks in `sobolev.py`; 9 new tests added in `test_sobolev_operators.py`; 279/279 pass |
| 2026-02-19 | Phase 2 | Bumped `pygeoinf>=1.3.3` → `>=1.4.2` in `pyproject.toml`; 279/279 pass |
| 2026-02-18 | Phase 4 | Updated `intervalinf-reference.md` (Sobolev section) to reflect fixed imports and new method docs |
| 2026-02-19 | Phase 4 | Updated `intervalinf-reference.md` pygeoinf dependency version |
