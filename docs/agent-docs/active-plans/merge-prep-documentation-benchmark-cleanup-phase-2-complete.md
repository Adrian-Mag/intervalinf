## Phase 2 Complete: Remove Benchmark Bloat from Runtime Code

All benchmark-era instrumentation timing code removed from production runtime files. Functional logic is preserved; no behavioural regressions (pygeoinf 420 pass, intervalinf 454 pass).

**Files created/changed:**
- `pygeoinf/pygeoinf/backus_gilbert.py`
- `pygeoinf/pygeoinf/convex_optimisation.py`
- `intervalinf/intervalinf/operators/sola.py`
- `pygeoinf/tests/test_dual_master_cost.py`
- `pygeoinf/docs/agent-docs/references/living/pygeoinf-reference.md`
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`

**Functions created/changed:**
- `DualMasterCostFunction.__init__` — removed `self._stats = DualMasterStats()` init
- `DualMasterCostFunction.value_and_subgradient` — removed all `time.perf_counter()` and stats-accumulation lines; functional fused oracle call preserved
- `DualMasterCostFunction._mapping` — removed timing lines
- `DualMasterCostFunction._finite_difference_gradient` — removed timing lines
- `ProximalBundleMethod.__init__` — removed `self._stats` attribute
- `ProximalBundleMethod.solve` — removed all timing/stats-accumulation lines; bundle logic preserved

**Classes removed:**
- `DualMasterStats` (dataclass, ~60 lines) — entirely deleted from `backus_gilbert.py`
- `DualMasterCostFunction.instrumentation_stats` property — deleted
- `DualMasterCostFunction.reset_instrumentation()` method — deleted
- `ProximalBundleStats` (dataclass, ~30 lines) — entirely deleted from `convex_optimisation.py`
- `ProximalBundleMethod.instrumentation_stats` property — deleted
- Benchmark-era "Phase N:" inline comments removed from `sola.py`

**Tests created/changed:**
- `pygeoinf/tests/test_dual_master_cost.py` — updated `test_fallback_branch_correctness_and_instrumentation`: removed `reset_instrumentation()` call and all stats assertions; retained the functional value-correctness assertion

**Review Status:** APPROVED (stale living-reference entries fixed in same pass)

**Git Commit Message:**
```
refactor(bloat): remove benchmark instrumentation from runtime code

- Delete DualMasterStats dataclass and all timing/stats accumulation from backus_gilbert.py
- Delete ProximalBundleStats dataclass and timing accumulation from convex_optimisation.py
- Remove benchmark-era "Phase N:" inline comments from sola.py
- Update test_dual_master_cost.py to drop removed stats assertions
- Clean stale DualMasterStats/ProximalBundleStats entries from living references

Plan: intervalinf/docs/agent-docs/active-plans/merge-prep-documentation-benchmark-cleanup-plan.md
Phase: 2 of 3
Related: intervalinf/docs/agent-docs/active-plans/merge-prep-documentation-benchmark-cleanup-phase-2-complete.md
```
