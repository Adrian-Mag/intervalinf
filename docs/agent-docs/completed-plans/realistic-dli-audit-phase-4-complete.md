## Phase 4 Complete: Diagnose and Propose Fixes

Phase 4 classified the prior-dominated collapse at N_d ≥ 15 as **EXPECTED_GEOMETRY**: property p0 sits in a complete geometric dead-zone (cos(g_k, b_0) ≈ 0 for all 30 catalog modes), and the transition for p1 is unchanged under 10× tighter noise, confirming that the collapse is geometric rather than SNR-driven. The notebook is not buggy; it simply runs more modes than are informative for these particular bump-function property targets.

**Files created/changed:**
- `intervalinf/demos/convex_analysis/realistic_dli_audit.py` — added `run_phase4_diagnosis()`, `_VERDICT_EXPLANATIONS`, called from `main()`
- `intervalinf/demos/convex_analysis/realistic_dli.ipynb` — markdown audit note inserted at cell 7 (before `N_d = 50` cell)
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` — updated with Phase 4 findings

**Functions created/changed:**
- `run_phase4_diagnosis()` — new; implements three-step diagnosis:
  - Step A: tight-noise sweep (σ_base = 0.01 vs 0.10) to discriminate SNR dilution from geometric cause
  - Step B: per-property cosine similarity table (30 modes × 2 properties), with per-property dead-zone detection (flag if max_cos[i] < 0.05 across all modes)
  - Step C: conditional verdict (EXPECTED_GEOMETRY / SNR_DILUTION / BOTH) and idempotent notebook annotation whose cause list is dynamically built from the diagnostic flags

**Key diagnostic findings:**
- Step A: transition N_d unchanged at 15 under tight noise → geometric, not SNR
- Step B: cos(b_0) = 0.0000 for all 30 modes → complete dead-zone for p0; cos(b_1) decays 0.87 → 0.31 but transition unchanged → geometric for p1 too
- Verdict: EXPECTED_GEOMETRY
- Notebook cell 7: evidence-based annotation, no false SNR dilution attribution

**Bugs fixed during review:**
1. MAJOR: notebook annotation hard-coded SNR dilution as a cause even when `_snr_flag=False` → fixed by building cause list conditionally
2. MAJOR: `_geom_flag` based on aggregate max-cosine missed complete per-property dead-zone → fixed by adding `_prop_deadzone` detection and `_any_deadzone` flag
3. MINOR: unused `import time as _time` in Phase 4 function → removed

**Review Status:** APPROVED_WITH_WARNINGS (minor: lint style warnings only)

**Git Commit Message:**
```
feat(convex-analysis): add Phase 4 diagnosis to DLI audit

- Add run_phase4_diagnosis() with tight-noise sweep, cosine table, verdict
- Detect per-property dead-zones; verdict = EXPECTED_GEOMETRY
- Insert evidence-based audit note in realistic_dli.ipynb before N_d=50 cell
- Fix notebook annotation to not assert SNR dilution when evidence is geometric
- Remove unused import; update living reference

Plan: intervalinf/docs/agent-docs/active-plans/realistic-dli-audit-plan.md
Phase: 4 of 4
Related: intervalinf/docs/agent-docs/active-plans/realistic-dli-audit-phase-4-complete.md
```
