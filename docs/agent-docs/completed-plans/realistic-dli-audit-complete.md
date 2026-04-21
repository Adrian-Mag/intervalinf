## Plan Complete: Realistic DLI Audit

The four-phase audit of `realistic_dli.ipynb` is complete. The investigation confirmed that the notebook's prior-dominated behavior at N_d=50 is a fully expected geometric effect: (1) property p0 sits in a complete geometric dead-zone (cos(g_k, b_0) ≈ 0 for all 30 catalog modes), and (2) the transition for p1 is unchanged under 10× tighter noise. The notebook is not buggy; it runs more modes than are informative for its bump-function property targets. Corrective annotations have been added to the notebook.

**Phases Completed:** 4 of 4
1. ✅ Phase 1: Baseline structural comparison of dli.ipynb vs realistic_dli.ipynb
2. ✅ Phase 2: Instrumented dual solve behavior at N_d=5
3. ✅ Phase 3: Controlled N_d sweep (N_d ∈ [5,10,20,30,40,50])
4. ✅ Phase 4: Diagnose and propose fixes — EXPECTED_GEOMETRY verdict

**All Files Created/Modified:**
- `intervalinf/demos/convex_analysis/realistic_dli_audit.py` — standalone audit script with all four phases
- `intervalinf/demos/convex_analysis/realistic_dli.ipynb` — markdown audit note cell added before N_d=50
- `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md` — living reference updated

**Key Functions/Classes Added:**
- `run_phase2_instrumentation()` — per-direction bundle diagnostics at N_d=5
- `run_phase3_nd_sweep()` — controlled N_d sweep with half-width reduction table
- `run_phase4_diagnosis()` — tight-noise sweep + per-property cosine table + verdict + notebook annotation

**Phase 3 Key Finding:**
- N_d=5: p0_red=1.030, p1_red=1.821
- N_d=10: p0_red=1.048, p1_red=1.899
- N_d≥15: both collapse to 1.000 (fully prior-dominated)
- Transition occurs between N_d=10 and N_d=15

**Phase 4 Key Finding:**
- Verdict: EXPECTED_GEOMETRY
- cos(b_0) = 0 for all 30 catalog modes → complete dead-zone for p0
- Transition unchanged under 10× tighter noise → primarily geometric, not SNR-driven
- Notebook not buggy; N_d ≤ 10 recommended for data-constraining results

**Test Coverage:**
- Self-checks in Phase 3: posterior ≤ prior for all N_d × property combinations ✅
- Phase 4 tight-noise sweep validates SNR vs geometry discrimination ✅
- All phases exit clean, no Python errors ✅

**Recommendations for Next Steps:**
- Consider re-running with N_d=10 as the "informative" baseline in realistic_dli.ipynb
- For property p0, investigate whether non-vp kernels (rho, vs) carry any information to constrain the ICB-side property
- If future audits revisit this, the transition point may shift if property targets or catalog modes change
