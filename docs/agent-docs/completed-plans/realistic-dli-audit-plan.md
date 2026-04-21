## Plan: Realistic DLI Audit

Investigate why `intervalinf/demos/convex_analysis/realistic_dli.ipynb` appears to solve faster than `intervalinf/demos/convex_analysis/dli.ipynb` despite a richer-looking forward model, and why changing `N_d` appears to have little effect on posterior bounds. The investigation will be kept separate from the notebooks by building standalone audit code under `intervalinf/demos/convex_analysis/` until the failure mode is understood.

**Phases 4 phases**
1. **Phase 1: Baseline the notebook formulations**
    - **Objective:** Extract and compare the actual DLI optimization structures used by `dli.ipynb` and `realistic_dli.ipynb`.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/realistic_dli_audit.py
    - **Tests to Write:** Run the audit script in baseline mode and verify it reports the notebook configuration, prior/data support types, property dimension, and solve settings for both formulations.
    - **Steps:**
        1. Create a standalone script under `intervalinf/demos/convex_analysis/` that encodes the baseline structural comparison between the two notebook setups.
        2. Record `N_d`, `N_p`, support-function types, solve settings, and the current posterior-vs-prior summaries in a machine-readable and human-readable form.
        3. Run the script to confirm the baseline comparison executes cleanly.

2. **Phase 2: Instrument the dual solve behavior**
    - **Objective:** Measure what the solver is actually doing in the realistic setup rather than relying on notebook wall-clock output.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/realistic_dli_audit.py
    - **Tests to Write:** Run the script in instrumentation mode and verify it records per-direction solve time, iteration counts, convergence flags, and lambda norms.
    - **Steps:**
        1. Reconstruct the current `realistic_dli` problem in the script without modifying the notebook.
        2. Log per-direction bundle diagnostics and compare prior and posterior support values numerically.
        3. Check whether the solve is effectively degenerating to a prior-dominated or near-zero-lambda problem.

3. **Phase 3: Run controlled experiments**
    - **Objective:** Test whether the suspicious behavior comes from modeling choices or from a library/implementation defect.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/realistic_dli_audit.py
    - **Tests to Write:** Run controlled script cases for selected `N_d`, matched noise assumptions, and simplified support-set configurations.
    - **Steps:**
        1. Compare `dli` and `realistic_dli` under matched `N_d`, `N_p`, and simplified assumptions where possible.
        2. Sweep `N_d` in the realistic formulation while holding other assumptions fixed.
        3. Determine whether the data term meaningfully changes the support values relative to the prior alone.

4. **Phase 4: Diagnose and propose fixes**
    - **Objective:** Convert the audit findings into a concrete explanation and, if necessary, a targeted fix.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/realistic_dli.ipynb, intervalinf/demos/convex_analysis/dli.ipynb, and/or pygeoinf sources only if warranted by the audit.
    - **Tests to Write:** Re-run the audit script and any affected notebook/script checks after any fix.
    - **Steps:**
        1. Classify the issue as expected geometry, bad experiment design, notebook modeling error, or library bug.
        2. Implement the smallest root-cause fix only if the audit justifies it.
        3. Summarize whether the realistic notebook results can be trusted and under what caveats.

**Open Questions 0 questions**
