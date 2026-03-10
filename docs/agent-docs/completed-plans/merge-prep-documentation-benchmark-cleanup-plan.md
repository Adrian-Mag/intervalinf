## Plan: Merge-Prep Documentation And Benchmark Cleanup

Prepare the `intervalinf` and `pygeoinf` repositories for merge into `convex_analysis` by updating permanent documentation, removing benchmark-era implementation bloat, and deleting temporary benchmark harnesses and generated outputs. The cleanup is split into three phases so documentation, runtime-code cleanup, and repository artifact cleanup can each be reviewed and committed independently.

**Phases 3**
1. **Phase 1: Update Permanent Documentation**
    - **Objective:** Update stable documentation so it reflects the final optimized DLI/SOLA state without referencing temporary benchmark scaffolding.
    - **Files/Functions to Modify/Create:** `intervalinf/README.md`; `intervalinf/demos/README.md`; `pygeoinf/README.md`; `pygeoinf/docs/source/pygeoinf.rst`; living references and any incorrect completed-plan links that point to stale locations
    - **Tests to Write:** none; validate links and documentation consistency only
    - **Steps:**
        1. Update README and demo-entry documentation to mention the stable convex-analysis / DLI workflow and the relevant optimized APIs.
        2. Update Sphinx/API documentation so `convex_analysis` and `convex_optimisation` are included in the published `pygeoinf` API surface.
        3. Correct any stale plan-document references and remove benchmark-script references from permanent docs.

2. **Phase 2: Remove Benchmark Bloat From Runtime Code**
    - **Objective:** Remove benchmark-oriented timing/instrumentation code and temporary optimization-campaign wording from runtime modules while preserving the intended functional behavior.
    - **Files/Functions to Modify/Create:** `pygeoinf/pygeoinf/backus_gilbert.py`; `pygeoinf/pygeoinf/convex_optimisation.py`; `intervalinf/intervalinf/operators/sola.py`; related tests and living references as needed
    - **Tests to Write:** focused regression tests needed to preserve DLI/SOLA behavior after removing instrumentation-only code
    - **Steps:**
        1. Add or adjust tests first so removal of timing/instrumentation code does not change algorithmic behavior.
        2. Remove timing counters, benchmark-only stats surfaces, and benchmark-era docstring language from the runtime modules.
        3. Re-run the targeted `pygeoinf` and `intervalinf` tests to confirm behavior is unchanged after cleanup.

3. **Phase 3: Remove Benchmark Harnesses And Generated Outputs**
    - **Objective:** Delete temporary benchmark scripts, CSV outputs, logs, and generated benchmark figures/results so the repositories are clean for merge.
    - **Files/Functions to Modify/Create:** benchmark scripts and generated artifacts under `intervalinf/rough_work/`; any docs that reference them; final merge-prep summary files if needed
    - **Tests to Write:** none; verify docs no longer reference deleted files and relevant package tests still pass
    - **Steps:**
        1. Remove benchmark harness scripts and all generated benchmark outputs that were created for the optimization campaign.
        2. Update permanent docs and living references so they do not point at deleted benchmark artifacts.
        3. Run final repository checks and summarize the merge-prep cleanup state.

**Open Questions 1**
1. None; user requested removal of both benchmark outputs and benchmark harnesses, and removal of timing/instrumentation bloat from runtime code.