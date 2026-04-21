## Phase 1 Complete: Baseline the notebook formulations

Built a standalone baseline audit for the two DLI notebooks and validated that it runs cleanly. The audit establishes that the current wall-clock comparison is confounded because `dli.ipynb` is running with `N_d = 50` while the current `realistic_dli.ipynb` state is running with `N_d = 5`, so the faster realistic solve is not yet evidence of a deeper solver defect by itself.

**Files created/changed:**
- intervalinf/demos/convex_analysis/realistic_dli_audit.py
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Functions created/changed:**
- ModelSpaceSummary
- SolverSettings
- SupportFunctionSettings
- SolveResults
- NotebookBaseline
- print_baseline_report
- print_comparison_table
- print_interpretation
- build_json_summary
- main

**Tests created/changed:**
- Baseline audit script execution for human-readable comparison output
- Baseline audit script execution for machine-readable JSON summary output
- Final script validation after cleanup with no file errors

**Review Status:** APPROVED

**Git Commit Message:**
feat(convex-analysis): add baseline realistic dli audit

- Add standalone baseline audit for dli notebook comparisons
- Record structural and solver summaries in script output
- Update living reference for the new audit workflow

Plan: intervalinf/docs/agent-docs/active-plans/realistic-dli-audit-plan.md
Phase: 1 of 4
Related: intervalinf/docs/agent-docs/active-plans/realistic-dli-audit-phase-1-complete.md