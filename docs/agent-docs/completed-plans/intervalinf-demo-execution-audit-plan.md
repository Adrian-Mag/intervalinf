## Plan: intervalinf demo execution audit

**Status:** completed  
**Created:** 2026-07-17

### Goal

Execute every maintained notebook under `intervalinf/demos/` against the clean
stabilized intervalinf and combined pygeoinf dependency state. Classify each
notebook as passing, failing because of a small product/API drift, blocked by an
external requirement, or unsuitable for bounded local execution. Repair small
API and resource-configuration issues with focused regressions, then re-execute
the affected notebook and the complete resource-capped test suite.

### Safety and scope

- Work only in `/home/adrian/PhD/Inferences/intervalinf-stabilize` on local
  branch `audit/intervalinf-demos`, based on product checkpoint `4840573`.
- Do not switch or clean the dirty mission worktree.
- Use the `inferences` conda environment with the stabilized intervalinf and
  pygeoinf compatibility worktrees first on `PYTHONPATH`.
- Execute one notebook at a time with `MPLBACKEND=Agg`, one native thread, no
  interactive display, and a bounded timeout.
- Do not launch the model-fusion notebook with its historical `n_jobs=16` on
  this laptop. Reduce demo parallelism before execution and record the
  deviation.
- Do not commit notebook outputs or generated figures. Executed copies and logs
  belong under `/tmp/intervalinf-demo-execution-audit/`.
- Keep this plan and all evidence local-only; never add agent documentation to
  the product branch.

### Phases

1. Inventory code cells, imports, optional dependencies, GUI calls, external
   files, and high-cost parameters for all notebooks.
2. Execute each bounded notebook sequentially and record the first failing
   cell, traceback, runtime, and classification.
3. For each small defect, add the narrowest useful regression first when the
   failure reflects runtime behavior, then apply the minimal source or notebook
   correction.
4. Re-execute every changed notebook and then re-run all notebooks from clean
   source copies.
5. Run focused tests, the full constrained test suite, Ruff on changed Python
   files, `git diff --check`, and a final product/confidentiality audit.
6. Update the living package reference, write a phase-completion report, and
   commit only product changes to the local audit branch.

### Notebook ledger

| Notebook | Initial result | Final result | Notes |
|---|---|---|---|
| `1_interval_domain_demo.ipynb` | pass | pass | 17 code cells |
| `2_functions_demo.ipynb` | pass | pass | 19 code cells |
| `3.1_kernel_functionals_demo.ipynb` | removed `Lebesgue(weight=...)` | pass | migrated to `WeightedLebesgue`; 15 cells |
| `3_lebesgue_space_demo.ipynb` | removed `Lebesgue(weight=...)` | pass | migrated to `WeightedLebesgue`; 16 cells |
| `4_function_and_basis_providers_demo.ipynb` | pass with `np.trapz` warning | pass | 14 cells; uses `np.trapezoid` |
| `5_gradient_operator_demo.ipynb` | pass with `np.trapz` warning | pass | 14 cells; uses `np.trapezoid` |
| `6_laplacian_operator_demo.ipynb` | executes but reports false large errors | pass with correct errors | fixed `-Delta` signs and eigenvalue ratio; 18 cells |
| `model_fusion/first_test.ipynb` | unsafe and three stale APIs | pass | one job, current inversion/push-forward/KL APIs; 11 cells |

### Exit gate

- Every maintained notebook has a reproducible execution result.
- Every repaired notebook executes from top to bottom in a fresh kernel.
- Small product/API problems have focused tests where appropriate.
- Remaining failures, if any, have explicit non-product blockers.
- The full intervalinf suite remains green.
- No outputs, local paths, agent infrastructure, or confidential PLI content
  enter the product commit.

### Current checkpoint

- All eight source notebooks execute sequentially from clean kernels: 124 of
  124 code cells complete with zero error outputs and zero deprecation warnings.
- Source notebooks retain zero outputs and zero execution counts.
- Full intervalinf suite: 557 passed in 11.63 seconds.
- Source distribution and wheel build successfully with the new `[demos]`
  dependency extra present in wheel metadata.
- `git diff --check` and the confidentiality/product audit pass.
- Next action: commit product-only notebook and packaging changes, move this
  local-only plan to `completed-plans/`, and record the commit ID.

### Closing note

Completed in product commit `62b73c9`. All eight maintained notebooks execute
from clean kernels after targeted notebook/API corrections; no library runtime
change was required. The plan moved to `completed-plans/` after the commit.
