## Phase 4 Complete: Independent Block Posteriors

`solve_block` and `solve_all_blocks` added to `full_spectrum_utils.py`. Each block is solved independently using a data-space `LinearBayesianInversion` with dense LU on the `(N_d × N_d)` normal operator. Optional thread-based parallelism via `joblib`. All 13 tests pass.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/full_spectrum_utils.py`
- `intervalinf/demos/old_demos/paper_demos/tests/test_full_spectrum_utils.py`
- `intervalinf/demos/old_demos/paper_demos/example.ipynb`

**Functions created/changed:**
- `solve_block(s, t, G_st, C_D_st, prior_st, d_st) -> GaussianMeasure` — module-level, pickle-safe; builds `LinearForwardProblem` + `LinearBayesianInversion` (data-space formalism), solves with `LUSolver()`
- `solve_all_blocks(forward_dict, prior_dict, split, n_jobs=1) -> dict` — iterates all blocks, uses `joblib.Parallel(prefer='threads')`

**Tests created/changed:**
- `test_solve_block_smoke_one_block` — checks return type is `GaussianMeasure`, expectation is 2-element nested list, covariance applied to zero doesn't raise
- `test_solve_block_posterior_reduces_uncertainty` — `‖G μ_post − d‖ ≤ ‖d‖` (posterior mean explains data better than zero prior)
- `test_solve_all_blocks_parallel_matches_serial` — `n_jobs=1` vs `n_jobs=2` produce identical `sigma_1` posterior means to `atol=1e-10`

**Review Status:** APPROVED

**Git Commit Message:**
```
feat(paper-demo): add solve_block and solve_all_blocks (Phase 4)

- Add solve_block: LinearForwardProblem + LinearBayesianInversion (data-space, LUSolver)
- Add solve_all_blocks: joblib thread-parallel loop over all (s,t) blocks
- 3 new tests: smoke, uncertainty-reduction, parallel-matches-serial (13/13 pass)
- Add Phase 4 notebook cells (markdown + timed solve_all_blocks call)

Plan: intervalinf/docs/agent-docs/active-plans/full-spectrum-splitting-pli-example-plan.md
Phase: 4 of 8
Related: intervalinf/docs/agent-docs/active-plans/full-spectrum-splitting-pli-example-phase-4-complete.md
```
