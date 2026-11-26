## intervalinf demo execution audit complete

**Plan:** `docs/agent-docs/active-plans/intervalinf-demo-execution-audit-plan.md`  
**Completed:** 2026-07-17

### Work completed

- Inventoried all eight maintained notebooks for dependencies, external data,
  GUI calls, and unsafe resource settings.
- Executed every notebook sequentially from a clean Jupyter kernel with Agg,
  native threads capped at one, and bounded cell and process timeouts.
- Migrated two notebooks from the removed `Lebesgue(weight=...)` API to the
  mass-weighted `WeightedLebesgue` abstraction.
- Updated model fusion from removed `LinearBayesianInference` behavior to
  `LinearBayesianInversion`, model posterior construction, and explicit
  Gaussian push-forward through the property operator.
- Updated `KLSampler.variance_function` property access and replaced 16-way
  model-fusion parallelism with deterministic one-job execution.
- Corrected negative-Laplacian analytical signs and the eigenvalue check that
  divided at zeros of even sine modes.
- Replaced all deprecated `np.trapz` calls in maintained notebooks.
- Added a `[demos]` dependency extra and an accurate interactive/headless demo
  command to `demos/README.md`.

### Execution evidence

```text
1_interval_domain_demo.ipynb:              17/17 code cells
2_functions_demo.ipynb:                    19/19
3.1_kernel_functionals_demo.ipynb:         15/15
3_lebesgue_space_demo.ipynb:               16/16
4_function_and_basis_providers_demo.ipynb: 14/14
5_gradient_operator_demo.ipynb:            14/14
6_laplacian_operator_demo.ipynb:           18/18
model_fusion/first_test.ipynb:             11/11

total: 124/124, zero error outputs, zero deprecation warnings
```

The corrected Laplacian eigenvalue errors are at most `4.8e-14`; the direct
spectral/FD comparison at `x=pi/2` reports errors `1.07e-14` and `1.70e-2`
respectively, rather than the stale sign-induced error of 18.

### Package evidence

```text
full intervalinf suite: 557 passed in 11.63 s
sdist: built
wheel: built
wheel [demos] metadata: verified
git diff --check: passed
source notebook outputs/execution counts: all zero
forbidden product/confidential paths: none
```

No library runtime change was needed. Existing mathematical tests already cover
the corrected weighted-space, KL variance, Bayesian push-forward, and
Laplacian behavior; full fresh-kernel notebook execution is the direct
regression evidence for the repaired notebook workflows.

### Deviations

The initial plan treated the model-fusion notebook only as unsafe because of
`n_jobs=16`. Once bounded, it also exposed three small pygeoinf/intervalinf API
drifts, all repaired without changing the intended inference mathematics.

Product commit: `62b73c9` on `audit/intervalinf-demos`.
