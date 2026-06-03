## Plan: Function-Space Prior Exploration

Explore prior families for the normal-mode tuner while preserving the continuous/function-space formulation used by pygeoinf and intervalinf. The plan compares existing and new operator-theoretic covariance families through their induced data-space covariance, posterior metrics, and function-space sample behavior; dense matrices are diagnostics only, not the prior representation.

**Branch Setup**
- intervalinf branch pointer created: `mission/20260520-prior-family-exploration`
- pygeoinf branch pointer created: `mission/20260520-prior-operator-exploration`
- Current working branches were not switched because both package worktrees already contain uncommitted work.

**Function-Space Guardrails**
- Priors must be `GaussianMeasure` objects on `pygeoinf.HilbertSpace` domains, with covariance supplied by `pygeoinf.LinearOperator` or an intervalinf operator subclass.
- New interval covariances must follow intervalinf templates: `SpectralOperator` for diagonal spectral covariances, `LinearOperator` for non-spectral or composite covariances, and existing provider abstractions for eigenfunctions/eigenvalues.
- New pygeoinf code is limited to generic Hilbert-space operator algebra if intervalinf cannot express the needed covariance cleanly.
- Dense arrays may be used for finite diagnostic reductions such as `G C G*`, effective rank, eigenspectra, and regression tests, but not as the primary prior object.
- Every candidate prior must be checked for self-adjointness, positivity, trace/effective-trace behavior, posterior covariance sanity, and sensitivity to basis refinement.

**Phases 7 phases**
1. **Phase 1: Reframe Diagnostics in Function Space**
    - **Objective:** Correct the audit interpretation so the infinite-dimensional inverse problem remains explicitly underdetermined, while measuring only the finite data-visible subspace induced by `G C G*`.
    - **Files/Functions to Modify/Create:** `intervalinf/work/prior-family-exploration/`, `intervalinf/work/prior-posterior-audit/FINAL_REPORT.md`, prior audit metric helpers copied or factored from the audit scripts if needed.
    - **Tests to Write:** `test_function_space_rank_language`, `test_data_space_effective_rank_metric`, `test_probe_contraction_uses_function_space_vectors`.
    - **Steps:**
        1. Read the prior-posterior audit outputs and identify metrics that should remain frozen.
        2. Define the diagnostic decomposition: full function space, prior-supported Cameron-Martin directions, data-visible directions, and null/near-null directions.
        3. Add tests for effective-rank and contraction helpers on small function-space examples.
        4. Update the audit report language if it still implies the full function space is determined by the data.

2. **Phase 2: Existing Bessel Family Expansion**
    - **Objective:** Exhaust the current `BesselSobolevInverse` covariance family before adding operators, using lower order, shorter length scales, mixture-like amplitude settings, and basis refinement.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/paper_demos/utils/prior_posterior_tuner.py`, `intervalinf/demos/old_demos/paper_demos/tuner_params.json` only through saved candidate configs, `intervalinf/work/prior-family-exploration/bessel_family_sweep.py`.
    - **Tests to Write:** `test_bessel_candidate_factory_preserves_domain`, `test_bessel_candidate_covariance_is_self_adjoint`, `test_bessel_effective_rank_increases_with_rougher_settings`.
    - **Steps:**
        1. Write tests around the candidate factory using real `Lebesgue` spaces and `BesselSobolevInverse` operators.
        2. Run a controlled grid over `s_order`, length, variance, boundary conditions, and `n_basis` on block `(0,0)`.
        3. Compare `chi_rms`, roughness, posterior probe standard deviations, `rank_eff(G C G*)`, and basis sensitivity.
        4. Shortlist Bessel-only priors that are less restrictive without numerical instability.

3. **Phase 3: Evaluate Existing Non-Bessel Templates**
    - **Objective:** Test available intervalinf covariance operators before implementing new ones: inverse Laplacian, radial inverse Laplacian, Sobolev inverse-mass covariances, and any existing reduced covariance operators that preserve the function-space semantics.
    - **Files/Functions to Modify/Create:** `intervalinf/operators/laplacian.py`, `intervalinf/operators/radial.py`, `intervalinf/operators/reduced.py`, `intervalinf/work/prior-family-exploration/existing_operator_priors.py`.
    - **Tests to Write:** `test_inverse_laplacian_covariance_prior_solves`, `test_radial_covariance_prior_matches_domain_weighting`, `test_existing_operator_prior_diagnostics_are_finite`.
    - **Steps:**
        1. Write a small prior registry that can wrap existing covariance operators into `GaussianMeasure` blocks.
        2. Use toy tests to verify adjoint consistency and positivity for each existing candidate.
        3. Run the representative block and compare data-space coverage against the Bessel baseline.
        4. Eliminate candidates that are not trace-class, not positive, or incompatible with the radial model-space geometry.

4. **Phase 4: Implement Composite Spectral Covariances If Needed**
    - **Objective:** Add reusable covariance operators only if existing templates cannot express a less restrictive prior with acceptable behavior.
    - **Files/Functions to Modify/Create:** `intervalinf/operators/covariance.py` or the nearest established operator module, `intervalinf/operators/__init__.py`, `intervalinf/tests/operators/test_covariance.py`; possible pygeoinf generic operator helper only if intervalinf composition is insufficient.
    - **Tests to Write:** `test_spectral_covariance_scales_eigenfunctions`, `test_spectral_covariance_is_self_adjoint_positive`, `test_spectral_covariance_trace_decay`, `test_mixture_covariance_adds_variances_without_losing_domain`, `test_operator_adjoint_consistency`.
    - **Steps:**
        1. Write failing tests for a `SpectralCovarianceOperator` that accepts an eigenfunction provider and a positive eigenvalue law.
        2. Implement the minimal intervalinf operator following the `SpectralOperator` pattern.
        3. Add optional positive weighted mixtures, such as smooth plus rough components, using existing pygeoinf operator composition when possible.
        4. Document trace-class requirements and reject non-decaying spectra with clear errors.
        5. Run intervalinf operator tests and the targeted normal-mode prior diagnostics.

5. **Phase 5: Integrate Prior Families Into the Tuner Path**
    - **Objective:** Make candidate priors selectable through the normal-mode tuner configuration without bypassing pygeoinf `GaussianMeasure` or intervalinf function spaces.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/paper_demos/utils/prior_posterior_tuner.py`, `intervalinf/demos/old_demos/paper_demos/prior_posterior_tuner_app.py`, `intervalinf/demos/old_demos/paper_demos/utils/full_spectrum_utils.py`, candidate config files under `intervalinf/work/prior-family-exploration/configs/`.
    - **Tests to Write:** `test_prior_registry_builds_bessel_prior`, `test_prior_registry_builds_composite_prior`, `test_tuner_prior_config_round_trips`, `test_posterior_solve_uses_operator_covariance`.
    - **Steps:**
        1. Add a typed prior-family config layer that maps JSON settings to function-space covariance operators.
        2. Preserve the current Bessel default as the backward-compatible path.
        3. Add candidate presets for rough Bessel, short-length Bessel, smooth-plus-rough mixture, and any successful non-Bessel operator.
        4. Verify posterior solves still call `LinearBayesianInversion` with `GaussianMeasure` covariance operators.

6. **Phase 6: Cross-Block and Synthetic Validation**
    - **Objective:** Verify that promising priors improve data-space coverage and posterior honesty across real blocks and controlled synthetic cases.
    - **Files/Functions to Modify/Create:** `intervalinf/work/prior-family-exploration/cross_block_validation.py`, `intervalinf/work/prior-family-exploration/synthetic_prior_recovery.py`, processed outputs under `intervalinf/work/prior-family-exploration/processed/`.
    - **Tests to Write:** `test_synthetic_recovery_uncertainty_covers_truth`, `test_cross_block_metrics_schema`, `test_candidate_prior_basis_refinement_stability`.
    - **Steps:**
        1. Validate candidates on `(0,0)`, `(2,0)`, and `(2,3)` using the frozen metrics.
        2. Run synthetic truth tests with known function-space samples from each prior family.
        3. Check whether credible intervals cover the synthetic truth in data-informed and weakly informed directions.
        4. Record candidates that reduce `chi_rms` by expanding data-space coverage rather than only inflating amplitude.

7. **Phase 7: Review, References, and Recommendation Package**
    - **Objective:** Produce a reviewed recommendation set for function-space priors and update the living references so future work does not drift back to discretize-first priors.
    - **Files/Functions to Modify/Create:** `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`, relevant pygeoinf living references if generic operator helpers are added, `intervalinf/work/prior-family-exploration/processed/final_prior_family_report.md`.
    - **Tests to Write:** Full targeted tests from Phases 1-6, plus `python -m pytest tests/operators tests/spaces` in intervalinf and targeted pygeoinf tests if pygeoinf changes are made.
    - **Steps:**
        1. Run the targeted tests first, then the relevant package test slices.
        2. Invoke code review with special attention to function-space correctness, self-adjointness, positivity, and trace-class behavior.
        3. Update living references for any new covariance operators, tuner config schema, or prior registry changes.
        4. Write final recommendations with candidate priors, failure modes, and tuner defaults.

**Open Questions 5 questions**
1. Should the first new family be a smooth-plus-rough mixture of existing Bessel covariances, or a generic spectral covariance with a user-supplied eigenvalue law?
2. Should radial weighting by `r^2 dr` be made explicit in new covariance tests before any normal-mode block runs?
3. What threshold should define meaningful data-space coverage: absolute eigenvalue, noise-normalized information ratio, or posterior contraction?
4. Should candidate priors be scored by marginal likelihood, predictive chi-rms, synthetic coverage, or a multi-objective Pareto rule?
5. Should the created branch pointers be switched to before implementation, or should implementation wait until current dirty work is committed/stashed by the user?
