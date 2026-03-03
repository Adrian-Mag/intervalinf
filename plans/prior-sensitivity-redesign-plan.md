## Plan: Prior Sensitivity Study Redesign

Redesign the prior sensitivity study script (previously `prior_sensitivity_study.py`,
now to be recreated from scratch) to support 7 decoupled prior-misspecification
cases (regularity R±, variance V±, mean M±, + reference), with:

- A **`FAST_MODE`** toggle (vp-only with real catalog kernels, ~30–60 s/case vs 3–4 min)
- **Selective per-case execution** via `CASES_TO_RUN`
- **Incremental CSV/MD persistence** so re-running one case preserves all others
- A **prior-predictive fairness check** (χ²/Nd) logged per case
- An **optional variance-normalisation** step for R± cases so that all cases can
  optionally be compared at equal property-space marginal σ

### Design decisions (from user)

- **Fast-mode forward operator**: use the **real seismic sensitivity kernel catalog**
  (`SensitivityKernelCatalog` + `SensitivityKernelProvider(kernel_type='vp')`),
  NOT `NormalModesProvider`.
- **Fast-mode true model**: reuse the **same `ran_array_vp[:10]` coefficients** as the
  realistic script.
- **Mean offset**: auto-computed in property-space σ units (~1.5σ).
- **R± variance normalisation**: include as an **optional step** (`NORMALISE_R_VARIANCE = False`).
- **Output folders**: two separate folders — `prior_sensitivity_fast/` and
  `prior_sensitivity/`.

### Source file

The original `prior_sensitivity_study.py` has been deleted from disk. It will be
recreated from scratch at:

```
intervalinf/demos/old_demos/paper_demos/prior_sensitivity_study.py
```

The script depends on `kernel_utils.py` in the same directory (still present) and
the kernel data directory at `../kernels_modeplotaat_Adrian/`.

---

## Phase 1: Script scaffold, imports, USER CONFIG block, FAST_MODE problem setup

### Objective

Create the new script with all imports, the full USER CONFIG block, and the two
problem-setup branches (`build_fast_problem` / `build_realistic_problem`), including
shared figures. No inference loop yet.

### Files/Functions to Create

- `prior_sensitivity_study.py` — entire file scaffolded

### What to build

#### 1.1 Imports (lines ~1–55)

Standard library: `os`, `sys`, `time`, `csv`, `warnings`, `datetime` from `pathlib`
import `Path`.

Third-party: `numpy`, `matplotlib` (with `Agg` backend), `seaborn`.

`intervalinf` imports (identical to what the old script used):
- `Lebesgue`, `IntervalDomain`, `LebesgueIntegrationConfig`, `IntegrationConfig`,
  `ParallelConfig`, `LebesgueSpaceDirectSum`, `Function`, `BoundaryConditions`,
  `KnownRegion`, `PartitionedLebesgueSpace`
- `intervalinf.operators`: `SOLAOperator`, `Laplacian`, `BesselSobolevInverse`
- `intervalinf.sampling`: `KLSampler`
- `intervalinf.providers`: `BumpFunctionProvider`, `NullFunctionProvider`

`kernel_utils` imports: `SensitivityKernelCatalog`, `SensitivityKernelProvider`,
`EARTH_RADIUS_KM`.

`pygeoinf` imports: `EuclideanSpace`, `RowLinearOperator`, `GaussianMeasure`,
`LinearForwardProblem`, `LinearBayesianInversion`, `CholeskySolver`,
`HilbertSpaceDirectSum`, `LinearOperator`.

#### 1.2 USER CONFIG block (lines ~60–160)

All tuneable parameters in a clearly delimited section:

```python
# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        USER CONFIGURATION                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

FAST_MODE = True                    # True = vp-only (~30-60 s/case)
                                    # False = full 5-component (~3-4 min/case)

CASES_TO_RUN = ["all"]              # ["all"] or subset e.g. ["Reference", "R+", "M-"]

PRIOR_PREDICTIVE_CHI2_THRESHOLD = 3.0   # warn if chi2/dof exceeds this

NORMALISE_R_VARIANCE = False        # if True, bisect overall_variance for R±
                                    # so property-space σ matches Reference

# ── Integration configs ──
# (Lebesgue, Laplacian, SOLA, BesselSobolev — same as old script)

# ── Parallelisation ──
# ParallelConfig(enabled=True, n_jobs=12)

# ── Prior case definitions ──
# PRIOR_CASES dict — see Phase 2 for the 7-case table
# CASE_COLORS dict
```

This section also contains `N`, `N_d`, `N_p`, `noise_level_fraction = 0.1`,
`OVERALL_VARIANCE_BASE = 10**1.5`, and all integration/parallel config objects.

#### 1.3 Output folder logic

```python
FIGURES_FOLDER = "prior_sensitivity_fast" if FAST_MODE else "prior_sensitivity"
OUTPUT_CSV = os.path.join(FIGURES_FOLDER, "prior_sensitivity_summary.csv")
OUTPUT_MD  = os.path.join(FIGURES_FOLDER, "prior_sensitivity_summary.md")
OUTPUT_TXT = os.path.join(FIGURES_FOLDER, "prior_sensitivity_detailed.txt")
os.makedirs(FIGURES_FOLDER, exist_ok=True)
```

#### 1.4 `build_fast_problem()` function

Creates a vp-only inference problem using real sensitivity kernels:

```
def build_fast_problem() -> ProblemTuple:
```

- `function_domain = IntervalDomain(0, EARTH_RADIUS_KM)`
- `M_vp = Lebesgue(N, function_domain, basis='cosine', integration_config=…, parallel_config=…)`
- Load `SensitivityKernelCatalog` from `_SCRIPT_DIR / '../kernels_modeplotaat_Adrian'`
- `vp_kernel_provider = SensitivityKernelProvider(M_vp, catalog, interpolation_method='cubic', include_discontinuities=True, kernel_type='vp')`
- `G = SOLAOperator(M_vp, D, vp_kernel_provider, integration_config=sola_cfg)`
- `width = 0.2 * EARTH_RADIUS_KM`; centres linearly spaced
- `target_provider = BumpFunctionProvider(M_vp, centers=centres, default_width=width)`
- `T = SOLAOperator(M_vp, P, target_provider, integration_config=sola_cfg)`
- True model: `ran_array_vp = np.zeros(N); ran_array_vp[:10] = np.random.RandomState(42).uniform(-1, 1, 10)`; `m_bar = M_vp.from_components(ran_array_vp)`
- Data: `d_bar = G(m_bar)`; `noise_std = noise_level_fraction * max(|d_bar|)`; seed 42; `d_tilde = d_bar + noise`
- `true_props = T(m_bar)`
- Returns `(M_vp, D, P, G, T, m_bar, d_tilde, d_bar, true_props, noise_variance, function_domain, catalog)` as a `ProblemResult` namedtuple

The `build_vp_prior()` function in fast mode returns a `GaussianMeasure` on `M_vp`
directly (no DirectSum wrapping).

#### 1.5 `build_realistic_problem()` function

Encapsulates the full 5-component setup from the old script verbatim:

```
def build_realistic_problem() -> ProblemTuple:
```

- Creates `M_vp`, `M_vs` (partitioned: IC + mantle), `M_rho`, `M_sigma_0`, `M_sigma_1`
- `M_model = HilbertSpaceDirectSum([LebesgueSpaceDirectSum([M_vp, M_vs, M_rho]), HilbertSpaceDirectSum([M_sigma_0, M_sigma_1])])`
- Forward operator: `G = RowLinearOperator([G_functions, G_euclidean])` with topo kernels
- Target operator: `T = RowLinearOperator([T_functions, T_euclidean])` with NullFunctionProviders for vs/rho/sigma
- Same true model coefficients, same noise generation
- Returns the same namedtuple shape

#### 1.6 Shared figures

A `generate_shared_figures(problem, figures_folder)` function that:
- In fast mode: 3 single-panel figures (true model vp, sensitivity kernels vp, target kernels vp)
- In realistic mode: multi-panel figures (3-panel true models, 5-panel sensitivity kernels, 3-panel target kernels + 2 scatter plots for topo) — identical to old script
- Also: synthetic observations scatter, data likelihood errorbar — same in both modes

#### 1.7 Validation

Run with `FAST_MODE = True`, `CASES_TO_RUN = ["Reference"]`, confirm:
- Script runs to the shared-figures stage without errors
- Shared figures are saved to `prior_sensitivity_fast/`
- Runtime for setup phase < 60 s

---

## Phase 2: Prior builder infrastructure + 7 decoupled case definitions

### Objective

Implement `build_vp_prior()`, `build_full_prior()`, `compute_reference_property_std()`,
the auto mean-offset calculation, the optional R± variance normalisation bisection,
and the `PRIOR_CASES` dict with all 7 entries.

### Files/Functions to Modify/Create

- `prior_sensitivity_study.py`: add the `PRIOR_CASES` dict, builder functions, and
  the pre-loop calibration step.

### What to build

#### 2.1 `PRIOR_CASES` dict

```python
PRIOR_CASES = {
    "Reference": {"s": 6.0, "ls": 30, "ov_mult": 1.0,   "offset_sigma": 0.0},
    "R+":        {"s": 12.0, "ls": 30, "ov_mult": 1.0,   "offset_sigma": 0.0},
    "R-":        {"s": 3.0, "ls": 10, "ov_mult": 1.0,   "offset_sigma": 0.0},
    "V+":        {"s": 6.0, "ls": 30, "ov_mult": 4.0,   "offset_sigma": 0.0},
    "V-":        {"s": 6.0, "ls": 30, "ov_mult": 0.25,  "offset_sigma": 0.0},
    "M+":        {"s": 6.0, "ls": 30, "ov_mult": 1.0,   "offset_sigma": +1.5},
    "M-":        {"s": 6.0, "ls": 30, "ov_mult": 1.0,   "offset_sigma": -1.5},
}

CASE_COLORS = {
    "Reference": "tab:blue",
    "R+":        "tab:green",
    "R-":        "tab:orange",
    "V+":        "tab:purple",
    "V-":        "tab:pink",
    "M+":        "tab:red",
    "M-":        "tab:brown",
}
```

Each case dict also has computed fields added at runtime:
- `"overall_variance"`: `OVERALL_VARIANCE_BASE * ov_mult` (possibly overridden for R± if `NORMALISE_R_VARIANCE`)
- `"mean_offset_model"`: auto-computed from `offset_sigma` (0 for non-M cases)

#### 2.2 `build_vp_prior(M_vp, s, length_scale, overall_variance, mean_offset_model, bcs, lap_kwargs, bs_kwargs)` function

Identical logic to the old `build_vp_prior` but with explicit `overall_variance` and
`mean_offset_model` args:

```python
def build_vp_prior(M_vp, s, length_scale, overall_variance,
                   mean_offset_model=0.0, n_kl=100, **kwargs):
    k = overall_variance ** (-0.5 / s)
    alpha = (length_scale ** 2) * (k ** 2)
    L = Laplacian(M_vp, bcs_vp, alpha, **lap_kwargs)
    C = BesselSobolevInverse(M_vp, M_vp, k, s, L, **bs_kwargs)
    m_0 = Function(M_vp, evaluate_callable=lambda x, _off=mean_offset_model: _off * np.ones_like(x))
    sampler = KLSampler(C, mean=m_0, n_modes=n_kl)
    return GaussianMeasure(covariance=C, expectation=m_0, sample=sampler.sample), sampler
```

#### 2.3 `build_full_prior(M_prior_vp)` function

- Fast mode: returns `M_prior_vp` directly.
- Realistic mode: wraps with fixed vs/rho/sigma priors via `GaussianMeasure.from_direct_sum`.

The fixed priors (vs_IC, vs_M, rho, sigma_0, sigma_1) are built once before the loop
(moved out of the old loop, which already did this).

#### 2.4 Auto mean-offset calculation

Before the inference loop, run once:

```python
# Build reference prior to calibrate mean-offset scale
M_prior_ref, _ = build_vp_prior(M_vp, s=6.0, ls=30, ov=OVERALL_VARIANCE_BASE, offset=0.0)
full_prior_ref = build_full_prior(M_prior_ref)

# Push reference prior through T to get property-space σ
prior_P_ref = full_prior_ref.affine_mapping(operator=T)
cov_P_ref = prior_P_ref.covariance.matrix(dense=True, parallel=True, n_jobs=…)
REF_PROP_STD = float(np.mean(np.sqrt(np.diag(cov_P_ref))))

# Compute T operator "norm" for converting property-σ to model-offset
T_mat = T.matrix(dense=True, parallel=True, n_jobs=…)  # (N_p, N) or (N_p, large)
T_row_norms = np.sqrt(np.sum(T_mat**2, axis=1))
T_mean_norm = float(np.mean(T_row_norms))

# For each M± case, compute offset in model units
for label, cfg in PRIOR_CASES.items():
    if cfg["offset_sigma"] != 0.0:
        cfg["mean_offset_model"] = cfg["offset_sigma"] * REF_PROP_STD / T_mean_norm
    else:
        cfg["mean_offset_model"] = 0.0
    cfg["overall_variance"] = OVERALL_VARIANCE_BASE * cfg["ov_mult"]

print(f"REF_PROP_STD = {REF_PROP_STD:.6f}")
print(f"T_mean_norm  = {T_mean_norm:.6f}")
for label, cfg in PRIOR_CASES.items():
    if cfg["offset_sigma"] != 0.0:
        print(f"  {label}: offset_model = {cfg['mean_offset_model']:.6f} "
              f"(= {cfg['offset_sigma']:.1f}σ in property space)")
```

#### 2.5 Optional R± variance normalisation

If `NORMALISE_R_VARIANCE = True`:

```python
def normalise_ov_for_property_std(target_std, s, ls, M_vp, T, bcs, lap_kwargs, bs_kwargs,
                                   parallel_cfg, tol=0.01, max_iter=20):
    """Bisect overall_variance so that mean property-space σ ≈ target_std."""
    ov_lo, ov_hi = 1e-2, 1e6
    for _ in range(max_iter):
        ov_mid = np.sqrt(ov_lo * ov_hi)  # geometric mean
        prior_vp, _ = build_vp_prior(M_vp, s, ls, ov_mid, 0.0)
        full_prior = build_full_prior(prior_vp)
        prior_P = full_prior.affine_mapping(operator=T)
        cov = prior_P.covariance.matrix(dense=True, parallel=True, n_jobs=…)
        current_std = float(np.mean(np.sqrt(np.diag(cov))))
        if abs(current_std - target_std) / target_std < tol:
            return ov_mid
        if current_std > target_std:
            ov_hi = ov_mid
        else:
            ov_lo = ov_mid
    warnings.warn(f"Bisection did not converge; using ov={ov_mid:.4f}")
    return ov_mid
```

Called for R+ and R- only, replacing their `overall_variance` in the config dict.
Costs ~2 extra prior pushforward evaluations per R case (fast in fast mode).

#### 2.6 Validation

Run Reference and M+ cases. Confirm:
- `REF_PROP_STD` and `T_mean_norm` are printed
- M+ auto-computed offset is ~1.5 × `REF_PROP_STD` / `T_mean_norm`
- Both cases produce `GaussianMeasure` objects without error

---

## Phase 3: Inference loop + selective execution + per-case figures

### Objective

Wire up the main inference loop with `CASES_TO_RUN` filtering, compute all diagnostics,
generate per-case figures.

### Files/Functions to Modify/Create

- `prior_sensitivity_study.py`: inference loop block, diagnostic computation, per-case
  figure generation.

### What to build

#### 3.1 Resolve `CASES_TO_RUN`

```python
cases_to_run = list(PRIOR_CASES.keys()) if "all" in CASES_TO_RUN else CASES_TO_RUN
print(f"Will run: {cases_to_run}")
```

#### 3.2 Inference loop

```python
solver = CholeskySolver(parallel=True, n_jobs=parallel_cfg.n_jobs)
results = {}   # will be populated from CSV in Phase 4

for label, cfg in PRIOR_CASES.items():
    if label not in cases_to_run:
        continue

    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"Running case: {label}")
    print(f"  s={cfg['s']}, ls={cfg['ls']}, ov={cfg['overall_variance']:.2f}, "
          f"offset_model={cfg['mean_offset_model']:.6f}")
    print(f"{'='*70}")

    # Build vp prior for this case
    M_prior_vp, sampler_vp = build_vp_prior(
        M_vp, cfg['s'], cfg['ls'], cfg['overall_variance'],
        cfg['mean_offset_model'])
    M_prior = build_full_prior(M_prior_vp)

    # Prior property uncertainty
    prior_P = M_prior.affine_mapping(operator=T)
    cov_P_prior = prior_P.covariance.matrix(dense=True, parallel=True, n_jobs=…)
    std_P_prior = np.sqrt(np.diag(cov_P_prior))

    # Prior predictive check (Phase 5 placeholder — just set chi2=0.0 for now)
    chi2_per_dof = 0.0

    # Forward problem + inference
    forward_problem = LinearForwardProblem(G, data_error_measure=gaussian_D_noise)
    bayes = LinearBayesianInversion(forward_problem, M_prior)
    posterior_model = bayes.model_posterior_measure(d_tilde, solver)
    m_tilde = posterior_model.expectation

    # Property posterior
    prop_post = posterior_model.affine_mapping(operator=T)
    p_tilde = prop_post.expectation
    cov_P = prop_post.covariance.matrix(dense=True, parallel=True, n_jobs=…)
    std_P = np.sqrt(np.diag(cov_P))

    elapsed = time.time() - t0

    # Diagnostics (same as old script)
    errors = p_tilde - true_props
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae  = float(np.mean(np.abs(errors)))
    cover = float(np.sum(np.abs(errors) <= 2*std_P) / N_p)
    mean_post_std  = float(np.mean(std_P))
    mean_prior_std = float(np.mean(std_P_prior))
    unc_reduction  = float(1.0 - mean_post_std / mean_prior_std)
    data_misfit    = float(np.linalg.norm(G(m_tilde) - d_tilde))
    min_eig        = float(np.linalg.eigvalsh(cov_P).min())

    row = dict(case=label, ..., chi2_per_dof=round(chi2_per_dof, 4), ...)
    results[label] = row
```

#### 3.3 Per-case figures

A `generate_case_figures(label, cfg, color, problem, prior, posterior, sampler, …)` function.

**Fast mode** — 4 figures per case, each single-panel:
1. **Prior measure on model space (vp)**: ±2σ band + prior samples + prior mean + true model
2. **Property prior distribution**: errorbar plot of prior properties ±2σ + true properties
3. **Model posterior (vp)**: posterior mean + true model + prior mean
4. **Property inference results**: errorbar plot of posterior properties ±2σ + true properties + prior properties

**Realistic mode** — 4 figures per case, multi-panel:
1. Prior measure: vp (1 panel) + vs partitioned (1 panel) + rho (1 panel) + σ₀ PDF (1 panel) + σ₁ PDF (1 panel) = 5 panels
2. Property prior distribution: single panel (same in both modes)
3. Model posterior: same 5-panel layout as prior measure
4. Property inference results: single panel (same in both modes)

Figures saved to `FIGURES_FOLDER / <label_slug>/` where
`label_slug = label.lower().replace('+', '_plus').replace('-', '_minus')`.

File names within each case directory:
- `prior_measure_on_model_space_(vp_&_vs_&_rho_&_sigmas).{png,pdf}` (realistic)
  or `prior_measure_on_model_space_(vp).{png,pdf}` (fast)
- `property_prior_distribution.{png,pdf}`
- `model_posterior_distribution_(…).{png,pdf}`
- `property_inference_results.{png,pdf}`

#### 3.4 Validation

Run all 7 cases in fast mode. Confirm:
- All 7 complete in total < 10 min
- Each case produces 4 PNG+PDF file pairs
- Console prints all diagnostics

---

## Phase 4: Incremental CSV/MD persistence

### Objective

Load existing CSV results at startup, merge only the newly-run cases, write back all
results so a single re-run preserves the other cached cases.

### Files/Functions to Modify/Create

- `prior_sensitivity_study.py`: startup CSV load, end-of-script write, TXT append.

### What to build

#### 4.1 Load existing results at startup

```python
results = {}
if Path(OUTPUT_CSV).exists():
    with open(OUTPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["case"]] = row
    print(f"Loaded {len(results)} existing results from {OUTPUT_CSV}")
```

#### 4.2 Upsert after each case

Already in Phase 3: `results[label] = new_row`.

#### 4.3 Write all results at end

```python
# Order by PRIOR_CASES key order; include cached + new
ordered_results = [results[k] for k in PRIOR_CASES if k in results]
```

Write CSV with `csv.DictWriter` (all fieldnames, same as old script + `chi2_per_dof`).
Write MD table with same format.

#### 4.4 Append-mode TXT log

```python
with open(OUTPUT_TXT, "a") as f:
    f.write(f"\n{'='*70}\n")
    f.write(f"Run at {datetime.datetime.now().isoformat()}\n")
    f.write(f"Cases: {cases_to_run}\n")
    f.write(f"{'='*70}\n\n")
    for line in log_lines:
        f.write(line + "\n")
```

#### 4.5 Console summary table

Same format as old script, printed for ALL results (cached + new), marking
cached rows with `(cached)` suffix on the case name.

```
==============================================================================================
Case              RMSE      MAE    Cover   UncRed     Misfit     MinEig  chi2/dof    Time
----------------------------------------------------------------------------------------------
Reference       0.0359   0.0220    50.0%    74.7%   504.1196   1.01e-09     1.02    30.2s
R+              0.0381   0.0242    25.0%    82.2%   546.0396  -2.25e-12     1.15    28.7s
R- (cached)     0.0150   0.0109   100.0%    67.5%   475.9222   7.88e-06     0.89    31.4s
...
==============================================================================================
```

#### 4.6 Validation

1. Run with `CASES_TO_RUN = ["Reference"]` → CSV has 1 data row
2. Run with `CASES_TO_RUN = ["R+"]` → CSV has 2 data rows (Reference preserved)
3. Run with `CASES_TO_RUN = ["R+"]` again → CSV still 2 rows, R+ values updated

---

## Phase 5: Prior predictive fairness check

### Objective

Before each inference, compute the normalised Mahalanobis distance Δ²/Nd
in data space and log it as a diagnostic column.

### Files/Functions to Modify/Create

- `prior_sensitivity_study.py`: new `prior_predictive_check()` function, integration
  into inference loop.

### What to build

#### 5.1 `prior_predictive_check()` function

```python
def prior_predictive_check(M_prior, G, d_tilde, C_D_matrix, parallel_cfg):
    """
    Compute the normalised prior-predictive Mahalanobis distance.

    Under a compatible prior: chi2/dof ~ 1.
    Large values indicate prior-data conflict.

    Parameters
    ----------
    M_prior : GaussianMeasure
        The full prior measure on model space.
    G : LinearOperator or RowLinearOperator
        Forward operator.
    d_tilde : array or list of arrays
        Observed data.
    C_D_matrix : ndarray (N_d, N_d)
        Data error covariance matrix.
    parallel_cfg : ParallelConfig

    Returns
    -------
    chi2_per_dof : float
        (d - G mu_0)^T S^{-1} (d - G mu_0) / N_d  where S = G C_0 G^T + C_D
    """
    # Prior mean prediction in data space
    mu_0 = M_prior.expectation
    d_prior = np.asarray(G(mu_0)).ravel()
    d_obs = np.asarray(d_tilde).ravel()
    N_d = len(d_obs)

    # Prior predictive covariance: S = G C0 G* + C_D
    prior_D = M_prior.affine_mapping(operator=G)
    S = prior_D.covariance.matrix(dense=True, parallel=True, n_jobs=parallel_cfg.n_jobs)
    S = S + C_D_matrix

    # Mahalanobis distance
    residual = d_obs - d_prior
    chi2 = float(residual @ np.linalg.solve(S, residual))
    return chi2 / N_d
```

#### 5.2 Integration into inference loop

Replace the Phase 3 placeholder `chi2_per_dof = 0.0` with:

```python
chi2_per_dof = prior_predictive_check(M_prior, G, d_tilde, C_D_matrix, parallel_cfg)
print(f"  Prior predictive check: chi2/dof = {chi2_per_dof:.4f}")
if chi2_per_dof > PRIOR_PREDICTIVE_CHI2_THRESHOLD:
    print(f"  WARNING: PRIOR PREDICTIVELY INCOMPATIBLE "
          f"(chi2/dof = {chi2_per_dof:.1f} > {PRIOR_PREDICTIVE_CHI2_THRESHOLD})")
```

#### 5.3 New CSV/MD column

Add `"chi2_per_dof"` to `fieldnames` list and MD column headers.

#### 5.4 Validation

- Reference: expect chi2/dof ≈ 1.0 (compatible prior)
- M± at ±1.5σ: expect chi2/dof slightly > 1 but < 3 (mildly incompatible)
- Hypothetical M+ at offset +3 (old script): would give chi2/dof >> 3 (rejected)

---

## Summary

| Phase | What | Key deliverable |
|---|---|---|
| 1 | Script scaffold + FAST_MODE | Two problem builders, shared figures |
| 2 | Prior builders + 7 cases + auto offset + optional R± normalisation | `PRIOR_CASES`, `build_vp_prior`, `normalise_ov_for_property_std` |
| 3 | Inference loop + selective execution + figures | Full diagnostic pipeline, per-case plots |
| 4 | Incremental persistence | CSV/MD survives partial re-runs |
| 5 | Prior predictive check | chi2/dof fairness gate |

Total estimated runtime for all 7 cases:
- **Fast mode**: ~5–7 min
- **Realistic mode**: ~21–28 min
