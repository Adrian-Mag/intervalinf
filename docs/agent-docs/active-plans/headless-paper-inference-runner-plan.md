# Plan: Headless Paper Inference Runner

Build a non-interactive, config-driven runner for the normal-mode paper demos in
`intervalinf/demos/old_demos/paper_demos/`.  The runner should reproduce the
computational capabilities of `prior_posterior_tuner_app.py` without the Tk GUI,
and should become the primary path for paper-quality results, prior/noise
sweeps, numerical-stability studies, and overnight batch runs.

The GUI remains useful for exploration, but it currently hides too much state in
callbacks and writes only GUI-oriented artifacts.  The paper workflow needs a
stable command-line entry point with explicit configuration, deterministic run
directories, rich figures, machine-readable diagnostics, and enough provenance
to rerun or audit every result.

## Implementation Status

Implemented on 2026-06-04 under
`intervalinf/demos/old_demos/paper_demos/headless/` with CLI entry points
`run_paper_inference.py` and `run_paper_sweep.py`. Baseline flat/weighted
configs and an example prior/noise/truncation sweep live in `configs/`, generated
artifacts are ignored under `runs/`, and focused config/sweep tests live in
`tests/test_headless_config.py` and `tests/test_headless_sweeps.py`.

---

## Current Situation

The useful compute path already exists, but it is split across UI callbacks,
remote helper scripts, and plotting helpers:

| Area | Existing code to reuse | Notes |
|---|---|---|
| Block inventory, forward operators, priors, posterior solves, property pushforward | `utils/full_spectrum_utils.py` | Main computational spine.  Already has `RadialSpecs`, block builders, `solve_block`, `solve_all_blocks`, `build_property_operator`, and `assemble_property_posterior`. |
| Prior hyperparameter translation | `visualization/prior_posterior_tuner.py::_build_custom_bessel_blocks` | Good starting point for `k/s/alpha` plus boundary conditions, but the headless path should drop the legacy `s_order/length/var` format. |
| GUI orchestration | `prior_posterior_tuner_app.py` | Has the desired workflow but state is embedded in Tk widgets, thread callbacks, and local/europa routing. |
| Remote batch-ish execution | `prior_posterior_tuner_remote.py` | Good reference for single-block, all-block, and inference tasks, but outputs are too narrow and config is not rich enough for paper studies. |
| Prior figures | `visualization/prior_viz.py` | `PriorViewer` and `_draw_prior_axes` can be driven headlessly with Matplotlib Agg. |
| Posterior radial/data-fit figures | `visualization/posterior_viz.py` | `_compute_posterior_display_data`, `compute_model_curves`, and `render_posterior_figure` are reusable. |
| Full-spectrum figures | `visualization/full_spectrum_viz.py` | `plot_block_posterior`, `plot_cmb_map`, `plot_equatorial_slice`, and `plot_property_posterior_summary` already cover several required paper outputs. |
| Target-kernel figures | `visualization/target_kernel_viz.py` | `TargetKernelViewer`, `_draw_geographic_axes`, and `_draw_st_axes` can generate target diagnostics without the interactive viewer. |
| Target definitions | `utils/property_targets.py` | Current API uses compact spherical caps and compact/boxcar radial kernels.  The runner must configure these explicitly instead of relying on GUI defaults. |

Main gaps:

- No first-class headless entry point for the complete workflow.
- No typed config schema covering data selection, prior families, noise scaling,
  integration accuracy, block selection, target definitions, plotting, random
  seeds, and operator truncation.
- No deterministic run directory contract.
- No complete output manifest with resolved config, code provenance, input data
  hashes, timings, spectral diagnostics, stability diagnostics, and figure
  inventory.
- No sweep driver that expands many configs and runs them unattended.
- Existing code still uses `n_basis` naming in places where the real concept is
  covariance-operator truncation, which risks confusing truncation with the
  model-space basis.  The headless runner should correct that terminology.
- Existing outputs are mostly display NPZ files; they do not yet record enough
  diagnostics for paper reproducibility or numerical accuracy studies.

---

## Design Goals

- **Reproducibility:** every run writes the input config, a resolved canonical
  config, git/code provenance, package versions, random seeds, data/kernel path
  fingerprints, wall-clock timings, and a complete artifact manifest.
- **Flexibility:** expose the knobs currently in the GUI plus numerical-study
  knobs: `s_max`, block filters, data subset rules, covariance truncation,
  weighted-vs-flat geometry, integration grids, SOLA grids, Bessel-Sobolev
  hyperparameters, boundary conditions, tau scaling, `sigma_var`,
  data-noise multiplier, synthetic data settings, posterior sample counts,
  covariance probe counts, target definitions, and figure selection.
- **Paper outputs:** save prior, target-kernel, posterior, data-fit,
  reconstructed-field, map, inference, spectrum, conditioning, timing, and
  metadata artifacts in a stable layout.
- **Operator-first semantics:** keep priors and posteriors as
  `GaussianMeasure`/`LinearOperator` objects.  Dense matrices are diagnostics
  only, not the primary model.
- **Basis-free model spaces:** the model spaces should remain basis-free.
  Finite truncation parameters belong to covariance operators and diagnostics,
  not to the definition of the model space itself.
- **Geometry awareness:** the runner must treat the two valid streams as
  separate defaults: flat `L2(dr)` with the ordinary Laplacian, and weighted
  `L2(r^2 dr)` with the radial Laplacian.  Their default `k`, `s`, `alpha`,
  and boundary-condition choices must be explicit and stream-specific.
- **Batch friendliness:** one CLI can run one config; a second CLI expands a
  sweep config into many concrete runs, can continue after individual sub-run
  failures, and is designed for simultaneous local and europa execution.
- **GUI compatibility:** factor shared logic so the GUI can eventually call the
  same pipeline, reducing drift between exploratory and paper workflows.

Non-goals for the first implementation:

- Do not replace `prior_posterior_tuner_app.py`.
- Do not add a database or workflow engine.
- Do not add new external dependencies unless a strong need appears.  Prefer
  Python 3.11 `tomllib` plus standard `json` for config parsing and existing
  NumPy/SciPy/Matplotlib tooling.
- Do not materialize full dense model covariance matrices.

---

## Proposed Layout

Create a small headless package under `paper_demos/`:

```text
intervalinf/demos/old_demos/paper_demos/
+-- configs/
|   +-- paper_baseline_flat.toml
|   +-- paper_baseline_weighted.toml
|   +-- sweep_prior_noise.toml
|   +-- README.md
+-- runs/                         # gitignored or user-managed output root
+-- headless/
|   +-- __init__.py
|   +-- config.py                 # typed config dataclasses + validation
|   +-- pipeline.py               # pure orchestration; no plotting side effects
|   +-- diagnostics.py            # spectra, conditioning, stability, timings
|   +-- figures.py                # Agg figure generation + save helpers
|   +-- artifacts.py              # run directory, manifest, hashes, provenance
|   +-- sweeps.py                 # config expansion and run queue
+-- run_paper_inference.py         # single-run CLI
+-- run_paper_sweep.py             # sweep CLI
```

The `headless/` code should import existing `utils/` and `visualization/`
modules through the same path setup pattern used by the visualization scripts.
It should not move the current utilities during Phase 1; refactoring can happen
after the headless path is tested.

---

## Config Contract

Use TOML for human-authored configs and write the resolved canonical config as
JSON inside the run directory.  The runner should also accept JSON config files
for machine-generated runs or compatibility with existing remote-script habits.
TOML parsing is available through Python 3.11 `tomllib`, so this avoids a new
dependency.

Example single-run config:

```toml
[run]
name = "baseline_smax4_flat"
output_root = "runs"
overwrite = false
seed = 0

[paths]
data_dir = "data/normal-mode-data"
kernel_dir = "data/normal-mode-kernels/kernels-all_PREM-layers_Adrian"

[data]
s_max = 4
blocks = "all"                    # or ["s0_t0", "s2_t0", "s4_t3"]
mode_filter = "kernel_catalog"
data_mode = "real"                # real | synthetic
max_data_per_block = 0            # 0 means no cap

[synthetic]
noise_scale = 1.0
null_amp = 0.0
seed = 0

[model_space]
covariance_truncation = 100
weighted = false
earth_radius_km = 6371.0
icb_radius_km = 1217.5
cmb_radius_km = 3480.0

[defaults]
stream = "flat"                   # flat | weighted

[integration.lebesgue.inner_product]
method = "trapz"
n_points = 1024

[integration.lebesgue.dual]
method = "trapz"
n_points = 1024

[integration.lebesgue.general]
method = "trapz"
n_points = 1024

[integration.sola]
method = "trapz"
n_points = 2048

[compute]
n_jobs = 1
threadpool_limit = 1
parallel_blocks = false

[prior]
family = "bessel"
sigma_var = 100.0

[prior.taus]
vp = 1.0
vs_IC = 1.0
vs_M = 1.0
rho = 1.0

[prior.hyper.vp]
k = 1.0
s = 3.0
alpha = 250000.0
bc = "neumann"

[prior.hyper.vs_IC]
k = 1.0
s = 3.0
alpha = 250000.0
bc = "neumann"

[prior.hyper.vs_M]
k = 1.0
s = 3.0
alpha = 250000.0
bc = "neumann"

[prior.hyper.rho]
k = 1.0
s = 3.0
alpha = 250000.0
bc = "neumann"

[noise]
data_noise_multiplier = 1.0

[[targets]]
type = "cap_bulk"
param = "vp"
lat_deg = 0.0
lon_deg = 0.0
cap_radius_deg = 15.0
r0_km = 4920.0
width_km = 400.0

[[targets]]
type = "boxcar_bulk"
param = "vs"
lat_deg = 45.0
lon_deg = 90.0
cap_radius_deg = 15.0
r_low_km = 3480.0
r_high_km = 3780.0

[[targets]]
type = "cap_cmb"
lat_deg = 0.0
lon_deg = 0.0
cap_radius_deg = 15.0

[outputs]
n_grid = 300
n_prior_samples = 8
n_posterior_samples = 8
n_covariance_probes = 20
save_npz = true
save_csv = true
save_figures = true
save_samples = true
figure_formats = ["png", "pdf"]

[diagnostics]
prior_spectra = true
data_spectra = true
normal_operator = true
posterior_contraction = true
condition_numbers = true
predictive_chi = true
operator_timing = true
truncation_refinement_compare = []
integration_refinement_compare = []
```

Validation rules:

- Reject unknown top-level sections and unknown target types.
- Require all four prior radial parameters: `vp`, `vs_IC`, `vs_M`, `rho`.
- Accept TOML and JSON inputs, but always write one canonical
  `resolved_config.json`.
- The model space must be basis-free; reject configs that try to define a model
  basis through truncation settings.
- `covariance_truncation` is the truncation used in covariance operators and
  related spectral approximations.  It is not a model-space basis size and must
  be described that way in code, config, tables, and figures.
- Require explicit boundary conditions `bc` for each radial prior component, or
  fill them from stream defaults in the resolved config.
- Drop the legacy `s_order/length/var` input format from the headless runner.
- Validate all positive numerical parameters before any expensive operator work.
- Make target labels deterministic and store them in the manifest.
- Expand relative paths relative to the config file location, not process CWD.
- Record the flat/weighted duality explicitly in the resolved config:
  inner-product family, operator family, and per-parameter default hyperparameters.

### Stream defaults

The runner should ship two baseline configs and treat them as separate streams:

- `paper_baseline_flat.toml`
  Uses plain `L2(dr)` model-space inner products and the ordinary Laplacian.
- `paper_baseline_weighted.toml`
  Uses weighted `L2(r^2 dr)` model-space inner products and the radial
  Laplacian.

Both streams must expose separate default `k`, `s`, `alpha`, `bc`, integration
settings, and diagnostic tolerances.  The resolved config must always state
which stream was used and which defaults were inherited.

The numerical defaults should match the current tuner app:

- Flat stream defaults: `k = 1.0`, `s = 3.0`, `alpha = 250000.0`,
  `tau = 1.0` for all components.
- Weighted stream defaults: `k = 1.0e-3`, `s = 3.0`, `alpha = 100.0`,
  `tau = 1.0` for all components.

Boundary conditions are not exposed in the tuner app UI, but the headless
runner must expose them and should default to the current paper-demo operator
defaults unless overridden:

- `vp`: `mixed_neumann_dirichlet`
- `vs_IC`: `neumann`
- `vs_M`: `mixed_neumann_dirichlet`
- `rho`: `mixed_neumann_dirichlet`

---

## Run Directory Contract

Each run writes to:

```text
runs/<run_name>__<timestamp-or-hash>/
+-- config/
|   +-- input.toml
|   +-- resolved_config.json
|   +-- config_diff_from_default.json
+-- metadata/
|   +-- manifest.json
|   +-- provenance.json
|   +-- input_fingerprints.json
|   +-- environment.txt
+-- arrays/
|   +-- blocks.npz
|   +-- prior_summary.npz
|   +-- posterior_summary.npz
|   +-- property_posterior.npz
|   +-- posterior_samples.npz
|   +-- diagnostics.npz
+-- tables/
|   +-- block_inventory.csv
|   +-- data_fit.csv
|   +-- timings.csv
|   +-- prior_spectra.csv
|   +-- data_spectra.csv
|   +-- posterior_contraction.csv
|   +-- stability_summary.csv
+-- figures/
|   +-- prior/
|   +-- target_kernels/
|   +-- posterior_blocks/
|   +-- data_fit/
|   +-- reconstructed_fields/
|   +-- inference/
|   +-- diagnostics/
+-- logs/
    +-- run.log
    +-- warnings.log
```

`manifest.json` should include every produced artifact with:

- relative path,
- artifact type,
- block or target identifier where applicable,
- config dependencies,
- creation time,
- file size,
- SHA256 hash for small files and optionally for large NPZ files.

`runs/` should be committed as an empty directory with a local `.gitignore`
that preserves the directory and ignores generated run contents.

---

## Pipeline API

The headless code should expose a pure, testable orchestration layer:

```python
def run_inference_from_config(config_path: Path) -> RunResult:
    config = load_config(config_path)
    return run_inference(config)

def run_inference(config: PaperRunConfig) -> RunResult:
    context = build_context(config)
    prior_result = build_prior(context)
    posterior_result = solve_posteriors(context, prior_result)
    inference_result = compute_property_inference(context, posterior_result)
    diagnostics = compute_diagnostics(context, prior_result, posterior_result, inference_result)
    artifacts = write_artifacts(context, prior_result, posterior_result, inference_result, diagnostics)
    return RunResult(config=config, run_dir=context.run_dir, artifacts=artifacts)
```

Recommended internal objects:

- `PaperRunConfig`: typed dataclasses for all TOML sections.
- `RunContext`: resolved paths, registry/catalog, `RadialSpecs`, blocks, split
  registries, targets, random generators, and artifact writer.
- `GeometryConfig`: explicit description of flat-vs-weighted geometry,
  associated Laplacian family, and stream defaults.
- `PriorResult`: shared Bessel blocks, prior viewer data, spectra, prior timing.
- `PosteriorResult`: `forward_dict`, per-block posterior measures, display data,
  posterior samples, predictive data fits, solve timings.
- `InferenceResult`: property posterior measure plus dense `mu_P`, `C_P`,
  `sigma_P`, `corr_P`, target names.
- `DiagnosticResult`: all spectra, condition numbers, residual metrics,
  contraction metrics, stability comparisons, and warnings.

The GUI callbacks should not be imported by the headless runner.  Shared helpers
such as `_build_scaled_data_noise_measure` should be moved into a small utility
module or duplicated briefly in Phase 1 and then deduplicated in Phase 2.

The pipeline should also expose a per-subrun failure boundary so sweep
execution can mark one run failed, write its partial logs/manifest, and proceed
to the next run without killing the whole overnight job.

---

## Required Figures

The runner should support a selected subset or all figures.

Mandatory outputs for every paper run:

1. Prior figures and prior samples.
2. Posterior mean model figures and posterior samples.
3. Inference solution figures and arrays.

Everything else may be optional by config.

Optional figure families:

1. **Prior figures**
   - One prior figure per representative block, using `PriorViewer` and
     `_draw_prior_axes`.
   - Per-parameter prior sample panels.
   - Prior eigenvalue/spectrum plots for each parameter.

2. **Target-kernel figures**
   - Geographic cap/radial profile for every target.
   - Per-block `(s,t)` coefficient plots for every target.
   - Reconstruction comparison: full `s <= s_max` versus available-block
     reconstruction.

3. **Posterior block figures**
   - Radial posterior mean plus samples for selected blocks.
   - Optional covariance-probed uncertainty bands using `plot_block_posterior`.
   - Data-fit panel with observed/synthetic data, assumed errors, predictions,
     standardized residuals, and chi/RMS annotation.

4. **Reconstructed fields**
   - Posterior-mean CMB topography map.
   - Optional CMB sample maps.
   - Optional equatorial/depth slices for `vp`, `vs`, and `rho`.
   - Optional posterior-sample reconstructed fields for uncertainty display.

5. **Inference figures**
   - Property posterior mean +/- sigma bar chart.
   - Property correlation heatmap.
   - Target-by-target SNR table figure.

6. **Diagnostics figures**
   - Prior eigenvalue decay per parameter.
   - `G C0 G*` data-space spectrum per block.
   - Noise-normalized information spectrum.
   - Posterior contraction by block/parameter/target.
   - Solve timing by block.
   - Residual diagnostics and outlier summaries.

All figures should use Matplotlib Agg and must close figures after saving to
avoid memory growth during overnight sweeps.

---

## Numerical And Performance Diagnostics

Minimum diagnostics to implement:

- Block inventory: `s`, `t`, number of observations, modes present, data-error
  ranges, whether the block was solved.
- Prior spectra: eigenvalues from each Bessel-Sobolev covariance block,
  effective rank, cumulative variance, trace proxy.
- Normal-operator diagnostics before inversion: approximation and stability
  metrics for the operator actually being inverted, since this is the main
  numerical failure point in practice.
- Data-space prior covariance: dense `G C0 G*` per block, eigenvalues,
  condition number, effective rank, diagonal range, noise-normalized spectrum.
- Approximated normal operator diagnostics: eigenvalues or singular-value
  surrogates, symmetry/self-adjointness checks where applicable, condition
  metrics, and refinement deltas under increased truncation/integration.
- Posterior contraction: selected probe variances prior versus posterior,
  contraction ratios, and target-space prior/posterior contraction.
- Data fit: posterior predictive residuals, standardized residuals using
  `data_noise_multiplier * error_vector`, RMS standardized residual,
  chi-squared per datum, maximum residual, and outlier count.
- Linear solve timing: build-forward time, build-prior time, solve time,
  display-data time, sample time, property-operator time, covariance-pushforward
  time, figure time.
- Operator/cache stats where available: SOLA fast-path/cache counters,
  Bessel object reuse, native thread limits, job count.
- Stability checks: symmetry/PSD checks for dense diagnostic covariances and
  property covariance, finite-value checks, and correlation diagonal checks.
- Optional refinement studies: rerun a smaller selected block set at higher
  covariance truncation or higher integration `n_points`, then compare the
  normal operator first, followed by posterior means, property posterior,
  residual metrics, and spectra.

Diagnostics should be written as both CSV tables and compact NPZ arrays where
appropriate.

Optional dense diagnostics that are likely useful:

- Dense normal-operator approximations for selected blocks.
- Dense `G C0 G*` matrices for selected blocks.
- Dense property-posterior covariance matrices.
- Dense blockwise posterior covariance restrictions on selected probe/test
  subspaces, not full model-space covariance matrices.
- Dense refinement-delta matrices for selected normal operators under higher
  truncation or integration settings.

---

## Sweep Support

Create `run_paper_sweep.py` for expanding a base config across parameter grids:

```toml
[sweep]
name = "prior_noise_grid"
base_config = "paper_baseline_flat.toml"
max_workers = 1

[[sweep.axis]]
path = "noise.data_noise_multiplier"
values = [0.5, 1.0, 2.0, 4.0]

[[sweep.axis]]
path = "prior.hyper.vp.alpha"
values = [1.0e4, 1.0e5, 1.0e6]

[[sweep.axis]]
path = "model_space.covariance_truncation"
values = [50, 100, 150]
```

Sweep behavior:

- Expand to concrete TOML configs in `runs/<sweep_name>/expanded_configs/`.
- Optionally expand to JSON as well when a machine-generated job manifest is
  requested.
- Give every run a deterministic name derived from axis values.
- Write `sweep_manifest.json` with run statuses, config paths, output dirs,
  start/end times, exit codes, and selected summary metrics.
- Support `--dry-run`, `--resume`, and `--limit N`.
- Continue after failures by default.  Individual run failures must mark the run
  as failed in the sweep manifest, preserve logs, and not stop sibling runs
  unless `--fail-fast` is explicitly requested.
- Support simultaneous local and europa execution through a shared sweep
  config format and per-run immutable configs.
- Europa support is first-class.  Submission must obey the workspace europa
  coordination rules: status check, free-space check, slot claim, registry
  check, and release on completion/abort.

Each execution backend should maintain its own sweep manifest.  Local and
europa sweeps are often unrelated and should not be forced into a single
combined manifest.

---

## Implementation Phases

### Phase 1: Config And Artifact Infrastructure

**Objective:** Add the headless directory, typed config loading, validation,
run-directory creation, logging, provenance capture, and a dry-run CLI.

**Files to create:**

- `headless/config.py`
- `headless/artifacts.py`
- `headless/pipeline.py`
- `run_paper_inference.py`
- `configs/paper_baseline_flat.toml`
- `configs/paper_baseline_weighted.toml`
- `configs/README.md`
- `tests/test_headless_config.py`

**Acceptance checks:**

- `python run_paper_inference.py --config configs/paper_baseline_flat.toml --dry-run`
  validates config and writes no expensive artifacts.
- Relative paths resolve from the config file location.
- Unknown config keys fail with clear errors.
- `resolved_config.json` is deterministic for the same input config.
- The resolved config records basis-free model spaces and covariance truncation
  terminology correctly.

### Phase 2: Single-Run Pipeline Without Figures

**Objective:** Run the complete computational workflow headlessly: load data,
enumerate/select blocks, build basis-free model-space specs, build targets,
build priors, solve posteriors, assemble property posterior, and save
arrays/tables/manifests.

**Files to create or modify:**

- `headless/pipeline.py`
- `headless/diagnostics.py`
- `headless/artifacts.py`
- `tests/test_headless_pipeline_smoke.py`

**Acceptance checks:**

- A small config with `s_max=2`, `covariance_truncation=15`, and one or two
  blocks runs
  end-to-end in the `inferences` environment.
- Property posterior arrays match `validate_local_vs_remote.py` for the same
  settings within explicit `np.testing.assert_allclose` tolerances.
- The run writes `property_posterior.npz`, `block_inventory.csv`,
  `data_fit.csv`, `timings.csv`, `manifest.json`, and `resolved_config.json`.

### Phase 3: Figure Generation

**Objective:** Generate the required prior, target, posterior, data-fit,
reconstructed-field, inference, and diagnostic figures from the headless
pipeline.

**Files to create or modify:**

- `headless/figures.py`
- Possibly small public render helpers in `visualization/prior_viz.py`,
  `visualization/target_kernel_viz.py`, and `visualization/posterior_viz.py`
  if private draw helpers need stable wrappers.
- `tests/test_headless_figures.py`

**Acceptance checks:**

- Uses `matplotlib.use("Agg")` before importing pyplot.
- Saves PNG and optional PDF outputs under `figures/`.
- Closes all figures after writing.
- Figure smoke tests verify files are non-empty and registered in
  `manifest.json`.

### Phase 4: Diagnostics And Stability Studies

**Objective:** Add spectral, conditioning, contraction, residual, timing, and
refinement diagnostics needed for paper-quality numerical assessment.

**Files to create or modify:**

- `headless/diagnostics.py`
- `headless/pipeline.py`
- `tests/test_headless_diagnostics.py`

**Acceptance checks:**

- Dense diagnostic matrices are finite, symmetric where expected, and PSD to an
  explicit numerical tolerance.
- Residual metrics use the same assumed noise as the solve:
  `data_noise_multiplier * error_vector`.
- Refinement comparisons produce clear metrics and fail only on invalid config,
  not on expected scientific differences.

### Phase 5: Sweep Runner

**Objective:** Add unattended many-run execution for prior/noise/truncation/integration
studies.

**Files to create:**

- `headless/sweeps.py`
- `run_paper_sweep.py`
- `configs/sweep_prior_noise.toml`
- `tests/test_headless_sweeps.py`

**Acceptance checks:**

- `--dry-run` lists all expanded run names and config diffs.
- `--limit 2` runs two concrete configs and writes a sweep manifest.
- `--resume` skips completed runs whose manifest says they finished
  successfully with the same resolved config hash.
- A failed sub-run does not stop the sweep unless `--fail-fast` is enabled.
- Europa submission and local execution can both consume the same expanded sweep
  configs and produce compatible manifests.

### Phase 6: GUI Reconciliation And Reference Update

**Objective:** Reduce drift between the GUI and headless runner once the
headless path is stable.

**Files to modify:**

- `prior_posterior_tuner_app.py` only where shared helpers can be imported from
  the headless or utility layer without changing GUI behavior.
- `prior_posterior_tuner_remote.py` if it can share config/pipeline helpers.
- `docs/agent-docs/references/living/intervalinf-reference.md`.

**Acceptance checks:**

- Existing paper-demo tests still pass.
- GUI local and remote computations still match `validate_local_vs_remote.py`.
- The living reference documents the headless runner, config contract, and
  output layout.

---

## Testing Strategy

Use targeted tests first, then broader checks:

```bash
conda run -n inferences python -m pytest intervalinf/demos/old_demos/paper_demos/tests
conda run -n inferences python -m pytest intervalinf/tests
```

Focused tests should use small settings:

- `s_max=2`
- `covariance_truncation=10` or `15`
- `n_grid=30` or `50`
- one to three blocks
- fixed random seeds

Numerical tests should use `np.testing.assert_allclose` with explicit
`rtol`/`atol`, especially for property posterior parity with the current remote
path.

---

## Risks And Mitigations

- **GUI-state leakage:** avoid importing `TunerApp`; move shared functions into
  plain modules.
- **Memory growth during sweeps:** close every figure, avoid storing full
  posterior samples unless configured, and write per-run artifacts promptly.
- **Sub-run failure propagation:** isolate each run in its own exception/logging
  boundary and always flush its partial manifest before moving on.
- **Dense diagnostic cost:** compute dense `G C G*` only in data space and dense
  property covariance only in target space.  Never dense-materialize full model
  covariances.
- **Terminology drift:** rename `n_basis` to `covariance_truncation` in the
  headless path so the code does not teach the wrong model-space semantics.
- **Geometry mismatch:** always bind flat `L2(dr)` to the ordinary Laplacian and
  weighted `L2(r^2 dr)` to the radial Laplacian in defaults, and record any
  deliberate override explicitly in the resolved config.
- **Hidden defaults:** write resolved configs and reject unknown keys.
- **Target drift:** configure every target explicitly, even when a baseline
  config mirrors `_make_default_targets`.
- **Thread oversubscription:** record and control `threadpool_limit`, `n_jobs`,
  and block-level parallelism separately.
- **Reproducibility versus timestamps:** use stable config hashes for resume and
  comparison; timestamps are only for human-readable directory names.

---

## Resolved Decisions

1. Accept both TOML and JSON configs.  TOML is the preferred human-authored
   format; JSON is supported for compatibility and machine generation.
2. Commit `runs/` as an empty directory with `.gitignore`.
3. Mandatory figure families are priors, posterior mean model plus samples, and
   inference solutions.  Other reconstructed-field figures are opt-in.
4. Europa support is required in the first implementation.
5. Truncation/integration acceptability should be judged primarily through the
   approximation quality and stability of the normal operator before inversion.
6. Sweep manifests should be per execution backend, not globally merged.
7. Stream defaults should match the current tuner app numeric defaults, with
   boundary-condition defaults inherited from the current hidden paper-demo
   operator defaults.

## Remaining Questions

1. Which optional dense diagnostics should be enabled by default, and which
   should be opt-in because of storage cost?
2. Should the runner write a lightweight per-backend summary index above the
   individual sweep manifests for easier browsing, or is the per-backend
   manifest alone enough?
