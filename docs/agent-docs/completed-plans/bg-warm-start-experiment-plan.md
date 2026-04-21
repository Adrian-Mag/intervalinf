## Plan: BG Warm-Start Bundle Experiment

Test whether Backus–Gilbert (BG) dual certificates can warm-start the ProximalBundleMethod
to reduce iteration count and wall time for DLI (Data-space Linear Inference) support-function
evaluation. Run a rectangular scaling sweep over (N_d, N_p) with multiple seeds, collect timing
and convergence data, and produce a decision report on whether the warm-start is worthwhile.

**Phases: 5**

1. **Phase 1: Experiment Scaffold & Artifact Layout**
    - **Objective:** Create the experiment folder at workspace root with all subfolders, a driver
      script skeleton, artifact manifest schema, and a smoke test that imports all needed packages.
    - **Files/Functions to Create:**
      - `bg_bundle_warm_start_experiment/` with subfolders: `configs/`, `raw_results/`, `processed/`,
        `figures/diagnostics/`, `figures/timing/`, `logs/`
      - `bg_bundle_warm_start_experiment/run_experiment.py` — main driver (skeleton)
      - `bg_bundle_warm_start_experiment/test_smoke.py` — importability smoke test
      - `bg_bundle_warm_start_experiment/README.md` — experiment description
    - **Tests to Write:**
      - `test_smoke_imports` — verifies pygeoinf, intervalinf, numpy, scipy importable
      - `test_folder_structure` — verifies expected subdirectories exist
    - **Steps:**
      1. Create all directories and placeholder files
      2. Write smoke test
      3. Run smoke test locally to confirm green
      4. Update .europa.yml pull_include for new folder (already done)

2. **Phase 2: Common Problem Generator**
    - **Objective:** Build a reusable problem generator mirroring `dli.ipynb` that takes
      `(N_d, N_p, width, seed)` and returns all objects needed for DLI solve.
    - **Files/Functions to Create:**
      - `bg_bundle_warm_start_experiment/problem_generator.py` containing `generate_problem(N_d, N_p, width, seed)` that returns a dataclass/namedtuple with M, G, T, m_bar, m_0, d_tilde, model_prior_support, data_error_support, and metadata.
    - **Tests to Write:**
      - `test_generate_problem_shapes` — verifies operator dimensions match N_d, N_p
      - `test_generate_problem_reproducibility` — same seed → same d_tilde
      - `test_generate_problem_small` — N_d=5, N_p=2 runs without error
    - **Steps:**
      1. Write tests in `bg_bundle_warm_start_experiment/test_problem_generator.py`
      2. Run tests (expect red)
      3. Implement `generate_problem()` following the pattern from `dli.ipynb`: LebesgueSpace, NormalModesProvider for G, BumpFunctionProvider for T, true model m̄(x) = exp(-((x-0.5)/0.5)²)sin(5πx) + x, prior center m_0(x) = x, Gaussian noise at σ = 0.1 × signal_rms, Ball supports with 1.05× tolerance.
      4. Run tests (expect green)

3. **Phase 3: Cold vs BG-Warm Runner**
    - **Objective:** Implement the runner that solves support values with cold start and BG warm
      start, recording iteration counts, wall time, and final objectives for each direction.
    - **Files/Functions to Create:**
      - `bg_bundle_warm_start_experiment/runner.py` containing:
        - `compute_bg_warm_start(problem) -> X_BG_adj` — builds the BG linear map X*_BG = T G* (G G* + αI)^{-1}
        - `run_cold(problem, directions, solver_params) -> RunResult` — solves with default x0=0
        - `run_bg_warm(problem, directions, solver_params, X_BG_adj) -> RunResult` — solves with x0 = X*_BG @ q for each direction
        - `RunResult` dataclass with per-direction data: iterations, wall_time, objective, converged
    - **Tests to Write:**
      - `test_bg_warm_start_formula` — X*_BG applied to unit vectors matches expected
      - `test_cold_run_returns_result` — cold run on small problem returns valid RunResult
      - `test_warm_run_returns_result` — warm run on small problem returns valid RunResult
      - `test_warm_start_fewer_iterations` — on trivial problem, warm start uses ≤ cold iterations
    - **Steps:**
      1. Write tests in `bg_bundle_warm_start_experiment/test_runner.py`
      2. Run tests (expect red)
      3. Implement `compute_bg_warm_start` using the formula from dli_vs_bg_polyhedral_comparison.ipynb: α = (data_radius/model_radius)², GG* + αI solved via Cholesky, X*_BG = T @ G.adjoint @ inv
      4. Implement `run_cold` and `run_bg_warm` wrapping solve_support_values with timing instrumentation
      5. Run tests (expect green)

4. **Phase 4: Scaling Sweep on Europa**
    - **Objective:** Run a rectangular grid sweep (N_d × N_p × seeds) on europa using its 32 cores.
      Save per-run results as NPZ files and a combined CSV summary.
    - **Files/Functions to Create:**
      - `bg_bundle_warm_start_experiment/run_experiment.py` — complete driver script that:
        - Accepts CLI args for grid specification or uses defaults
        - Default grid: N_d ∈ {10, 25, 50, 100, 200}, N_p ∈ {2, 5, 10, 20, 50}, seeds = [42, 123, 314]
        - Uses multiprocessing.Pool with europa's 32 cores
        - Saves per-run NPZ to `raw_results/<Nd>_<Np>_<seed>_{cold,warm}.npz`
        - Saves combined CSV to `processed/sweep_summary.csv`
        - Logs progress to `logs/sweep.log`
      - `bg_bundle_warm_start_experiment/configs/default_grid.json` — default sweep config
    - **Tests to Write:**
      - `test_driver_single_point` — run_experiment on a single (N_d=5, N_p=2, seed=42) point locally
    - **Steps:**
      1. Write the driver script with CLI interface
      2. Test single-point locally
      3. Push to europa and submit the full sweep
      4. Monitor via europa logs
      5. Pull results

5. **Phase 5: Diagnostic Analysis & Decision Report**
    - **Objective:** Generate figures and a written report answering: does BG warm-start help, and
      at what problem scale?
    - **Files/Functions to Create:**
      - `bg_bundle_warm_start_experiment/analysis.py` containing:
        - `load_results(results_dir)` — loads all NPZ + CSV into a DataFrame
        - `plot_iteration_heatmap(df)` — cold vs warm iteration ratio as (N_d, N_p) heatmap
        - `plot_timing_heatmap(df)` — cold vs warm wall-time ratio as (N_d, N_p) heatmap
        - `plot_convergence_comparison(df, N_d, N_p)` — per-direction convergence curves for selected cases
        - `plot_speedup_curve(df)` — speedup vs problem size (N_d × N_p)
        - `generate_report(df)` — writes `processed/decision_report.md`
      - Figures saved to `figures/diagnostics/` and `figures/timing/`
    - **Tests to Write:**
      - `test_load_results` — loads mock NPZ files correctly
    - **Steps:**
      1. Write analysis script
      2. Run locally on pulled results
      3. Generate all figures and decision report

**Open Questions**
1. Should we also test warm-starting with a scaled λ_BG (e.g., 0.5 × X*_BG q) to see if overshoot hurts? → Defer to follow-up experiment if exact BG warm-start shows promise.
2. Should the N_theta directions be shared between cold and warm runs for fair comparison? → Yes, use identical directions from problem generator.
