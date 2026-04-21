## Plan: BG Warm-Start Diagnostic Investigation

Enhance the warm-start experiment to capture per-direction diagnostic metrics (BG bounds vs DLI bounds, BG certificates vs DLI certificates), fix incremental saving, re-run with diagnostics, analyze WHY BG warm-start doesn't help, and computationally test alternative warm-start strategies.

**Grid:** N_d ∈ {10, 25, 50, 100}, N_p ∈ {2, 5, 10, 20}, seeds = [42, 123, 314] → 48 combos

**Phases: 6**

1. **Phase 1: Enhance Runner with Per-Direction Diagnostics**
    - **Objective:** Modify runner.py to capture BG bound values, DLI bound values, certificate norms, and certificate distances for every direction — enabling comparison of BG vs optimal DLI certificates.
    - **Files/Functions to Modify:**
      - `bg_bundle_warm_start_experiment/runner.py`: Extend `DirectionResult` with `bg_bound`, `dli_bound`, `bg_cert_norm`, `dli_cert_norm`, `cert_distance`, `relative_cert_distance`. Modify `_run_mode` to accept a `bg_operator` and evaluate `cost(lambda_bg)` after each direction solve. Both `run_cold` and `run_bg_warm` compute diagnostics via the shared cost object.
      - `bg_bundle_warm_start_experiment/test_runner.py`: Add diagnostic-metric tests.
    - **Tests to Write:**
      - `test_direction_result_has_diagnostics` — DirectionResult includes bg_bound, cert_distance fields
      - `test_bg_bound_geq_dli_bound` — For every direction, bg_bound ≥ dli_bound (BG is suboptimal)
      - `test_certificate_distance_nonnegative` — cert_distance ≥ 0
      - `test_relative_cert_distance_finite` — relative distance is finite when DLI cert is nonzero
    - **Steps:**
      1. Write failing tests
      2. Extend DirectionResult dataclass with new fields
      3. Modify `_run_mode`: after each solve, compute lambda_bg = bg_operator.adjoint(q), bg_bound = cost(lambda_bg), cert_distance = D.distance(x_best, lambda_bg), norms via D.norm
      4. Both run_cold and run_bg_warm now always build bg_operator and pass it in
      5. Run tests to green

2. **Phase 2: Fix Incremental Saving in Driver**
    - **Objective:** Modify run_experiment.py to write CSV incrementally (append after each combo) and save new diagnostic arrays in NPZ files.
    - **Files/Functions to Modify:**
      - `bg_bundle_warm_start_experiment/run_experiment.py`: Write CSV header at start, append each row as futures complete. Extend NPZ saving with diagnostic arrays. Add aggregate diagnostic columns to CSV.
    - **Tests to Write:**
      - `test_incremental_csv_survives_partial` — Verify CSV has rows after partial sweep
    - **Steps:**
      1. Write failing test
      2. Refactor run_sweep: open CSV at start with header, append inside as_completed loop
      3. Extend NPZ saving with bg_bounds, dli_bounds, cert_distances, bg_cert_norms, dli_cert_norms, relative_cert_distances arrays
      4. Add CSV columns: mean_bg_bound_gap, mean_relative_cert_distance, max_relative_cert_distance
      5. Update default grid to the reduced grid
      6. Run tests to green

3. **Phase 3: Enhanced Analysis with Diagnostic Plots**
    - **Objective:** Add analysis functions that plot certificate distances and bound gaps to test the hypothesis that larger N_d → worse BG certificates → warm-start less effective.
    - **Files/Functions to Modify:**
      - `bg_bundle_warm_start_experiment/analysis.py`: Add:
        - `plot_cert_distance_vs_nd()` — Certificate distance vs N_d grouped by N_p
        - `plot_bound_tightness_vs_nd()` — BG bound / DLI bound ratio vs N_d
        - `plot_cert_distance_vs_speedup()` — Correlation: cert distance vs observed speedup
        - `plot_per_direction_comparison()` — Polar plots for selected (N_d,N_p) pairs: (10,2), (25,5), (50,10)
      - Update `generate_decision_report()` with "Diagnostic Analysis" section
    - **Tests to Write:**
      - `test_diagnostic_analysis_runs` — Analysis functions execute on mock data without error
    - **Steps:**
      1. Write failing test with synthetic diagnostic data
      2. Implement the four plot functions
      3. Update decision report generator
      4. Run tests to green

4. **Phase 4: Run Diagnostic Sweep on Europa**
    - **Objective:** Submit enhanced experiment on europa, pull results, generate diagnostic analysis.
    - **Steps:**
      1. europa status (VPN check)
      2. Push updated code
      3. Claim europa slot
      4. Submit sweep (48 combos, estimate ~20 min from previous timing data)
      5. Wait appropriately based on time estimate
      6. Pull results
      7. Run analysis locally
      8. Release europa slot

5. **Phase 5: Deep Analysis & Conclusions**
    - **Objective:** Synthesize diagnostic results into conclusions about WHY warm-start fails.
    - **Files/Functions to Create:**
      - `bg_bundle_warm_start_experiment/processed/diagnostic_report.md`
    - **Steps:**
      1. Analyze certificate distance trends vs N_d
      2. Analyze bound tightness trends
      3. Compute correlation between certificate distance and speedup
      4. Write diagnostic report with mechanistic explanation
      5. Identify candidate alternative strategies for Phase 6

6. **Phase 6: Test Alternative Warm-Start Strategies**
    - **Objective:** Computationally test 2-3 alternative warm-start strategies on a modest grid to see if any outperform both cold-start and BG warm-start.
    - **Files/Functions to Create:**
      - `bg_bundle_warm_start_experiment/alt_strategies.py`: Functions for alternative initial points (e.g., scaled BG certificate, BG with tuned α, convex combo of BG and zero)
      - `bg_bundle_warm_start_experiment/test_alt_strategies.py`
    - **Tests to Write:**
      - `test_alt_strategy_returns_valid_lambda` — Each strategy produces a valid D-element
      - `test_alt_strategy_run_completes` — Each strategy completes a small problem
    - **Steps:**
      1. Based on Phase 5 findings, implement 2-3 alternative initial-point strategies
      2. Write a small comparative sweep (N_d ∈ {25, 50}, N_p ∈ {5, 10}, 1 seed) testing cold, bg_warm, and each alternative
      3. Run locally or on europa
      4. Analyze and report which (if any) alternative shows promise
      5. Update diagnostic report with conclusions

**Open Questions:** None — all resolved by user.
