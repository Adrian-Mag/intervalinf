## Plan: Full-Spectrum Splitting-Function PLI Example

Build `intervalinf/demos/old_demos/paper_demos/example.ipynb` solving the entire normal-mode splitting PLI problem from `idea.txt`: a direct sum of independent radial blocks over all visible $(s,t)$ harmonics in the data, with 3D bulk volumetric-average properties and 2D CMB **angular-bump** surface properties, both built by spherical-harmonic projection of physical target kernels. The $(s,t)$ block-diagonal structure of $G$, $C_0$, $C_D$ is exploited so the posterior factorizes and each block is a 1D radial inverse problem; per-block posteriors are then combined into a **full** property posterior $\mathcal{N}(\pt, \Cp)$ via $\Tau_\mathcal{I} C^{\mathrm{post}}_\mathcal{M} \Tau_\mathcal{I}^*$.

**Design principles**

- Open design choices (radial inner-product weight, prior power spectrum, property suite) are localised behind small factories so they can be revised without rewriting the rest.
- 3D model-space posterior visualisation is treated as **optional** and expensive; full property-space posterior is treated as **standard** and cheap (small $N_p$).
- Performance-critical structure (shared kernel quadrature mesh per parameter, shared Bessel-Sobolev factorisation per parameter scaled by $\tau_{p,s}^2$) is set up from Phase 2 so Phase 8 has fast paths to enable.

**Answered design questions (from chat)**

1. Working harmonic range: start with even $s \le 4$, all $t$; scale up later.
2. Radial inner-product weight: $w \equiv 1$ (with a `weight=` hook for future $r^2$).
3. Prior power spectrum $\tau_{p,s}^2$: flat in $s$ first, decay knob exposed.
4. Inter-interface covariance: code for general $N_\Sigma$, instantiate with $N_\Sigma = 1$ (CMB only).
5. Property suite: 3 bulk bumps × 2 horizontal locations + 2 antipodal CMB angular bumps.
6. Surface targets: **angular bumps** (smooth Gaussian-like caps), not hard spherical-cap indicators.
7. Property posterior: build **full** $\Cp = \Tau_\mathcal{I} C^{\mathrm{post}}_\mathcal{M} \Tau_\mathcal{I}^*$ as a dense $N_p \times N_p$ matrix; 3D model-space posterior visualisation is opt-in.

**Phases**

1. **Phase 1: Spectral inventory and block-indexing infrastructure**
    - **Objective:** Enumerate visible $(s,t)$ blocks $\mathcal{I}$, pick a working subset (even $s \le s_{\max} = 4$, all $t$), build a thin block-indexing layer that maps each $(s,t)$ to its data sub-vector, kernel sub-catalog, model space, prior, noise covariance, and forward operator.
    - **Files/Functions to Modify/Create:**
        - New `paper_demos/full_spectrum_utils.py` with `BlockIndex`, `enumerate_blocks(reg_full, s_max) -> list[(s,t)]`, `block_data_split(reg_full, blocks) -> dict[(s,t), reg_st]`.
        - New `paper_demos/example.ipynb` (title cell + spectral-inventory cell + diagnostic plot of data-count-per-$(s,t)$).
        - Reuse `normal_mode_kernel_utils.NormalModeDataRegistry.filter_by_st` from `paper_demos/`.
    - **Tests to Write** (in `paper_demos/tests/test_full_spectrum_utils.py`, run from `paper_demos/`):
        - `test_enumerate_blocks_respects_smax_and_parity`: only even $s$ with $0 \le s \le s_{\max}$ appear.
        - `test_enumerate_blocks_skips_empty_blocks`: $(s,t)$ pairs with zero observations are excluded.
        - `test_block_data_split_partitions_registry`: union of per-block data equals full registry; intersections are empty.
    - **Steps:**
        1. Write the three tests against expected `BlockIndex` API; run, see them fail.
        2. Implement `enumerate_blocks` and `block_data_split` in `full_spectrum_utils.py`.
        3. Run tests, confirm pass.
        4. Add notebook cells: load full registry, call `enumerate_blocks(reg_full, s_max=4)`, produce a bar plot of $|\dt_{st}|$ per $(s,t)$.

2. **Phase 2: Per-block radial model space, inner product, and forward operator**
    - **Objective:** For each $(s,t) \in \mathcal{I}$, build $\modelspace_{st} = \modelspace^{v_p}_{st} \oplus \modelspace^{v_s,\mathrm{IC}}_{st} \oplus \modelspace^{v_s,M}_{st} \oplus \modelspace^{\rho}_{st} \oplus \mathbb{R}^{N_\Sigma}$ and block forward operator $G_{st}: \modelspace_{st} \to \dataspace_{st}$. All radial spaces share the same `function_domain` and SOLA quadrature mesh — this is the foundation of the kernel-eval fast path used in Phase 8.
    - **Files/Functions to Modify/Create:**
        - `paper_demos/full_spectrum_utils.py`: `make_radial_space(n, domain, weight=1.0, basis='ND')`, `build_block_forward(s, t, reg_st, catalog, radial_specs) -> (G_st, C_D_st, D_st)`.
        - Notebook cell: loop over $\mathcal{I}$, store `forward_dict[(s,t)] = (G_st, C_D_st, M_st, D_st)`.
    - **Tests to Write:**
        - `test_make_radial_space_unit_weight_matches_lebesgue`: `make_radial_space(..., weight=1.0)` returns a `Lebesgue` whose inner product agrees with `Lebesgue` directly (1e-12).
        - `test_block_forward_against_raw_kernel_integration`: for one $(s,t)$ and one $\alpha$, $[G_{st}\,m]_\alpha$ equals the manual $\int K_{\alpha s}^{v_p}\,\delta v_{p,st}\,dr + \dots$ to 1e-10.
        - `test_block_forward_shapes`: codomain dimension equals number of $(\alpha,s,t)$ data points in the block.
    - **Steps:**
        1. Write tests; run; see them fail.
        2. Implement `make_radial_space` with `weight` parameter (only `weight=1.0` supported in this phase; raise `NotImplementedError` otherwise with a TODO pointing to `r^2`-elliptic covariances).
        3. Implement `build_block_forward` reusing the existing `NormalModeSplittingKernelProvider` / `SOLAOperator` pattern from `example_2.ipynb`.
        4. Run tests; confirm pass.
        5. Notebook cell: build `forward_dict`.

3. **Phase 3: Per-block prior covariance with shared per-parameter factorisation**
    - **Objective:** Build $C_{0,st}$ for each block as a Bessel-Sobolev radial covariance $(k_p^2 I + \alpha_p(-\Delta_r))^{-q_p/2}$ scaled by a degree-dependent factor $\tau_{p,s}^2$ from a user-supplied callable `prior_power_spectrum(p, s)`. The radial Bessel-Sobolev operator depends only on the parameter $p$, not on $(s,t)$ — build it **once per parameter** and share across blocks (one FEM stiffness matrix and one eigendecomposition per parameter, reused $|\mathcal{I}|$ times). Interface blocks use a scalar $\mathbb{R}^{N_\Sigma}$ covariance $\mathbf{C}_\Sigma$.
    - **Files/Functions to Modify/Create:**
        - `paper_demos/full_spectrum_utils.py`: `build_shared_bessel_blocks(radial_specs) -> dict[param, BesselSobolevInverse]`, `prior_power_spectrum_default(p, s) -> float` (flat in $s$ with per-parameter scalar; documented as the user-tunable hook), `build_block_prior(s, t, shared_bessel, C_sigma, tau_fn) -> GaussianMeasure`.
        - Notebook cells: instantiate `shared_bessel`, define `prior_power_spectrum`, build `prior_dict[(s,t)]`.
        - Markdown cell flagging that swapping to $w(r)=r^2$ requires a radial-elliptic covariance not yet available in `intervalinf` (forward-pointer to a future plan).
    - **Tests to Write:**
        - `test_shared_bessel_is_built_once_per_parameter`: identity check that `shared_bessel['vp']` is the same object across blocks.
        - `test_block_prior_covariance_scales_with_tau`: doubling $\tau_{p,s}^2$ doubles the diagonal action of $C_{0,st}^p$ on a test function (1e-10).
        - `test_block_prior_is_block_diagonal`: a draw from `block_prior` has independent components across the direct-sum parts (verified by sample correlation over $N=200$ draws).
    - **Steps:**
        1. Tests; run; fail.
        2. Implement `build_shared_bessel_blocks` using existing `BesselSobolevInverse` ctor.
        3. Implement `build_block_prior` composing scalar scaling + shared `BesselSobolevInverse` + $\mathbf{C}_\Sigma$.
        4. Tests pass.
        5. Notebook cells.

4. **Phase 4: Independent block posteriors in parallel**
    - **Objective:** For each $(s,t)$, build a `LinearForwardProblem` + `LinearBayesianInversion` and store the posterior measure $\mu_{st}^{\dt}$. Run blocks in parallel with `joblib.Parallel` (blocks are pure, share nothing computationally beyond read-only `shared_bessel`). Posterior covariance is left in operator form per block (no `.matrix()` calls) — dense materialisation is deferred to Phase 6 where it costs $O(N_p \cdot \dim\mathcal M_{st})$ rather than $O(\dim\mathcal M_{st}^2)$.
    - **Files/Functions to Modify/Create:**
        - `paper_demos/full_spectrum_utils.py`: `solve_block(s, t, G_st, C_D_st, prior_st, d_st) -> GaussianMeasure` (pickle-safe entry point), `solve_all_blocks(forward_dict, prior_dict, data_dict, n_jobs) -> dict[(s,t), GaussianMeasure]`.
        - Notebook cell: timed call to `solve_all_blocks` with a `tqdm` progress bar.
    - **Tests to Write:**
        - `test_solve_block_smoke_one_block`: a single $(s,t)$ block runs end-to-end and returns a `GaussianMeasure` with finite mean and finite trace.
        - `test_solve_block_posterior_reduces_uncertainty`: posterior covariance trace ≤ prior covariance trace on the chosen $(s,t)$.
        - `test_solve_all_blocks_parallel_matches_serial`: same posterior means/covs (1e-10) whether `n_jobs=1` or `n_jobs=4`.
    - **Steps:**
        1. Tests; run; fail.
        2. Implement `solve_block`.
        3. Implement `solve_all_blocks` with `joblib.Parallel`.
        4. Tests pass.
        5. Notebook cell.

5. **Phase 5: Property operators via spherical-harmonic projection of 3D / angular targets**
    - **Objective:** Build the coefficient-space property operator $\Tau_\mathcal{I} = \Tau_{3\mathrm{D}} \pi_\mathcal{I}$ as a `RowLinearOperator` over blocks, with two target families:
        1. **Bulk volumetric averages.** $T(r,\xi) = h(r)\,b(\xi - \xi_0)$ where $h$ is a radial bump and $b$ is a smooth angular bump (Gaussian on the sphere $\propto \exp(-\arccos(\hat\xi\cdot\hat\xi_0)^2/(2\sigma^2))$). Project to $\{T_{st}(r)\}$ via `pyshtools` analysis of $b$ then multiply by $h(r)$.
        2. **CMB topography angular bumps.** $B(\xi) = b(\xi - \xi_0)$ with the same smooth angular-bump kernel. Project to $\{B_{st}\}$ via `pyshtools` analysis. **Not** spherical-cap indicators.
    - All bumps are *real-valued and rotationally well-defined*: project an axisymmetric template once at the pole and rotate via `pyshtools` Wigner-d / `SHRotateRealCoef` to each $\hat\xi_0$. Use the geophysics-standard $4\pi$-normalised real spherical-harmonic convention; document the normalisation choice in a markdown cell.
    - **Files/Functions to Modify/Create:**
        - New `paper_demos/property_targets.py`: `angular_bump(theta, sigma)` (axisymmetric template), `angular_bump_coeffs(sigma, s_max)` (analytic `pyshtools` analysis at the pole), `rotate_coeffs(coeffs_pole, xi_0)`, `radial_bump(r, r0, sigma_r)`, `bulk_target_projection(h, xi_0, sigma_ang, blocks)`, `cmb_target_projection(xi_0, sigma_ang, blocks)`.
        - `paper_demos/full_spectrum_utils.py`: `build_property_operator(targets, forward_dict) -> RowLinearOperator` that assembles one column per $(s,t)$.
        - Notebook cells: define 3 radial bumps × 2 horizontal locations (6 bulk targets) + 2 antipodal CMB angular bumps (2 surface targets); build $\Tau_\mathcal{I}$.
    - **Tests to Write** (`paper_demos/tests/test_property_targets.py`):
        - `test_angular_bump_coeffs_unit_norm_at_pole`: the axisymmetric template at the pole has $t=0$ only (within numerical tolerance) and integrates to the expected total mass.
        - `test_rotation_preserves_inner_product`: $\|\text{rotate}(c, \xi_0)\| = \|c\|$ to 1e-10.
        - `test_bulk_target_projection_roundtrip`: synthesising $\sum_{st} T_{st}(r) Y_s^t(\xi)$ reproduces $h(r)b(\xi-\xi_0)$ on a coarse sphere mesh to within harmonic truncation error.
        - `test_property_operator_shape`: $\Tau_\mathcal{I}: \modelspace_\mathcal{I} \to \mathbb{R}^{N_p}$ has the right codomain dimension.
        - `test_property_operator_pushforward_one_block`: for a single block, $\Tau_\mathcal{I}\,m$ equals direct numerical integration of $T_{3D}\,(\pi_\mathcal{I} m)$ over the sphere × radius (within 1% — limited by truncation and sphere quadrature).
    - **Steps:**
        1. Tests; fail.
        2. Implement angular bump + rotation helpers using `pyshtools.SHGrid` analysis + `SHCoeffs.rotate`.
        3. Implement bulk + CMB projection functions.
        4. Implement `build_property_operator` returning a `RowLinearOperator` whose blocks are existing `intervalinf` linear forms on the radial components plus scalar-multiplication blocks on the interface components.
        5. Tests pass.
        6. Notebook cells defining the 8-property suite + building $\Tau_\mathcal{I}$.

6. **Phase 6: Full property posterior $\mathcal{N}(\pt, \Cp)$**
    - **Objective:** Compute the **full dense** property posterior:
        - Mean: $\pt = \sum_{(s,t)}\Tau_{st}\,\widetilde m_{st}$ (cheap — one matvec per block).
        - Covariance: $\Cp = \sum_{(s,t)}\Tau_{st}\,C^{\mathrm{post}}_{st}\,\Tau_{st}^*$ as an $N_p \times N_p$ dense matrix. For each block, apply $C^{\mathrm{post}}_{st}$ to each of the $N_p$ rows of $\Tau_{st}$, then assemble the $N_p \times N_p$ Gram matrix. Total cost $O(|\mathcal{I}|\cdot N_p \cdot \text{block-solve-cost})$, dominated by $N_p$ posterior-covariance applies per block — well within budget since $N_p = 8$.
    - **Files/Functions to Modify/Create:**
        - `paper_demos/full_spectrum_utils.py`: `assemble_property_posterior(property_op, model_posterior_dict) -> GaussianMeasure` (Euclidean, dense $\Cp$).
        - Notebook cell building `mu_P = assemble_property_posterior(...)` and printing diagnostics (mean, std, full correlation matrix as a heatmap).
    - **Tests to Write:**
        - `test_assemble_property_posterior_mean_matches_pushforward`: $\pt$ from `assemble_property_posterior` equals $\Tau_\mathcal{I}\,\widetilde m$ assembled across blocks (1e-10).
        - `test_assemble_property_posterior_cov_symmetric_psd`: $\Cp$ is symmetric (1e-10) and PSD (smallest eigenvalue $\ge -1$e-10).
        - `test_assemble_property_posterior_cov_matches_monte_carlo`: dense $\Cp$ agrees with $N=10\,000$-sample RTO estimate to within 3$\sigma$ Monte-Carlo error per entry.
    - **Steps:**
        1. Tests; fail.
        2. Implement `assemble_property_posterior` with per-block $\Tau_{st}\,C^{\mathrm{post}}_{st}\,\Tau_{st}^*$ accumulation.
        3. Tests pass.
        4. Notebook cell + correlation-heatmap figure.

7. **Phase 7: Visualisation and reporting**
    - **Objective:** Produce headline figures:
        1. Per-block posterior radial fields for one representative $(s,t)$ (e.g. $(2,0)$): mean + ±1$\sigma$ envelope per parameter $\{v_p, v_s\text{-IC}, v_s\text{-M}, \rho\}$ + scalar bar for $\delta\Sigma_{\mathrm{CMB}}$.
        2. (**Optional, opt-in via boolean flag**) 3D model-space posterior visualisation: equatorial slices of $\delta v_s$ posterior mean and pointwise std, synthesised via `pyshtools` on the retained $(s,t)$ harmonics. Flagged as expensive; off by default.
        3. CMB topography posterior map (always on, cheap): mean + std on a sphere grid.
        4. Property posterior diagnostics: bar plot of mean ± std per property, plus the $N_p \times N_p$ correlation heatmap from Phase 6.
    - **Files/Functions to Modify/Create:**
        - New `paper_demos/full_spectrum_viz.py`: `plot_block_posterior`, `plot_cmb_map`, `plot_equatorial_slice` (gated), `plot_property_posterior_summary`.
        - Notebook viz cells (one per figure).
    - **Tests to Write:**
        - `test_plot_block_posterior_returns_figure`: smoke test, no exception, returns a `Figure`.
        - `test_plot_cmb_map_returns_figure`: smoke test.
        - `test_plot_equatorial_slice_gated_off_by_default`: helper returns `None` when `enabled=False`.
    - **Steps:**
        1. Smoke tests; fail.
        2. Implement plotting helpers reusing patterns from `example_2.ipynb`.
        3. Tests pass.
        4. Notebook cells render the four figures.

8. **Phase 8: Performance review and targeted fast paths**
    - **Objective:** Profile end-to-end with `cProfile`; verify the structural fast paths from Phases 2–4 are actually hit; if profile justifies it, add one paper-demo-specific shared-eigendecomposition sampler.
    - Known structural fast paths to confirm:
        1. SOLA `_kernel_eval_cache` and `stats['fast_path_hits']` count > 0 across blocks (same `function_domain`, same `n_points`).
        2. `shared_bessel[p]` is reused across all $(s,t)$ for parameter $p$.
        3. `joblib.Parallel` in Phase 4 produces ≥ 4× speedup on 12 cores at $|\mathcal{I}| \ge 8$.
    - **Potential custom fast path** (added only if profile justifies it): `BlockBesselSobolevSampler` that shares the Laplacian eigendecomposition across all blocks of the same parameter — one eigendecomp, $|\mathcal{I}|$ scaled samples.
    - **Files/Functions to Modify/Create:**
        - New `paper_demos/profile_full_spectrum.py`: `cProfile`-driven script that runs the whole pipeline at a small $|\mathcal{I}|$ and dumps a sorted `pstats` table.
        - Notebook timing cell at end (final wall-clock + per-phase breakdown).
        - Possibly `paper_demos/full_spectrum_utils.py::BlockBesselSobolevSampler`.
    - **Tests to Write:**
        - `test_sola_fast_path_hits_across_blocks`: after running Phase 4, `G_st.stats['fast_path_hits']` is > 0 for at least one block (or, more precisely, kernel-eval cache reuse occurs across blocks sharing the same parameter).
        - `test_shared_bessel_object_identity_in_pipeline`: in the assembled `prior_dict`, the underlying `BesselSobolevInverse` for parameter `vp` is the same object across all $(s,t)$.
        - (Conditional) `test_block_bessel_sampler_matches_independent_samplers`: if `BlockBesselSobolevSampler` is added, its draws match the slow per-block sampler statistics within Monte-Carlo error.
    - **Steps:**
        1. Write the two structural tests; run; pass or fail and fix.
        2. Run `cProfile` on the full pipeline; record the top 20 hotspots.
        3. Decide whether `BlockBesselSobolevSampler` is justified (criterion: prior-sampling > 15% of total runtime).
        4. If yes: implement + test + integrate. If no: document the decision in a notebook markdown cell.
        5. Re-time end-to-end; report final wall-clock and per-phase breakdown.

**Open Questions**

1. **Sphere quadrature for `pyshtools` analysis of angular bumps**: Driscoll–Healy grid `N` parameter — tied to $s_{\max}$ via $N \ge 2(s_{\max}+1)$, but headroom $N = 4(s_{\max}+1)$ recommended; confirm at Phase 5.
2. **Real-SH normalisation convention**: `pyshtools` default (`4pi`, geodesy) vs. orthonormal. Recommended `4pi` for downstream physical interpretability; pin in `property_targets.py`.
3. **Joblib backend**: `loky` (default, safe with `pyshtools`) vs. `threading` (faster if no GIL contention — unlikely here); confirm at Phase 4.
4. **Profile granularity**: `cProfile` line-level vs. function-level — start function-level for the dispatch picture, drop to `line_profiler` only if a single hotspot dominates.
5. **3D model-space visualisation cost ceiling**: cap the optional Phase 7 slice at a coarse $\theta\times\phi$ grid (e.g. $90\times180$) so even when enabled it stays under ~30 s.


---
**Status: completed** (moved from active-plans during the 2026-07-09 agent-docs revival;
phase-completion documents exist in completed-plans/ — see files with this plan's prefix).
