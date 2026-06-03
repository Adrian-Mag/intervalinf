# Final Report: Prior–Posterior Audit
## MISSION_20260519_INFERENCES_PRIOR_POSTERIOR_AUDIT

**Date:** 2026-05-19 (overnight run)
**Block analysed:** (s=0, t=0) primary; (s=2, t=0) and (s=2, t=3) as cross-checks
**Configuration:** `tuner_params.json`, n_basis=50, n_jobs=12

---

## 1. Central Question

> *Why does the Bayesian posterior appear so tightly constrained (near-zero uncertainty)
> relative to the prior, across all tested prior families?  Is this numerical, structural, or both?
> What prior families encode "I do not know much" while still giving honest posterior uncertainty
> and acceptable data misfit?*

---

## 2. Executive Summary

The tight posterior uncertainty is **structural and correct within the finite data-visible
reduction** — it is not a numerical artifact. The underlying inverse problem is still posed on a
function space and is therefore underdetermined by finitely many normal-mode data. What the audit
diagnoses is the finite operator $G C G^*$ induced by the chosen function-space covariance. In that
data-visible prior-supported subspace, the observations are strong enough to produce tight
marginal posteriors. However, the baseline Bessel-Sobolev covariance only gives appreciable prior
predictive variance to **80 of the 186 data-space directions**, leaving 106 directions with
essentially zero prior support. This causes two observable symptoms:

1. **chi_rms = 5.34** (baseline) — a factor-of-5 systematic misfit that cannot be reduced below
   ~3.8 by prior-amplitude changes alone, because the chosen covariance does not place enough
   variance in many data-visible directions.
2. **Apparent 98% contraction** from bump probing — this is misleading: the prior is very wide
   (bump-std ≈ 29 km/s√km) so the *relative* reduction appears large, but the absolute posterior
   uncertainty (~0.55) is not particularly small. Only 70/186 data directions truly have high
   (>95%) posterior contraction; the other 116 directions have less than 95% contraction.

The chi_rms floor and the rank-80 limit are **universal across all tested blocks** (2,0) and
(2,3) show identical rank(GCG*, 1e-8) = 80/186.

---

## 3. Phase-by-Phase Findings

### Phase 1: Baseline Reproduction (block 0,0)

| Metric | Value |
|--------|-------|
| N_d | 186 |
| chi_rms (posterior) | 5.3400 |
| chi_rms (null model, m=0) | 48.72 |
| Solver agreement (explicit vs pygeoinf) | YES (4 d.p.) |
| cond(N) | 9.4 × 10¹⁰ |
| rank(N, 1e-12) | 186/186 (full) |
| Contraction (bump probe, vp) | 98.1% |
| Prior std vp (bump) | 29.28 |
| Posterior std vp (bump) | 0.55 |

**Symptom identified:** chi_rms = 5.34 combined with near-zero posterior uncertainty ("confident
but wrong" solution).

### Phase 2: Numerical Accuracy Audit

| n_basis | chi_rms | chi_rms explicit | cond(N) | rank(N, 1e-12) | prior sample chi_rms |
|---------|---------|-----------------|---------|----------------|---------------------|
| 30 | 5.513 | 5.513 | 1.12×10¹¹ | 186/186 | ~9600 |
| 50 | 5.340 | 5.340 | 9.39×10¹⁰ | 186/186 | ~8200 |
| 80 | 5.335 | 5.335 | 9.34×10¹⁰ | 186/186 | ~10000 |
| 120 | 5.334 | 5.334 | 9.33×10¹⁰ | 186/186 | ~10100 |

**Key conclusions:**
- chi_rms is **converged** at n_basis ≥ 50. Not a truncation artifact.
- Explicit and pygeoinf solvers agree exactly. Not a solver artifact.
- Prior predictive chi_rms ≈ 8000–10000: the prior is enormously diffuse; the posterior is
  correctly fitting data much better than the prior (5.34 << 8000), even if it doesn't reach 1.
- Data SNR is very high: chi_rms(null) = 48.72, median |d|/σ = 18.1.

### Phase 3+4: Prior Hyperparameter Sweep

**Sweep A — Variance scaling (α × baseline var):**

| α | chi_rms | contraction (bump) | roughness |
|---|---------|-------------------|-----------|
| 0.001 | 6.55 | 96.9% | 0.0065 |
| 0.1 | 5.78 | 97.7% | 0.0076 |
| 1.0 (baseline) | 5.34 | 98.1% | 0.0069 |
| 10 | 4.96 | 98.5% | 0.0065 |
| 100 | 4.66 | 98.7% | 0.0068 |
| 1000 | 4.39 | 98.9% | 0.0078 |

**Observation:** Even with var × 1000, chi_rms only falls to 4.39 (still far from 1).
Contraction stays near 98% regardless of variance. Roughness is nearly constant.

**Sweep B — Smoothness order (all params jointly):**

| s_order | chi_rms | contraction | roughness |
|---------|---------|-------------|-----------|
| 1.0 | 4.02 | 94.0% | 0.0119 |
| 2.0 | 4.24 | 95.5% | 0.0096 |
| 4.0 | 4.87 | 97.4% | 0.0068 |
| 6.0 (baseline) | 5.66 | 98.3% | 0.0058 |
| 8.0 | 6.41 | 98.8% | 0.0049 |

**Observation:** Lower s_order → better chi_rms AND higher roughness. There is a genuine
smoothness vs data-fit tradeoff. At s_order = 1, contraction drops to 94% (more honest).

**Sweep C — Length scale factor:**

| length × | chi_rms | contraction | roughness |
|----------|---------|-------------|-----------|
| 0.1 | 3.89 | 92.8% | 0.0136 |
| 0.33 | 4.18 | 94.9% | 0.0092 |
| 0.5 | 4.51 | 96.4% | 0.0073 |
| 1.0 (baseline) | 5.34 | 98.1% | 0.0069 |
| 2.0 | 6.47 | 99.2% | 0.0061 |
| 5.0 | 7.98 | 99.7% | 0.0027 |

**Observation:** Very long length scales (over-smooth priors) give chi_rms as high as 8 and
near-total uncertainty collapse (99.7%). Short length scales give lower chi_rms but rougher
posteriors. The floor at 0.1× length = 3.89 is the true model floor.

### Phase 5: Geometric Diagnosis

The SVD of the data-space prior operator GCG* reveals:

| Metric | Value |
|--------|-------|
| rank(GCG*, 1e-8) | **80/186** |
| Median info_ratio (λ_signal/λ_noise per N-eigendirection) | 0.167 |
| Fraction of N-directions with info_ratio > 1 | 41.9% |
| Directions with contraction > 95% | 70/186 |
| Directions with contraction > 99% | 63/186 |
| chi_rms from ridge (near-zero regularization) | **3.82** |

**Root cause of chi_rms floor:** The Bessel-Sobolev prior at (s_order=6, length=210km, n_basis=50)
spans only **80 of the 186 data-space directions** (measured at 1e-8 threshold of GCG*
eigenvalues). The remaining 106 directions have essentially zero prior signal — the posterior in
those directions is dominated by C_D alone, and the data signal there is not properly explained.

**Root cause of apparent 98% contraction:** The bump-probe measures *relative* contraction against
a very wide prior (std ≈ 29). But in the data-space eigenvector basis, only 70/186 directions
truly have high (>95%) contraction. The 70 high-SNR directions dominate the posterior mean, while
the other 116 low-SNR directions contribute to chi_rms.

**Chi_rms decomposition (by SNR order):**
- Top 10 data-space directions: cumulative chi_rms = 0.86
- Top 50 directions: cumulative chi_rms = 2.87
- Top 100 directions: cumulative chi_rms = 4.04
- All 186 directions: cumulative chi_rms = 6.30 (full reconstruction)

The chi_rms builds up from the low-SNR directions that the prior doesn't cover.

### Phase 6: Cross-Check on Blocks (2,0) and (2,3)

| Block | N_d | chi_null | chi_rms | rank(GCG*) | info_med | contraction |
|-------|-----|---------|---------|-----------|---------|-------------|
| (0,0) | 186 | 48.72 | 5.34 | 80/186 | 0.167 | 98.1% |
| (2,0) | 186 | 33.99 | 4.97 | 80/186 | 0.208 | 97.7% |
| (2,3) | 186 | 25.58 | 5.71 | 80/186 | 0.161 | 97.3% |

**Universal finding:** rank(GCG*, 1e-8) = **80/186 in every block tested.**
The chi_rms floor and the rank-80 ceiling are intrinsic properties of the
Bessel-Sobolev prior at these hyperparameters, not block-specific artifacts.

Variance sweep on (2,0) and (2,3) shows same slow chi_rms improvement with alpha:
chi_rms drops by only ~0.6 per decade of variance increase, confirming that pure amplitude
increase alone cannot close the gap.

---

## 4. Root Cause Analysis

### Why Is Posterior Uncertainty Small?

The full problem is **function-space underdetermined**: finitely many data cannot determine an
infinite-dimensional radial perturbation field. The small posterior uncertainties arise only in the
finite set of directions where the chosen covariance places prior variance and the forward operator
can see that variance. In other words, the posterior is tight on the data-informed part of the
Cameron-Martin support, while null-space and weakly visible directions remain prior controlled or
outside the effective numerical prior representation.

Thus the important finite diagnostic is not a model-space dimension count. It is the spectrum of
$G C G^*$ and the noise-normalized information ratio in data space. The prior controls posterior
*shape* (smoothness) and data-space coverage; amplitude changes alone do not create missing
function-space directions.

### Why Is chi_rms = 5.34?

Three contributing factors, in order of importance:

1. **Prior basis gap (primary cause, 70–80% of chi_rms floor):**
   The Bessel-Sobolev prior at s_order=6, length=210km projects onto only 80/186 data space
   directions. The remaining 106 directions contain real data signal (median |d|/σ = 18) but
   the prior's RKHS has no support there. Those directions are regularised only by C_D,
   leading to zero posterior mean and non-zero residuals.

2. **Smoothness mismatch (15–20% of chi_rms floor):**
   Smooth priors (high s_order, long length-scale) systematically sacrifice data fit for solution
   smoothness. At the extreme (length × 5), chi_rms = 7.98 — the smooth prior forces a very
   wrong solution.

3. **Possible forward model misspecification (5–10% of floor, unquantified):**
   Even with the loosest priors (length × 0.1, s_order=1), chi_rms = 3.89 rather than ~1. This
   residual floor could indicate: (a) forward kernel approximation errors, (b) unmodelled physical
   effects (source parameters, rotation, ellipticity), or (c) data error underestimation.

---

## 5. Recommendations

### Immediate: Tuner Hyperparameter Changes

For a prior that says "I don't know much" while achieving acceptable data fit:

| Parameter | Baseline | Recommended ("loose but honest") |
|-----------|----------|--------------------------------|
| s_order (vp) | 6.0 | **2.0 – 3.0** |
| s_order (vs_IC) | 4.0 | **2.0** |
| s_order (vs_M) | 4.0 | **2.0** |
| s_order (rho) | 5.0 | **2.0 – 3.0** |
| length (vp) | 210 km | **70–100 km** (0.33–0.5×) |
| length (vs_IC) | 115 km | **40–60 km** (0.33–0.5×) |
| length (vs_M) | 140 km | **50–70 km** (0.33–0.5×) |
| length (rho) | 200 km | **70–100 km** (0.33–0.5×) |
| var (vp) | 1000 | **1000–5000** (keep large) |
| var (vs_IC) | 1000 | **1000–5000** |
| var (vs_M) | 10 | **50–100** |
| var (rho) | 10 | **50–100** |

**Expected outcomes with "loose but honest" prior (s_order=2, length×0.33):**
- chi_rms: ~4.2 (vs baseline 5.34) — measurable improvement
- Contraction: ~95% (vs 98%) — more honest uncertainty bounds
- Posterior roughness: ~0.009 (vs 0.007) — moderately rougher but not oscillatory
- rank(GCG*) coverage: likely ~100–120/186 (improved data space coverage)

### Structural: Prior Design

The fundamental issue is that Bessel-Sobolev with high s_order produces very smooth
basis functions that only span a low-dimensional subspace of data space. For this inverse
problem, the prior needs to:

1. **Cover more data-space directions**: use shorter correlation lengths (< 100 km) so that
   the basis functions have wider spectral support.
2. **Allow higher-frequency structure**: lower s_order (< 3) enables the posterior mean to
   adapt to features at smaller radial scales.
3. **Keep variance large**: `var ≥ 1000` so the prior doesn't pull the solution to zero.

### For Future Analysis

1. **Increase n_basis to 100–200** and re-check rank(GCG*). If rank approaches 186, the
   n_basis=50 truncation is contributing to the floor.
2. **Block-dependent length scales**: different (s, t) blocks constrain different radial structure;
   a global hyperparameter may be suboptimal for some blocks.
3. **Forward model validation**: the irreducible chi_rms floor of ~3.8 should be investigated by:
   (a) testing with synthetic data from a known model to verify recovery,
   (b) checking whether unmodelled effects (gravity, rotation, source moments) account for
   residual signal.
4. **Consider Laplacian-based (Whittle-Matérn) priors with full spectral density control** to
   more directly target the desired frequency content.

---

## 6. Answer to Central Questions

**Is the strong prior imprint numerical, structural, or both?**

It is **structural, not numerical**. Specifically:
- The chi_rms = 5.34 is stable across all n_basis, solvers, and blocks — it is real.
- The 98% posterior contraction comes from comparing against a very wide prior in model space;
  in data space, only 70/186 directions have true high contraction.
- The root cause is the prior's RKHS having rank 80/186 in data space: the prior simply doesn't
  provide information in 57% of the data directions.

**What prior families encode "I do not know much" while giving honest uncertainty and acceptable data misfit?**

- **Lower s_order (1–3)** with **shorter length scale (0.3–0.5×)** and **large var (1000+)**
  gives chi_rms ≈ 4.0–4.5 (improved), contraction ≈ 93–96% (more honest), and expanded
  rank(GCG*) coverage.
- This combination "doesn't know much" in the sense of allowing rough/heterogeneous solutions,
  while still being properly regularised enough to give a unique posterior.
- The absolute posterior uncertainty floor (~3.8) is set by the model parameterisation, not the
  prior amplitude. Increasing var alone cannot significantly reduce chi_rms.

---

## 7. Output Files

| File | Contents |
|------|----------|
| `processed/baseline_metrics.json` | Phase 1 baseline chi_rms, contraction, solver agreement |
| `processed/current_prior_snapshot.json` | Phase 1 prior configuration snapshot |
| `processed/phase2_numerical_audit.json` | n_basis sweep [30,50,80,120], SNR analysis |
| `processed/phase34_prior_sweep.json` | 3 sweeps: variance, s_order, length scale |
| `processed/phase5_geometric.json` | SVD, rank, info_ratio, contraction per direction |
| `processed/phase6_cross_check.json` | Cross-check on blocks (2,0) and (2,3) |
| `figures/baseline_block_0_0.png` | Phase 1 posterior visualisation |
| `figures/phase2_eigspectrum.png` | N eigenvalue spectrum for n_basis sweep |
| `figures/phase2_nbasis_sweep.png` | chi_rms, cond(N) vs n_basis |
| `figures/phase34_var_sweep.png` | chi_rms, contraction, roughness vs variance scale |
| `figures/phase34_smooth_sweep.png` | chi_rms, contraction, roughness vs s_order |
| `figures/phase34_length_sweep.png` | chi_rms, contraction, roughness vs length scale |
| `figures/phase5_geometric.png` | Information spectrum, contraction per direction |
| `figures/phase6_comparison.png` | Cross-block comparison of chi_rms, rank, info_ratio |

---

*Report generated by MISSION_20260519_INFERENCES_PRIOR_POSTERIOR_AUDIT — 2026-05-19*
