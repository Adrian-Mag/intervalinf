# Conversation Summary Through 2026-06-07

This document summarizes the full conversation that led to the recent
`intervalinf` and paper-demo prior work. It combines:

- the scientific discussion,
- the package-architecture discussion,
- the normal-mode run analysis discussion,
- the prior-design discussion,
- the implementation work that was completed,
- the remaining open technical questions.

The goal is to preserve both the reasoning and the resulting code state.

## 1. Early conceptual discussion

The conversation began with questions about `pygeoinf`, Gaussian measures, and
Bayesian inversion for normal operators.

The main conceptual thread was:

- whether covariance-factor constructions are really necessary,
- whether one could instead make the covariance operator itself more robust,
- and whether this is effectively the same as manually symmetrizing a matrix
  after it is computed.

The important distinction that emerged is:

- a mathematically well-defined covariance operator should already be
  self-adjoint and positive,
- while post-hoc matrix symmetrization is a numerical repair step,
- so the two ideas are related but not identical.

This set up a recurring theme of the conversation:

The right long-term fix is to improve the prior and operator design, rather
than relying on numerical cleanup after the fact.

## 2. pygeoinf / intervalinf package-structure discussion

There was then a large design discussion about the future structure of
`pygeoinf` and related packages ahead of a major refactor toward
`pygeoinf 2.0`.

The user described the intended conceptual structure for the package family:

1. A basic inversion layer built from:
   - model space,
   - forward map,
   - data space.

   This supports point-estimator inversion such as least-norm, least-squares,
   and Tikhonov methods. The central object is a cost function, and there is no
   uncertainty quantification at this level.

2. A property-inference layer:
   - if a property space and property map are added,
   - a point solution can be pushed into the property space.

3. A probabilistic/Bayesian layer:
   - if data noise is modeled probabilistically,
   - then the forward relation becomes a likelihood kernel,
   - and once a prior measure is added on the model space, Bayesian inversion
     becomes possible,
   - with optional pushforward to a property-space posterior measure.

4. A set-based inference layer:
   - if noise is modeled by feasible sets rather than probability measures,
   - and the model prior is also a set,
   - then one gets a set-based Bayesian or feasible-posterior construction,
   - which can again be pushed to property space.

The thesis in `/home/adrian/PhD/thesis` was identified as the main source for
this conceptual structure.

The software-architecture concerns raised by the user included:

- too many core functions in a single folder,
- `symmetric` sitting in the same crowded area,
- `intervalinf` and `pygeoinf3D` living as separate packages rather than
  obvious submodules,
- oversized files,
- and the lack of a clean conceptual layout for future team growth.

This discussion established the long-term direction:

The package family should present a much clearer progression from deterministic
inversion to Bayesian inference to set-based inference, with property-space
pushforwards treated as first-class objects.

## 3. Europa run analysis and prior concerns

The conversation then shifted to the results of a Europa run whose data lived
under the `work` folder.

Several diagnostics were discussed:

- graphing convergence across truncation levels `50`, `100`, and `200`,
- identifying which run had the best data fit,
- isolating promising groups of runs,
- and interpreting the effects of truncation, noise multiplier, and prior
  choice.

The practical conclusions were:

- truncation `100` looked sufficient,
- a scientifically plausible noise multiplier should lie between `2` and `4`,
  with a preference for `2`,
- and the biggest lever was not truncation or noise, but the prior.

The radial-weighted prior was not abandoned. Instead, the run was treated as
useful evidence about how truncation and noise scaling behave before revisiting
the prior family more seriously.

## 4. Physical prior-design discussion

A large part of the scientific conversation focused on finding a defensible
prior for the normal-mode problem.

The user’s physical position was:

- start with the lowest Sobolev order `s` first, because that assumes the least
  smoothness,
- keep the boundary conditions as they were,
- and think much more carefully about the physical interpretation of `k` and
  `alpha`.

The motivating physical quantities were described as 3-D perturbation fields
such as:

```math
\delta \ln v_p(r,\omega),
\qquad
\delta \ln m = \frac{\delta m}{m_0}.
```

The reasoning was:

- because the inverse problem is linearized, the perturbations should usually
  be only a few percent,
- literature typically suggests amplitudes below about `10%`,
- so prior samples with amplitudes of `0.8` or `1.0` in `\delta \ln m` look
  physically wrong if the models truly are log-perturbations.

This led to an important scientific uncertainty:

- perhaps the models were actually `\delta m` rather than `\delta \ln m`,
- but even if the data labeling was wrong, the prior still needed to be
  defensible on physical grounds.

That led naturally to the calibration question:

How should one choose prior amplitudes after spherical-harmonic separation and
radial expansion so that reconstructed fields stay within physically reasonable
amplitude ranges?

## 5. Review of an external calibration proposal

The user then supplied a long externally generated proposal for calibrating
spherical-harmonic radial Gaussian priors so that reconstructed fields have a
target amplitude.

That proposal had several strong core ideas:

- treat the covariance shape as fixed and calibrate amplitudes on top,
- estimate amplitudes through Monte Carlo rather than guessing from covariance
  entries,
- and verify full-field reconstructed amplitudes instead of only per-block
  amplitudes.

The current implementation eventually adopted a simpler version of the same
philosophy:

- the weighted radial covariance shape is fixed first,
- a degree spectrum is imposed,
- then Monte Carlo is used to calibrate the scalar amplitudes.

## 6. Prior calibration plan and implementation

The user requested a formal plan for prior calibration, followed by a full
implementation.

The implemented calibration code now lives in the paper-demo utilities and is
also wired into the tuner app.

### 6.1 Mathematical idea

For each component

```math
p \in \{\mathrm{vp}, \mathrm{vs}, \rho\},
```

the reconstructed 3-D field is written as a sum over real spherical-harmonic
blocks:

```math
f_p(r,\omega)
=
\sum_{s,t} m_{p,s,t}(r) Y_{s,t}(\omega).
```

In the weighted radial case:

- each degree `s` uses its own radial reference covariance
  `C^{ref}_{p,s}`,
- built from `RadialLaplacian(..., ell=s)`,
- with shared `k`, `alpha`, and Sobolev order across degrees,
- and a calibrated scalar amplitude `tau_{p,s}` placed on top.

The calibration imposes a degree spectrum

```math
w_s = (1+s)^{-\beta}
```

and writes

```math
\tau_{p,s} = a_p w_s.
```

It then estimates the empirical prior-predictive quantile of

```math
\max_{r,\omega} |f_p(r,\omega)|
```

at unit scale `a_p = 1`, and chooses

```math
a_p = \frac{A_p}{Q_q},
```

where:

- `A_p` is the target amplitude,
- `Q_q` is the Monte Carlo `q`-quantile of the unit-scale maxima.

Update on 2026-06-08: `vs_IC` and `vs_M` should not be calibrated as two
independent physical fields. They are now treated as one split `vs` field whose
calibration maximum is `max(max_IC, max_M)`, while the implementation still
stores separate `vs_IC` and `vs_M` tau lookup entries for compatibility with the
block-prior builder. The split radial reference blocks are also normalized
before estimating the shared `vs` scale, so the weighted `s=0` inner-core
Neumann zero mode does not force the mantle monopole amplitude to be
artificially small.

This yields the final calibrated amplitudes

```math
\tau_{p,s} = a_p (1+s)^{-\beta}.
```

### 6.2 Code-level implementation

The calibration code introduced:

- `PriorCalibrationConfig`,
- `PriorCalibrationResult`,
- per-component calibration results,
- radial synthesis matrices built from the degree-aware covariance eigenpairs,
- full-field Monte Carlo sampling on a radial and angular grid,
- CSV / NPZ / plot outputs summarizing the resulting `tau` values.

It also writes outputs such as:

- `tau_by_degree.csv`,
- `tau_st.csv`,
- `component_amplitude_summary.csv`,
- `calibration_samples.npz`,
- diagnostic figures and sample maps.

### 6.3 Tuner integration

The user requested that prior calibration run automatically during prior
construction so that `tau` values no longer needed to be chosen by hand.

That was implemented in the prior-posterior tuner:

- building the prior now runs the calibration,
- `k`, `alpha`, and Sobolev order are synchronized across ST blocks,
- the user can inspect calibrated prior maps,
- and the calibrated `tau` values are used automatically.

### 6.4 Performance discussion

Sampling reconstructed fields initially took too long.

The user pushed for execution times on the order of seconds rather than a
minute or more.

This led to discussion and implementation around:

- reducing the number of depths sampled for prior-map viewing,
- separating fast prior-map radial-depth controls from the denser posterior
  display path,
- and clarifying what `s_max`, truncation, and depth sampling mean in the
  tuner app.

The user also found an inconsistency where the chosen prior-depth count did not
match the apparent visualization state. That prompted fixes so the viewer uses
the actual sampled depth set rather than pretending a denser continuous depth
grid exists.

## 7. Degree-aware weighted radial priors

The next major technical topic was whether the weighted case properly supports
nonzero spherical-harmonic degree in the radial Laplacian.

At the start of this work:

- the radial Laplacian implementation only properly supported `ell = 0`,
- but the weighted normal-mode prior should really use `ell = s` for the ST
  block of degree `s`.

The user asked:

- whether the current radial-weighted case works,
- how hard general `ell` support would be,
- whether fast packages exist for Bessel functions,
- and then requested a detailed implementation plan.

## 8. General `ell` radial Laplacian implementation

This became one of the biggest implementation tasks in the conversation.

### 8.1 Mathematical target

For the separated spherical problem, the radial operator is

```math
L_\ell f
=
-f'' - \frac{2}{r} f' + \frac{\ell(\ell+1)}{r^2} f.
```

The eigenfunctions are expressed in terms of spherical Bessel functions:

```math
j_\ell(kr), \qquad y_\ell(kr).
```

Regular domains starting at `r = 0` use only the regular branch `j_\ell`,
while shell domains use combinations `A j_\ell + B y_\ell`.

### 8.2 Code changes

The implementation introduced:

- general radial mode and provider classes in
  `intervalinf/intervalinf/providers/radial.py`,
- general-`ell` eigenvalue and spectrum wiring in
  `intervalinf/intervalinf/operators/radial.py`,
- spherical Bessel / derivative helpers based on SciPy,
- shell-domain determinant equations and root-finding with `brentq`,
- weighted `r^2 dr` normalization of radial modes,
- correct finite-difference inclusion of the centrifugal term
  `ell(ell+1)/r^2`,
- and safeguards so the old radial Dirichlet DST fast path is only used for
  `ell = 0`.

### 8.3 Tests

New tests covered:

- regular and shell radial modes for `ell > 0`,
- boundary conditions,
- weighted orthonormality,
- positivity and monotonicity of eigenvalues,
- correct Bessel-Sobolev inverse scaling,
- and explicit regression that the old fast path is not incorrectly used for
  `ell > 0`.

### 8.4 Outcome

This removed the foundational blocker for degree-aware weighted priors.

After that, the weighted prior could finally be made consistent with the
spherical-harmonic block structure:

Each `(s,t)` block now uses a radial covariance built with `ell = s`.

## 9. Degree-aware shared Bessel blocks everywhere

Once the general radial Laplacian worked, the next problem was making sure the
entire prior-building stack actually used degree-aware covariances everywhere.

This included:

- `build_shared_bessel_blocks`,
- `build_block_prior`,
- prior calibration,
- prior visualization,
- the Tk tuner app,
- the notebook tuner,
- remote Europa execution,
- headless paper runners,
- validation helpers.

Initially, some weighted builders still had a fallback that silently created a
degree-zero family if no degrees were supplied.

That was removed.

The final rule is now:

- in weighted mode, explicit ST degrees must be supplied,
- degree-aware shared Bessel blocks are keyed by `(component, degree)`,
- and the code no longer silently defaults to `ell = 0` for weighted priors.

This was an important correctness fix because it closed the last path by which
weighted ST-block priors could quietly revert to the wrong radial operator.

## 10. Prior-calibration documentation

After the implementation, the user requested a detailed mathematical and
algorithmic explanation of how calibration works now that the weighted case has
the more complicated degree-aware prior structure.

That produced the living reference note:

- `intervalinf/docs/agent-docs/references/living/prior-calibration-explainer.md`

This note explains:

- the block structure,
- the degree-specific covariance operators,
- the degree decay `w_s = (1+s)^(-beta)`,
- the unit-scale Monte Carlo calibration,
- and the final scalar calibration law
  `tau_{p,s} = a_p (1+s)^(-beta)`.

## 11. Plan-management and documentation cleanup

The conversation also included plan-document maintenance work.

This included:

- moving completed phase documents from `active-plans` to `completed-plans`,
- moving the `general-ell-radial-laplacian` plan to completed state,
- updating living references so documentation matched the implementation,
- and preserving provenance links inside the archived plan files.

In particular:

- several `full-spectrum-splitting-pli-example-phase-*.md` files were archived,
- several `realistic-dli-notebook-phase-*.md` files were archived,
- the main intervalinf living reference was updated repeatedly to reflect:
  - prior calibration,
  - headless runner support,
  - degree-aware weighted priors,
  - general-`ell` radial support,
  - and the new calibration explainer.

## 12. Thesis access

The user asked whether the thesis folder was accessible.

It was confirmed that:

- `/home/adrian/PhD/thesis` is readable,
- but `/home/adrian/PhD/Inferences/thesis` does not exist from the current
  environment.

This matters because the thesis was identified earlier as an important source
for the conceptual structure of the package family.

## 13. Main technical outcomes

By the end of the conversation, the main concrete outcomes were:

1. The weighted radial prior now has the right structural form for the ST-block
   model:
   - one degree-aware radial covariance shape per `(component, degree)`,
   - with `ell = s`.

2. Prior calibration is automatic and physically interpretable:
   - unit-scale Monte Carlo sampling,
   - component-wise target amplitudes,
   - degree decay controlled by `beta`,
   - final `tau_{p,s}` values derived automatically.

3. The prior-posterior tuner app is now much closer to the desired workflow:
   - synchronized `k`, `alpha`, and Sobolev order,
   - automatic calibration,
   - prior maps,
   - all-block and pushforward inference support,
   - better handling of prior-depth sampling.

4. The radial-laplacian implementation is substantially more correct and more
   general than at the start of the conversation.

5. The intervalinf agent docs are cleaner and more consistent, with completed
   work archived properly and new explanatory notes added.

## 14. Remaining scientific open questions

Several important scientific issues remain open.

### 14.1 Physical interpretation of amplitudes

It is still not fully resolved whether the model perturbations should be
interpreted as:

- `\delta \ln m`, or
- `\delta m`.

This matters because it changes how physically large a prior sample is allowed
to be.

### 14.2 Choice of `k` and `alpha`

The calibration solves only the amplitude problem.
It does not decide the best values of:

- the Bessel parameter `k`,
- the stiffness / correlation parameter `alpha`,
- the Sobolev order,
- or the boundary conditions.

Those remain prior-design questions requiring physical reasoning and empirical
testing.

### 14.3 Relation between 3-D amplitude bounds and separated coefficients

The calibration uses reconstructed-field Monte Carlo, which is good, but there
is still a deeper modeling question:

What amplitude assumptions on the original 3-D field should imply what prior
statistics after spherical-harmonic and radial decomposition?

This was one of the core motivations for the calibration effort.

## 15. Remaining software open questions

The software work also left several larger future tasks open.

### 15.1 pygeoinf 2.0 package restructuring

The high-level package architecture still needs the major refactor discussed
earlier:

- cleaner module structure,
- clearer conceptual layering,
- easier future growth.

### 15.2 Full scientific prior search

The current calibration machinery is a tool for exploring priors, not the final
answer to prior design.

The larger scientific program still needs:

- physically motivated prior-family comparison,
- stronger amplitude reasoning,
- and broader empirical comparison of weighted and flat priors.

### 15.3 Remaining unrelated test issues

During the implementation work, the targeted tests for the new radial/prior
features passed, but broader intervalinf testing still had unrelated preexisting
issues, including:

- one collection-time missing module issue involving `pygeoinf.matrix_function`,
- and a set of existing SOLA cache / instrumentation failures.

These were not introduced by the new weighted-prior implementation, but they
remain outstanding in the broader test landscape.

## 16. Recommended reading order for someone joining later

For a new collaborator trying to reconstruct the work quickly, the most useful
order is:

1. Read the package-level overview in
   `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`.
2. Read the prior-calibration explanation in
   `intervalinf/docs/agent-docs/references/living/prior-calibration-explainer.md`.
3. Read the completed general-ell plan in
   `intervalinf/docs/agent-docs/completed-plans/general-ell-radial-laplacian-plan.md`.
4. Inspect the weighted paper-demo utilities:
   - `full_spectrum_utils.py`,
   - `prior_calibration.py`,
   - `prior_viz.py`,
   - `prior_posterior_tuner.py`,
   - `prior_posterior_tuner_app.py`,
   - `prior_posterior_tuner_remote.py`.
5. Revisit the scientific open questions:
   - physical amplitude interpretation,
   - defensible `k` / `alpha`,
   - and how prior smoothness should be justified physically.

## 17. Bottom-line summary

The conversation moved from broad conceptual questions about Gaussian priors and
Bayesian inversion to a concrete, technically correct implementation of
degree-aware weighted radial priors with automatic amplitude calibration.

Scientifically, the core theme was:

Do not pick prior amplitudes by hand if they can instead be tied to physically
interpretable reconstructed-field amplitudes.

Technically, the core result was:

The weighted normal-mode prior now uses the correct spherical degree in the
radial operator, calibrates its amplitudes automatically, and exposes that
workflow through the tuner and the supporting paper-demo tooling.

