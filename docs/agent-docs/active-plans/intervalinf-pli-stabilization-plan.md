## Plan: Stabilize intervalinf for PLI Paper
**Status:** active **Created:** 2026-07-17
**Canonical path:** `/home/adrian/PhD/Inferences/intervalinf/docs/agent-docs/active-plans/intervalinf-pli-stabilization-plan.md`

### Goal

Recover the wanted implementation on `mission/lowering-materialization` into a
clean, stable, product-only `intervalinf` branch that can support the active
`PLI_paper` computations. Preserve the scientific and performance improvements,
remove branch-history and artifact pollution from the publishable result, repair
known correctness gaps, and establish a reproducible compatibility contract with
`pygeoinf`.

The recovery must not modify or switch the existing dirty mission worktree. It
must use isolated worktrees, constrained local computation, deterministic tests,
and explicit numerical gates. The final publishable branch must contain no agent
documentation, AI infrastructure, generated research outputs, or `work/`
artifacts.

### Local-only status and confidentiality

This plan is a local execution ledger. It must not be pushed or included in a
public branch. The same applies to all `docs/agent-docs/` content, agent
configuration, hooks, and AI infrastructure.

The current mission checkout is the plan-holding worktree. It will remain on
`mission/lowering-materialization`; all stabilization work will occur in sibling
worktrees. This keeps the canonical plan available at the absolute path above
even while implementation work moves between branches.

### Continuity protocol

Every agent or human resuming this campaign must do the following before reading
or changing implementation files:

1. Read this plan from its canonical absolute path.
2. Read the `Current checkpoint` table below.
3. Run `git worktree list` and verify that the recorded worktrees and branches
   still match reality.
4. Check both `intervalinf` and `pygeoinf` worktree status without switching
   branches.
5. Update the checkpoint before starting a new phase, before changing worktrees,
   and before ending a work session.
6. Record tests, numerical evidence, deviations, blockers, and the next exact
   action in this file as work proceeds.

Do not create independent copies of this plan. A stabilization worktree may have
an untracked, locally excluded pointer file or symlink to this canonical path,
but the plan itself has one source of truth.

### Current checkpoint

| Field | Current value |
|---|---|
| Campaign state | Actionable local work complete; external and deferred gates remain |
| Active phase | Phase 8 publication gate: product cleanup verified, awaiting a compatible pygeoinf release |
| Plan-holding worktree | `/home/adrian/PhD/Inferences/intervalinf` |
| Plan-holding branch | `mission/lowering-materialization` |
| Mission HEAD | `91c9c5c` |
| Mission dirty source patch | 6 insertions / 6 deletions across 4 source files |
| Stabilization worktree | `/home/adrian/PhD/Inferences/intervalinf-stabilize` |
| Stabilization branch | `stabilize/intervalinf-for-pli`; Phase 8 checkpoint `4840573` |
| Demo audit branch | `audit/intervalinf-demos`; product tip `62b73c9`, based on `4840573` |
| pygeoinf current checkout | `fix/solver-numerical-robustness` at `bafb88f` |
| pygeoinf functional branch | `refactor/functional-vector-updates` at `fec12f5` |
| pygeoinf compatibility worktree | `/home/adrian/PhD/Inferences/pygeoinf-functional-solvers` at `66c127b` |
| Last completed action | Audited all eight maintained demos and committed targeted notebook/API repairs as `62b73c9`; 124/124 code cells and all 557 package tests pass, with source notebooks remaining output-free |
| Next exact action | After pygeoinf releases the functional vector-update contract, set the exact release floor, rerun public CI and the package build, then close Phase 8; schedule full `s_max=14` PLI reproduction separately when requested |
| Blockers | No tagged pygeoinf release contains the required functional vector-update contract; full `s_max=14` reproduction is intentionally deferred; four heavy local PLI tests remain deferred for resource reasons |

### Situation summary

#### intervalinf branch structure

The branch history is a strict stack:

```text
origin/main (9996213)
  -> convex_analysis (fa426dc locally; origin at 0e08616)
    -> fix/laplacian-double-alpha (4dbc230)
      -> mission/lowering-materialization (91c9c5c)
```

`origin/main` has not advanced since 2026-01-12. Every development branch is
zero commits behind it, so there is no divergent-main rebase problem. The
mission branch contains all wanted development commits but is local-only and
must not be pushed as-is.

The local mission worktree currently contains approximately 315 modified tracked
paths and 15 untracked paths. Most are generated demos and figures. Four source
files contain a small uncommitted patch dated 2026-06-23:

```text
intervalinf/core/functions.py
intervalinf/operators/sola.py
intervalinf/operators/spectral_helpers.py
intervalinf/sampling/kl_sampler.py
```

Those changes suppress repeated inner domain checks inside already-validated
composite function evaluations. They may be intentional performance work, but
they require focused correctness and timing tests before inclusion.

#### Publishable mission delta

The useful library delta from `origin/main` to the mission tip is tractable:

| Category | Changed paths |
|---|---:|
| Runtime package | 24 |
| Tests | 14 |
| Demos | 1,304 |
| Agent documentation | 107 |
| `work/` artifacts | 1,354 |
| `rough_work/` | 7 |

The runtime and test changes are about 8,069 inserted lines and 310 deleted
lines. The branch size is therefore mostly research artifacts, generated output,
and local documentation rather than library code.

#### PLI paper dependency

The active manuscript repository is `/home/adrian/PhD/PLI_paper`. Its working
tree is clean and its local `main` is 88 commits ahead of `origin/main`. Its
instructions prohibit pushes and public network access.

The active paper pipeline directly imports mission-only `intervalinf` features,
including:

- `WeightedLebesgue` and mass-weighted Riesz maps;
- `RadialLaplacian` with general spherical-harmonic degree;
- `BesselSobolevInverse` on weighted radial spaces;
- weighted `KLSampler` behavior;
- basis-free `Lebesgue` model spaces;
- SOLA operators and semantic data-space Bayesian inversion.

The current conda environment imports `intervalinf` and `pygeoinf` directly from
their working checkouts. Returning to the old `intervalinf/main` API would break
the active paper workflow.

The active PLI path does not call `compute_reduced_covariance()` or the reduced
covariance adapter. Its Bayesian solve constructs the data-space normal operator
through semantic `pygeoinf` operators. The known weighted reduced-covariance bug
is therefore a library release blocker, but is not by itself evidence that the
existing paper results are numerically wrong.

#### Provenance gap

PLI run provenance currently computes the repository root incorrectly and
records `git_head` and `git_branch` as `null` in inspected run artifacts. It also
does not record the `intervalinf` or `pygeoinf` import path, commit, or dirty
patch hash. Existing result directories therefore cannot prove which dependency
states produced them.

Final paper-critical outputs must be regenerated after provenance is repaired
and the dependency commits are pinned. Existing outputs remain valuable
comparison baselines, but cannot be treated as fully reproducible evidence.

#### pygeoinf compatibility

The two relevant pygeoinf branches both descend directly from current
`origin/main` but diverge from one another:

```text
origin/main (b97486c)
  -> fix/solver-numerical-robustness (bafb88f)
  -> refactor/functional-vector-updates (fec12f5)
```

The PLI environment currently sees the solver branch, where the abstract
`HilbertSpace.axpy` contract is still in-place. The intended interval-function
model requires the functional vector-update branch. Stabilization must test a
local integration of both changes. If the solver PR enters `main` during this
campaign, the compatibility branch must be rebuilt from the updated remote main.

### Design decisions

1. **Reconstruct from `origin/main`, do not clean the mission history in place.**
   Apply a path-limited final-state delta for runtime code and tests. Preserve the
   mission branch locally for archaeology.
2. **Keep PLI code in the PLI repository.** The paper demos have evolved into a
   separate scientific workflow. Do not copy the 1,300-file demo tree into the
   clean library branch.
3. **Use semantic invariants, not only regression snapshots.** Adjointness,
   symmetry, positivity, finiteness, and operator equivalence are required
   evidence for scientific code.
4. **Treat the current paper outputs as comparison data, not a truth oracle.** A
   changed result must be explained mathematically; it must not be forced back
   to a historically unpinned value.
5. **Support PLI locally before upstream review completes.** The stabilization
   integration branch may be used by PLI once its gates pass. Upstream changes
   can later be split into reviewable PRs.
6. **No branch transitions in the dirty mission checkout.** Sibling worktrees
   are mandatory.
7. **No expensive parallel local work.** Use one Python process and one native
   numerical thread until a workload is explicitly moved to Europa.

### Files and components

The reconstruction covers these product areas:

```text
intervalinf/core/
  config.py
  domain.py
  functions.py
  materialization.py

intervalinf/spaces/
  lebesgue.py
  weighted_lebesgue.py
  sobolev.py
  forms.py
  __init__.py

intervalinf/operators/
  base.py
  laplacian.py
  radial.py
  bessel.py
  sola.py
  reduced.py
  spectral_helpers.py
  _impl/fast_spectral.py
  __init__.py

intervalinf/providers/
  base.py
  radial.py
  functions/fem.py
  functions/step.py

intervalinf/sampling/
  kl_sampler.py

intervalinf/__init__.py
tests/
pyproject.toml
README.md
.github/workflows/tests.yml
```

PLI integration work is limited to local reproducibility and validation support
under `/home/adrian/PhD/PLI_paper/paper_demos/`. Manuscript prose and scientific
claims are outside this stabilization plan.

### Resource policy

All local Python validation must begin with:

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMBA_NUM_THREADS=1
```

Additional constraints:

- use `conda run -n inferences python ...`;
- do not use pytest-xdist or `-n auto`;
- do not regenerate broad notebook or figure collections locally;
- run focused tests before package-wide tests;
- run only one significant test or benchmark process at a time;
- record command duration and peak-memory concerns when known;
- move full paper configurations to Europa only after local gates pass and the
  user explicitly approves the remote run.

### Phases

#### Phase 0 - Preserve and freeze the current effective state

**Objective:** Make the current source state reproducible without modifying the
shared dirty checkout.

Actions:

1. Save a source-only patch for the four modified runtime files outside the
   repository's publishable tree and record its SHA-256 hash.
2. Inventory meaningful untracked source/notebook files separately from
   generated output. Do not delete, stash, or commit them.
3. Record the current intervalinf and pygeoinf SHAs, import paths, branch names,
   dirty status, Python version, and numerical-library versions.
4. Select a small set of existing PLI configurations and output arrays to use
   as comparison baselines; record hashes without copying the 2.4 GB run tree.
5. Write the Phase 0 completion record with all preservation paths and hashes.

**Exit gate:** The mission checkout can be lost without losing the source patch
or the information required to reconstruct the effective environment.

#### Phase 1 - Create isolated worktrees and compatibility branches

**Objective:** Establish clean work locations without disturbing active agents.

Actions:

1. Create a sibling intervalinf worktree from `origin/main` on local branch
   `stabilize/intervalinf-for-pli`.
2. Create a local pygeoinf compatibility worktree that combines the functional
   vector-update branch with the solver robustness commit.
3. Add a local, untracked pointer from the stabilization worktree to this
   canonical plan and exclude the pointer through local Git metadata.
4. Verify all worktrees and branches with `git worktree list` and clean status.
5. Configure test commands to select packages with `PYTHONPATH`, avoiding
   changes to the shared editable conda installation.

**Exit gate:** Clean intervalinf and pygeoinf worktrees exist, import the intended
source trees, and the original worktrees are unchanged.

#### Phase 2 - Reconstruct the runtime and test snapshot

**Objective:** Transfer the wanted product state without transferring polluted
history or unrelated artifacts.

Apply the final `origin/main...mission/lowering-materialization` delta only for
`intervalinf/` and `tests/`. Work in the following independently reviewable
cohorts:

1. Core function behavior, support metadata, configuration, and materialization.
2. Lebesgue, Sobolev, forms, and weighted-space representation.
3. Spectral helpers, Laplacian, radial, Bessel, and provider machinery.
4. SOLA optimization, weighted adjoints, Gram assembly, and reduced operators.
5. KL sampling, package exports, and integration cleanup.

For each cohort:

- inspect the final diff against both main and mission;
- retain established public APIs unless a correctness issue requires change;
- run focused tests before committing the cohort;
- update the local living reference in the same unit of work;
- write a phase evidence record before advancing.

Do not transfer `demos/`, `docs/agent-docs/`, `work/`, `rough_work/`, generated
figures, or branch-local hooks during this phase.

**Exit gate:** The clean branch matches the wanted mission runtime snapshot and
its transferred tests, with no unrelated paths.

#### Phase 3 - Resolve cross-package vector-update compatibility

**Objective:** Make interval functions reliable under the intended functional
`ax` and `axpy` contract while documenting transitional behavior.

Actions:

1. Add intervalinf integration tests that exercise `Function` vectors through
   pygeoinf solvers, direct sums, Gaussian operations, and operator application.
2. Run those tests against the current solver branch and the local combined
   functional-plus-solver branch.
3. Ensure callers always retain returned vectors where the functional contract
   requires it.
4. Confirm that mutable NumPy-backed spaces retain their optimized in-place
   implementation without changing results.
5. If the solver PR is merged, rebuild the combined test branch from the new
   pygeoinf main and repeat the compatibility gate.

**Exit gate:** The PLI-critical interval vector paths pass against the intended
pygeoinf integration state, and any unsupported older pygeoinf state fails with
a clear version or contract boundary rather than silently computing bad results.

#### Phase 4 - Correctness hardening

**Objective:** Resolve known defects and reproduce design risks before changing
their behavior.

Actions:

1. Fix weighted reduced covariance so `C` acts on the adjoint representative
   `M^{-1} k_j`; add plain, weighted-Lebesgue, and Sobolev semantic-equivalence
   tests.
2. Reproduce incompatible equal-space behavior for different basis
   representations. Decide whether to strengthen `__eq__`, introduce an
   explicit representation-compatibility predicate, or both.
3. Reproduce basis-free `dim=0` failures in coordinate-only pygeoinf paths.
   Add explicit capability checks or guarded errors for observed failures; do
   not redesign the entire Hilbert-space API without evidence.
4. Validate the six-line `check_domain=False` patch. Confirm that the outer
   `Function` still enforces domain membership, nested calls do not accept
   invalid inputs, and repeated mesh evaluation benefits measurably.
5. Verify weighted SOLA adjointness and Gram identities, radial eigenfunction
   orthogonality, Bessel covariance self-adjointness/positivity, KL variance,
   and finite outputs at boundary and near-singular radial points.
6. Resolve the historically recorded SOLA instrumentation failures. A stable
   branch may not finish with unexplained failing tests.

**Exit gate:** All known correctness issues have either a tested fix or a
documented, deliberately bounded behavior with clear user-facing errors.

#### Phase 5 - Resource-constrained intervalinf validation

**Objective:** Establish correctness and performance evidence for the clean
library branch.

Validation ladder:

1. Import and public API smoke tests.
2. Core and space tests.
3. Spectral, radial, weighted, SOLA, and reduced-operator tests.
4. Cross-package functional-vector tests.
5. Full intervalinf suite in one process. The source currently contains about
   506 test functions and historical caches show roughly 658 collected cases.
6. Focused process-time benchmarks for function composition, SOLA repeated
   evaluation, weighted Bessel application, and KL evaluation.

Comparison rules:

- use fixed seeds and explicit `rtol`/`atol`;
- compare semantic outputs before timing outputs;
- use multiple timing repetitions and report variance;
- investigate material regressions rather than weakening thresholds;
- record every command and result in the phase completion document.

**Exit gate:** Zero unexplained failures, numerical invariants pass, and no
material PLI-critical performance regression remains.

#### Phase 6 - Repair PLI provenance and run downstream acceptance tests

**Objective:** Make paper runs attributable to exact dependency states and use
the paper as a downstream integration suite.

Actions in the confidential PLI repository require its local instructions and
must never use network access or push changes.

1. Correct the PLI repository-root calculation used by provenance collection.
2. Record PLI, intervalinf, and pygeoinf commit IDs, branch names, dirty state,
   source import paths, dirty-patch hashes, Python version, and package versions.
3. Add tests for provenance collection using temporary repositories or mocked
   command outputs.
4. Run the paper-demo tests with one process and capped numerical threads. The
   current suite has 108 tests after adding two provenance regressions; 104 pass
   locally in 73.65 s. Four all-block tests are deferred as recorded in the
   checkpoint because one requires parallel workers and three require the same
   multi-minute full property solve.
5. Run a one-block weighted smoke inference.
6. Run a deterministic reduced weighted pipeline with small covariance
   truncation, no broad plotting, and `n_jobs=1`.
7. Compare posterior means, covariances, standardized residuals, property
   outputs, symmetry, positivity, and finiteness with preserved baselines.

**Exit gate:** A new reduced PLI run records complete provenance and passes all
scientific and numerical acceptance checks on pinned source states.

#### Phase 7 - Reproduce paper-critical configurations

**Objective:** Replace unpinned final evidence with reproducible outputs from the
stabilized dependency state.

Actions:

1. Select the smallest set of configurations that feed current manuscript
   claims and figures.
2. Freeze resolved configs, seeds, input fingerprints, and dependency commits.
3. Run local reduced cases first.
4. After explicit user approval, use Europa for full configurations. Follow the
   PLI repository's status, disk, claim, registry, safe-pull, inspection, and
   release procedures.
5. Compare new and historical outputs. Trace any difference from the earliest
   operator-stage divergence rather than only comparing final figures.
6. Install only reviewed result directories into the PLI checkout.

**Exit gate:** Every retained paper-critical result is tied to a complete
provenance record and its numerical differences from prior results are explained.

**2026-07-17 checkpoint:** The active P1 manuscript and figure scripts resolve to
three paper-critical configurations:

```text
paper_synthetic_weighted_anisotropic_smax14.toml
paper_synthetic_weighted_anisotropic_noise1x_smax14.toml
paper_synthetic_weighted_anisotropic_smoothprior_smax14.toml
```

Their existing complete run directories have hashes `167eca2239de`,
`752e1cee4b58`, and `367f40ec0029`, respectively. All three record null PLI Git
heads and no dependency states, so they are preserved comparison baselines, not
reproducible final evidence.

A laptop-safe reduction of the baseline config selected only `s2_t0`, used
covariance truncation 4, reduced calibration sampling, one worker, and no
figures or posterior samples. It exposed an eager degree-zero lookup in split-vs
calibration when only a nonzero degree is selected. A red-green regression moved
that lookup into the non-per-degree path; all eight calibration tests then pass.
The refined 256/512-point reduced run completes in 53 s. All numeric artifacts
are finite, property covariance is exactly symmetric and positive, the
symmetrized data normal is positive, `chi_rms=0.47448`, and the recovered target
is `-0.10111` posterior standard deviations from truth. Quadrature refinement
reduced relative normal asymmetry by 6.45x while changing the property mean by
only `3.48e-9`. The post-fix constrained paper-demo suite reports 105 passed and
4 deliberately deselected in 81.71 s.

Europa execution was approved by the user on 2026-07-17. After an initial VPN
timeout, connectivity returned and the remote disk gate reported 2.8 TB free.
Dedicated remote snapshots import intervalinf at `d12e9d4` and pygeoinf at
`66c127b`; 554 remote intervalinf tests pass in 6.03 s. The sync excluded and
removed agent documentation, AI infrastructure, Git data, and caches. Because
the remote PLI tree is intentionally Git-less, run provenance records the live
remote import paths plus an explicit synced-source snapshot.

The initially submitted three-run `s_max=14` job (PID `2488686`) was too large
for an acceptance check and was stopped at 4 of 120 blocks. No result from that
incomplete run is accepted as evidence. It was replaced by
`paper_synthetic_weighted_anisotropic_reduced_smax2_repro20260717`, retaining
the weighted degree-aware priors, prior calibration, noise model, all six
available blocks through `s_max=2`, and all nine property targets. The reduced
run completed on Europa in 64 s and its 18 manifest artifacts passed hash,
finiteness, covariance symmetry, positive-semidefiniteness-to-roundoff, and
standardized-residual checks. Its source snapshot records PLI dirty hash
`7a263028...`, intervalinf `d12e9d4`, pygeoinf `66c127b`, and the dedicated
remote import paths. This is reduced end-to-end acceptance evidence only; it
does not satisfy the Phase 7 exit gate for final `s_max=14` production results.

#### Phase 8 - Product and publication cleanup

**Objective:** Produce a reviewable, product-only branch without blocking local
paper work on upstream review timing.

Actions:

1. Repair README examples so constructors, imports, boundary conditions, and
   operator application syntax are executable.
2. Align `requires-python`, CI, and the pygeoinf dependency floor. The initial
   supported environment should be Python 3.12 unless evidence supports more.
3. Replace placeholder authors and repository URLs.
4. Add appropriate product-level ignore rules for `work/` and generated output.
5. Add only small, maintainable user-facing examples needed to document weighted
   and radial APIs. Do not restore the historical paper-demo tree.
6. Run formatting, lint, `git diff --check`, full tests, and package build checks.
7. Audit the final tree and history for forbidden local paths, agent docs, AI
   infrastructure, hooks, large binaries, generated figures, and confidential
   PLI content.
8. Keep the clean integration branch local until the user explicitly chooses to
   publish. Later upstream PRs may be split by coherent feature group.

**Exit gate:** The local stabilization branch is clean, reproducible, usable by
PLI, and safe to turn into product PRs without exposing local infrastructure.

**2026-07-17 checkpoint:** All locally actionable cleanup is verified. The root
README now uses executable continuous, Laplacian, and weighted-radial examples;
the demo index names only retained notebooks; Python, NumPy, SciPy, repository,
author, license, Ruff, and CI metadata match the tested product state; and
ignore rules protect local research and agent infrastructure. Saved output was
removed from eight notebooks while preserving every source cell exactly. Three
README regression tests pass, the full suite reports 557 passed in 12.20 s,
changed Python files pass Ruff, `git diff --check` passes, both sdist and wheel
build, and an installed-wheel smoke test passes. The wheel contains only the
package and license, while the reviewed sdist contains no agent infrastructure,
local paths, binaries, generated figures, or confidential PLI content.

Repository-wide Ruff is not yet clean: 459 pre-existing findings remain across
legacy source and tests, predominantly import ordering and Python-3.12 typing
modernization. They were recorded rather than mass-rewritten during a targeted
publication cleanup. Phase 8 cannot pass its public-CI release gate until a
tagged pygeoinf version contains the functional vector-update contract. The
metadata therefore records 1.8.2 as the current tagged baseline while README,
runtime, CI, and this plan explicitly state that 1.8.2 alone is insufficient
for basis-free workflows. The exact floor must be set to the first compatible
release rather than guessed in advance.

The product-only checkpoint is commit `4840573` on
`stabilize/intervalinf-for-pli`. The worktree is clean and the commit contains
no agent documentation. The canonical plan and living-reference updates remain
local-only in the protected mission worktree.

### Dynamic decision gates

#### Baseline gate

If the clean mission snapshot fails before any correction, classify each
failure as a known historical failure, environmental incompatibility, or real
mission regression. Do not copy a failing state forward without a recorded
disposition.

#### pygeoinf gate

If functional vector updates change PLI results, compare the first affected
vector operation and its mathematical invariant. Do not preserve the old result
by discarding returned vectors or by reintroducing mutation assumptions.

#### Representation gate

If strengthening space equality breaks legitimate operator composition, split
mathematical-space equality from representation compatibility rather than
weakening all checks or comparing basis objects ad hoc.

#### Basis-free gate

If PLI uses a generic coordinate-only path with a basis-free space, first route
that workflow through semantic operators where appropriate. Add a capability API
to pygeoinf only when multiple concrete failures justify the abstraction.

#### Performance gate

If the clean implementation is materially slower, profile only the affected
focused workload. Preserve the semantic implementation while adding a validated
specialized fast path. Never trade weighted adjoint or covariance correctness
for historical timing.

#### Paper-output gate

If stabilized outputs differ from existing PLI outputs, keep both artifacts,
identify the first numerical divergence, and determine which result satisfies
the governing operator identities. Historical plots are not automatically the
reference truth.

#### Remote-compute gate

If local validation cannot finish within laptop constraints, reduce the test
case further. Use Europa only after the local reduced case passes and the user
explicitly approves the remote state change.

### Acceptance criteria

The campaign is complete only when all of the following hold:

1. The original mission worktree and its uncommitted source work remain
   preserved.
2. A clean product branch contains the wanted runtime functionality and focused
   tests without the mission branch's artifacts or agent infrastructure.
3. intervalinf passes its full constrained test suite with no unexplained
   failures.
4. Weighted and basis-free scientific paths satisfy explicit mathematical
   invariants.
5. Functional pygeoinf vector updates and solver robustness coexist in the
   tested PLI dependency state.
6. PLI's paper-demo tests and a reduced weighted end-to-end inference pass.
7. New PLI results record exact PLI, intervalinf, and pygeoinf provenance.
8. Final paper-critical outputs are reproduced or their differences are
   scientifically explained.
9. README, packaging, CI, and public metadata describe the actual supported
   system.
10. No agent docs, AI infrastructure, generated research outputs, or
    confidential paper content appear in the publishable branch.

### Open questions

1. Will pygeoinf's solver robustness PR merge before the compatibility phase?
2. What release/version will first contain the functional vector-update
   contract, and therefore what dependency floor should intervalinf publish?
3. Should mathematical equality and coordinate-representation compatibility be
   separate public concepts in pygeoinf, or can intervalinf solve the observed
   cases locally?
4. Which existing PLI run directories directly feed the current manuscript and
   therefore require final regeneration?
5. Does the six-line domain-check bypass produce a material speedup after the
   current materialization and caching paths are reconstructed?
6. Which small intervalinf demos should remain as maintained public examples,
   given that the full paper workflow now belongs to the confidential PLI
   repository?

### Update log

| Date | Update |
|---|---|
| 2026-07-17 | Created from the completed read-only recovery audit; no execution or branch changes performed. |
| 2026-07-17 | Began Phase 0 after verifying both package worktrees and rereading the TDD protocol. |
| 2026-07-17 | Completed Phase 0; patch and manifest preserved in Git-local and external local-only recovery storage. |
| 2026-07-17 | Completed Phase 1; clean intervalinf and combined functional-plus-solver pygeoinf worktrees verified. |
| 2026-07-17 | Completed Phase 2; reconstructed 38 product/test paths and isolated the six known SOLA failures. |
| 2026-07-17 | Completed Phase 3; functional interval updates pass through direct sums, weighted spaces, CG, and Gaussian accumulation, with a clear legacy-contract boundary. |
| 2026-07-17 | Phase 4 checkpoint: fixed weighted reduced covariance after reproducing the unweighted-moment error; 19 reduced tests pass. |
| 2026-07-17 | Phase 4 checkpoint: separated operational representation compatibility from dimension/domain coincidence; 128 focused tests pass. |
| 2026-07-17 | Phase 4 checkpoint: bounded basis-free coordinate-only paths with explicit errors; 130 focused tests pass. |
| 2026-07-17 | Phase 4 checkpoint: validated nested domain-check bypass; outer guards remain effective, no new SOLA failures, focused wrapper timing improved from 29.934 to 23.835 microseconds. |
| 2026-07-17 | Phase 4 checkpoint: resolved the six historical SOLA failures; full SOLA file is 160/160 green and weighted/radial invariants are 19/19 green. |
| 2026-07-17 | Completed Phase 4; all 554 intervalinf tests pass under the constrained local resource policy. |
| 2026-07-17 | Completed Phase 5; full validation is green and focused deterministic benchmarks record semantic equivalence plus timing evidence. |
| 2026-07-17 | Phase 8 checkpoint: all local product cleanup and artifact verification pass; closure waits on the first pygeoinf release with functional vector updates. |
| 2026-07-17 | Completed the derivative demo-execution audit on `audit/intervalinf-demos` at `62b73c9`; all eight maintained notebooks pass from clean kernels. |
