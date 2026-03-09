# SOLAOperator Phase 1 Audit

## Scope

This note records the Phase 1 implementation and dependency audit for the
`SOLAOperator_speedup` project. The goal of Phase 1 was to understand the exact
current behavior of `intervalinf.operators.SOLAOperator`, identify preserved
invariants, and pinpoint the most likely sources of runtime overhead before any
benchmarking or implementation changes.

## Files audited

- `intervalinf/intervalinf/operators/sola.py`
- `intervalinf/intervalinf/core/functions.py`
- `intervalinf/intervalinf/core/domain.py`
- `intervalinf/intervalinf/providers/base.py`
- `intervalinf/intervalinf/providers/functions/data.py`
- `intervalinf/intervalinf/providers/functions/smooth.py`
- `intervalinf/intervalinf/spaces/lebesgue.py`
- `intervalinf/intervalinf/spaces/forms.py`
- `pygeoinf/pygeoinf/linear_operators.py`
- `pygeoinf/pygeoinf/direct_sum.py`
- `pygeoinf/pygeoinf/backus_gilbert.py`
- `pygeoinf/pygeoinf/convex_analysis.py`
- `intervalinf/rough_work/benchmark_dli_solvers.py`

## Exact forward path

For `G = SOLAOperator(domain=M, codomain=D)`, `G(f)` proceeds as follows:

1. `LinearOperator.__call__` dispatches to `SOLAOperator._mapping`.
2. `SOLAOperator._mapping` calls `SOLAOperator._apply_kernels(f)`.
3. `_apply_kernels` loops over every kernel index.
4. For each kernel it:
   - retrieves the kernel from a list or provider,
   - creates a fresh product callable `f(x) * k_i(x)`,
   - wraps that callable in a new `Function`,
   - integrates it numerically with the configured method and point count.
5. The result is stored into the output vector component by component.

The implementation is therefore mathematically continuous in semantics, but in
runtime behavior it performs one numerical integration per kernel per call.

## Exact adjoint path

`SOLAOperator` supplies a dual mapping rather than a direct adjoint mapping. The
adjoint action is derived through the `pygeoinf.LinearOperator` dual-to-adjoint
machinery:

1. convert the Euclidean vector to a dual element,
2. reconstruct a kernel-based `LinearFormKernel`,
3. map that dual element back into the primal model space via `domain.from_dual`.

For `Lebesgue`, this effectively produces a function equal to the weighted sum of
the kernels. For `Sobolev`, the inverse Riesz map is applied, so the adjoint
semantics depend on the space as intended.

The reconstructed function is itself a callable that loops over all contributing
kernels when evaluated, which becomes expensive when the adjoint output is then
integrated or normed repeatedly downstream.

## Direct-sum behavior

`SOLAOperator.for_direct_sum(...)` constructs one SOLA block per subspace and
returns a `RowLinearOperator` that sums block outputs. Kernel restriction is done
through provider restriction or `Function.restrict(...)` on each kernel. Any
future optimization must preserve this decomposition and its adjoint semantics.

## Preserved invariants

- Forward semantics remain integration-based and continuous-first in the public API.
- Linearity must be preserved in both the forward and adjoint paths.
- Adjoint correctness must continue to be space-aware, especially for Sobolev spaces.
- Provider-backed kernels remain authoritative; caching must not change which kernel
  is used.
- Support metadata semantics must be preserved, including `support=[]` effectively
  representing the zero function and `support=None` meaning no compact-support hint.
- The non-vectorized callable fallback path must remain correct, even if it remains slower.

## Current inefficiencies

### Forward application

- One fresh product callable and one fresh `Function` allocation per kernel.
- One fresh quadrature mesh build per kernel integration.
- Repeated evaluation of the same input function across all kernels.
- Kernel object caching exists, but kernel-on-mesh values are not cached.

### Support handling

- The product integrand built inside `_apply_kernels` carries no support metadata,
  so integration always occurs over the full domain even when both function and
  kernel have compact support.
- Current support-intersection behavior elsewhere often drops compact-support
  information when one operand has `support=None`, which limits the performance
  benefit of support-aware logic.

### Adjoint downstream cost

- Reconstructed adjoint functions evaluate as Python loops over kernels.
- In `DualMasterCostFunction.value_and_subgradient(...)`, these reconstructed
  functions are repeatedly normed and integrated, amplifying the evaluation cost.

## Key downstream hotspot

`DualMasterCostFunction.value_and_subgradient(...)` is the most important realistic
hot path for optimization work. In a typical call it performs:

- one `G.adjoint(lambda)` evaluation,
- one or more norm/integration-heavy support-function computations using the
  resulting function,
- one `G(v)` forward evaluation for the support point.

This makes it the most relevant end-to-end benchmark target for Phase 2.

## Test coverage findings

- Dedicated functional tests for `SOLAOperator` are currently minimal to nonexistent.
- Existing operator tests in `tests/operators/test_operators.py` only verify importability.
- There is no SOLA-specific test module covering forward correctness, adjoint
  identities, compact-support behavior, provider-backed kernels, or integration-method behavior.

This gap makes a strong baseline test phase mandatory before optimization.

## Current knobs and defaults

The current implementation is driven primarily by the following configuration paths:

- `SOLAOperator(..., integration_config=IntegrationConfig(method='simpson', n_points=1000))`
- `SOLAOperator(..., cache_kernels=False)` by default
- provider-backed kernels are fetched lazily through `get_kernel(i)`
- vectorization behavior is currently determined inside `IntervalDomain.integrate(...)`
  by trying array evaluation first and falling back to scalar iteration if needed

In practical terms, this means current runtime depends strongly on:

- the selected integration method,
- the number of quadrature points,
- whether kernels and input functions support array evaluation,
- whether kernels are global or compactly supported,
- and whether repeated calls reuse the same operator and settings.

## Integration-method inconsistency

- `IntegrationConfig` advertises a `quad` method name.
- Domain-level integration behavior is implemented with a different naming/behavior
  convention (`adaptive` in the current research notes).
- This mismatch is now explicitly in scope for cleanup during the speedup project.

### Current method-name behavior summary

| Layer | Current names observed | Notes |
|------|-------------------------|-------|
| `IntegrationConfig` | `simpson`, `trapz`, `quad` | Public config surface currently advertises `quad` |
| `IntervalDomain.integrate(...)` | `simpson`, `trapz`, `adaptive` | Internal integration dispatch uses a different adaptive-method name |
| `SOLAOperator` integration call path | passes config method through | No normalization layer currently reconciles the naming mismatch |

Phase 3 should make these names internally consistent and explicitly tested.

## Recommended Phase 2 benchmark surfaces

1. Isolated forward operator timings for global and compact-support kernels.
2. Isolated adjoint construction and adjoint-evaluation timings.
3. End-to-end `DualMasterCostFunction.value_and_subgradient(...)` timings.
4. Direct-sum/discontinuity cases using `SOLAOperator.for_direct_sum(...)`.
5. Accuracy comparisons against higher-accuracy integration settings.
6. Adjoint-consistency residuals under explicit tolerances.

### Draft benchmark taxonomy

The benchmark surfaces should be split into two categories:

1. **Pure SOLA microbenchmarks**
   - These exist to understand the operator itself.
   - They should not treat model-basis size as a primary axis.
   - If a `Lebesgue` space object is required by the current implementation,
     its basis dimension should be held fixed as an ambient implementation detail,
     not swept as if SOLA itself were being benchmarked as a basis projection.

2. **Downstream workflow benchmarks**
   - These exist to understand realistic costs in current PLI/DLI-style pipelines.
   - Here, ambient model-space choices can matter because they influence the
     behavior of the surrounding space implementation and downstream function/norm
     evaluations, even though they are not part of SOLA's public semantics.

### Pure SOLA microbenchmark axes

| Axis | Baseline values to include |
|------|----------------------------|
| Integration method | `simpson`, `trapz`, cleaned-up adaptive/quad case |
| `n_points` | 200, 500, 1000, 2000 |
| Data dimension `N_d` | 1, 5, 20, 50, 200 |
| Kernel source | provider-backed, direct function list, direct callable list |
| Kernel family | global-support modes, compact-support bumps |
| Function representation | callable-based, compact-support callable |
| Topology | standard space, direct-sum/discontinuity case |

### Downstream workflow axes

| Axis | Example values |
|------|----------------|
| Property dimension `N_p` | 1, 2, 5, 20 |
| End-to-end hotspot | `DualMasterCostFunction.value_and_subgradient(...)` |
| Ambient space implementation details | fixed by default; only varied in a dedicated downstream study if needed |

## Phase 1 conclusion

The audit confirms that `SOLAOperator` is a strong candidate for acceleration,
especially in repeated optimization workloads such as PLI and DLI. The most likely
high-value improvements are shared-mesh/batched fixed-grid evaluation, support-aware
integration restriction, and reusable cache structures for repeated calls. Benchmarking
should explicitly distinguish pure operator costs from downstream workflow costs so that
future optimizations do not quietly reframe SOLA as a basis-discretized operator.