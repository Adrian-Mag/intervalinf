## Phase 4 Complete: Correctness hardening

**Plan:** `docs/agent-docs/active-plans/intervalinf-pli-stabilization-plan.md`  
**Completed:** 2026-07-17

### Work completed

- Fixed reduced covariance on mass-weighted domains by applying the model
  covariance to `G._adjoint_kernel(j)`, i.e. to `M^-1 k_j` rather than `k_j`.
- Added plain, weighted-Lebesgue, and Sobolev semantic-equivalence coverage for
  `G C G*`, plus rejection of coordinate operators using a different basis.
- Added `Lebesgue.is_representation_compatible()` and made operational equality
  include named and direct-callable basis representations.
- Stopped basis-free `dim=0` spaces from silently mapping nonzero functions to
  empty coordinates or through zero-size matrix operators to zero.
- Validated and applied the preserved nested `check_domain=False` optimization.
  Reconstructed SOLA adjoints are now attached to the model space so the outer
  domain guard remains active.
- Corrected SOLA fixed-grid classification so global kernels remain on the
  shared cached batch even when the input function has compact support.
- Replaced an invalid equality assertion between two different fixed-grid
  quadrature problems with analytic and path-specific accuracy assertions.

### Commits

```text
257b715 fix(reduced): respect weighted model adjoints
0ad7156 fix(spaces): distinguish basis representations
c5f33e4 fix(spaces): guard basis-free coordinate paths
5df8049 perf(functions): skip redundant nested domain checks
7b926c4 fix(sola): restore global kernel batching
```

### Red evidence

The weighted identity-covariance reproduction returned the unweighted moment
matrix instead of the semantic mass-weighted result:

```text
actual   = [[1.000000, 0.500000],
            [0.500000, 0.333333]]
expected = [[0.693147, 0.306853],
            [0.306853, 0.193147]]
max absolute difference = 0.30685282
```

Additional red tests showed:

- equal dimension/domain sine and cosine spaces compared equal;
- distinct direct-callable bases compared equal;
- basis-free `to_components()` returned an empty array and a `(0,0)` matrix
  operator could return the zero function;
- all four nested-evaluation probes requested a redundant default domain check;
- the reconstructed mission snapshot had five SOLA cache/counter failures and
  one support-quadrature comparison failure.

### Final evidence

```text
reduced operators:
19 passed in 1.41s

spaces + reduced + functional compatibility after representation hardening:
128 passed in 11.78s

basis-free coordinate guard focused set:
130 passed in 11.33s

domain-bypass focused set:
136 passed in 9.22s

SOLA:
160 passed in 1.56s

weighted SOLA + weighted/radial Bessel invariants:
19 passed in 1.46s

complete intervalinf suite, one process and capped native threads:
554 passed in 12.68s
```

Focused repeated-mesh timing for one validated wrapper on 2,001 points, median
of seven repeats with 1,000 evaluations per repeat:

```text
redundant nested check: 29.934 microseconds/evaluation
validated inner bypass: 23.835 microseconds/evaluation
speedup: 1.256x
```

### Deviations

The preserved SOLA line could not be applied safely by itself because
`_reconstruct_function()` returned a standalone `Function`, whose outer
evaluation does not check model-space membership. The clean implementation
attaches that result to `self._domain` before disabling nested kernel checks.

The historical compact-support test assumed identical numerical values when
integrating a smooth function on `[0.2, 0.8]` and a discontinuous zero-extended
function on `[0,1]` with the same point count. The restricted result matched the
analytic integral; the full-domain mesh missed the discontinuities. The revised
test keeps a tight analytic tolerance on the support-aware path, exact
consistency between identical global paths, and an explicit looser accuracy
bound on the discontinuous global path.

### Next phase

Run the Phase 5 validation ladder from import smoke tests through the full suite,
then add focused, repeatable benchmarks for the PLI-critical function, SOLA,
weighted Bessel, and KL paths. Record semantic equivalence before timing.
