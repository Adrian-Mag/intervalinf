# intervalinf

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

Function spaces and operators on one-dimensional intervals, built on
[pygeoinf](https://github.com/da380/pygeoinf).

`intervalinf` provides continuous and basis-backed representations of functions
on intervals. Its main components are Lebesgue and Sobolev spaces, spectral and
finite-difference operators, weighted radial spaces, SOLA operators, and
Karhunen-Loeve sampling.

## Installation

`intervalinf` requires Python 3.12 or newer and is currently installed from a
source checkout:

```bash
git clone https://github.com/Adrian-Mag/intervalinf.git
cd intervalinf
python -m pip install -e ".[dev]"
```

Basis-free spaces require pygeoinf's functional vector-update contract, in
which `HilbertSpace.ax()` and `HilbertSpace.axpy()` return the updated vector.
That contract is newer than pygeoinf 1.8.2 and has not yet appeared in a tagged
release. Until it is released, install intervalinf alongside a pygeoinf source
checkout that contains the contract. Basis-free construction checks this at
runtime and fails with an explicit compatibility error instead of silently
returning incorrect results.

## Continuous Functions

No finite basis is required for direct function evaluation and integration:

```python
import numpy as np

from intervalinf import Function, IntervalDomain, Lebesgue

domain = IntervalDomain(0.0, 1.0)
space = Lebesgue(0, domain, basis=None)

f = Function(space, evaluate_callable=lambda x: x**2)
g = Function(space, evaluate_callable=lambda x: x**3)

inner_product = space.inner_product(f, g)
np.testing.assert_allclose(inner_product, 1.0 / 6.0, rtol=1e-6, atol=1e-10)
```

### Declared discontinuities and split quadrature

`Function.breakpoints` records conservative interior integration splits. It
does not try to discover jumps in arbitrary callables, but boxcar and random
step providers declare their own joins and function arithmetic preserves their
union. Use the opt-in split Gauss--Legendre path when a quadrature rule must
not sample a pointwise convention at a discontinuity:

```python
from intervalinf import Function, IntegrationConfig, QuadratureRule

step = Function(
    space,
    evaluate_callable=lambda x: (np.asarray(x) >= 0.5).astype(float),
    breakpoints=(0.5,),
)
rule = QuadratureRule.split_gauss_legendre(
    domain, breakpoints=step.breakpoints, n_points=32
)
integral = step.integrate(quadrature_rule=rule)
np.testing.assert_allclose(integral, 0.5, rtol=1e-14, atol=1e-14)

split_space = Lebesgue(
    0,
    domain,
    basis=None,
    integration_config=IntegrationConfig(
        method="split_gauss_legendre", n_points=32
    ),
)
```

For a discrete SOLA adjoint test, pass the *same* explicit rule to
`SOLAOperator.apply_with_quadrature_rule` and
`Lebesgue.inner_product(..., quadrature_rule=rule)`. This is separate from the
continuous mathematical adjoint and currently provides the exact discrete
identity for ordinary (unweighted) `Lebesgue` spaces.

Set a positive dimension and choose a basis when coefficients or spectral
operators are needed:

```python
import numpy as np

from intervalinf import BoundaryConditions, Function, IntervalDomain, Lebesgue
from intervalinf.operators import Laplacian

domain = IntervalDomain(0.0, np.pi)
space = Lebesgue(32, domain, basis="sine")
f = Function(space, evaluate_callable=np.sin)

laplacian = Laplacian(space, BoundaryConditions.dirichlet())
laplacian_f = laplacian(f)
np.testing.assert_allclose(laplacian_f(np.pi / 2.0), 1.0, rtol=1e-6, atol=1e-10)
```

Operators are called on vectors with `operator(vector)`. The `@` operator is
reserved for composing operators.

## Weighted Radial Spaces

`WeightedLebesgue` represents weighted inner products through pygeoinf's mass
operator abstraction. For a spherical radial coordinate, the weight is `r**2`:

```python
from intervalinf import BoundaryConditions, IntervalDomain, WeightedLebesgue
from intervalinf.operators import RadialLaplacian

radial_domain = IntervalDomain(1.0, 2.0)
radial_space = WeightedLebesgue(
    0,
    radial_domain,
    weight=lambda r: r**2,
    inverse_weight=lambda r: 1.0 / r**2,
    basis=None,
)
radial_laplacian = RadialLaplacian(
    radial_space,
    BoundaryConditions.dirichlet(),
    1.0,
    dofs=16,
    ell=2,
)
assert radial_laplacian.get_eigenvalue(0) > 0.0
```

Use a strictly positive lower radius, or provide a regularized inverse weight,
when `1 / r**2` would otherwise be singular at the origin.

## Public Imports

Core objects and spaces are available from `intervalinf`. Operators and
sampling utilities are exposed by their subpackages:

```python
from intervalinf import Function, IntervalDomain, Lebesgue, Sobolev
from intervalinf.operators import BesselSobolev, Laplacian, SOLAOperator
from intervalinf.sampling import KLSampler
```

The notebooks in [`demos/`](demos/) provide longer examples.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
