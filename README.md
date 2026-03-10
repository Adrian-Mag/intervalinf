# intervalinf

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Function spaces and operators on 1D intervals** - built on [pygeoinf](https://github.com/yourusername/pygeoinf).

## Overview

`intervalinf` provides concrete implementations of Hilbert spaces for functions defined on 1D intervals. It is designed with a **continuous-first philosophy**: discretization is optional, not forced.

### Key Features

- **Lebesgue (L²) and Sobolev spaces** on intervals with various boundary conditions
- **Differential operators**: Laplacian, gradient, Bessel-Sobolev
- **SOLA operator**: optimized kernel-averager for Deterministic Linear Inference (DLI)
- **Spectral methods** with fast transforms (DST, DCT, DFT)
- **FEM solvers** for elliptic PDEs
- **KL expansion sampling** for Gaussian measures
- **Radial operators** for spherical geometry applications

### Design Philosophy

Unlike many discretization frameworks, `intervalinf` allows functions to be represented **continuously**:

```python
from intervalinf import IntervalDomain, Lebesgue, Function

# Create a domain and space WITHOUT discretization
domain = IntervalDomain(0, 1)
space = Lebesgue(function_domain=domain, basis='none')

# Functions are callables, not coefficient vectors
f = Function(space, evaluate_callable=lambda x: x**2)

# Inner products computed via numerical integration, not matrix multiplication
g = Function(space, evaluate_callable=lambda x: x**3)
ip = space.inner_product(f, g)  # ∫₀¹ x² · x³ dx = 1/6
```

When you **need** a basis (for operators, inference, etc.), you can opt in:

```python
# Opt into discretization when needed
space = Lebesgue(dim=50, function_domain=domain, basis='sine')
```

## Installation

```bash
pip install intervalinf
```

Or for development:

```bash
git clone https://github.com/yourusername/intervalinf.git
cd intervalinf
pip install -e ".[dev]"
```

## Quick Start

```python
from intervalinf import IntervalDomain, Lebesgue, Function, Laplacian
import numpy as np

# Define the domain [0, π]
domain = IntervalDomain(0, np.pi)

# Create L² space with sine basis (Dirichlet BCs)
space = Lebesgue(dim=50, function_domain=domain, basis='sine')

# Create a function
f = Function(space, evaluate_callable=np.sin)

# Apply the Laplacian
L = Laplacian(space)
Lf = L @ f  # -sin(x)
```

## Package Structure

```
intervalinf/
├── core/           # Domain, boundary conditions, Function class
├── spaces/         # Lebesgue, Sobolev spaces
├── operators/      # Laplacian, gradient, Bessel-Sobolev, SOLA
├── providers/      # Basis function providers (sine, cosine, hat, etc.)
├── sampling/       # KL expansion sampler
└── utils/          # Utilities
```

## Demos

The `demos/` directory contains Jupyter notebooks illustrating the package's features:

- **`demos/1_interval_domain_demo.ipynb`** — domain basics, meshing and integration
- **`demos/2_functions_demo.ipynb`** — creating and evaluating `Function` objects
- **`demos/3_lebesgue_space_demo.ipynb`** — L² space construction and inner products
- **`demos/3.1_kernel_functionals_demo.ipynb`** — kernel-based linear forms
- **`demos/4_function_and_basis_providers_demo.ipynb`** — basis providers (sine, cosine, hat)
- **`demos/5_gradient_operator_demo.ipynb`** — gradient operator
- **`demos/6_laplacian_operator_demo.ipynb`** — Laplacian and inverse Laplacian

### Convex-Analysis / DLI Demos

`demos/convex_analysis/` contains end-to-end workflow notebooks for the
combined `intervalinf` + `pygeoinf` inversion stack:

- **`dli.ipynb`** — Deterministic Linear Inference via dual proximal-bundle optimization
- **`dli_vs_bg_polyhedral_comparison.ipynb`** — side-by-side DLI vs Backus-Gilbert admissible-region comparison
- **`bg_with_errors_minkowski.ipynb`** — Backus-Gilbert with data errors using Minkowski-sum support-function algebra
- **`bg_with_errors_minkowski_multi_nd.ipynb`** — multi-N_d comparison of BG admissible regions

## Relationship to pygeoinf

`intervalinf` is built on top of `pygeoinf` and provides concrete implementations of its abstract base classes:

| pygeoinf (abstract) | intervalinf (concrete) |
|---------------------|------------------------|
| `HilbertSpace` | `Lebesgue` |
| `MassWeightedHilbertSpace` | `Sobolev` |
| `LinearOperator` | `Laplacian`, `Gradient`, etc. |
| `LinearForm` | `LinearFormKernel` |

## License

BSD-3-Clause
