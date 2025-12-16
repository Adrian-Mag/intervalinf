# intervalinf

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Function spaces and operators on 1D intervals** - built on [pygeoinf](https://github.com/yourusername/pygeoinf).

## Overview

`intervalinf` provides concrete implementations of Hilbert spaces for functions defined on 1D intervals. It is designed with a **continuous-first philosophy**: discretization is optional, not forced.

### Key Features

- **Lebesgue (L²) and Sobolev spaces** on intervals with various boundary conditions
- **Differential operators**: Laplacian, gradient, Bessel-Sobolev
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
