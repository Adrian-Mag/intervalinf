# intervalinf Demos

Interactive Jupyter notebooks demonstrating the capabilities of the `intervalinf` package
for interval-based function spaces and probabilistic inference.

## Demo Organization

### 1. Foundations
- **01_interval_domain.ipynb** - Intervals, domains, and integration
- **02_functions.ipynb** - Creating and manipulating functions on intervals
- **03_function_arithmetic.ipynb** - Function operations and calculus

### 2. Function Spaces
- **04_lebesgue_spaces.ipynb** - L² spaces and basis representations
- **05_sobolev_spaces.ipynb** - Sobolev spaces and smoothness priors
- **06_weighted_spaces.ipynb** - Mass-weighted inner products

### 3. Operators
- **07_laplacian.ipynb** - Laplacian operator and boundary conditions
- **08_inverse_laplacian.ipynb** - Green's functions and covariance operators
- **09_bessel_sobolev.ipynb** - Bessel potential operators

### 4. Probabilistic Linear Inference
- **10_gaussian_measures.ipynb** - Gaussian measures on function spaces
- **11_kl_expansion.ipynb** - Karhunen-Loève sampling
- **12_linear_inference.ipynb** - Bayesian inference with linear observations

### 5. Advanced Topics
- **13_discontinuities.ipynb** - Handling discontinuous functions
- **14_boundary_conditions.ipynb** - Advanced boundary condition handling
- **15_radial_operators.ipynb** - Spherical coordinate operators

## Running the Demos

Ensure you have `intervalinf` installed:

```bash
pip install -e /path/to/intervalinf
```

Then launch Jupyter:

```bash
jupyter lab demos/
```

## Prerequisites

Each notebook includes the necessary imports. The package dependencies are:
- numpy
- scipy
- matplotlib (for visualization)
