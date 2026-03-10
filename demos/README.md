# intervalinf Demos

Interactive Jupyter notebooks demonstrating the capabilities of the `intervalinf` package
for interval-based function spaces and probabilistic inference.

## Core Package Notebooks

| Notebook | Topic |
| :------- | :---- |
| `1_interval_domain_demo.ipynb` | `IntervalDomain` basics: meshing, integration, subdomain operations |
| `2_functions_demo.ipynb` | Creating and evaluating `Function` objects in standalone and space-attached modes |
| `3_lebesgue_space_demo.ipynb` | `Lebesgue` L² space construction, inner products, and basis representations |
| `3.1_kernel_functionals_demo.ipynb` | Kernel-based linear forms (`LinearFormKernel`) |
| `4_function_and_basis_providers_demo.ipynb` | Basis/eigenvalue providers (sine, cosine, hat, FEM, smooth) |
| `5_gradient_operator_demo.ipynb` | `Gradient` operator: construction, adjoint, and spectral properties |
| `6_laplacian_operator_demo.ipynb` | `Laplacian` and `InverseLaplacian`: Green's functions and covariance operators |

## Convex-Analysis / DLI Demos (`convex_analysis/`)

End-to-end workflow notebooks combining `intervalinf` operators with `pygeoinf`'s
convex-analysis inversion stack.

| Notebook | Topic |
| :------- | :---- |
| `convex_analysis/dli.ipynb` | Deterministic Linear Inference (DLI) via dual proximal-bundle optimization |
| `convex_analysis/dli_vs_bg_polyhedral_comparison.ipynb` | Side-by-side DLI vs Backus-Gilbert admissible-region comparison with configurable `N_d`, `N_p` |
| `convex_analysis/bg_with_errors_minkowski.ipynb` | Backus-Gilbert admissible regions with data errors using Minkowski-sum support-function algebra |
| `convex_analysis/bg_with_errors_minkowski_multi_nd.ipynb` | Multi-`N_d` comparison of BG admissible regions |

## Running the Demos

Ensure you have `intervalinf` and `pygeoinf` installed:

```bash
conda activate inferences3
# or: pip install -e /path/to/intervalinf -e /path/to/pygeoinf
```

Then launch Jupyter from the project root:

```bash
jupyter lab intervalinf/demos/
```

## Prerequisites

Each notebook includes the necessary imports. Core dependencies:
- numpy
- scipy
- matplotlib (for visualization)
- pygeoinf (required by convex-analysis notebooks)
