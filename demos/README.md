# intervalinf Demos

These Jupyter notebooks demonstrate the public `intervalinf` API. Install the
development dependencies from the repository root before launching Jupyter:

```bash
python -m pip install -e ".[dev]"
jupyter lab demos/
```

## Foundations

- `1_interval_domain_demo.ipynb`: interval construction, meshing, and
  integration.
- `2_functions_demo.ipynb`: callable and coefficient-backed functions.
- `3_lebesgue_space_demo.ipynb`: Lebesgue spaces and basis representations.
- `3.1_kernel_functionals_demo.ipynb`: kernel-defined linear forms.
- `4_function_and_basis_providers_demo.ipynb`: function and spectral provider
  APIs.

## Operators

- `5_gradient_operator_demo.ipynb`: gradient construction and application.
- `6_laplacian_operator_demo.ipynb`: Laplacian boundary conditions and spectral
  application.

`model_fusion/first_test.ipynb` is an experimental model-fusion notebook rather
than part of the ordered introductory sequence.
