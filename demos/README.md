# intervalinf Demos

These Jupyter notebooks demonstrate the public `intervalinf` API. Install the
demo dependencies from the repository root before launching Jupyter:

```bash
python -m pip install -e ".[demos]"
jupyter lab demos/
```

The notebooks are also safe to execute headlessly and sequentially:

```bash
MPLBACKEND=Agg python -m nbconvert \
  --to notebook --execute --ExecutePreprocessor.timeout=120 \
  --output-dir=/tmp/intervalinf-demos \
  demos/*.ipynb demos/model_fusion/*.ipynb
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
