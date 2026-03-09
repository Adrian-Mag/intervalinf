## Plan: Continuous DLI Property Inference

Refactor the DLI notebook so property-space inference is performed with the continuous model space and continuous operators directly, without introducing a Galerkin model discretization. This keeps the notebook aligned with the package message that property inference does not require a model-space inversion grid, while deferring literal `dim=0` support to a later library change.

**Phases 3**
1. **Phase 1: Remove Galerkin DLI Path**
    - **Objective:** Replace the notebook's discrete `EuclideanSpace(N)` model-space inference block with direct use of the continuous model space `M` and continuous operators `G` and `T`.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/pli_demos/dli.ipynb`
    - **Tests to Write:** Re-run the notebook inference cell and final output checks to confirm continuous DLI executes successfully.
    - **Steps:**
        1. Edit the DLI inference cell to remove `G.matrix(...)`, `T.matrix(...)`, `LinearOperator.from_matrix(...)`, and `M.to_components(...)` usage.
        2. Build `DualMasterCostFunction` and `ChambollePockSolver` directly from `M`, `G`, `T`, `model_prior_support`, and `data_error_support`.
        3. Run the updated notebook cells and confirm final admissible-property bounds and figures are produced.

2. **Phase 2: Make Notebook Basis-Free**
    - **Objective:** Configure the notebook model space so no explicit basis is used for inference.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/old_demos/pli_demos/dli.ipynb`
    - **Tests to Write:** Re-run the full notebook after changing the model-space construction to basis-free mode.
    - **Steps:**
        1. Change the model-space setup to use the supported basis-free `Lebesgue` configuration.
        2. Remove any remaining notebook logic that depends on coefficients or basis components.
        3. Run the notebook again and confirm property-space DLI remains correct.

3. **Phase 3: Evaluate Literal Zero-Dimension Support**
    - **Objective:** Determine and, if desired later, implement true `dim=0` support in the library rather than the notebook.
    - **Files/Functions to Modify/Create:** Likely `intervalinf/intervalinf/spaces/lebesgue.py`, `pygeoinf/pygeoinf/convex_optimisation.py`, and related tests.
    - **Tests to Write:** Solver and space tests for continuous operators with no model-basis bookkeeping.
    - **Steps:**
        1. Identify solver and space assumptions that require `model_space.dim > 0`.
        2. Define consistent semantics for infinite-dimensional or basis-free spaces in the current API.
        3. Implement and test those semantics before updating notebooks to use literal `dim=0`.

**Open Questions 2**
1. Should basis-free function spaces be represented long-term as `dim=None` or `dim=np.inf` rather than `dim=0`?
2. For future library support, should solver norm-estimation avoid any dependence on coefficient-space random initialization?
