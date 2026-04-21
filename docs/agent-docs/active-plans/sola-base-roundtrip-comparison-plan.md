## Plan: sola-base Roundtrip Comparison

Refactor the `sola_demo` notebook so the discrete model uses the same physical cell basis as `sola-base`, then export the resulting discrete problem to `sola-base`, run the external solver, and compare all directly comparable outputs against the `intervalinf` and `pygeoinf` solution.

**Phases 4 phases**
1. **Phase 1: Align Discrete Basis**
    - **Objective:** Replace the orthonormal boxcar discretization in the notebook with a physical cell basis plus a mass-weighted discrete model space.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/sola-base_demo/sola_demo.ipynb`
    - **Tests to Write:** Re-run the discrete setup cell and the discrete solve cell to confirm unimodularity and discrete reconstruction still work.
    - **Steps:**
        1. Rewrite the discretization markdown to describe the physical cell basis and mass matrix.
        2. Build a dedicated discrete Lebesgue cell space and a `MassWeightedHilbertSpace` coefficient space.
        3. Rebuild `G_disc` and `T_disc` using `LinearOperator.from_formal_adjoint`.

2. **Phase 2: Export sola-base Inputs**
    - **Objective:** Create the exact `sola-base` input bundle from the notebook’s discrete problem.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/sola-base_demo/sola_demo.ipynb`
    - **Tests to Write:** Validate the exported matrix and vector shapes and file existence for `G`, `d`, `dstd`, `V`, `T/T_k`, `ks_to_solve`, and `parameters.cfg`.
    - **Steps:**
        1. Add notebook cells to create a clean export folder inside `sola-base`.
        2. Write the sparse and dense file formats expected by `sola-base`.
        3. Derive and write the `eta` value used for the comparable run.

3. **Phase 3: Run sola-base Pipeline**
    - **Objective:** Execute preprocessing, LSQR inversion, postprocessing, and averaging-kernel generation on the exported problem.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/sola-base_demo/sola_demo.ipynb`
    - **Tests to Write:** Execute the run cell and check that `sola-base` produces `SOLUTION_k.txt`, `m_<eta>`, `mstd_<eta>`, and `A/A_k` outputs.
    - **Steps:**
        1. Compile the `sola-base` binaries if they are missing.
        2. Run `sola_preproc.py`, `sola_lsqr.py`, `sola_postproc.py`, and `sola_ak.py` from the notebook.
        3. Confirm the output files are present and loadable.

4. **Phase 4: Compare Outputs**
    - **Objective:** Compare notebook and `sola-base` outputs to confirm the exported problem is the same discrete SOLA problem.
    - **Files/Functions to Modify/Create:** `intervalinf/demos/sola-base_demo/sola_demo.ipynb`, `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
    - **Tests to Write:** Numerical comparisons with explicit tolerances for property means, uncertainties, averaging kernels, and data-space weights.
    - **Steps:**
        1. Load `sola-base` outputs in the notebook.
        2. Compare them against `p_mean_disc`, `p_std_disc`, and the notebook’s resolving kernels.
        3. Update the living reference to mention the new round-trip demo coverage.

**Open Questions 2 questions**
1. Use the notebook itself as the full export and run driver, or split file-writing into a helper script later if the cell becomes too long?
2. Treat solver agreement as strict numerical tolerance rather than bitwise identity because `sola-base` uses LSQR while the notebook uses dense direct linear algebra?