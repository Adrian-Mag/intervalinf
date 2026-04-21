## Plan: Synthetic vp vs DLI Notebook

Create a new convex-analysis demo notebook that sits between `dli.ipynb` and `realistic_dli.ipynb`: synthetic and lightweight, but using a two-block direct-sum model `vp ⊕ vs` with block forward and property operators. The notebook will use `NormalModesProvider` only, keep the prior and data-confidence sets simple via `BallSupportFunction`, and provide a clear end-to-end DLI example with plots and solver output.

**Phases 3 phases**
1. **Phase 1: Build notebook scaffold and block spaces**
    - **Objective:** Create a new notebook with imports, numerical configuration, mathematical setup, and a two-component direct-sum model space.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/synthetic_vp_vs_dli.ipynb
    - **Tests to Write:** Validate the notebook JSON and confirm the setup cells define `M_vp`, `M_vs`, `M_model`, `D`, and `P` cleanly.
    - **Steps:**
        1. Create a notebook with an introduction explaining the synthetic vp/vs direct-sum setup.
        2. Define a common interval domain and two basis-free Lebesgue spaces `M_vp` and `M_vs`.
        3. Assemble `M_model = HilbertSpaceDirectSum([M_vp, M_vs])`, together with the data and property spaces.

2. **Phase 2: Add synthetic operators and synthetic data**
    - **Objective:** Build a block forward operator and property operator using synthetic normal modes and generate reproducible synthetic data.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/synthetic_vp_vs_dli.ipynb
    - **Tests to Write:** Confirm `G(m_bar)` and `T(m_bar)` return arrays of the expected shape and that plots render for kernels and synthetic data.
    - **Steps:**
        1. Build `G_vp` from `NormalModesProvider` and `G_vs` from `NormalModesProvider` with a different seed to make the setup slightly richer than `dli.ipynb`.
        2. Build `T_vp` from `BumpFunctionProvider` and `T_vs` from `NullFunctionProvider`, then assemble block operator `T`.
        3. Create a seeded synthetic true model for both `vp` and `vs`, generate noisy data, and visualize kernels and observations.

3. **Phase 3: Solve the DLI problem and document the demo**
    - **Objective:** Solve the deterministic linear inference problem using simple ball support functions, present prior/posterior property bounds, and update the living reference.
    - **Files/Functions to Modify/Create:** intervalinf/demos/convex_analysis/synthetic_vp_vs_dli.ipynb, intervalinf/docs/agent-docs/references/living/intervalinf-reference.md
    - **Tests to Write:** Validate that the notebook parses cleanly, the DLI solve cells are syntactically sound, and the final results section reports posterior bounds and reduction factors.
    - **Steps:**
        1. Use `BallSupportFunction` for both the model prior and data-confidence set.
        2. Solve upper and lower support values in the directions `±e_i` with `ProximalBundleMethod`.
        3. Plot prior-vs-posterior bounds, summarize the results, and update the living reference to include the new notebook.

**Open Questions 0 questions**
