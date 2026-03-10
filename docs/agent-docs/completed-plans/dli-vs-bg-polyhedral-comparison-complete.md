## Plan Complete: DLI vs BG Polyhedral Comparison

This plan delivered a reproducible comparison notebook that builds DLI and Backus-Gilbert admissible regions on the same synthetic problem, approximates both regions with the same property-space directional sampling budget, and overlays the resulting polyhedra exactly. The final notebook includes comparative diagnostics, figure export, reproducibility fixes, and living-reference updates so the demo inventory reflects the finished comparison workflow.

**Phases Completed:** 3 of 3
1. ✅ Phase 1: Create Shared Setup and DLI Polyhedral Path
2. ✅ Phase 2: Add BG Polyhedral Path and Exact Overlay Plot
3. ✅ Phase 3: Add Comparative Diagnostics and Final Polish

**All Files Created/Modified:**
- intervalinf/demos/convex_analysis/dli_vs_bg_polyhedral_comparison.ipynb
- intervalinf/docs/agent-docs/references/living/intervalinf-reference.md

**Key Functions/Classes Added:**
- Notebook cells constructing the shared DLI/BG setup
- Notebook cells constructing DLI and BG polyhedral admissible regions
- Notebook cells for LP-based width diagnostics and exported comparison figures

**Test Coverage:**
- Total tests written: notebook validation and assertion cells across all 3 phases
- All tests passing: ✅

**Recommendations for Next Steps:**
- Use the notebook as the reference demo when comparing DLI and BG admissible geometry under matched property-space angular sampling.
- If the demo is expanded further, keep the shared-setup cells aligned with the existing DLI notebook and update the living reference in the same change.