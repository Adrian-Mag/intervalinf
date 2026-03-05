# Agent Configuration for intervalinf

## Plan Directory
`docs/agent-docs/`

All agent-oriented materials live in this directory:
- **`docs/agent-docs/active-plans/`** — In-progress plans
- **`docs/agent-docs/completed-plans/`** — Finished projects (reference archive)
- **`docs/agent-docs/references/`** — Research reports, exploration findings
- **`docs/agent-docs/theory/`** — Theory documents and research papers

## Package Context
**intervalinf** provides concrete implementations of the abstract `HilbertSpace` interface from `pygeoinf`, operating on 1D intervals.

### Key Features
- Lebesgue (L²) and Sobolev spaces on 1D intervals
- Differential operators: Laplacian, gradient (Bessel-based)
- Spectral methods with fast transforms
- Hat/spline basis providers for FEM-style discretisation
- Function support metadata propagation (compact-support arithmetic)
- **pygeoinf depends on intervalinf for concrete spaces in examples**

### Python Requirements
- Python ≥ 3.11
- Core dependencies: numpy, scipy, matplotlib
- Dev dependencies: pytest, ruff
- Install: `pip install -e .[dev]`

### Running Tests
```bash
cd intervalinf
python -m pytest tests/
```

## Theory Documents
Mathematical foundations shared with pygeoinf (same paper library):
- **`docs/agent-docs/theory/theory.txt`** — Main theory document "DLI as Convex Analysis problems"
- **`docs/agent-docs/theory/`** — 18 PDF research papers

## Package Quick References
All files matching `docs/agent-docs/active-plans/*-reference.md` are **condensed reference documents** that agents **must read first** before exploring individual source files.

**Read rule:** Before exploring any source files, read every `*-reference.md` in `docs/agent-docs/active-plans/`.
- If reference files exist → use them; only read individual source files for details they don't cover.
- If no reference files exist → proceed with normal file exploration.

**Update rule:** After changes, **update all `*-reference.md` files** to reflect additions, removals, or modifications. Stale references are actively harmful.

## Related Packages
**pygeoinf** at `../pygeoinf/`
- Abstract Hilbert space framework that intervalinf implements
- Convex analysis, support functions, inversion algorithms
- **intervalinf provides concrete spaces for pygeoinf examples**

## Commit Message Convention
Follow the workspace-level convention in `../COMMIT_CONVENTION.md`.
Always include `Plan:` and `Phase:` fields in feature/fix commits.
