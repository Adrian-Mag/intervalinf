# Windsurf/Cascade Instructions for intervalinf

Historical Copilot-oriented package instructions are archived in `AGENTS_copilot.md`.

## Package Role

`intervalinf` provides concrete implementations of abstract `pygeoinf` Hilbert-space concepts on 1D intervals.

## Key Features

- Lebesgue and Sobolev spaces on 1D intervals.
- Differential operators such as Laplacian and gradient.
- Spectral methods and fast transforms.
- Hat/spline basis providers for FEM-style discretisation.
- Function support metadata propagation and compact-support arithmetic.

## Orientation

Before exploring source files, read every living reference:

```text
intervalinf/docs/agent-docs/references/living/*-reference.md
```

Currently expected:

```text
intervalinf/docs/agent-docs/references/living/intervalinf-reference.md
```

Never use `references/legacy/` unless explicitly requested.

## Plan Directory

Use:

```text
intervalinf/docs/agent-docs/
```

Expected subdirectories are `active-plans/`, `completed-plans/`, `references/`, and `theory/`.

## Development Rules

- Use Python >= 3.11; the workspace default conda environment is `inferences`.
- Core dependencies are numpy, scipy, and matplotlib.
- Run package tests with `python -m pytest tests/` from the `intervalinf/` directory.
- Use `np.testing.assert_allclose` with explicit tolerances for numerical tests.
- Update living references after code changes.
- Follow the workspace commit convention in `../COMMIT_CONVENTION.md`.
