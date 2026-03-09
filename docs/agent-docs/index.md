# Agent Documentation

This folder contains planning documents, research summaries, and theory materials intended for **agent-driven development** in intervalinf. User-facing documentation lives in the parent `docs/` folder (if present) or the repository root `README.md`.

## Structure

### `active-plans/`
Plans currently under development or awaiting implementation.

- **Purpose:** Real-time tracking of in-progress work
- **Lifecycle:** Plans move to `completed-plans/` when finished

### `completed-plans/`
Archived plans and phase summaries from finished projects.

- **Purpose:** Historical record, knowledge base, git commit traceability
- **Retention:** Keep indefinitely; useful for understanding past design decisions
- **Current contents:** SOLA speedup phase-complete records and the intervalinf agent-docs audit Phase 1 completion record

### `references/`
Exploration reports, research summaries, and **living reference documents**.

- **Purpose:**
  - Intermediate findings used to inform planning
  - **Living architecture references** (`*-reference.md`) — describe package structure, class hierarchies, APIs
- **Key file:** `references/living/intervalinf-reference.md` — agents **must read this first** before exploring source files
- **Retention:** Keep indefinitely; references especially valuable for agent learning

### `theory/`
Research papers and main theory document (`theory.txt`).

- **Purpose:** Mathematical foundations for implementation validation
- **Contents:** 19 PDF papers on inverse problems, Backus-Gilbert methods, convex analysis
- **Audience:** Theory-Validator-subagent, developers needing mathematical context

---

## How Agents Use This Folder

1. **Conductor (Atlas)**: Reads `references/living/intervalinf-reference.md` first; writes plans to `active-plans/`
2. **Sisyphus (Implementer)**: Reads reference file before coding; updates it after changes
3. **Oracle (Researcher)**: Consults `theory/theory.txt` and PDFs for mathematical context
4. **Theory-Validator**: Reads theory docs to validate correctness of operators and spaces
5. **Code-Review**: Checks phase-complete summaries against implemented code

## Quick Start for Agents

```
1. Read: docs/agent-docs/references/living/intervalinf-reference.md
2. Check active plan: docs/agent-docs/active-plans/intervalinf-convex-update-plan.md
3. Consult theory: docs/agent-docs/theory/theory.txt
```

---

**Last Updated:** 2026-03-09
