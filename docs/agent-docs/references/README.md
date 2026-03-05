# References

This folder contains **living reference documents** and **legacy archives** for agent-driven development in intervalinf.

## Structure

### `living/`
**Active, maintained architecture and API references.**

- `intervalinf-reference.md` — Current package architecture, spaces, operators, providers, APIs
  - **Agents must read this FIRST before exploring source files**
  - Updated: immediately after code changes (by Sisyphus-subagent)
  - Lifecycle: never becomes stale (updated every phase)

### `legacy/`
**Archived artifacts: superseded plans and research reports.**

Currently empty — will accumulate as plans complete.

---

## For Agents

### Reading Strategy
1. **Always start with:** `living/intervalinf-reference.md`
2. **Then check:** `../active-plans/intervalinf-convex-update-plan.md` for current work
3. **Consult legacy only if:** Debugging past design decisions

### Updating Living References
After implementing features:
1. Open `living/intervalinf-reference.md`
2. Update affected sections (new classes, changed signatures, new files, patterns)
3. Add timestamp: `<!-- Last updated: YYYY-MM-DD by Sisyphus -->`
4. Commit with plan reference

### Archiving Artifacts
When a plan completes or research is superseded:
1. Move to `legacy/<category>/`
2. Add deprecation notice explaining archive reason
3. Commit: `chore: archive <name> to legacy/`

---

**See also:** [Agent Docs Index](../index.md)
