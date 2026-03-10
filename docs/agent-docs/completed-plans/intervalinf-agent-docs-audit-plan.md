## Plan: intervalinf agent-docs audit refresh

Audit and reconcile the agent-oriented documentation for `intervalinf` so that entrypoint guidance, plan metadata, and living references all match the current package layout. The work is split into small documentation-only phases so each step leaves the agent-doc system more reliable for future development.

**Phases 3**
1. **Phase 1: Fix agent-doc navigation rules**
    - **Objective:** Make the main agent entrypoints agree on where living references reside and what agents should read first.
    - **Files/Functions to Modify/Create:** `intervalinf/AGENTS.md`, `intervalinf/docs/agent-docs/index.md`, `intervalinf/docs/agent-docs/references/README.md`
    - **Tests to Write:** No code tests; validate by cross-checking paths and instructions against the current docs tree.
    - **Steps:**
        1. Identify contradictory path guidance in the package agent entrypoints.
        2. Update the read rules and structure descriptions to point to `docs/agent-docs/references/living/intervalinf-reference.md`.
        3. Re-read the edited docs to confirm they agree with the current folder structure.

2. **Phase 2: Reconcile stale plan/archive status**
    - **Objective:** Remove outdated status statements and align the convex-update plan with its own changelog.
    - **Files/Functions to Modify/Create:** `intervalinf/docs/agent-docs/index.md`, `intervalinf/docs/agent-docs/active-plans/intervalinf-convex-update-plan.md`
    - **Tests to Write:** No code tests; validate by comparing documented status with existing files and changelog entries.
    - **Steps:**
        1. Update stale statements about completed plans and archive contents.
        2. Mark convex-update Phase 4 complete, consistent with the recorded reference updates.
        3. Re-check plan metadata for internal consistency.

3. **Phase 3: Refresh living reference audit notes**
    - **Objective:** Improve the living reference with the missing test-layout guidance and a final consistency pass.
    - **Files/Functions to Modify/Create:** `intervalinf/docs/agent-docs/references/living/intervalinf-reference.md`
    - **Tests to Write:** No code tests; validate by checking that the architecture overview reflects the actual source and test tree.
    - **Steps:**
        1. Add a concise tests-tree overview for agent navigation.
        2. Reconcile any remaining package-structure mismatches discovered during the audit.
        3. Re-read key sections for consistency with the current codebase.

**Open Questions 1**
1. Convex-update Phase 4 status? Mark complete based on logged reference updates / keep open for future reference work