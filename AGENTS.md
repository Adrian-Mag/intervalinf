# Windsurf/Cascade Instructions for intervalinf

<!-- BEGIN two-machine-workflow: managed by thesis/scripts/apply-agent-rules.py from thesis/docs/agent-docs/references/living/two-machine-workflow.md; edit the source and re-run -->

## Two-machine workflow: laptop + europa (author rules, 2026-09-16)

The author works on two machines the way a two-person team would:

- **laptop**: repos in `~/PhD/<repo>`. Laptop agents edit code here.
- **europa** (office PC, main machine): repos in `/disks/data/phd/<repo>`. These are full git clones,
  where the author and europa agents edit. Heavy jobs always run on europa.

Each machine has its own clone. **Git is the only way code moves between the two clones.** Never
copy code between them with rsync or scp. Never work on the same branch on both machines at once.

### Start of every session (both machines)
1. Run `git status`, then `git fetch <sync remote>` (see the table below). If the current branch is
   behind, run `git pull --ff-only`. If it has diverged, or the tree holds uncommitted changes you
   did not make, stop and ask the author.
2. Work on a task branch (`<area>/<task>`), not directly on `main`, unless the author says otherwise.
   Check with the author before taking a branch the other machine may be using.

### While working
3. Make small, coherent commits (WIP commits are fine). Follow the repo's commit conventions.
4. Code and small text files go through git. Large outputs, data, pickles, generated PDFs and
   build products stay gitignored and live on europa.

5. **Branch collision check.** A branch whose tip on the sync remote moved in the last 12 hours
   and was not pushed by you is in use on the other machine: ask the author before checking it
   out or committing to it.

### Before stopping or handing over to the other machine
6. Commit, then `git push <sync remote> <branch>` without force. Tell the author the branch name.
7. Merge into `main` only when the author asks, and only after the thesis or paper builds cleanly.

### Git network permission
This is a narrow exception to the offline and no-GitHub rules elsewhere in this file.
- **Allowed:** `git fetch`, `git pull --ff-only` and `git push` to this repo's sync remote in the
  table. Never `--force`, never delete remote branches, never push tags without the author's
  request.
- **Not allowed:** other remotes, `gh` commands (except `gh auth status`), uploads, or anything
  else the offline rules forbid.

| Repo | Sync remote | Notes |
|---|---|---|
| thesis | `origin` (Adrian-Mag/thesis) | private |
| BGP_Paper | `origin` (Adrian-Mag/BGP_Paper) | private |
| PLI_paper | `origin` (Adrian-Mag/PLI_Paper) | private |
| Inferences | `origin` (Adrian-Mag/Inferences) | private |
| Inferences/intervalinf, intervalinf-stabilize worktree | `origin` (Adrian-Mag/intervalinf) | **public**; work branches approved by the author |
| Inferences/pygeoinf and its worktrees | `backup` (Adrian-Mag/pygeoinf-backup) | private; **never push to `origin` (da380, third party)** |
| Every other repo (pygeoinf3D, AxiSEM3D and its sub-repos, Ascension, Adrian-Mag.github.io, control-testing, ...) | none | commit only; the author syncs |

`backup/wip-<date>-<machine>-<branch>` branches are written automatically by the hourly snapshot
script (`thesis/scripts/backup-snapshot.sh`). Never push, delete or base work on them.

### Jobs on europa
- **From the laptop:** `europa push` rsyncs this working tree to `~/data/PhD/<project>` on europa.
  Then use `europa submit`, `status`, `logs` and `pull` as described in the `europa` skill.
  `~/data/PhD` is a job area, not a clone: never edit code there.
- **On europa itself:** run `europa claim`, `submit`, `status` and `release` inside
  `/disks/data/phd/<project>`. This is local mode: there is no push or pull.
- Both sides share one job registry. Always `claim` before submitting and `release` afterwards.
- **Python env:** laptop-submitted jobs use `inferences2` as named in `.europa.yml`. Local-mode jobs
  on europa map `inferences2` to `/disks/data/envs/inferences-main`, an exact copy of the laptop
  `inferences` env.

### Results and watchers
- **Big outputs stay on europa** (backed up there); the laptop pulls a specific path with
  `europa pull <path>` when it needs one. Text outputs (logs, STATUS/REPORT notes, small JSON)
  are pulled or committed if the project tracks them.
- **Every job longer than about an hour gets a watcher agent on europa**, launched with
  `europa-watch` from a mission written from the watcher template. A watcher runs under
  `claude -p` and never ends its turn while the job lives. Details: the `europa` and
  `long-missions` skills (source `thesis/tools/claude-skills/`, installed on both machines by
  `thesis/scripts/install-claude-skills.sh`) and
  `references/living/two-machine-operations.md`.

### Not synced between machines
- **Zotero:** undecided (2026-09-16). Agents must not add, edit or delete Zotero items on either
  machine; read-only lookups are fine.
- **Claude memory:** separate on each machine. **Skills:** `europa` and `long-missions` are
  versioned in `thesis/tools/claude-skills/` and installed on both machines by
  `thesis/scripts/install-claude-skills.sh` (`--check` reports drift); other skills are separate.

<!-- END two-machine-workflow -->


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
