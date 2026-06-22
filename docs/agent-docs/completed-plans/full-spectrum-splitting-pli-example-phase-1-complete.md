## Phase 1 Complete: Spectral Inventory and Block-Indexing Infrastructure

Block-indexing layer for the full-spectrum PLI example is in place. Three tests confirm correct enumeration (even $s \le 4$, non-empty blocks only, real-harmonic $t$ convention) and clean partitioning of the data registry.

**Files created/changed:**
- `intervalinf/demos/old_demos/paper_demos/full_spectrum_utils.py` (new)
- `intervalinf/demos/old_demos/paper_demos/tests/__init__.py` (new, empty)
- `intervalinf/demos/old_demos/paper_demos/tests/test_full_spectrum_utils.py` (new)
- `intervalinf/demos/old_demos/paper_demos/example.ipynb` (new)

**Functions created/changed:**
- `BlockIndex` — frozen dataclass `(s: int, t: int)` for a real-harmonic $(s,t)$ spectral block
- `enumerate_blocks(reg, s_max)` — returns sorted list of non-empty `BlockIndex` for even $s \le s_{\max}$
- `block_data_split(reg, blocks)` — thin wrapper returning `{BlockIndex: NormalModeDataRegistry}` via `filter_by_st`

**Tests created/changed:**
- `test_enumerate_blocks_respects_smax_and_parity` — only even $s$ in $[0, s_{\max}]$, $t \in [0,2s]$
- `test_enumerate_blocks_skips_empty_blocks` — every returned block is non-empty; every covered $(s,t)$ pair appears
- `test_block_data_split_partitions_registry` — union of sub-registries equals full covered set; no duplicates

**Key finding during implementation:**
`t` in the `.new` data files is a **real-harmonic component index** $0 \le t \le 2s$ (not bipolar azimuthal $-s \le t \le s$). The plan spec assumed bipolar convention; the implementation was corrected to match the actual data format. All downstream phases should use `0 ≤ t ≤ 2s`.

**Review Status:** APPROVED (sort-key docstring nit fixed post-review: changed `(s, |t|, t)` key to `(s, t)` to match non-negative real-harmonic convention)

**Git Commit Message:**
```
feat(paper-demos): Phase 1 — block-indexing infrastructure for full-spectrum PLI

- Add BlockIndex frozen dataclass (s, t real-harmonic component index)
- Add enumerate_blocks(reg, s_max): even s, non-empty (s,t) blocks only
- Add block_data_split(reg, blocks): thin filter_by_st wrapper
- Add 3 TDD tests, all passing (1.09s)
- Add example.ipynb: spectral inventory + data-count bar plot
- Fix: t is real-harmonic index 0<=t<=2s, not bipolar azimuthal

Plan: intervalinf/docs/agent-docs/active-plans/full-spectrum-splitting-pli-example-plan.md
Phase: 1 of 8
Related: intervalinf/docs/agent-docs/completed-plans/full-spectrum-splitting-pli-example-phase-1-complete.md
```
