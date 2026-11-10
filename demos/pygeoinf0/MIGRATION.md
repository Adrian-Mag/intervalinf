# intervalinf demos on pygeoinf0

| old | new | status |
|---|---|---|
| `../convex_analysis/dli.ipynb` (intervalinf + pygeoinf 1.x: `SOLAOperator`, `BallSupportFunction`, `PrimalKKTSolver`) | `dli_pygeoinf0.py` | ported and verified (below) |

The old notebook is kept unchanged. The rest of the demos are listed, with their pygeoinf0 equivalents, in the
migration inventory (europa, 2026-11-11, `/disks/data/tmp/migration-inventory-other.md`).

## dli.ipynb → dli_pygeoinf0.py

The port builds the same problem with the same seeds:
- the model space L2([0, 1]);
- 50 "normal-mode" data kernels, using NormalModesProvider's formulas and its seed 2, including the up to 50 modes and
  Gaussian envelopes of width 1–5 %;
- 20 unit-mass bump targets of support width 0.2;
- the true model, and the noise draw with seed 42;
- the model ball, centred on m0(x) = x with radius 1.05 ||m_bar − m0||, and the data ball of radius 1.05 ||noise||.

The bounds come from one data-space decomposition: `pg.decompose(G, D.gaussian()).ellipsoid_intervals(T, a, r,
datum, centre)`. Each end carries a certificate, which is EXACT when the primal and dual values agree.

**Measured** (europa, 2026-11-11; `migration/compare.py`, optionally given a re-run from `migration/rerun_old.py`):

| quantity | notebook (saved output) | pygeoinf0 |
|---|---|---|
| properties inside their intervals | 20/20 | 20/20 |
| mean absolute error of the interval centres | 0.2479 | 0.2479 |
| RMS error of the centres | 0.2836 | 0.2836 |
| largest half-width | 1.2711 | 1.2711 |
| mean half-width | 0.9999 | 0.9999 |
| data radius / model radius | 0.7685 / 0.5753 | 0.7685 / 0.5753 |
| prior-only width | 2.989 | 2.9894 |
| certificates | 40/40 KKT solves converged | all 40 ends EXACT |
| time | 21.3 s (saved), 14.1 s (re-run) | 0.6 s |

The port reproduces every number the notebook printed.

**The old stack today gives different bounds.** Re-running the notebook on today's intervalinf (9a12d26) and
pygeoinf runtime returns intervals with a mean half-width of 1.4920. That is almost the prior-only half-width (1.4947),
as if the data constraint had dropped out, although every KKT solve reports convergence. The rest of the re-run agrees
with the port to quadrature accuracy: the data to 9e-8, the model radius to 2e-9, the prior bounds to 3e-8. So the
difference is in the old KKT route, not in the problem. That is a regression in the old libraries since the notebook was
saved. pygeoinf0 agrees with the saved, published output.

**Notes for porting more of these demos.**
- The two end targets touch the boundary of [0, 1]. The function bank's `bump` refuses that (its support must lie
  inside), so the port builds the targets from their formula with `breaks=` at the support ends.
- `support=` on an interval fails in pygeoinf0 at main 8a6ec56: an error in the Legendre fit's scout. It's reported in
  the migration notes as a pygeoinf0 gap.
- The notebook's markdown describes an r²-weighted L2 space, but its code uses plain L2([0, 1]), and so does the port.
  In pygeoinf0 the weighted space would be `pg.L2(pg.LebesgueMeasure(I).with_density(lambda r: r**2))`.
