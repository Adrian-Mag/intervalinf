"""Compare the pygeoinf0 DLI port with the notebook's saved output and with a re-run of the old notebook.

    python demos/pygeoinf0/migration/compare.py [old_dli.npz]

The notebook's saved summary (demos/convex_analysis/dli.ipynb, last cell) is read from the .ipynb itself; old_dli.npz
(optional) holds a re-run of the notebook's cells (lower, upper, prior_lower, prior_upper, true_props, d_bar, d_tilde,
data_radius, model_radius).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(here))
import dli_pygeoinf0 as dl  # noqa: E402

pb = dl.build()
res, plo, phi = dl.bounds(pb)
lo, hi = np.asarray(res.lower), np.asarray(res.upper)
true = pb["T"](pb["m_bar"]).values
c, h = (lo + hi) / 2, (hi - lo) / 2
e = np.abs(c - true)
mine = {"captured": int(np.sum((lo <= true) & (true <= hi))), "MAE": e.mean(), "RMS": np.sqrt((e ** 2).mean()),
        "max half-width": h.max(), "mean half-width": h.mean()}

nb = json.loads((here.parent / "convex_analysis" / "dli.ipynb").read_text())
text = "".join("".join(o.get("text", "")) for cell in nb["cells"] if cell["cell_type"] == "code"
               for o in cell.get("outputs", []))
pick = {"captured": r"captured by admissible intervals: (\d+)/", "MAE": r"Mean absolute error of interval centers: ([\d.]+)",
        "RMS": r"RMS error of interval centers: ([\d.]+)", "max half-width": r"Max interval half-width: ([\d.]+)",
        "mean half-width": r"Average interval half-width: ([\d.]+)"}
print(f"{'quantity':<18}{'notebook (saved)':>18}{'pygeoinf0':>14}")
for k, pat in pick.items():
    m = re.search(pat, text)
    print(f"{k:<18}{(m.group(1) if m else '-'):>18}{mine[k]:>14.4f}")
for k, pat in [("data radius", r"Deterministic data radius: ([\d.]+)"), ("model radius", r"Deterministic model radius: ([\d.]+)"),
               ("prior width", r"Prior width range: \[([\d.]+)")]:
    m = re.search(pat, text)
    v = {"data radius": pb["data_radius"], "model radius": pb["model_radius"], "prior width": (phi - plo).min()}[k]
    print(f"{k:<18}{(m.group(1) if m else '-'):>18}{v:>14.4f}")
print("certificates:", sorted(set(res.status_lower) | set(res.status_upper)))
if len(sys.argv) > 1:
    old = np.load(sys.argv[1])
    rel = lambda a, b: float(np.abs(np.asarray(a, float) - b).max() / np.abs(b).max())
    print("re-run of the old notebook today: max rel diff of d_bar %.1e, model radius %.1e, prior bounds %.1e, "
          "DLI lower %.2f, upper %.2f" % (rel(pb["d_bar"], old["d_bar"]), rel(pb["model_radius"], old["model_radius"]),
                                         max(rel(plo, old["prior_lower"]), rel(phi, old["prior_upper"])),
                                         rel(lo, old["lower"]), rel(hi, old["upper"])))
    ho = (old["upper"] - old["lower"]) / 2
    print(f"  its mean half-width {ho.mean():.4f} (the saved notebook: see above); prior-only half-width "
          f"{((old['prior_upper'] - old['prior_lower']) / 2).mean():.4f}")
