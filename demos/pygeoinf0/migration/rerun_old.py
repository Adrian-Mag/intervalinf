"""Re-run the old notebook's code cells headless (figures to a temporary folder) and save its numbers.

    PYTHONPATH=<intervalinf checkout> conda run -n inferences python demos/pygeoinf0/migration/rerun_old.py old_dli.npz
"""
import json
import os
import sys
import tempfile
from pathlib import Path

nb = Path(__file__).resolve().parents[2] / "convex_analysis" / "dli.ipynb"
cells = ["".join(c["source"]) for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"]
out = os.path.abspath(sys.argv[1])
os.chdir(tempfile.mkdtemp(prefix="dli_old_"))
src = "import matplotlib\nmatplotlib.use('Agg')\n" + "\n\n".join(cells).replace("plt.show()", "plt.close('all')")
src += f"""
np.savez({out!r}, upper=upper_bounds, lower=lower_bounds, prior_upper=prior_upper, prior_lower=prior_lower,
         true_props=np.asarray(true_props), d_bar=np.asarray(d_bar), d_tilde=np.asarray(d_tilde),
         data_radius=data_radius, model_radius=model_radius, solve_time=solve_elapsed)
"""
exec(compile(src, str(nb), "exec"), {"__name__": "__main__"})
