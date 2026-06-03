# Paper Demo Visualization Scripts

This folder contains visualization-domain code for the old normal-mode paper demos: standalone viewers, interactive apps, and plotting helper modules.

The split is by responsibility, not only by whether a file has a runnable UI. A module such as `full_spectrum_viz.py` is a helper, but it is still a plotting helper, so it belongs here rather than in `../utils/`.

Shared data/model-building helpers stay in `../utils/`. The visualization modules import `../utils/` through `_path_setup.py` so they can still be run directly as scripts from this folder or from `paper_demos/`.
