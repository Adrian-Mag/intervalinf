"""Path setup for standalone paper-demo visualization scripts."""

from __future__ import annotations

import sys
from pathlib import Path


VISUALIZATION_DIR = Path(__file__).resolve().parent
DEMO_DIR = VISUALIZATION_DIR.parent
UTILS_DIR = DEMO_DIR / "utils"

for path in (VISUALIZATION_DIR, UTILS_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
