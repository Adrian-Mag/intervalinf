#!/bin/bash
# Runner script for paper demo notebooks on europa.
# Run from: ~/data/PhD/Inferences/intervalinf/demos/old_demos/paper_demos/
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Working directory: $PWD ==="
echo "=== Python: $(python --version) ==="
echo "=== nbconvert: $(jupyter nbconvert --version) ==="
echo ""

echo "=== [1/2] Starting example_1.ipynb ==="
jupyter nbconvert \
    --execute \
    --to notebook \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    example_1.ipynb
echo "=== [1/2] example_1.ipynb DONE ==="
echo ""

echo "=== [2/2] Starting example_2.ipynb ==="
jupyter nbconvert \
    --execute \
    --to notebook \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    example_2.ipynb
echo "=== [2/2] example_2.ipynb DONE ==="
echo ""

echo "=== ALL NOTEBOOKS COMPLETE ==="
