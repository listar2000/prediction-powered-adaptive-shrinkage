#!/usr/bin/env bash
# End-to-end reproduction of the post-fix PAS Table 3 results.
#
#   ./src/scripts/post-fix/reproduce.sh            # K = 200, both share_var settings
#   TRIALS=5 ./src/scripts/post-fix/reproduce.sh   # quick smoke run
#
# Run from the repository root. Requires the Amazon HDF5 and Galaxy CSV under
# data/ (see pas.config.DATA_PATHS) and pdflatex with booktabs for the PDF step.
set -euo pipefail

TRIALS="${TRIALS:-200}"
WORKERS="${WORKERS:-6}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Running benchmark (K=${TRIALS}, ${WORKERS} workers, share_var off and on)"
uv run python "${HERE}/run_post_fix_tables.py" \
    --trials "${TRIALS}" --num-workers "${WORKERS}"

echo
echo "==> Rendering LaTeX tables and compiling the PDF"
uv run python "${HERE}/make_latex_tables.py"

echo
echo "==> Outputs"
ls -1 "${HERE}"/post_fix_tables.{tex,pdf} "${HERE}"/post_fix_tables_bodies.tex \
      "${HERE}"/results/summary.csv
