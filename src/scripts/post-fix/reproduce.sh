#!/usr/bin/env bash
# End-to-end reproduction of the post-fix PAS Tables 2 and 3 and Figure 3.
#
#   ./src/scripts/post-fix/reproduce.sh            # K = 200
#   TRIALS=5 ./src/scripts/post-fix/reproduce.sh   # quick smoke run
#
# Run from the repository root. Requires the Amazon HDF5 and Galaxy CSV under
# data/ (see pas.config.DATA_PATHS) and pdflatex with booktabs for the PDF step.
# The synthetic cells need no data files.
set -euo pipefail

TRIALS="${TRIALS:-200}"
WORKERS="${WORKERS:-5}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Running benchmarks (K=${TRIALS}, ${WORKERS} workers: 3 real + 2 synthetic)"
uv run python "${HERE}/run_post_fix_tables.py" \
    --trials "${TRIALS}" --num-workers "${WORKERS}"

echo
echo "==> Rendering LaTeX tables and compiling the PDF"
uv run python "${HERE}/make_latex_tables.py"

echo
echo "==> Rendering Figure 3, its shrinkage close-up, and PT-variance diagnostic"
uv run python "${HERE}/../paper_plots/plot_synthetic_params.py"

echo
echo "==> Outputs"
ls -1 "${HERE}"/post_fix_tables.{tex,pdf} "${HERE}"/post_fix_tables_bodies.tex \
      "${HERE}"/results/summary.csv "${HERE}"/results/synthetic_params.csv \
      "${HERE}"/results/synthetic_variances.csv \
      "${HERE}"/../../../images/synthetic_params.pdf \
      "${HERE}"/../../../images/synthetic_omegas_zoomed.pdf \
      "${HERE}"/../../../images/synthetic_variances.pdf
