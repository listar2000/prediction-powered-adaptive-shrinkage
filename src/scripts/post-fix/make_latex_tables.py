"""Render the post-fix results as LaTeX tables in the PAS paper's layout.

Reads ``results/summary.csv`` (produced by ``run_post_fix_tables.py``) and writes
a standalone, compilable document plus the bare table bodies for pasting into
the paper.

Emits four tables: the post-fix Table 3 (nine estimator rows against three
dataset column-pairs, MSE and %% Improved) and the post-fix Table 2 (seven rows,
two synthetic predictors, MSE only), each followed by its published counterpart
for reference. Neither published table is directly comparable to its revised
counterpart; see the note in the document.

Usage (from repo root)::

    uv run python src/scripts/post-fix/make_latex_tables.py
    uv run python src/scripts/post-fix/make_latex_tables.py --no-published
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_post_fix_tables import (  # noqa: E402
    PUBLISHED,
    PUBLISHED_SYNTHETIC,
    ROWS,
    SYNTHETIC_ROWS,
)

# Row label as it should appear in LaTeX.
ROW_TEX = {
    "mle": "Classical",
    "pred_mean": "Prediction Avg",
    "ppi": "PPI",
    "pt": "PT",
    "shrinkage_only": "Shrink Classical",
    "shrinkage_mean": "Shrink Avg",
    "pas": r"\texttt{PAS} \textbf{(ours)}",
    "uni_pt": r"\texttt{UniPT} \textbf{(ours)}",
    "uni_pas": r"\texttt{UniPAS} \textbf{(ours)}",
}

DATASET_TEX = {
    "amazon_base": r"\textbf{Amazon (base $f$)}",
    "amazon_tuned": r"\textbf{Amazon (tuned $f$)}",
    "galaxy": r"\textbf{Galaxy}",
    "synthetic_f1": r"\textbf{$f_1(x) = x^2$}",
    "synthetic_f2": r"\textbf{$f_2(x) = |x|$}",
}

REAL_ORDER = ["amazon_base", "amazon_tuned", "galaxy"]
SYNTHETIC_ORDER = ["synthetic_f1", "synthetic_f2"]


def _cell(value: float, se: float, best: bool, precision: int = 3) -> str:
    body = f"{value:.{precision}f} $\\pm$ {se:.{precision}f}"
    return f"\\textbf{{{body}}}" if best else body


def _collect(summary: pd.DataFrame, order: list, rows: list) -> dict:
    """(dataset, estimator) -> (mse, mse_se, impr, impr_se)."""
    out = {}
    for dataset in order:
        for _, key in rows:
            row = summary[(summary["dataset"] == dataset)
                          & (summary["estimator"] == key)]
            if row.empty:
                continue
            row = row.iloc[0]
            out[(dataset, key)] = (
                float(row["mse_mean_e3"]), float(row["mse_se_e3"]),
                float(row["improved_pct_mean"]), float(row["improved_pct_se"]),
            )
    return out


def _published_cells(order: list, rows: list) -> dict:
    out = {}
    for dataset in order:
        for _, key in rows:
            if dataset in PUBLISHED_SYNTHETIC:
                mse, mse_se = PUBLISHED_SYNTHETIC[dataset][key]
                out[(dataset, key)] = (mse, mse_se, None, None)
            else:
                out[(dataset, key)] = PUBLISHED[dataset][key]
    return out


def make_table(cells: dict, caption: str, label: str, order: list, rows: list,
               with_improved: bool = True) -> str:
    keys = [key for _, key in rows]
    n_cols = len(order) * (2 if with_improved else 1)

    # Bold the best value per column. "% Improved" skips the Classical row,
    # whose cell reads "baseline".
    best_mse = {d: min(keys, key=lambda k: cells[(d, k)][0]) for d in order}
    best_impr = {}
    if with_improved:
        for d in order:
            cand = [k for k in keys if k != "mle" and cells[(d, k)][2] is not None]
            best_impr[d] = max(cand, key=lambda k: cells[(d, k)][2]) if cand else None

    if with_improved:
        header_groups = " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{DATASET_TEX[d]}}}" for d in order)
        cmidrules = " ".join(
            f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(order)))
        metric_header = " & ".join(
            r"\textbf{MSE} \( (\times 10^{-3}) \) & \textbf{\% Improved $\uparrow$}"
            for _ in order)
    else:
        header_groups = ""
        cmidrules = ""
        metric_header = " & ".join(
            f"{DATASET_TEX[d]} \\textbf{{MSE}} \\( (\\times 10^{{-3}}) \\)"
            for d in order)

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \vskip 0.05in",
        r"    \footnotesize",
        f"    \\begin{{tabular}}{{l{'c' * n_cols}}}",
        r"    \toprule",
    ]
    if with_improved:
        lines += [f"    & {header_groups} \\\\", f"    {cmidrules}"]
    lines += [f"    \\textbf{{Estimator}} & {metric_header} \\\\", r"    \midrule"]

    for key in keys:
        row_cells = []
        for d in order:
            mse, mse_se, impr, impr_se = cells[(d, key)]
            row_cells.append(_cell(mse, mse_se, key == best_mse[d]))
            if with_improved:
                if key == "mle" or impr is None:
                    row_cells.append("baseline")
                else:
                    row_cells.append(
                        _cell(impr, impr_se, key == best_impr[d], precision=1))
        lines.append(f"    {ROW_TEX[key]} & " + " & ".join(row_cells) + r" \\")

    lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


PREAMBLE = r"""% Standalone render of the post-fix PAS results (Tables 2 and 3).
% Build:  pdflatex post_fix_tables.tex
%
% Generated by src/scripts/post-fix/make_latex_tables.py -- edit that script
% rather than this file.
\documentclass[11pt]{article}
\usepackage[margin=0.6in,landscape,a4paper]{geometry}
\usepackage{booktabs}
\usepackage{float}  % [H]: pin tables where they are written, so the trailing
                    % note cannot be split around a migrating float
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[T1]{fontenc}
\pagestyle{empty}
\setlength{\tabcolsep}{4pt}
\setlength{\abovecaptionskip}{4pt}
\setlength{\belowcaptionskip}{0pt}
\setlength{\textfloatsep}{12pt}
\setlength{\intextsep}{12pt}

\begin{document}

\begin{center}
{\large\bfseries PAS Tables 2 and 3, recomputed after the estimator and pseudo-ground-truth fixes}\\[3pt]
{\small $K = 200$ replicates. Real data: default split seed (42; replicate $k$ uses seed $42+k$), \texttt{train\_test\_split} $=0.2$.
Synthetic: $m = 200$, $n_j = 20$, $N_j = 80$, split seed 4321, with corrected
analytical true second moments (no Monte Carlo moment approximation), $c=0.05$,
and $\psi=0.2$.
Real-data second moments are per problem (\texttt{share\_var=False}).}
\end{center}
"""

NOT_COMPARABLE_NOTE = r"""
\vspace{0.4em}
\hrule
\vspace{0.5em}
{\scriptsize
\noindent\textbf{The evaluation target changed; the real-data table is not
comparable row-for-row with the published Table~3.} The pseudo-ground-truth now
averages \emph{all} labels for a problem (Appendix~E.4, $\dot\theta_j :=
T_j^{-1}\sum_i \dot Y_{ij}$) instead of the unlabelled split only, so every
estimator is evaluated against a different pseudo ground-truth. The synthetic
target $\theta_j = \eta_j^2$ remains known exactly, but the revised experiment
uses $\psi=0.2$ rather than the published $\psi=0.1$, so its published table is
also included only as a reference.

\noindent\textbf{Second moments.} All real-data cells use the per-problem
estimators of Appendix~C.1, reported as \texttt{share\_var=False} --- the
configuration behind the paper's numbers. On the synthetic model the true
second moments are evaluated from their corrected closed forms (Appendix~E.1),
without a Monte Carlo moment approximation, so that choice does not arise.
\textbf{Bolding} marks the best value in each column.
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=HERE / "results" / "summary.csv")
    parser.add_argument("--out-tex", type=Path, default=HERE / "post_fix_tables.tex")
    parser.add_argument(
        "--bodies-out", type=Path, default=HERE / "post_fix_tables_bodies.tex",
        help="Bare table bodies, for pasting into the paper.",
    )
    parser.add_argument("--no-published", action="store_true",
                        help="Omit the published reference tables.")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    trials = int(summary["trials"].iloc[0])
    present = set(summary["dataset"])
    real = [d for d in REAL_ORDER if d in present]
    synth = [d for d in SYNTHETIC_ORDER if d in present]

    post_fix, reference = [], []

    if real:
        post_fix.append(make_table(
            _collect(summary, real, ROWS),
            caption=(
                f"Post-fix Table 3: results aggregated over $K = {trials}$ replicates "
                r"on two real-world datasets with three predictor settings --- "
                r"Amazon review ratings with \texttt{BERT-base} and \texttt{BERT-tuned} "
                r"predictors, and spiral galaxy fractions with a \texttt{ResNet50} "
                r"predictor. Metrics are "
                r"reported with $\pm$ 1 standard error."
            ),
            label="tbl:postfix_real", order=real, rows=ROWS,
        ))
        reference.append(make_table(
            _published_cells(real, ROWS),
            caption=(
                r"\emph{Reference only:} the published Table 3, computed before the "
                r"fixes. Not directly comparable with the post-fix table because the "
                r"pseudo-ground-truth changed; see the note on the preceding page."
            ),
            label="tbl:published_real", order=real, rows=ROWS,
        ))

    if synth:
        post_fix.append(make_table(
            _collect(summary, synth, SYNTHETIC_ROWS),
            caption=(
                f"Post-fix Table 2: MSE over $K = {trials}$ replicates of the "
                r"synthetic model with the good predictor $f_1(x) = x^2$ and the "
                r"flawed predictor $f_2(x) = |x|$, using $m=200$ and the corrected "
                r"analytical true second moments, $c=0.05$, and $\psi=0.2$. "
                r"UniPT/UniPAS are omitted, as in "
                r"the paper, since the second moments are known here. Metrics are "
                r"reported with $\pm$ 1 standard error."
            ),
            label="tbl:postfix_synthetic", order=synth, rows=SYNTHETIC_ROWS,
            with_improved=False,
        ))
        reference.append(make_table(
            _published_cells(synth, SYNTHETIC_ROWS),
            caption=(
                r"\emph{Reference only:} the published Table 2, which used "
                r"$\psi=0.1$ rather than the revised $\psi=0.2$."
            ),
            label="tbl:published_synthetic", order=synth, rows=SYNTHETIC_ROWS,
            with_improved=False,
        ))

    bodies = "\n\n".join(post_fix + ([] if args.no_published else reference)) + "\n"
    args.bodies_out.write_text(bodies)
    print(f"Saved table bodies -> {args.bodies_out}")

    # Post-fix tables and the explanatory note share page 1; the published
    # reference tables (if requested) get their own page.
    body = "\n\n".join(post_fix) + "\n" + NOT_COMPARABLE_NOTE
    if not args.no_published and reference:
        body += "\n\\clearpage\n" + "\n\n".join(reference) + "\n"
    args.out_tex.write_text(PREAMBLE + "\n" + body + "\n\\end{document}\n")
    print(f"Saved standalone document -> {args.out_tex}")

    if args.no_compile:
        return
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", args.out_tex.name],
        cwd=args.out_tex.parent, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("pdflatex failed:\n" + result.stdout[-2500:])
        raise SystemExit(1)
    for suffix in (".aux", ".log"):
        args.out_tex.with_suffix(suffix).unlink(missing_ok=True)
    print(f"Compiled -> {args.out_tex.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
