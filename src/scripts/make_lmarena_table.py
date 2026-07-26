"""Build the LMArena CI appendix table from saved raw results.

Loads `raw.csv` produced by `run_lmarena_ci.py` and emits a LaTeX table
matching the column layout of `tbl:synthetic_amazon` in the paper appendix
(minus Oracle, since LMArena has no oracle).

Columns: Classic, PredMean, PPI, PT, RobustEB, DoubleShrink, PT-P, PT-NP.
Rows: per alpha, three metrics -- coverage (%), length, length-ratio (vs Classic).
Each cell is reported as ``mean $\\pm$ 1 Monte-Carlo SE'' over the random
labeled/unlabeled splits.

Usage:

    uv run python src/scripts/make_lmarena_table.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# (column header in LaTeX, internal method name)
TABLE_COLUMNS = [
    ("Classic",  "mle_ci"),
    ("PredMean", "pred_mean_ci"),
    ("PPI",      "ppi_ci"),
    ("PT",       "pt_ci"),
    ("RobustEB", "robust_eb_ci"),
    ("DoubleShrink", "double_shrinkage_mle_ci"),
    ("PT-P",     "eb_pt_ci"),
    ("PT-NP",    "eb_npmle_pt_ci"),
]

ALPHA_LABEL_FMT = "{:g}"  # 0.10 -> "0.1"

CLASSIC = "mle_ci"


def _per_alpha_metrics(raw: pd.DataFrame, methods) -> pd.DataFrame:
    """Return per-(alpha, method) mean and SE for coverage, length, and
    length-ratio (each computed per-trial, then averaged across trials)."""
    records = []
    for alpha, group in raw.groupby("alpha"):
        # Per-trial Classical width to compute per-trial length ratios.
        classic_wid_trial = group[f"{CLASSIC}_width"].astype(float)
        for method in methods:
            cov = group[f"{method}_coverage"].astype(float)
            wid = group[f"{method}_width"].astype(float)
            ratio = wid / classic_wid_trial
            records.append({
                "alpha": float(alpha),
                "method": method,
                "cov_mean": cov.mean(),
                "cov_se":   cov.sem(),
                "wid_mean": wid.mean(),
                "wid_se":   wid.sem(),
                "ratio_mean": ratio.mean(),
                "ratio_se":   ratio.sem(),
            })
    return pd.DataFrame(records)


def _fmt_pct(mean, se):
    # Note: the trailing "%" is hoisted into the row label "coverage (%)" so
    # individual cells stay as bare numbers.
    return f"${100.0 * mean:.1f}{{\\scriptstyle\\,\\pm\\,{100.0 * se:.1f}}}$"


def _fmt_pm(mean, se, prec=3):
    return f"${mean:.{prec}f}{{\\scriptstyle\\,\\pm\\,{se:.{prec}f}}}$"


def make_latex_table(raw: pd.DataFrame) -> str:
    methods = [m for _, m in TABLE_COLUMNS]
    summary = _per_alpha_metrics(raw, methods)

    alphas = sorted(raw["alpha"].unique())

    # Header
    n_cols = len(TABLE_COLUMNS)
    align = "c" + "c" * (1 + n_cols)  # alpha col + metric col + method cols

    header_methods = " & ".join(f"\\textbf{{{label}}}" for label, _ in TABLE_COLUMNS)
    lines = [
        "\\begin{table}[H]",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\caption{LMArena CI benchmark across $\\alpha\\in\\{0.01,0.05,0.10,0.20,0.30\\}$, "
        "averaged over 200 random labeled/unlabeled splits with $n=298$ pairwise "
        "LLM problems. Each cell reports mean $\\pm$ 1 Monte-Carlo SE. "
        "PredMean is the prediction-only interval; PT denotes the power-tuned "
        "PPI baseline; RobustEB is the Armstrong et al. robust EBCI applied "
        "to PT; DoubleShrink is the MLE double-shrinkage robust EBCI of "
        "Rosenman et al.; PT-P and PT-NP are the rebiased PT estimators with "
        "parametric (Gaussian) and NPMLE bias priors. Length-ratio is normalized "
        "by the Classical (CLT) interval.}",
        "\\label{tbl:lmarena_ci}",
        "\\makebox[\\textwidth][c]{",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        f"$\\alpha$ & & {header_methods} \\\\",
        "\\hline",
    ]

    for alpha in alphas:
        cov_row, wid_row, ratio_row = [], [], []
        for _, method in TABLE_COLUMNS:
            row = summary[(summary["alpha"] == alpha) & (summary["method"] == method)].iloc[0]
            cov_row.append(_fmt_pct(row["cov_mean"], row["cov_se"]))
            wid_row.append(_fmt_pm(row["wid_mean"], row["wid_se"], prec=3))
            ratio_row.append(_fmt_pm(row["ratio_mean"], row["ratio_se"], prec=3))

        alpha_label = ALPHA_LABEL_FMT.format(alpha)
        lines.append(
            f"\\multirow{{3}}{{*}}{{{alpha_label}}} & coverage (\\%)  & "
            + " & ".join(cov_row) + " \\\\"
        )
        lines.append(
            "& length        & " + " & ".join(wid_row) + " \\\\"
        )
        lines.append(
            "& len-ratio     & " + " & ".join(ratio_row) + " \\\\"
        )
        if alpha != alphas[-1]:
            lines.append("\\hline")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    default = Path("/tmp/lmarena_ci_results")
    parser.add_argument("--in-dir", type=Path, default=default)
    parser.add_argument("--out", type=Path,
                        default=default / "lmarena_appendix_table.tex")
    args = parser.parse_args()

    raw = pd.read_csv(args.in_dir / "raw.csv")
    tex = make_latex_table(raw)
    args.out.write_text(tex)
    print(f"Saved table -> {args.out}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
