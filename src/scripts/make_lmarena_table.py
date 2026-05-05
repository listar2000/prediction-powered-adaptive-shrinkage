"""Build the LMArena CI appendix table from saved raw results.

Loads `raw.csv` produced by `run_lmarena_ci.py` and emits a LaTeX table
matching the column layout of `tbl:synthetic_amazon` in the paper appendix
(minus Oracle, since LMArena has no oracle).

Columns: Classic, PPI, PPI-P, PPI-NP, UniPT, UniPT-P, UniPT-NP.
Rows: per alpha, three metrics -- coverage (%), length, length-ratio (vs Classic).

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
    ("PPI",      "ppi_ci"),
    ("PPI-P",    "eb_ppi_ci"),
    ("PPI-NP",   "eb_npmle_ppi_ci"),
    ("UniPT",    "pt_ci"),
    ("UniPT-P",  "eb_unipt_ppi_ci"),
    ("UniPT-NP", "eb_npmle_unipt_ppi_ci"),
]

ALPHA_LABEL_FMT = "{:g}"  # 0.10 -> "0.1"


def _summary(raw: pd.DataFrame, methods) -> pd.DataFrame:
    records = []
    for alpha, group in raw.groupby("alpha"):
        for method in methods:
            cov = group[f"{method}_coverage"].astype(float)
            wid = group[f"{method}_width"].astype(float)
            records.append({
                "alpha": float(alpha),
                "method": method,
                "mean_coverage": cov.mean(),
                "mean_width": wid.mean(),
            })
    return pd.DataFrame(records)


def _fmt_pct(x):
    return f"{100.0 * x:.1f}\\%"


def _fmt_len(x):
    return f"{x:.3f}"


def _fmt_ratio(x):
    return f"{x:.3f}"


def make_latex_table(raw: pd.DataFrame) -> str:
    methods = [m for _, m in TABLE_COLUMNS]
    summary = _summary(raw, methods)

    alphas = sorted(raw["alpha"].unique())

    # Header
    n_cols = len(TABLE_COLUMNS)
    align = "c" + "c" * (1 + n_cols)  # alpha col + metric col + method cols

    header_methods = " & ".join(f"\\textbf{{{label}}}" for label, _ in TABLE_COLUMNS)
    lines = [
        "\\begin{table}[H]",
        "\\small",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\caption{LMArena CI benchmark across $\\alpha\\in\\{0.01,0.05,0.10,0.20,0.30\\}$, "
        "averaged over 200 random splits with $m=298$ pairwise LLM problems. "
        "Length-ratio is normalized by the Classical (CLT) interval.}",
        "\\label{tbl:lmarena_ci}",
        "\\makebox[\\textwidth][c]{",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        f"$\\alpha$ & & {header_methods} \\\\",
        "\\hline",
    ]

    for alpha in alphas:
        cov_row = []
        wid_row = []
        # Length-ratio normalized to the Classical width at this alpha.
        classic_wid = summary[
            (summary["alpha"] == alpha) & (summary["method"] == "mle_ci")
        ]["mean_width"].iloc[0]

        ratio_row = []
        for _, method in TABLE_COLUMNS:
            row = summary[(summary["alpha"] == alpha) & (summary["method"] == method)]
            cov = row["mean_coverage"].iloc[0]
            wid = row["mean_width"].iloc[0]
            ratio = wid / classic_wid
            cov_row.append(_fmt_pct(cov))
            wid_row.append(_fmt_len(wid))
            ratio_row.append(_fmt_ratio(ratio))

        alpha_label = ALPHA_LABEL_FMT.format(alpha)
        lines.append(
            f"\\multirow{{3}}{{*}}{{{alpha_label}}} & coverage  & " + " & ".join(cov_row) + " \\\\"
        )
        lines.append(
            "& length    & " + " & ".join(wid_row) + " \\\\"
        )
        lines.append(
            "& len-ratio & " + " & ".join(ratio_row) + " \\\\"
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
    parser.add_argument("--in-dir", type=Path,
                        default=Path(__file__).parent / "lmarena_ci_results")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "lmarena_appendix_table.tex")
    args = parser.parse_args()

    raw = pd.read_csv(args.in_dir / "raw.csv")
    tex = make_latex_table(raw)
    args.out.write_text(tex)
    print(f"Saved table -> {args.out}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
