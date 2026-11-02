import argparse
from pathlib import Path

import pandas as pd


def _binarize_outcome(
    series: pd.Series, swapped: pd.Series
) -> pd.Series:
    series = series.astype(str)
    is_a = series == "model_a"
    is_b = series == "model_b"
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")

    result.loc[~swapped & is_a] = 1
    result.loc[~swapped & is_b] = 0
    result.loc[swapped & is_a] = 0
    result.loc[swapped & is_b] = 1

    return result


def convert_summary(
    input_path: Path, output_path: Path, min_comparisons: int = 1
) -> pd.DataFrame:
    if min_comparisons < 1:
        raise ValueError("min_comparisons must be positive")
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input summary contains no comparisons")

    model_a = df["model_a"].astype(str)
    model_b = df["model_b"].astype(str)

    swapped = model_a > model_b
    ordered_a = model_a.where(~swapped, model_b)
    ordered_b = model_b.where(~swapped, model_a)

    winner = _binarize_outcome(df["winner"], swapped)
    if winner.isna().any():
        raise ValueError("Winners must all be model_a or model_b")

    # Auto-detect prediction mode: numeric values indicate Bradley-Terry mode
    pred_numeric = pd.to_numeric(df["prediction"], errors="coerce")
    if pred_numeric.notna().all():
        if not pred_numeric.between(0, 1).all():
            raise ValueError("Probability predictions must be finite and within [0, 1]")
        # Bradley-Terry: continuous predictions in [0, 1]
        prediction = pred_numeric.astype(float)
        prediction = prediction.where(~swapped, 1.0 - prediction)
    elif df["prediction"].isin(["model_a", "model_b"]).all():
        # Binary: string labels "model_a" / "model_b"
        prediction = _binarize_outcome(df["prediction"], swapped)
    else:
        raise ValueError("Predictions must be all numeric probabilities or all model_a/model_b labels; remove failed rows first")

    pair_keys = pd.MultiIndex.from_arrays([ordered_a, ordered_b])
    group_id = pd.factorize(pair_keys, sort=False)[0]

    out = pd.DataFrame(
        {
            "id": df["id"],
            "model_a": ordered_a,
            "model_b": ordered_b,
            "winner": winner,
            "prediction": prediction,
            "group_id": group_id,
        }
    )

    counts = out.groupby(["model_a", "model_b"]).size()
    too_small = counts[counts < min_comparisons]
    if not too_small.empty:
        raise ValueError(f"Pairs with fewer than {min_comparisons} valid comparisons:\n{too_small.to_string()}")

    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize LLM pair comparisons: order models, binarize outcomes, "
            "and assign group IDs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the normalized CSV",
    )
    parser.add_argument("--min-comparisons", type=int, default=1,
                        help="Require this many final comparisons per pair (use 100 for the paper)")
    args = parser.parse_args()

    convert_summary(args.input, args.output, args.min_comparisons)


if __name__ == "__main__":
    main()
