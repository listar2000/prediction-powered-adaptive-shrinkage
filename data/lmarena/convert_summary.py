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


def convert_summary(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    model_a = df["model_a"].astype(str)
    model_b = df["model_b"].astype(str)

    swapped = model_a > model_b
    ordered_a = model_a.where(~swapped, model_b)
    ordered_b = model_b.where(~swapped, model_a)

    winner = _binarize_outcome(df["winner"], swapped)
    prediction = _binarize_outcome(df["prediction"], swapped)

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
    args = parser.parse_args()

    convert_summary(args.input, args.output)


if __name__ == "__main__":
    main()
