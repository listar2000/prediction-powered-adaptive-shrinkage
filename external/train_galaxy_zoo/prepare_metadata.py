"""Build the train/test metadata used by the Galaxy Zoo ResNet experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "galaxy_zoo"
SOURCE_DIR = ASSETS_DIR / "metadata" / "source"
CLEANED_DIR = ASSETS_DIR / "metadata" / "cleaned"
TOP_TRAIN_BINS = 20

LABEL_FILES = [
    SOURCE_DIR / "gz2_train.csv",
    SOURCE_DIR / "gz2_valid.csv",
    SOURCE_DIR / "gz2_test.csv",
]
FILENAME_MAPPING = SOURCE_DIR / "gz2_filename_mapping.csv"
BIN_MAPPING = SOURCE_DIR / "gz2sample.csv"


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def prepare_metadata() -> None:
    for path in [*LABEL_FILES, FILENAME_MAPPING, BIN_MAPPING]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    label_frames = []
    for path in LABEL_FILES:
        frame = pd.read_csv(path)
        require_columns(frame, {"galaxyID", "label1"}, path)
        label_frames.append(frame)
    combined = pd.concat(label_frames, ignore_index=True)

    filename_mapping = pd.read_csv(FILENAME_MAPPING)
    require_columns(filename_mapping, {"asset_id", "objid"}, FILENAME_MAPPING)
    combined = combined.merge(
        filename_mapping[["asset_id", "objid"]],
        left_on="galaxyID",
        right_on="asset_id",
        how="inner",
    ).drop(columns=["asset_id"])
    combined = combined.rename(columns={"objid": "dr7objid"})
    if "sample" in combined.columns:
        combined = combined.drop(columns=["sample"])

    bin_mapping = pd.read_csv(BIN_MAPPING)
    require_columns(bin_mapping, {"OBJID", "WVT_BIN"}, BIN_MAPPING)
    combined = combined.merge(
        bin_mapping[["OBJID", "WVT_BIN"]],
        left_on="dr7objid",
        right_on="OBJID",
        how="inner",
    ).drop(columns=["OBJID"])

    bin_counts = combined.groupby("WVT_BIN").size().sort_values(ascending=False)
    train_bins = set(bin_counts.head(TOP_TRAIN_BINS).index)
    train = combined[combined["WVT_BIN"].isin(train_bins)].copy()
    test = combined[~combined["WVT_BIN"].isin(train_bins)].copy()

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(CLEANED_DIR / "gz2_final.csv", index=False)
    train.to_csv(CLEANED_DIR / "gz2_final_train.csv", index=False)
    test.to_csv(CLEANED_DIR / "gz2_final_test.csv", index=False)
    print(f"All matched galaxies: {len(combined)}")
    print(f"Training galaxies ({TOP_TRAIN_BINS} WVT bins): {len(train)}")
    print(f"Held-out galaxies: {len(test)}")


if __name__ == "__main__":
    prepare_metadata()
