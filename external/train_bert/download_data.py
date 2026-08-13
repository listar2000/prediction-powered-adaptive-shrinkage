"""Download the Amazon Fine Food Reviews dataset into external/assets."""

import shutil
import zipfile
from pathlib import Path

import kagglehub


ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
REVIEWS_PATH = ASSETS_DIR / "Reviews.csv"
DATASET_NAME = "snap/amazon-fine-food-reviews"


def download_data():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if REVIEWS_PATH.exists():
        print(f"Dataset already exists at: {REVIEWS_PATH}")
        return REVIEWS_PATH

    # Download latest version. KaggleHub normally extracts the archive in its cache.
    path = Path(kagglehub.dataset_download(DATASET_NAME))
    print("Path to dataset files:", path)

    reviews_files = list(path.rglob("Reviews.csv"))
    if reviews_files:
        shutil.copy2(reviews_files[0], REVIEWS_PATH)
    else:
        # Keep this fallback for KaggleHub versions that leave the archive zipped.
        archive_paths = (
            [path] if zipfile.is_zipfile(path) else path.rglob("*.zip")
        )
        for archive_path in archive_paths:
            with zipfile.ZipFile(archive_path) as archive:
                review_member = next(
                    (
                        member
                        for member in archive.namelist()
                        if Path(member).name == "Reviews.csv"
                    ),
                    None,
                )
                if review_member is not None:
                    with archive.open(review_member) as source, REVIEWS_PATH.open(
                        "wb"
                    ) as target:
                        shutil.copyfileobj(source, target)
                    break
        else:
            raise FileNotFoundError(f"Could not find Reviews.csv under {path}")

    print(f"Reviews.csv saved to: {REVIEWS_PATH}")
    return REVIEWS_PATH


if __name__ == "__main__":
    download_data()
