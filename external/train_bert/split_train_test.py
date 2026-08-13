"""
Script that does the following:

1. Load Reviews.csv from external/assets.
2. Group the data by the `ProductId` column.
3. Use the 200 products with the highest number of reviews for evaluation.
4. Remove rows with missing `Summary`, `Text`, and `Score` fields.
5. Store product metadata in the parquet files alongside the reviews.
6. Save all other products to train.parquet and the top products to eval.parquet.
"""

from pathlib import Path

import pandas as pd


# determine how many products to save for evaluation
N = 200
RANDOM_STATE = 123456

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
REVIEWS_PATH = ASSETS_DIR / "Reviews.csv"
TRAIN_DATA_PATH = ASSETS_DIR / "train.parquet"
EVAL_DATA_PATH = ASSETS_DIR / "eval.parquet"


def shuffle_data(dataframe):
    return (
        dataframe.groupby("ProductId", group_keys=False, sort=False)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def split_train_test():
    # 1. Load the Reviews.csv file with pandas
    df = pd.read_csv(REVIEWS_PATH)

    # 2. Group the data by the `ProductId` column
    grouped = df.groupby("ProductId")

    # 4. Remove rows with missing `Summary`, `Text`, and `Score` fields
    df_cleaned = df.dropna(subset=["Summary", "Text", "Score"])

    # Regroup after cleaning the data
    grouped_cleaned = df_cleaned.groupby("ProductId")
    product_review_counts = grouped_cleaned.size()

    # 3. Find the top-N products with the highest number of reviews
    top_n_products = product_review_counts.nlargest(N).index

    # Keep the former metadata fields in each parquet row. Parquet compression keeps
    # the repeated review count compact while allowing eval groups to be reconstructed.
    df_cleaned = df_cleaned.copy()
    df_cleaned["Number of Reviews"] = df_cleaned["ProductId"].map(
        product_review_counts
    )
    output_columns = [
        "ProductId",
        "Number of Reviews",
        "Summary",
        "Text",
        "Score",
    ]

    # 6. Save the top products together in the evaluation parquet
    eval_data = shuffle_data(df_cleaned[df_cleaned["ProductId"].isin(top_n_products)])

    # Save reviews for all other products from the cleaned dataframe
    train_data = shuffle_data(df_cleaned[~df_cleaned["ProductId"].isin(top_n_products)])

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_parquet(TRAIN_DATA_PATH, index=False)
    eval_data.to_parquet(EVAL_DATA_PATH, index=False)

    print(f"Training data saved to '{TRAIN_DATA_PATH}'.")
    print(f"Evaluation data saved to '{EVAL_DATA_PATH}'.")


if __name__ == "__main__":
    split_train_test()
