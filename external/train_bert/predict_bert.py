"""
Make predictions of review ratings using either the original (bad) BERT model or
the fine-tuned model. Ground-truth and predicted ratings are stored by ProductId
in HDF5 files under external/assets.
"""

import argparse

import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_bert import (
    ASSETS_DIR,
    EVAL_DATA_PATH,
    FINETUNED_MODEL_DIR,
    PRETRAINED_MODEL_DIR,
    download_original_model_if_needed,
)


BATCH_SIZE = 128
OUTPUT_FILES = {
    "bad": ASSETS_DIR / "predict_bad.h5",
    "finetuned": ASSETS_DIR / "predict_finetuned.h5",
}


class InferenceDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}


def get_predictions(model_type):
    """Generate predictions from either the original or fine-tuned model."""
    if model_type not in OUTPUT_FILES:
        raise ValueError(f"Unknown model type: {model_type}")

    download_original_model_if_needed()
    model_dir = (
        PRETRAINED_MODEL_DIR if model_type == "bad" else FINETUNED_MODEL_DIR
    )
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory '{model_dir}' does not exist. Run train_bert.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(PRETRAINED_MODEL_DIR), use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    eval_df = pd.read_parquet(EVAL_DATA_PATH)
    product_groups = eval_df.groupby("ProductId", sort=False)
    print("Number of products:", product_groups.ngroups)

    output_file = OUTPUT_FILES[model_type]
    with h5py.File(output_file, "w") as output:
        print(f"Starting {model_type} predictions for each product...")
        for product_id, product_reviews in tqdm(
            product_groups, total=product_groups.ngroups
        ):
            product_reviews = product_reviews.copy()
            product_reviews["Summary"] = (
                product_reviews["Summary"].fillna("").astype(str)
            )
            product_reviews["Text"] = (
                product_reviews["Text"].fillna("").astype(str)
            )
            texts = (
                product_reviews["Summary"] + ". " + product_reviews["Text"]
            ).tolist()
            true_scores = product_reviews["Score"].values

            encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            dataset = InferenceDataset(encodings)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

            predicted_scores = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    outputs = model(**batch)
                    predictions = torch.argmax(outputs.logits, dim=-1) + 1
                    predicted_scores.extend(predictions.cpu().numpy())

            output.create_dataset(
                name=str(product_id),
                data=[true_scores, predicted_scores],
                dtype="i",
            )

    print(f"Predictions saved in '{output_file}' file.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        nargs="?",
        choices=["bad", "finetuned", "both"],
        default="both",
        help="Model whose predictions should be generated (default: both).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_types = ["bad", "finetuned"] if args.model == "both" else [args.model]
    for selected_model in model_types:
        get_predictions(selected_model)
