from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# Some constants
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
TRAIN_DATA_PATH = ASSETS_DIR / "train.parquet"
EVAL_DATA_PATH = ASSETS_DIR / "eval.parquet"
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
PRETRAINED_MODEL_DIR = ASSETS_DIR / "bert-base-multilingual-uncased-sentiment"
FINETUNED_MODEL_DIR = ASSETS_DIR / "finetuned_model"


class AmazonReviewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx] - 1)
        return item

    def __len__(self):
        return len(self.labels)


def download_original_model_if_needed():
    """Download and cache the original tokenizer and model under external/assets."""
    model_exists = (PRETRAINED_MODEL_DIR / "config.json").exists() and (
        (PRETRAINED_MODEL_DIR / "model.safetensors").exists()
        or (PRETRAINED_MODEL_DIR / "pytorch_model.bin").exists()
    )
    tokenizer_exists = (PRETRAINED_MODEL_DIR / "tokenizer_config.json").exists()

    if model_exists and tokenizer_exists:
        print(f"Using original model from '{PRETRAINED_MODEL_DIR}'.")
        return

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading original model '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(str(PRETRAINED_MODEL_DIR))
    model.save_pretrained(str(PRETRAINED_MODEL_DIR))
    print(f"Original model saved to '{PRETRAINED_MODEL_DIR}'.")


def get_train_dataset(tokenizer, n_chunks=100, save_indices=True):
    # Step 1: Load the data
    print("Loading and preprocessing the data...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    # shuffle the data
    train_df = train_df.sample(frac=1.0, random_state=123456)

    # Ensure all entries in 'Summary' and 'Text' are strings, replacing NaNs with empty strings
    train_df["Summary"] = train_df["Summary"].fillna("").astype(str)
    train_df["Text"] = train_df["Text"].fillna("").astype(str)

    # Combine the summary and text
    train_texts = (train_df["Summary"] + ". " + train_df["Text"]).tolist()
    train_labels = train_df["Score"].tolist()

    # Check for unexpected data types or issues
    if not all(isinstance(text, str) for text in train_texts):
        raise ValueError("All entries in 'train_texts' must be strings.")

    print(f"Loaded {len(train_texts)} texts and {len(train_labels)} labels.")
    print("Sample text:", train_texts[0])

    print("Tokenizing the data...")
    # Step 2: Prepare the dataset for the model
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=512
    )
    train_dataset = AmazonReviewsDataset(train_encodings, train_labels)

    for batch in train_dataset:
        assert (
            "input_ids" in batch
            and "attention_mask" in batch
            and "labels" in batch
        ), "Dataset is missing required keys"
        break

    return train_dataset


def get_eval_dataset(tokenizer, n_products: int = 100):
    # pick n of the top-N products with most reviews for evaluation
    eval_df = pd.read_parquet(EVAL_DATA_PATH)
    product_ids = eval_df["ProductId"].drop_duplicates().values
    eval_product_ids = np.random.choice(product_ids, n_products, replace=False)
    eval_df = eval_df[eval_df["ProductId"].isin(eval_product_ids)].copy()

    eval_df["Summary"] = eval_df["Summary"].fillna("").astype(str)
    eval_df["Text"] = eval_df["Text"].fillna("").astype(str)
    eval_texts = (eval_df["Summary"] + ". " + eval_df["Text"]).tolist()
    eval_labels = eval_df["Score"].tolist()

    eval_encodings = tokenizer(
        eval_texts, truncation=True, padding=True, max_length=512
    )
    eval_dataset = AmazonReviewsDataset(eval_encodings, eval_labels)
    return eval_dataset


def fine_tune_training(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    epochs: int = 10,
    batch_size: int = 32,
):
    # Step 3: Set up the model for fine-tuning
    model = AutoModelForSequenceClassification.from_pretrained(
        str(PRETRAINED_MODEL_DIR)
    )

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Using {device_count} GPUs.")
        model.to(torch.device("cuda"))
    else:
        raise ValueError("CUDA is not available. Please use a GPU for training.")

    training_args = TrainingArguments(
        output_dir=str(ASSETS_DIR / "finetune_model" / "results"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(ASSETS_DIR / "finetune_model" / "logs"),
        logging_steps=20,
        save_total_limit=4,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=2000,
        report_to="none",
        dataloader_num_workers=8,
        dataloader_drop_last=True,
    )

    # Step 4: Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # Step 5: Save the model
    trainer.save_model(str(FINETUNED_MODEL_DIR))
    print(f"Fine-tuned model saved to '{FINETUNED_MODEL_DIR}'.")


if __name__ == "__main__":
    torch.manual_seed(123456)
    np.random.seed(123456)
    download_original_model_if_needed()
    tokenizer = AutoTokenizer.from_pretrained(
        str(PRETRAINED_MODEL_DIR), use_fast=True
    )
    train_dataset = get_train_dataset(tokenizer)
    eval_dataset = get_eval_dataset(tokenizer)
    fine_tune_training(train_dataset, eval_dataset, batch_size=128, epochs=2)
