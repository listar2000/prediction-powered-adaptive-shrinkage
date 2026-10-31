"""Prepare a deterministic product holdout, pinned BERT model, and token cache.

Run download_and_dedup_data.py first. This CPU-only step preserves all reviews,
including empty titles and literal 'NA' strings, and orders each split by row Id.
"""
import argparse
import json
from pathlib import Path
import time

from common import (ASSETS_DIR, MODEL_NAME, MODEL_REVISION, MAX_LENGTH,
                    prepared_manifest, save_json, sha256)


def partition_reviews(frame, holdout_products=200):
    """Rank by review count, break ties by ProductId, and retain original row IDs."""
    import numpy as np

    if not frame["Id"].is_unique:
        raise ValueError("Review Id values must be unique")
    if not frame["Score"].isin(range(1, 6)).all():
        raise ValueError("Review scores must be integer ratings from 1 to 5")
    if frame.duplicated(["Summary", "Text", "Score", "UserId"]).any():
        raise ValueError("Run download_and_dedup_data.py before preparing the splits")
    counts = frame.groupby("ProductId").size().rename("reviews").reset_index()
    counts = counts.sort_values(["reviews", "ProductId"], ascending=[False, True])
    if not 0 < holdout_products < len(counts):
        raise ValueError("Holdout size must leave at least one training product")
    heldout = counts.head(holdout_products).ProductId.tolist()
    is_heldout = frame.ProductId.isin(heldout)
    partitions = {"train": frame[~is_heldout].copy(), "eval": frame[is_heldout].copy()}
    for name, part in partitions.items():
        part = part.sort_values("Id").reset_index(drop=True)
        part["review_index"] = np.arange(len(part))
        part["Number of Reviews"] = part.ProductId.map(frame.groupby("ProductId").size())
        partitions[name] = part
    return partitions, counts, heldout


def main(args):
    from datasets import Dataset, DatasetDict
    from huggingface_hub import snapshot_download
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    root = args.root.resolve()
    started = time.perf_counter()
    source = (args.source or root / "Reviews.csv").resolve()
    source_hash = sha256(source)
    out = root / "prepared"
    if (out / "manifest.json").exists():
        manifest = prepared_manifest(root)
        if manifest["source_sha256"] != source_hash or manifest["eval_products"] != args.holdout_products:
            raise ValueError("Existing preparation uses different inputs; choose a fresh --root")
        if not (root / "base_model/config.json").is_file():
            raise FileNotFoundError("Prepared base model is missing; choose a fresh --root")
        print(json.dumps(manifest, indent=2), flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("Incomplete preparation exists; choose a fresh --root or remove its prepared/ directory")
    if args.tokenizer_workers < 1:
        raise ValueError("--tokenizer-workers must be positive")
    out.mkdir(parents=True, exist_ok=True)
    # Preserve strings such as "NA" and empty titles; do not silently drop reviews.
    frame = pd.read_csv(source, keep_default_na=False,
                        dtype={"ProductId": str, "UserId": str, "Summary": str, "Text": str})
    partitions, counts, heldout = partition_reviews(frame, args.holdout_products)
    snapshot_download(MODEL_NAME, revision=MODEL_REVISION, local_dir=root / "base_model",
                      allow_patterns=["config.json", "tokenizer*", "vocab.txt", "special_tokens_map.json", "pytorch_model.bin", "model.safetensors"])
    tokenizer = AutoTokenizer.from_pretrained(root / "base_model", use_fast=True)
    tokenized = {}
    lengths = {}
    for name, part in partitions.items():
        part.to_parquet(out / f"{name}.parquet", index=False)
        text = (part.Summary + ". " + part.Text).tolist()
        dataset = Dataset.from_dict({"text": text, "labels": (part.Score - 1).tolist(),
                                     "review_index": part.review_index.tolist()})
        def encode(batch):
            values = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)
            values["length"] = [len(x) for x in values["input_ids"]]
            return values
        dataset = dataset.map(encode, batched=True, batch_size=1024, num_proc=args.tokenizer_workers,
                              remove_columns=["text"], desc=f"Tokenize {name}")
        tokenized[name] = dataset
        lens = np.array(dataset["length"])
        lengths[name] = {"mean": float(lens.mean()), "p50": float(np.median(lens)),
                         "p90": float(np.percentile(lens, 90)), "at_max_length": int((lens == MAX_LENGTH).sum())}
    DatasetDict(tokenized).save_to_disk(out / "tokens")
    counts.to_csv(out / "product_counts.csv", index=False)
    manifest = {"source_path": str(source), "source_sha256": source_hash,
                "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
                "train_rows": len(partitions["train"]), "eval_rows": len(partitions["eval"]),
                "train_products": int(partitions["train"].ProductId.nunique()), "eval_products": args.holdout_products,
                "heldout_products": heldout, "lengths": lengths, "max_length": MAX_LENGTH,
                "parquet_sha256": {n: sha256(out / f"{n}.parquet") for n in partitions},
                "preparation_seconds": time.perf_counter() - started}
    save_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ASSETS_DIR)
    parser.add_argument("--source", type=Path, help="Deduplicated Reviews.csv (default: ROOT/Reviews.csv)")
    parser.add_argument("--holdout-products", type=int, default=200)
    parser.add_argument("--tokenizer-workers", type=int, default=8)
    main(parser.parse_args())
