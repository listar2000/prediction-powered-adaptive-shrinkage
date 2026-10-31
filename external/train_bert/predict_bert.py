"""Evaluate base/epoch checkpoints on the same product holdout; save aligned HDF5/parquet."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

from common import (ASSETS_DIR, SEED, configure_cuda, environment, prepared_manifest,
                    save_json, seed_everything, sha256)

from datasets import load_from_disk
import h5py
import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def main(args):
    if args.per_device_batch < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and worker count nonnegative")
    if not 1 <= int(os.environ.get("WORLD_SIZE", "1")) <= 8:
        raise ValueError("Use between 1 and 8 GPU processes")
    precision = configure_cuda(args.precision)
    root = Path(args.root).resolve()
    run = root / "runs" / args.run_name
    manifest = prepared_manifest(root)
    training_config = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").exists() else {}
    seed = training_config.get("seed", SEED) if args.seed is None else args.seed
    if training_config and training_config["data_manifest"]["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Training and evaluation were prepared from different source data")
    seed_everything(seed, tf32=precision["tf32"])
    checkpoints = {}
    if any(name != "base" for name in args.models):
        checkpoints = json.loads((run / "checkpoints.json").read_text())
    models = {}
    for name in args.models:
        if name == "base":
            models[name] = str(root / "base_model")
        else:
            # Resolve within this run so a downloaded run directory stays portable.
            checkpoint = run / "checkpoints" / Path(checkpoints[name.removeprefix("epoch")]).name
            if not checkpoint.is_dir():
                raise FileNotFoundError(checkpoint)
            models[name] = str(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(root / "base_model", use_fast=True)
    data = load_from_disk(root / "prepared/tokens")["eval"].sort("length")
    review_indices = np.asarray(data["review_index"])
    frame = pd.read_parquet(root / "prepared/eval.parquet")
    assert len(frame) == manifest["eval_rows"] and frame.ProductId.nunique() == manifest["eval_products"]
    assert sorted(review_indices.tolist()) == list(range(len(frame)))
    inverse_order = np.argsort(review_indices)
    summaries = {}
    for name, model_path in models.items():
        seed_everything(seed, tf32=precision["tf32"])
        model = AutoModelForSequenceClassification.from_pretrained(model_path, attn_implementation="sdpa")
        training_args = TrainingArguments(
            output_dir=str(run / "evaluation_tmp"),
            per_device_eval_batch_size=args.per_device_batch, **precision,
            dataloader_num_workers=args.num_workers, dataloader_pin_memory=True,
            dataloader_drop_last=False, seed=seed, data_seed=seed,
            report_to="none", disable_tqdm=True,
            ddp_find_unused_parameters=False,
        )
        trainer = Trainer(model=model, args=training_args, processing_class=tokenizer,
                          data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
        if trainer.is_world_process_zero():
            save_json(run / "prediction_config.json", {
                "environment": environment(), "arguments": training_args.to_dict(),
                "seed": seed, "models": models, "data_manifest": manifest,
            })
        started = time.perf_counter()
        predictions = trainer.predict(data, metric_key_prefix=name)
        if trainer.is_world_process_zero():
            logits = np.asarray(predictions.predictions)[inverse_order]
            labels = np.asarray(predictions.label_ids)[inverse_order]
            assert logits.shape == (len(frame), 5)
            assert np.array_equal(labels + 1, frame.Score.to_numpy())
            assert np.isfinite(logits).all()
            predicted = logits.argmax(axis=1) + 1
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            actual = frame.Score.to_numpy()
            output = frame[["Id", "ProductId", "UserId", "review_index", "Score"]].copy()
            output["prediction"] = predicted
            output["expected_rating"] = probabilities @ np.arange(1, 6)
            for i in range(5):
                output[f"probability_{i+1}"] = probabilities[:, i]
            output["correct"] = predicted == actual
            output["squared_error"] = (predicted - actual) ** 2
            folder = run / "predictions"
            folder.mkdir(parents=True, exist_ok=True)
            output.to_parquet(folder / f"{name}.parquet", index=False)
            per_product = output.groupby("ProductId").agg(reviews=("Score", "size"), accuracy=("correct", "mean"), mse=("squared_error", "mean"))
            per_product.to_csv(folder / f"{name}_per_product.csv")
            with h5py.File(folder / f"{name}.h5", "w") as h5:
                h5.attrs["checkpoint"] = name
                h5.attrs["seed"] = seed
                h5.attrs["rating_scale"] = "1-5 integer star ratings"
                for product, part in output.groupby("ProductId", sort=True):
                    h5.create_dataset(str(product), data=np.vstack([part.Score, part.prediction]), dtype="i")
            confusion = np.zeros((5, 5), dtype=int)
            np.add.at(confusion, (actual - 1, predicted - 1), 1)
            summaries[name] = {
                "checkpoint_path": model_path, "rows": len(frame), "products": int(frame.ProductId.nunique()),
                "accuracy": float(np.mean(predicted == actual)),
                "correct_reviews": int(np.sum(predicted == actual)),
                "product_macro_accuracy": float(per_product.accuracy.mean()),
                "mse": float(np.mean((predicted - actual) ** 2)),
                "mae": float(np.mean(np.abs(predicted - actual))),
                "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
                "prediction_seconds": time.perf_counter() - started,
                "evaluation_parquet_sha256": sha256(root / "prepared/eval.parquet"),
                "trainer_metrics": predictions.metrics,
            }
            save_json(run / "evaluation_summary.json", summaries)
            print(json.dumps({name: summaries[name]}, indent=2), flush=True)
        del trainer, model, predictions
        gc.collect()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ASSETS_DIR)
    parser.add_argument("--run-name", default="bert")
    parser.add_argument("--models", nargs="+", choices=["base", "epoch1", "epoch2"], default=["base", "epoch1", "epoch2"])
    parser.add_argument("--per-device-batch", type=int, default=128)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--seed", type=int, help="Default: training run's seed, or 123456 for base-only evaluation")
    parser.add_argument("--num-workers", type=int, default=2)
    main(parser.parse_args())
