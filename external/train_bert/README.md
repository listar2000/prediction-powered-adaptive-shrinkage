# Amazon review training and prediction

These scripts prepare the deduplicated SNAP corpus, train BERT for two epochs,
and evaluate the base model and both epoch checkpoints on the same held-out
products. They run on a single machine with 1–8 NVIDIA CUDA GPUs. No cloud SDK is
needed. The reference run used Python 3.12, CUDA 12.8, and eight H100 GPUs.

## Setup and preparation

From the repository root, create an environment on your GPU machine:

```sh
uv venv --python 3.12 external/assets/venv
uv pip install --python external/assets/venv/bin/python -r external/train_bert/requirements.txt
source external/assets/venv/bin/activate

python external/train_bert/download_and_dedup_data.py
python external/train_bert/split_train_test.py
```

Dependencies and the pretrained model revision are pinned. Preparation holds out
the 200 products with the most reviews, breaking ties by ascending ProductId.
All other products form the training set. It preserves empty titles and literal
`NA` strings, sorts each partition by original review Id, and does not shuffle or
drop reviews during preparation. Tokenization uses `Summary + ". " + Text`, a
maximum of 512 tokens, and no fixed padding. Training pads each batch dynamically.

The default workspace is the ignored `external/assets/` directory. Preparation
writes `prepared/train.parquet`, `prepared/eval.parquet`, `prepared/tokens/`, and
`prepared/manifest.json`, and downloads the pinned model to `base_model/`.
It replaces the previous splitter's top-level `train.parquet`/`eval.parquet`
workflow; old split files are not consumed by the new trainer.

For an existing deduplicated CSV, use:

```sh
python external/train_bert/split_train_test.py \
  --source /path/to/Reviews.csv --root external/assets
```

`--holdout-products` and `--tokenizer-workers` are configurable (defaults 200 and
8). Use a fresh `--root` when changing the dataset or holdout. A completed cache
is reused only if its source checksum, split checksums, and settings match.

### Optional cross-split token cleanup

To also remove residual copies that BERT sees as identical inputs:

```sh
python external/train_bert/download_and_dedup_data.py \
  --output-dir external/assets/token_clean --remove-token-overlap
python external/train_bert/split_train_test.py --root external/assets/token_clean
```

Pass `--root external/assets/token_clean` to training and prediction as well.
The flag runs after the three standard deduplication policies. It uses the pinned
tokenizer on `Summary + ". " + Text`, truncated to 512 tokens, and removes **both
sides** of cross-split matches whose token IDs, UserId, and Score agree. Timestamp
need not agree. Matches with different authors or ratings are deliberately kept.
This is a selective duplicate-review cleanup, not a guarantee that no identical
model inputs remain across the split.

The holdout is the top 200 products after standard deduplication. The cleanup
refuses to proceed if removals would change membership of that set. Use the same
`--holdout-products` value for both scripts when overriding 200. Review IDs and
relative row order are preserved. `token_overlap_removed.csv` lists the removed
rows; `dedup_summary.json` records the matched pairs and tokenizer revision.

On the reference corpus this removes 18 training and 18 held-out reviews, leaving
341,112 training and 52,706 held-out reviews. Two input-matching pairs with different
authors or ratings remain by design. The flag is off by default; ordinary
deduplication retains its previous behavior and standard-library-only dependency.
The cleaned CSV SHA256 is
`2971af7fd425b8a30ce241f0d1733b18555ef525bc561f90543477f07056f7e7`.
With the same eight-H100 settings and seed as the reference run, measured holdout
accuracies were 66.5693% (base), 75.9477% (epoch 1), and 76.6175% (epoch 2).

## Train and predict

For eight GPUs, matching the reference training settings:

```sh
torchrun --standalone --nproc_per_node=8 external/train_bert/train_bert.py \
  --run-name bert --global-batch 256 --per-device-batch 32 --precision bf16
torchrun --standalone --nproc_per_node=8 external/train_bert/predict_bert.py \
  --run-name bert --precision bf16
```

For one GPU:

```sh
torchrun --standalone --nproc_per_node=1 external/train_bert/train_bert.py --run-name bert_single
torchrun --standalone --nproc_per_node=1 external/train_bert/predict_bert.py --run-name bert_single
```

Use `CUDA_VISIBLE_DEVICES` to select GPUs. All workers in a run should use the same
GPU type. Pass the same `--root` to preparation, training, and prediction if using
a custom workspace. Training refuses to overwrite a nonempty run directory.

Training uses two epochs, seed 123456, learning rate 5e-5, 500 warmup steps, linear
learning-rate decay, weight decay 0.01, fused AdamW, length-grouped batches, and
gradient clipping at norm 1. No early stopping or validation-based model selection
is used. The default precision is BF16 on GPUs with native support and FP16 on
older GPUs; `--precision fp32` disables mixed precision. TF32 is enabled on
Ampere and newer GPUs. These options apply to prediction as well.

The default batch plan keeps the effective batch at 256 whenever possible:

| GPUs | Per-device batch | Accumulation steps | Effective batch |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 8 | 256 |
| 2 | 32 | 4 | 256 |
| 3 | 17 | 5 | 255 |
| 4 | 32 | 2 | 256 |
| 5 | 17 | 3 | 255 |
| 6 | 21 | 2 | 252 |
| 7 | 18 | 2 | 252 |
| 8 | 32 | 1 | 256 |

For 3, 5, 6, or 7 GPUs, equal-sized batches cannot divide 256 evenly, so the
default rounds down to the nearest multiple of GPU count. The actual batch plan
is printed and recorded. An explicit `--global-batch` must divide evenly and is
never silently changed. Learning rate is not scaled with batch size.

On a smaller-memory GPU, reduce `--per-device-batch` to a divisor of the per-GPU
effective batch (for example, 16 with 1, 2, 4, or 8 GPUs); accumulation increases
automatically. Prediction has its own `--per-device-batch` (default 128), which
can also be reduced. Both scripts accept `--num-workers` (default 2, per process).

Seeds, model revision, data checksums, software versions, and source checksums are
recorded. A fixed seed does **not** guarantee identical weights across GPU counts,
GPU types, precision modes, or library/driver versions. Changing the microbatch
also changes length grouping and stochastic operations. The eight-H100 settings
above match the reference configuration; GPU kernels are not forced bitwise
deterministic. Distributed sampling may repeat a few tail examples to balance
workers; no input rows are dropped.

## Outputs

Each run lives under `external/assets/runs/<run-name>/`:

- `checkpoints/`: one checkpoint per epoch, including model, tokenizer, optimizer,
  scheduler, and RNG state. `checkpoints.json` maps epochs to checkpoint paths.
- `run_config.json` and `training_summary.json`: training provenance and timings.
- `prediction_config.json` and `evaluation_summary.json`: evaluation provenance,
  exact-star accuracy, product-macro accuracy, MAE, MSE, and confusion matrices.
- `predictions/{base,epoch1,epoch2}.h5`: one array per ProductId, shaped
  `(2, number_of_reviews)`, with true then argmax-predicted integer ratings (1–5).
  These are compatible with `AmazonReviewDataset(file_path=Path(...))`.
- Matching parquet files retain review IDs, class probabilities, expected ratings,
  and per-review errors; matching `_per_product.csv` files summarize each product.

`base.h5` corresponds to the former `predict_bad.h5` / `amazon_review_raw.h5`;
`epoch2.h5` corresponds to `predict_finetuned.h5` / `amazon_review.h5`. Epoch 1 is
an additional checkpoint. Existing files under `data/amazon/` are not overwritten.

Prediction restores original review order after sorting by sequence length for
efficiency. By default it evaluates all three models; `--models base` can evaluate
the pretrained model without trained checkpoints. Resume interrupted training
with `--resume-from-checkpoint /path/to/checkpoint-N` and the original run name,
data, seed, GPU count, and batch/precision settings. A positive `--max-steps` runs
a timing probe without saving epoch checkpoints; use a separate run name.

The reference deduplicated dataset has 341,130 training and 52,724 held-out reviews.
Its CSV SHA256 is
`38f8cef570894340ab88157b4eae98a3c43fd02c396ad1c5e6f3cfcab5fb514e`.
Measured held-out accuracies were 66.5617% (base), 75.5766% (epoch 1), and 76.6861%
(epoch 2). These counts and results describe the reference inputs, not hard-coded
requirements for other deduplication thresholds or holdout sizes.

## Lightweight checks

```sh
python -m unittest discover -s external/train_bert -p 'test_*.py'
```

These checks exercise batch planning and split invariants without loading a model,
downloading data, or using GPUs.
