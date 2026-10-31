"""Shared settings, GPU batch planning, and reproducibility metadata."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random

# Match the reference run; callers can override these environment settings.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
SEED = 123456
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"
MAX_LENGTH = 512
GLOBAL_BATCH = 256


def plan_batch(world_size, global_batch=None, per_device_batch=None):
    """Use a default global batch at most 256, divisible by the GPU count.

    Choose the largest per-device divisor at most 32. Explicit batch sizes must
    divide exactly; an explicit global target is never silently adjusted.
    """
    if not 1 <= world_size <= 8:
        raise ValueError("Use between 1 and 8 GPU processes")
    effective = (GLOBAL_BATCH // world_size) * world_size if global_batch is None else global_batch
    if effective <= 0 or effective % world_size:
        raise ValueError("--global-batch must be positive and divisible by the GPU count")
    per_rank = effective // world_size
    if per_device_batch is None:
        per_device_batch = max(n for n in range(1, min(32, per_rank) + 1) if per_rank % n == 0)
    if per_device_batch <= 0 or per_rank % per_device_batch:
        raise ValueError("--per-device-batch times GPU count must divide the global batch")
    return {"world_size": world_size, "global_batch": effective,
            "per_device_batch": per_device_batch,
            "gradient_accumulation_steps": per_rank // per_device_batch}


def configure_cuda(precision="auto"):
    """Select the local CUDA device and precision supported by this GPU."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Training and prediction require NVIDIA CUDA GPUs")
    if "LOCAL_RANK" not in os.environ and torch.cuda.device_count() > 1:
        raise RuntimeError("Use torchrun --nproc_per_node=N, or expose one GPU with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
    if precision == "auto":
        precision = "bf16" if bf16_supported else "fp16"
    if precision == "bf16" and not bf16_supported:
        raise ValueError("This GPU does not support native BF16; use --precision fp16 or fp32")
    return {"bf16": precision == "bf16", "fp16": precision == "fp16",
            "tf32": torch.cuda.get_device_capability()[0] >= 8}


def seed_everything(seed=SEED, tf32=True):
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def prepared_manifest(root):
    """Reject mismatched inputs rather than mixing splits and cached tokens."""
    root = Path(root)
    manifest = json.loads((root / "prepared/manifest.json").read_text())
    if manifest["model_revision"] != MODEL_REVISION or manifest["max_length"] != MAX_LENGTH:
        raise ValueError("Prepared model revision or token length differs; prepare a fresh root")
    for split, expected in manifest["parquet_sha256"].items():
        if sha256(root / "prepared" / f"{split}.parquet") != expected:
            raise ValueError(f"Prepared {split} parquet checksum differs from its manifest")
    if not (root / "prepared/tokens/dataset_dict.json").is_file():
        raise FileNotFoundError("Missing token cache; run split_train_test.py first")
    return manifest


def environment():
    import torch

    return {
        "versions": {p: importlib.metadata.version(p) for p in
                     ["torch", "transformers", "accelerate", "datasets", "numpy", "pandas", "pyarrow"]},
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
