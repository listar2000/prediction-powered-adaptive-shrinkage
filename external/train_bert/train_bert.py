"""Seeded single-/multi-GPU BERT training; invoke with torchrun on any platform."""
import argparse
import json
import os
from pathlib import Path
import time

from common import (ASSETS_DIR, SEED, GLOBAL_BATCH, configure_cuda, environment,
                    plan_batch, prepared_manifest, save_json, seed_everything)

from datasets import load_from_disk
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments)


class RunCallback(TrainerCallback):
    def __init__(self, folder, probe, global_batch):
        self.folder, self.probe = folder, probe
        self.global_batch = global_batch
        self.start = None
        self.measured_steps = 0
        self.elapsed = None
        self.epoch_checkpoints = {}

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if state.global_step == 5:
                torch.cuda.synchronize()
                self.start = time.perf_counter()
            elif self.start is not None:
                torch.cuda.synchronize()
                self.elapsed = time.perf_counter() - self.start
                self.measured_steps = state.global_step - 5
                if self.probe or state.global_step % 100 == 0:
                    save_json(self.folder / "progress.json", {
                        "step": state.global_step, "epoch": state.epoch,
                        "measured_seconds": self.elapsed,
                        "steady_examples_per_second": self.measured_steps * self.global_batch / self.elapsed,
                    })

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            epoch = int(round(state.epoch))
            assert abs(state.epoch - epoch) < 1e-6, state.epoch
            self.epoch_checkpoints[str(epoch)] = str(Path(args.output_dir) / f"checkpoint-{state.global_step}")
            save_json(self.folder / "checkpoints.json", self.epoch_checkpoints)


def main(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    batch = plan_batch(world, args.global_batch, args.per_device_batch)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    precision = configure_cuda(args.precision)
    seed_everything(args.seed, tf32=precision["tf32"])
    root = Path(args.root).resolve()
    run = root / "runs" / args.run_name
    # Only rank zero checks: other ranks may arrive after it starts writing logs.
    if (int(os.environ.get("RANK", "0")) == 0 and run.exists()
            and any(run.iterdir()) and not args.resume_from_checkpoint):
        raise ValueError("Run directory is not empty; choose a new --run-name or resume a checkpoint")
    manifest = prepared_manifest(root)
    probe = args.max_steps > 0
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"batch": batch, "precision": precision, "seed": args.seed}), flush=True)
        if args.global_batch is None and batch["global_batch"] != GLOBAL_BATCH:
            print(f"Using global batch {batch['global_batch']} so it divides evenly across {world} GPUs.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(root / "base_model", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(root / "base_model", attn_implementation="sdpa")
    assert model.config.num_labels == 5
    data = load_from_disk(root / "prepared/tokens")["train"]
    if len(data) != manifest["train_rows"]:
        raise ValueError("Training token count differs from the prepared manifest")
    callback = RunCallback(run, probe, batch["global_batch"])
    if args.resume_from_checkpoint and (run / "checkpoints.json").exists():
        callback.epoch_checkpoints = json.loads((run / "checkpoints.json").read_text())
    training_args = TrainingArguments(
        output_dir=str(run / "checkpoints"),
        num_train_epochs=2,
        max_steps=args.max_steps,
        per_device_train_batch_size=batch["per_device_batch"],
        gradient_accumulation_steps=batch["gradient_accumulation_steps"],
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        **precision,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=False,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_drop_last=False,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        eval_strategy="no",
        save_strategy="no" if probe else "epoch",
        save_total_limit=2,
        save_safetensors=True,
        logging_strategy="steps",
        logging_steps=5 if probe else 50,
        report_to="none",
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=data,
                      processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
                      callbacks=[callback])
    if trainer.is_world_process_zero():
        run.mkdir(parents=True, exist_ok=True)
        save_json(run / "run_config.json", {
            "environment": environment(), "arguments": training_args.to_dict(),
            "seed": args.seed, **batch, "probe": probe,
            "train_rows": len(data), "attn_implementation": model.config._attn_implementation,
            "data_manifest": manifest,
            "reproducibility": "Fixed seed and pinned inputs/software; GPU kernels are not forced bitwise deterministic.",
        })
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        runtime = time.perf_counter() - started
        summary = {
            **result.metrics, "wall_seconds": runtime, "global_step": trainer.state.global_step,
            "completed_epochs": trainer.state.epoch, **batch,
            "max_gpu_allocated_gib_rank0": torch.cuda.max_memory_allocated() / 2**30,
            "timed_steps": callback.measured_steps, "timed_seconds": callback.elapsed,
            "steady_examples_per_second": callback.measured_steps * batch["global_batch"] / callback.elapsed if callback.elapsed else None,
            "epoch_checkpoints": callback.epoch_checkpoints,
        }
        if not probe:
            assert set(callback.epoch_checkpoints) == {"1", "2"}
        save_json(run / "training_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ASSETS_DIR)
    parser.add_argument("--run-name", default="bert")
    parser.add_argument("--global-batch", type=int, help="Default: largest multiple of GPU count at most 256")
    parser.add_argument("--per-device-batch", type=int, help="Default: largest compatible microbatch at most 32")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value runs a timing probe without checkpoints")
    parser.add_argument("--resume-from-checkpoint")
    main(parser.parse_args())
