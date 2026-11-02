# LMArena preparation

Run these commands from the repository root with `datasets`, `transformers`,
`httpx`, `pandas`, `python-dotenv`, and `tqdm` installed. The reward-model server
also requires SGLang and a suitable GPU.

The paper uses all languages, clear human preferences, single-turn text-only
conversations, and Bradley–Terry probabilities at temperature 1:

```sh
python data/lmarena/summarize_dataset.py \
  --single-turn-only --min-comparisons 100 \
  --output data/lmarena/sample_data/llm_pair_summary_single_turn.csv

python -m sglang.launch_server --config data/lmarena/skywork_server_args.yaml
```

Once the server is ready, run in another terminal:

```sh
python data/lmarena/run_judge_skywork.py \
  --pairs-csv data/lmarena/sample_data/llm_pair_summary_single_turn.csv \
  --output-dir data/lmarena/results_skywork \
  --single-turn-only --prediction-mode bradley-terry --temperature 1 \
  --max-concurrent 128

python data/lmarena/convert_summary.py \
  --input data/lmarena/results_skywork/summary.csv \
  --output data/lmarena/clean_data/clean_summary_v2.csv \
  --min-comparisons 100
```

The scorer's existing CLI default remains binary; specify Bradley–Terry mode as
above for the paper. The converter preserves comparison order and alphabetically
orders the two models, flipping outcomes and probabilities together. It rejects
invalid predictions and undersized final groups instead of silently dropping them.
The existing probability dataset has 40,663 comparisons across 298 groups.

Dataset loading and the default model tokenizer are pinned in `revisions.py`;
the server YAML pins the same model revision:

| Asset | Default revision |
| --- | --- |
| LMArena dataset | `6322995ab34d7c2693e3f47dd13fa5caa0789a74` |
| Skywork reward model | `cba2f842f3f1af2f1b2f0d35e794d789976390c5` |

Checked against upstream history on 2026-11-01: the dataset's data files have
not changed since its 2025-08-01 upload, and the model's runtime files have not
changed since 2025-07-01. These pins preserve the upstream contents available
before the repository's recorded LMArena work in early 2026. This inference is
supported by file-hash comparisons, rather than an original inference-run manifest.

To override, pass the same `--dataset-revision <commit>` to summarization and
scoring. For the model, set `revision: <commit>` in the server YAML and pass that
same commit through the judge's `--model-revision` option. The latter pins the
client tokenizer; it does not change or verify the already-running server.
A custom `--model-name` does not inherit the Skywork revision.

Summarization writes a JSON sidecar with the requested revision and dataset
fingerprint. Scoring writes `run_metadata.json` with arguments, dataset metadata,
valid-row totals, and exclusion counts. A null revision (for a custom model)
means it was not pinned; the tokenizer commit is recorded only when exposed by
the tokenizer. Detailed
per-pair CSVs retain failed row IDs and parsing/scoring failure markers. These
records describe new runs; they do not establish the causes of historical exclusions.

`download_sample.py` exports only two example rows and is not a preparation step.
