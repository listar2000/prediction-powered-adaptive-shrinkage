"""
Stage 2b: Skywork Reward Model Judge for LMArena pairwise comparisons.

Uses the Skywork-Reward-V2 model (trained via Bradley-Terry) which directly outputs
a scalar reward score for each user-assistant conversation. This replaces the generic
LLM-judge approach (manual prompt + parsing) with a purpose-built reward model.

This script processes ALL pairs from a CSV file in a single run, loading the dataset
once and efficiently utilizing the SGLang inference server (which handles data
parallelism internally).

Outputs:
    - Per-pair detailed CSVs in {output_dir}/pairs/ (includes prompts/responses)
    - A single summary CSV at {output_dir}/summary.csv (concise: id, models, scores, winner)

Usage:
    1. Start the SGLang server:
       python -m sglang.launch_server --config-file data/lmarena/skywork_server_args.yaml

    2. Run this script:
       python data/lmarena/run_judge_skywork.py \\
           --pairs-csv data/lmarena/sample_data/llm_pair_summary.csv \\
           --output-dir data/lmarena/results_skywork \\
           --language en --single-turn-only
"""

import argparse
import asyncio
import os
import re

import httpx
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm

import dotenv

dotenv.load_dotenv()


DEFAULT_PORT = 30000
DEFAULT_MODEL = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"


# ---------------------------------------------------------------------------
# Dataset loading & filtering
# ---------------------------------------------------------------------------
def load_lmarena_dataset() -> pd.DataFrame:
    """Load the full LMArena dataset from HuggingFace."""
    ds = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
    return ds.to_pandas()


def _is_single_turn(conv: list[dict]) -> bool:
    return (
        len(conv) == 2
        and conv[0]["role"] == "user"
        and conv[1]["role"] == "assistant"
        and len(conv[0]["content"]) == 1
        and len(conv[1]["content"]) == 1
        and conv[0]["content"][0]["type"] == "text"
        and conv[1]["content"][0]["type"] == "text"
    )

    
def apply_global_filters(
    df: pd.DataFrame,
    language: str | None = None,
    single_turn_only: bool = False,
) -> pd.DataFrame:
    """Apply global filters: valid winners, language, turn count."""
    # Only clear winners
    df = df[df["winner"].isin(["model_a", "model_b"])].copy()

    if language is not None:
        df = df[df["language"] == language]

    if single_turn_only:
        df = df[(df["conversation_a"].apply(_is_single_turn) & df["conversation_b"].apply(_is_single_turn))]

    return df.reset_index(drop=True)


def filter_for_pair(df: pd.DataFrame, llm_a: str, llm_b: str) -> pd.DataFrame:
    """Filter dataset to comparisons between a specific pair (in either order)."""
    mask = (
        ((df["model_a"] == llm_a) & (df["model_b"] == llm_b))
        | ((df["model_a"] == llm_b) & (df["model_b"] == llm_a))
    )
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Content extraction & reward-model text formatting
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Extract plain text from content that may be a string or structured list/array.

    The dataset stores message content as either a plain string or a list of
    typed blocks like ``[{"type": "text", "text": "..."}]``.
    """
    if isinstance(content, str):
        return content
    try:
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item["text"]
    except (TypeError, KeyError):
        pass
    # Last resort – stringify
    return str(content)


def extract_conversation_content(row: pd.Series) -> tuple[str, str, str] | None:
    """Extract the first user prompt and both model responses from a dataset row.

    Works for single-turn conversations. Returns ``(user_prompt, response_a,
    response_b)`` or ``None`` on failure.
    """
    try:
        conv_a = row["conversation_a"]
        conv_b = row["conversation_b"]

        user_prompt = _extract_text(conv_a[0]["content"])
        response_a = _extract_text(conv_a[1]["content"])
        response_b = _extract_text(conv_b[1]["content"])

        return user_prompt, response_a, response_b
    except Exception:
        return None


def format_reward_text(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    response: str,
) -> str:
    """Format a (user, assistant) pair for the Skywork reward model.

    Applies the tokenizer's chat template and strips a leading BOS token if
    present (required by Skywork-Reward-V2).
    """
    conv = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(conv, tokenize=False)
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token) :]
    return text


# ---------------------------------------------------------------------------
# Async scoring via SGLang /classify endpoint
# ---------------------------------------------------------------------------

async def score_comparison(
    client: httpx.AsyncClient,
    classify_url: str,
    model_name: str,
    text_a: str,
    text_b: str,
    task_idx: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, float | None, float | None]:
    """Score both responses for a single comparison via the /classify endpoint."""
    async with semaphore:
        payload = {"model": model_name, "text": [text_a, text_b]}
        try:
            resp = await client.post(classify_url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            score_a = data[0]["embedding"][0]
            score_b = data[1]["embedding"][0]
            return task_idx, score_a, score_b
        except Exception as e:
            print(f"  Error on task {task_idx}: {e}")
            return task_idx, None, None


async def judge_pair(
    client: httpx.AsyncClient,
    classify_url: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    df_pair: pd.DataFrame,
    semaphore: asyncio.Semaphore,
    pair_label: str,
) -> list[dict]:
    """Judge all comparisons for a single LLM pair and return result dicts."""
    results: list[dict | None] = [None] * len(df_pair)
    tasks = []
    skipped = 0

    for idx, (_, row) in enumerate(df_pair.iterrows()):
        content = extract_conversation_content(row)
        if content is None:
            skipped += 1
            results[idx] = {
                "id": row["id"],
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "winner": row["winner"],
                "prediction": "parse_error",
                "score_a": None,
                "score_b": None,
                "language": row.get("language", ""),
                "user_prompt": "",
                "response_a": "",
                "response_b": "",
            }
            continue

        user_prompt, response_a, response_b = content
        text_a = format_reward_text(tokenizer, user_prompt, response_a)
        text_b = format_reward_text(tokenizer, user_prompt, response_b)

        task = asyncio.create_task(
            score_comparison(
                client, classify_url, model_name, text_a, text_b, idx, semaphore
            )
        )
        tasks.append((idx, row, user_prompt, response_a, response_b, task))

    if skipped > 0:
        print(f"  Skipped {skipped} rows due to parsing errors")

    # Await all tasks with progress bar
    for idx, row, user_prompt, response_a, response_b, task in tqdm(
        tasks, desc=f"  Judging {pair_label}"
    ):
        task_idx, score_a, score_b = await task

        if score_a is not None and score_b is not None:
            prediction = "model_a" if score_a >= score_b else "model_b"
        else:
            prediction = "score_error"

        results[task_idx] = {
            "id": row["id"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "winner": row["winner"],
            "prediction": prediction,
            "score_a": score_a,
            "score_b": score_b,
            "language": row.get("language", ""),
            "user_prompt": user_prompt,
            "response_a": response_a,
            "response_b": response_b,
        }

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Sanitize a string for use in a filename."""
    return re.sub(r"[^\w\-]", "_", name)


def print_pair_summary(results_df: pd.DataFrame) -> None:
    """Print accuracy stats for a single pair."""
    valid = results_df[results_df["prediction"].isin(["model_a", "model_b"])]
    if len(valid) > 0:
        acc = (valid["prediction"] == valid["winner"]).mean()
        print(f"  Accuracy: {acc:.1%} ({len(valid)} valid predictions)")


async def run_pipeline(args) -> None:
    """Main async pipeline: load once, judge every pair, save results."""
    # Load tokenizer
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    print("Loading LMArena dataset...")
    df = load_lmarena_dataset()
    print(f"Loaded {len(df)} rows")

    # Apply global filters
    df = apply_global_filters(
        df, language=args.language, single_turn_only=args.single_turn_only
    )
    print(f"After global filters: {len(df)} rows")

    # Read pairs CSV
    pairs_df = pd.read_csv(args.pairs_csv)
    print(f"Will process {len(pairs_df)} pairs from {args.pairs_csv}")

    if args.max_pairs is not None:
        pairs_df = pairs_df.head(args.max_pairs)
        print(f"  (limited to first {len(pairs_df)} pairs)")

    # Prepare output directories
    pairs_dir = os.path.join(args.output_dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    # Concurrency & HTTP setup
    classify_url = f"http://localhost:{args.port}/classify"
    semaphore = asyncio.Semaphore(args.max_concurrent)
    all_summary_rows: list[pd.DataFrame] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        for pair_idx, pair_row in pairs_df.iterrows():
            llm_a = pair_row.iloc[0]  # first column = llm_a
            llm_b = pair_row.iloc[1]  # second column = llm_b
            pair_label = f"{llm_a} vs {llm_b}"

            df_pair = filter_for_pair(df, llm_a, llm_b)

            if len(df_pair) == 0:
                print(f"\n[{pair_idx}] {pair_label}: no comparisons found, skipping.")
                continue

            if args.debug:
                df_pair = df_pair.head(10)

            print(f"\n[{pair_idx}] {pair_label}: {len(df_pair)} comparisons")

            results = await judge_pair(
                client,
                classify_url,
                args.model_name,
                tokenizer,
                df_pair,
                semaphore,
                pair_label,
            )

            results_df = pd.DataFrame(results)

            # Save detailed per-pair CSV
            pair_filename = (
                f"{sanitize_filename(llm_a)}_vs_{sanitize_filename(llm_b)}.csv"
            )
            results_df.to_csv(os.path.join(pairs_dir, pair_filename), index=False)

            # Accumulate concise summary rows (no prompts / responses)
            summary_cols = [
                "id", "model_a", "model_b", "winner",
                "prediction", "score_a", "score_b",
            ]
            all_summary_rows.append(results_df[summary_cols])

            print_pair_summary(results_df)

    # ---- Save overall summary ----
    if all_summary_rows:
        summary_df = pd.concat(all_summary_rows, ignore_index=True)
        summary_path = os.path.join(args.output_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 50)
        print("Overall Summary")
        print("=" * 50)
        print(f"Pairs processed : {len(all_summary_rows)}")
        print(f"Total comparisons: {len(summary_df)}")

        valid = summary_df[summary_df["prediction"].isin(["model_a", "model_b"])]
        if len(valid) > 0:
            acc = (valid["prediction"] == valid["winner"]).mean()
            print(f"Overall accuracy : {acc:.1%} ({len(valid)} valid)")

        print(f"\nSaved summary to {summary_path}")
        print(f"Saved per-pair results to {pairs_dir}/")
    else:
        print("\nNo pairs had any comparisons. Nothing was saved.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Skywork Reward Model judge on LMArena pairwise comparisons"
    )
    parser.add_argument(
        "--pairs-csv",
        type=str,
        required=True,
        help="CSV with LLM pairs (first two columns are the model names)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/lmarena/results_skywork",
        help="Output directory for results",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Filter by language (e.g., 'en')",
    )
    parser.add_argument(
        "--single-turn-only", "-s",
        action="store_true",
        help="Only include single-turn conversations",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="SGLang server port",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name for the Skywork reward model",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=64,
        help="Maximum concurrent requests to the server",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit number of pairs to process (useful for testing)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only process first 10 rows per pair",
    )
    args = parser.parse_args()

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
