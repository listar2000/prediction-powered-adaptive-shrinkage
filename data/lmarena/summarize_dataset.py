"""
Stage 1: Dataset Analysis - Summarize LLM pair comparisons from LMArena dataset.

This script reads the lmarena-ai/arena-human-preference-140k dataset and produces
a ranked table of (LLM P, LLM Q, comparisons) where each row represents a unique
pair of LLMs and the number of times they were matched against each other.

Only comparisons with clear winners (model_a or model_b) are included.
LLM pairs are alphabetically ordered (smaller name first).

Example usage:
python data/lmarena/summarize_dataset.py --output data/lmarena/sample_data/llm_pair_summary_all_languages.csv --min-comparisons 50
"""

import argparse
import os
from datasets import load_dataset
import pandas as pd
import dotenv

from run_judge_skywork import _is_single_turn

dotenv.load_dotenv()


def load_lmarena_dataset() -> pd.DataFrame:
    """Load the full LMArena dataset from HuggingFace."""
    ds = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
    return ds.to_pandas()


def filter_single_turn_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataset to only include single-turn conversations.

    which is defined by ``conversation_a`` and ``conversation_b`` columns are lists with exactly 2 elements.
    we further check that the first element is a dict with role "user" and the second one a dict with role "assistant".
    """
    return df[df["conversation_a"].apply(_is_single_turn) & df["conversation_b"].apply(_is_single_turn)]


def summarize_comparisons(
    df: pd.DataFrame,
    language: str | None = None,
    min_comparisons: int = 1,
    single_turn_only: bool = False,
) -> pd.DataFrame:
    """
    Summarize the dataset into a ranked table of LLM pair comparisons.
    
    Args:
        df: The raw LMArena DataFrame
        language: Optional language filter (e.g., 'en', 'zh')
        min_comparisons: Minimum number of comparisons to include a pair
    
    Returns:
        DataFrame with columns: llm_p, llm_q, comparisons
        Sorted by comparisons in descending order
    """
    # Filter to only include clear winners
    valid_winners = ["model_a", "model_b"]
    df_filtered = df[df["winner"].isin(valid_winners)].copy()
    
    # Apply language filter if specified
    if language is not None:
        df_filtered = df_filtered[df_filtered["language"] == language]

    # Filter to only include single-turn conversations if specified
    if single_turn_only:
        df_filtered = filter_single_turn_only(df_filtered)

    # Apply ordering (vectorized approach for speed)
    mask = df_filtered["model_a"] <= df_filtered["model_b"]
    df_filtered["llm_a"] = df_filtered["model_a"].where(mask, df_filtered["model_b"])
    df_filtered["llm_b"] = df_filtered["model_b"].where(mask, df_filtered["model_a"])
    
    # Group and count
    summary = (
        df_filtered.groupby(["llm_a", "llm_b"])
        .size()
        .reset_index(name="comparisons")
    )
    
    # Filter by minimum comparisons
    summary = summary[summary["comparisons"] >= min_comparisons]
    
    # Sort by comparisons descending
    summary = summary.sort_values("comparisons", ascending=False).reset_index(drop=True)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Summarize LMArena dataset into LLM pair comparisons"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/lmarena/llm_pair_summary.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Filter by language (e.g., 'en', 'zh', 'es')"
    )
    parser.add_argument(
        "--min-comparisons", "-m",
        type=int,
        default=1,
        help="Minimum number of comparisons to include a pair"
    )
    parser.add_argument(
        "--single-turn-only", "-s",
        action="store_true",
        help="Only include single-turn conversations"
    )
    parser.add_argument(
        "--top-n", "-n",
        type=int,
        default=None,
        help="Only output the top N pairs by comparison count"
    )
    args = parser.parse_args()
    
    print("Loading LMArena dataset...")
    df = load_lmarena_dataset()
    print(f"Loaded {len(df)} rows")
    
    print("Summarizing comparisons...")
    summary = summarize_comparisons(
        df,
        language=args.language,
        min_comparisons=args.min_comparisons,
        single_turn_only=args.single_turn_only,
    )
    
    if args.top_n is not None:
        summary = summary.head(args.top_n)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    summary.to_csv(args.output, index=False)
    print(f"Saved summary to {args.output}")
    print(f"Total unique LLM pairs: {len(summary)}")
    print(f"\nTop 10 pairs by comparison count:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
