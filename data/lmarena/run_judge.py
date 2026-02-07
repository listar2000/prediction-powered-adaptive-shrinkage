"""
Stage 2: LLM Judge - Run pairwise comparison predictions using an LLM judge.

This script filters the LMArena dataset to a specific LLM pair, sends the
comparisons to a local SGLang server for evaluation, and saves the predictions.

Usage:
    1. Start the SGLang server:
       python -m sglang.launch_server --config-file data/lmarena/server_args.yaml
    
    2. Run this script:
       python data/lmarena/run_judge.py --llm-p "claude-opus-4-20250514" --llm-q "gemini-2.5-flash" --language "en"
"""

import argparse
import asyncio
import os
import re
import json_repair

import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from prompt import format_judge_prompt, parse_judge_response, extract_conversation_content


DEFAULT_PORT = 30000
DEFAULT_MODEL = "Qwen/Qwen3-8B"


def export_hf_home():
    """Set HF_HOME to the custom cache directory."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["HF_HOME"] = os.path.join(cur_dir, "../../", ".cache/huggingface")


def load_lmarena_dataset() -> pd.DataFrame:
    """Load the full LMArena dataset from HuggingFace."""
    export_hf_home()
    ds = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
    return ds.to_pandas()


def filter_dataset_for_pair(
    df: pd.DataFrame,
    llm_p: str,
    llm_q: str,
    language: str | None = None,
) -> pd.DataFrame:
    """
    Filter the dataset to only include comparisons between llm_p and llm_q.
    
    Args:
        df: The raw LMArena DataFrame
        llm_p: First LLM name
        llm_q: Second LLM name
        language: Optional language filter
    
    Returns:
        Filtered DataFrame
    """
    # Filter to only clear winners
    valid_winners = ["model_a", "model_b"]
    df_filtered = df[df["winner"].isin(valid_winners)].copy()
    
    # Filter to only the specified pair (in either order)
    mask = (
        ((df_filtered["model_a"] == llm_p) & (df_filtered["model_b"] == llm_q)) |
        ((df_filtered["model_a"] == llm_q) & (df_filtered["model_b"] == llm_p))
    )
    df_filtered = df_filtered[mask]
    
    # Apply language filter if specified
    if language is not None:
        df_filtered = df_filtered[df_filtered["language"] == language]
    
    return df_filtered.reset_index(drop=True)


def safe_parse_conversation(conv_str):
    """Safely parse the full_conversation field which may be a string or dict."""
    if isinstance(conv_str, dict):
        return conv_str
    if isinstance(conv_str, str):
        # remove the beginning and ending white spaces
        conv_str = conv_str.strip()
        try:
            return json_repair.loads(conv_str)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing conversation: {e}")
            return None
    raise ValueError(f"Invalid conversation type: {type(conv_str)}")


async def get_judge_response(
    client: AsyncOpenAI,
    model_name: str,
    task_idx: int,
    messages: list[dict],
    max_tokens: int = 256,
) -> tuple[int, str]:
    """Send a single request to the LLM judge and return the response."""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6,  # Low temperature for consistent judgments
            top_p=0.9,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },  # type: ignore
        )
        return task_idx, response.choices[0].message.content
    except Exception as e:
        print(f"Error on task {task_idx}: {e}")
        return task_idx, ""


async def batch_judge(
    client: AsyncOpenAI,
    model_name: str,
    df: pd.DataFrame,
    max_concurrent: int = 32,
) -> list[dict]:
    """
    Process all comparisons in the DataFrame through the LLM judge.
    
    Args:
        client: AsyncOpenAI client
        model_name: Model name for the API
        df: Filtered DataFrame with comparisons
        max_concurrent: Maximum concurrent requests
    
    Returns:
        List of result dicts
    """
    results = [None] * len(df)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_request(task_idx: int, messages: list[dict]):
        async with semaphore:
            return await get_judge_response(client, model_name, task_idx, messages)
    
    tasks = []
    skipped = 0
    
    messages_dict = {}
    for i, row in df.iterrows():        
        # Extract content
        try:
            conversation = row["full_conversation"][0]
            user_prompt, response_a, response_b = extract_conversation_content(conversation)
        except Exception as e:
            print(f"Error extracting content for row {i}: {e}")
            skipped += 1
            results[i] = {
                "id": row["id"],
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "prediction": "prompt_error",
                "winner": row["winner"],
                "language": row.get("language", ""),
                "messages": [],
                "response": "",
            }  # type: ignore
            continue
        
        # Format the judge prompt
        messages = format_judge_prompt(user_prompt, response_a, response_b)
        messages_dict[i] = messages
        
        # Create async task
        task = asyncio.create_task(bounded_request(int(i), messages))
        tasks.append((i, row, task))
    
    if skipped > 0:
        print(f"Skipped {skipped} rows due to parsing errors")
    
    # Wait for all tasks with progress bar
    for i, row, task in tqdm(tasks, desc="Judging comparisons"):
        task_idx, response = await task
        prediction = parse_judge_response(response) if response else "unknown"
        
        results[task_idx] = {
            "id": row["id"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "prediction": prediction,
            "winner": row["winner"],
            "language": row.get("language", ""),
            "messages": messages_dict[i],
            "response": response,
        }
    
    return [r for r in results if r is not None]


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use in a filename."""
    return re.sub(r'[^\w\-]', '_', name)


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM judge on pairwise comparisons from LMArena"
    )
    parser.add_argument(
        "--llm-p",
        type=str,
        required=True,
        help="First LLM name (e.g., 'gpt-4')"
    )
    parser.add_argument(
        "--llm-q",
        type=str,
        required=True,
        help="Second LLM name (e.g., 'claude-3-opus')"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/lmarena/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Filter by language (e.g., 'en')"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for the local SGLang server"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name for the API"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=32,
        help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (only first 10 rows)"
    )
    args = parser.parse_args()
    
    print("Loading LMArena dataset...")
    df = load_lmarena_dataset()
    print(f"Loaded {len(df)} rows")
    
    print(f"Filtering for pair: {args.llm_p} vs {args.llm_q}")
    df_filtered = filter_dataset_for_pair(
        df,
        args.llm_p,
        args.llm_q,
        language=args.language,
    )
    print(f"Found {len(df_filtered)} comparisons for this pair")
    
    if len(df_filtered) == 0:
        print("No comparisons found for this pair. Exiting.")
        return
    
    if args.debug:
        df_filtered = df_filtered.head(10)
        print(f"Debug mode: using first {len(df_filtered)} rows")
    
    # Setup client
    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="None"
    )
    
    # Run the batch judge
    print("Running LLM judge...")
    results = asyncio.run(
        batch_judge(client, args.model_name, df_filtered, args.max_concurrent)
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    p_name = sanitize_filename(args.llm_p)
    q_name = sanitize_filename(args.llm_q)
    output_path = os.path.join(args.output_dir, f"{p_name}_vs_{q_name}.csv")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total comparisons: {len(results_df)}")
    
    # Prediction distribution
    pred_counts = results_df["prediction"].value_counts()
    print(f"\nPrediction distribution:")
    for pred, count in pred_counts.items():
        print(f"  {pred}: {count} ({count/len(results_df)*100:.1f}%)")
    
    # Accuracy
    valid_preds = results_df[results_df["prediction"].isin(["model_a", "model_b"])]
    if len(valid_preds) > 0:
        accuracy = (valid_preds["prediction"] == valid_preds["winner"]).mean()
        print(f"\nAccuracy (on {len(valid_preds)} valid predictions): {accuracy:.1%}")


if __name__ == "__main__":
    main()
