"""
LLM Judge Prompt for predicting human preferences in LMArena comparisons.

This module contains the prompt template for instructing an LLM judge to
evaluate pairwise comparisons and predict which response the human preferred.
"""

SYSTEM_PROMPT = """You are an impartial judge evaluating responses from two AI assistants (Model A and Model B) to a user's prompt.

Your task is to determine which model's response is better based on:
1. Helpfulness: Does the response address the user's request?
2. Accuracy: Is the information correct and factual?
3. Clarity: Is the response well-organized and easy to understand?
4. Completeness: Does it fully answer the question without unnecessary content?

IMPORTANT: You must output ONLY one of these two options:
- "WINNER: model_a" if Model A's response is better
- "WINNER: model_b" if Model B's response is better

Do not explain your reasoning. Output only the winner line."""

USER_PROMPT_TEMPLATE = """## User's Prompt
{user_prompt}

---

## Model A's Response
{response_a}

---

## Model B's Response
{response_b}

---

Based on the responses above, which model provided the better response?"""


def format_judge_prompt(user_prompt: str, response_a: str, response_b: str) -> list[dict]:
    """
    Format the judge prompt as a list of messages for the OpenAI-compatible API.
    
    Args:
        user_prompt: The original user's prompt/question
        response_a: Model A's response
        response_b: Model B's response
    
    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    user_content = USER_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt,
        response_a=response_a,
        response_b=response_b,
    )
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_judge_response(response: str) -> str:
    """
    Parse the judge's response to extract the winner.
    
    Args:
        response: The raw response from the judge LLM
    
    Returns:
        'model_a', 'model_b', or 'unknown' if parsing fails
    """
    response_lower = response.strip().lower()
    
    # Look for explicit winner declaration
    if "winner: model_a" in response_lower:
        return "model_a"
    elif "winner: model_b" in response_lower:
        return "model_b"
    
    # Fallback: look for model mentions at the end
    lines = response_lower.strip().split("\n")
    last_line = lines[-1] if lines else ""
    
    if "model_a" in last_line and "model_b" not in last_line:
        return "model_a"
    elif "model_b" in last_line and "model_a" not in last_line:
        return "model_b"
    
    return "unknown"


def extract_conversation_content(full_conversation: dict) -> tuple[str, str, str]:
    """
    Extract the user prompt and both model responses from the full_conversation field.
    
    The full_conversation is a dict with keys 'user', 'model_side_a', 'model_side_b'.
    Each value is a dict with 'role' and 'content' (which is a numpy array of dicts).
    
    Args:
        full_conversation: The full_conversation dict from the dataset
    
    Returns:
        Tuple of (user_prompt, response_a, response_b) as strings
    """
    # Extract user prompt
    user_content = full_conversation.get("user", {}).get("content", [])
    if hasattr(user_content, "__iter__") and len(user_content) > 0:
        user_prompt = user_content[0].get("text", "") if isinstance(user_content[0], dict) else str(user_content[0])
    else:
        user_prompt = str(user_content)
    
    # Extract Model A's response
    model_a_content = full_conversation.get("model_side_a", {}).get("content", [])
    if hasattr(model_a_content, "__iter__") and len(model_a_content) > 0:
        response_a = model_a_content[0].get("text", "") if isinstance(model_a_content[0], dict) else str(model_a_content[0])
    else:
        response_a = str(model_a_content)
    
    # Extract Model B's response
    model_b_content = full_conversation.get("model_side_b", {}).get("content", [])
    if hasattr(model_b_content, "__iter__") and len(model_b_content) > 0:
        response_b = model_b_content[0].get("text", "") if isinstance(model_b_content[0], dict) else str(model_b_content[0])
    else:
        response_b = str(model_b_content)
    
    return user_prompt, response_a, response_b
