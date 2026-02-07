"""
LLM Judge Prompt for predicting human preferences in LMArena comparisons.

This module contains the prompt template for instructing an LLM judge to
evaluate pairwise comparisons and predict which response the human preferred.
"""

import re

SYSTEM_PROMPT = """You are an impartial judge evaluating responses from two AI assistants (Model A and Model B) to a human asker's prompt.

Your task is to determine which model's response is better based on:
1. Helpfulness: Does the response address the user's request?
2. Accuracy: Is the information correct and factual?
3. Clarity: Is the response well-organized and easy to understand?
4. Completeness: Does it fully answer the question without unnecessary content?

IMPORTANT: You must output ONLY one of these two options, wrapped in <answer> tags:

- "<answer>model_a</answer>" if Model A's response is better
- "<answer>model_b</answer>" if Model B's response is better

No need to explain your reasoning. Output only the answer within the <answer> tags."""

USER_PROMPT_TEMPLATE = """## Human Asker's Prompt
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
        'model_a', 'model_b', or 'parsing_error' if parsing fails
    """
    response_lower = response.strip().lower()

    # use regex to extract the answer within the <answer> tags
    match = re.search(r"<answer>(.*?)</answer>", response_lower)
    if match:
        return match.group(1)
    return "parsing_error"


def extract_conversation_content(conversation: dict) -> tuple[str, str, str]:
    """
    Extract the user prompt and both model responses from the conversation field.
    
    The conversation is a dict with keys "user", "model_side_a", "model_side_b".
    
    Args:
        conversation: The conversation dict from the dataset
    
    Returns:
        Tuple of (user_prompt, response_a, response_b) as strings
    """
    def _extract_first_text_content(content_lst: list[dict]) -> str:
        for content in content_lst:
            if "type" in content and content["type"] == "text":
                return content["text"]
        raise ValueError("No text content found")

    # Extract user prompt
    user_content = conversation.get("user", {}).get("content", [])
    user_prompt = _extract_first_text_content(user_content)
    
    # Extract Model A's response
    model_a_content = conversation.get("model_side_a", {}).get("content", [])
    response_a = _extract_first_text_content(model_a_content)
    
    # Extract Model B's response
    model_b_content = conversation.get("model_side_b", {}).get("content", [])
    response_b = _extract_first_text_content(model_b_content)
    
    return user_prompt, response_a, response_b
