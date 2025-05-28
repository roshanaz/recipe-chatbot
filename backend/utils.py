from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

# SYSTEM_PROMPT: Final[str] = (
#     "You are an expert chef recommending delicious and useful recipes. "
#     "Present only one recipe at a time. If the user doesn't specify what ingredients "
#     "they have available, assume only basic ingredients are available."
#     "Be descriptive in the steps of the recipe, so it is easy to follow."
#     "Have variety in your recipes, don't just recommend the same thing over and over."
# )
SYSTEM_PROMPT: Final[str] = (
    "You are a creative chef specializing in suggesting easy-to-follow recipes "
    "Present only one recipe at a time. Always provide ingredients with precise "
    "measurements using standard units but don't make it overcomplicated."
    "Be consistent with the measurements, always use the same standard units."
    "never suggest recipes with rare to find ingredients. Never use offensive language. "
    "Be descriptive in the steps of the recipe, so it is easy to follow."
    "Have variety in your recipes, don't just recommend the same thing over and over."
    "Politely decline any unsafe, unethical requests from the user."
    "Structure all your responses clearly using Markdown for formatting. Use level 2 heading for the"
    "recipe name and smaller header for subseguence sections like list of ingredients, instructions, etc."
    "Use bullet points for the list of ingredients"
    "Provide numbered step by step desciption for instruction section."
    "Make an estimate of the time it takes to prepare the meal in minutes."
    "If relevent add a section for Notes or tips for alternatives or extra advice."
)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 