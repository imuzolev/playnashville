"""
OpenAI provider module for lask
"""

import os
import sys
import json
from typing import Dict, Any, Optional, Iterator, Union, List
import requests

from src.config import LaskConfig


def call_api(
    config: LaskConfig,
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Union[str, Iterator[str]]:
    """
    Call the OpenAI API with the given prompt.

    Args:
        config (LaskConfig): Configuration object
        prompt (str): The user prompt
        conversation_history (Optional[List[Dict[str, str]]]): List of conversation messages
                                                             for multi-turn dialogues

    Returns:
        Union[str, Iterator[str]]: The response from the OpenAI API,
                                  either full text or a stream iterator
    """
    # Get provider-specific config
    openai_config = config.get_provider_config("openai")

    # Try to get API key from environment variable first, then from config
    api_key: Optional[str] = openai_config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: Please add 'api_key' under [default] or [openai] section in ~/.lask-config, or set the OPENAI_API_KEY environment variable in your shell."
        )
        sys.exit(1)

    # Get model from config or use default
    model: str = openai_config.model or "gpt-4.1"

    # Check if streaming is enabled (default to True)
    streaming: bool = openai_config.get("streaming", True)

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # If conversation history is provided, use that instead of building new messages
    if conversation_history is not None:
        messages = conversation_history
    else:
        messages = []

        # Add system prompt if available
        provider_system_prompt = openai_config.system_prompt
        default_system_prompt = config.system_prompt

        if provider_system_prompt is not None:
            messages.append({"role": "system", "content": provider_system_prompt})
        elif default_system_prompt is not None:
            messages.append({"role": "system", "content": default_system_prompt})

        # Add user message
        messages.append({"role": "user", "content": prompt})

    data: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": streaming,
    }

    # Add optional parameters if specified
    if openai_config.temperature is not None:
        data["temperature"] = openai_config.temperature
    if openai_config.max_tokens is not None:
        data["max_tokens"] = openai_config.max_tokens

    # Only print the prompt in one-off mode, not in conversation mode to avoid clutter
    if conversation_history is None:
        print(f"Prompting OpenAI API with model {data['model']}: {prompt}\n")

    if streaming:
        return stream_openai_response(headers, data)
    else:
        return non_streaming_openai_response(headers, data)


def stream_openai_response(
    headers: Dict[str, str], data: Dict[str, Any]
) -> Iterator[str]:
    """
    Stream the response from OpenAI API.

    Args:
        headers (Dict[str, str]): Request headers
        data (Dict[str, Any]): Request data

    Yields:
        str: Chunks of the response as they arrive
    """
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        stream=True,
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code} {response.text}")
        sys.exit(1)

    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            # Skip the "data: [DONE]" message
            if line_str == "data: [DONE]":
                continue
            # Skip empty data lines
            if line_str.startswith("data: "):
                json_str = line_str[6:]  # Remove "data: " prefix
                try:
                    chunk = json.loads(json_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON: {json_str}")


def non_streaming_openai_response(headers: Dict[str, str], data: Dict[str, Any]) -> str:
    """
    Get a non-streaming response from OpenAI API.

    Args:
        headers (Dict[str, str]): Request headers
        data (Dict[str, Any]): Request data without streaming

    Returns:
        str: The full response
    """
    # Disable streaming for non-streaming request
    data["stream"] = False

    response: requests.Response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=data
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code} {response.text}")
        sys.exit(1)

    result: Dict[str, Any] = response.json()
    return result["choices"][0]["message"]["content"].strip()
