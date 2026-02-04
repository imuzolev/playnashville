"""
Anthropic provider module for lask
"""

import os
import sys
import json
from typing import Dict, Any, Optional, Union, Iterator, List
import requests

from src.config import LaskConfig


def call_api(
    config: LaskConfig,
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Union[str, Iterator[str]]:
    """
    Call the Anthropic API with the given prompt.

    Args:
        config (LaskConfig): Configuration object
        prompt (str): The user prompt
        conversation_history (Optional[List[Dict[str, str]]]): List of conversation messages
                                                             for multi-turn dialogues

    Returns:
        Union[str, Iterator[str]]: The response from the Anthropic API,
                                  either full text or a stream iterator

    Raises:
        Exception: If there's an error calling the Anthropic API
    """
    # Get provider-specific config
    anthropic_config = config.get_provider_config("anthropic")

    # Get API key
    api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY") or anthropic_config.api_key
    if not api_key:
        print(
            "Error: Please set the ANTHROPIC_API_KEY environment variable or add 'api_key' under [anthropic] section in ~/.lask-config"
        )
        sys.exit(1)

    # Get model (Claude by default)
    model: str = anthropic_config.model or "claude-3-opus-20240229"

    # Check if streaming is enabled (default to True)
    streaming: bool = anthropic_config.get("streaming", True)

    headers: Dict[str, str] = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    # If conversation history is provided, use that instead of building new messages
    if conversation_history is not None:
        messages = conversation_history
    else:
        messages = []

        # Add system prompt if available
        provider_system_prompt = anthropic_config.system_prompt
        default_system_prompt = config.system_prompt

        # For Anthropic, system prompts are handled differently
        # Claude uses a "system" role message at the start of the conversation
        if provider_system_prompt is not None:
            messages.append({"role": "system", "content": provider_system_prompt})
        elif default_system_prompt is not None:
            messages.append({"role": "system", "content": default_system_prompt})

        # Add user message
        messages.append({"role": "user", "content": prompt})

    data: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": anthropic_config.max_tokens or 4096,
        "stream": streaming,
    }

    if anthropic_config.temperature is not None:
        data["temperature"] = anthropic_config.temperature

    # Only print the prompt in one-off mode, not in conversation mode to avoid clutter
    if conversation_history is None:
        print(f"Prompting Anthropic API with model {model}: {prompt}\n")

    if streaming:
        return stream_anthropic_response(headers, data)
    else:
        return non_streaming_anthropic_response(headers, data)


def stream_anthropic_response(
    headers: Dict[str, str], data: Dict[str, Any]
) -> Iterator[str]:
    """
    Stream the response from Anthropic API.

    Args:
        headers (Dict[str, str]): Request headers
        data (Dict[str, Any]): Request data

    Yields:
        str: Chunks of the response as they arrive
    """
    response = requests.post(
        "https://api.anthropic.com/v1/messages", headers=headers, json=data, stream=True
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
                    if "type" in chunk and chunk["type"] == "content_block_delta":
                        if "delta" in chunk and "text" in chunk["delta"]:
                            yield chunk["delta"]["text"]
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse JSON: {json_str}")


def non_streaming_anthropic_response(
    headers: Dict[str, str], data: Dict[str, Any]
) -> str:
    """
    Get a non-streaming response from Anthropic API.

    Args:
        headers (Dict[str, str]): Request headers
        data (Dict[str, Any]): Request data without streaming

    Returns:
        str: The full response
    """
    # Disable streaming for non-streaming request
    data["stream"] = False

    response: requests.Response = requests.post(
        "https://api.anthropic.com/v1/messages", headers=headers, json=data
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code} {response.text}")
        sys.exit(1)

    result: Dict[str, Any] = response.json()
    return result["content"][0]["text"]
