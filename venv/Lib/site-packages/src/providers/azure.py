"""
Azure OpenAI provider module for lask
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
    Call the Azure OpenAI API with the given prompt.

    Args:
        config (LaskConfig): Configuration object
        prompt (str): The user prompt
        conversation_history (Optional[List[Dict[str, str]]]): List of conversation messages
                                                             for multi-turn dialogues

    Returns:
        Union[str, Iterator[str]]: The response from the Azure OpenAI API,
                                  either full text or a stream iterator

    Raises:
        Exception: If there's an error calling the Azure OpenAI API
    """
    # Get provider-specific config
    azure_config = config.get_provider_config("azure")

    # Get API key
    api_key: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY") or azure_config.api_key
    if not api_key:
        print(
            "Error: Please set the AZURE_OPENAI_API_KEY environment variable or add 'api_key' under [azure] section in ~/.lask-config"
        )
        sys.exit(1)

    # Get required Azure-specific parameters
    resource_name: Optional[str] = azure_config.resource_name
    if not resource_name:
        print(
            "Error: Please set 'resource_name' under [azure] section in ~/.lask-config"
        )
        sys.exit(1)

    # Check if streaming is enabled (default to True)
    streaming: bool = azure_config.get("streaming", True)

    deployment_id: Optional[str] = azure_config.deployment_id
    if not deployment_id:
        print(
            "Error: Please set 'deployment_id' under [azure] section in ~/.lask-config"
        )
        sys.exit(1)

    api_version: str = azure_config.api_version or "2023-05-15"

    # Construct the API URL
    endpoint: str = f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}/chat/completions?api-version={api_version}"

    headers: Dict[str, str] = {"api-key": api_key, "Content-Type": "application/json"}

    # If conversation history is provided, use that instead of building new messages
    if conversation_history is not None:
        messages = conversation_history
    else:
        messages = []

        # Add system prompt if available
        provider_system_prompt = azure_config.system_prompt
        default_system_prompt = config.system_prompt

        if provider_system_prompt is not None:
            messages.append({"role": "system", "content": provider_system_prompt})
        elif default_system_prompt is not None:
            messages.append({"role": "system", "content": default_system_prompt})

        # Add user message
        messages.append({"role": "user", "content": prompt})

    data: Dict[str, Any] = {
        "messages": messages,
        "stream": streaming,
    }

    # Add optional parameters if specified
    if azure_config.temperature is not None:
        data["temperature"] = azure_config.temperature
    if azure_config.max_tokens is not None:
        data["max_tokens"] = azure_config.max_tokens

    # Only print the prompt in one-off mode, not in conversation mode to avoid clutter
    if conversation_history is None:
        print(f"Prompting Azure OpenAI API with deployment {deployment_id}: {prompt}\n")

    if streaming:
        return stream_azure_response(endpoint, headers, data)
    else:
        return non_streaming_azure_response(endpoint, headers, data)


def stream_azure_response(
    endpoint: str, headers: Dict[str, str], data: Dict[str, Any]
) -> Iterator[str]:
    """
    Stream the response from Azure OpenAI API.

    Args:
        endpoint (str): The Azure OpenAI API endpoint
        headers (Dict[str, str]): Request headers
        data (Dict[str, Any]): Request data

    Yields:
        str: Chunks of the response as they arrive
    """
    response = requests.post(endpoint, headers=headers, json=data, stream=True)

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


def non_streaming_azure_response(
    endpoint: str, headers: Dict[str, str], data: Dict[str, Any]
) -> str:
    """
    Get a non-streaming response from Azure OpenAI API.

    Args:
        endpoint (str): The Azure OpenAI API endpoint
        headers (Dict[str, str]): Request headers
        data (Dict[str, Any]): Request data without streaming

    Returns:
        str: The full response
    """
    # Disable streaming for non-streaming request
    data["stream"] = False

    response: requests.Response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code != 200:
        print(f"Error: {response.status_code} {response.text}")
        sys.exit(1)

    result: Dict[str, Any] = response.json()
    return result["choices"][0]["message"]["content"].strip()
