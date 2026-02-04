"""
Provider modules for lask

Providers handle communication with various LLM APIs and support both:
- One-off prompts
- Conversation history for multi-turn dialogues in REPL mode
"""

from importlib import import_module
from typing import Union, Iterator, List, Dict, Optional
from types import ModuleType

from src.config import LaskConfig


def get_provider_module(provider_name: str) -> ModuleType:
    """
    Dynamically import and return the provider module based on provider name.

    Args:
        provider_name (str): The name of the provider (e.g., 'openai', 'anthropic', 'aws', 'azure')

    Returns:
        ModuleType: The imported provider module

    Raises:
        ImportError: If the provider module cannot be imported
    """
    try:
        return import_module(f"src.providers.{provider_name}")
    except ImportError:
        raise ImportError(
            f"Provider '{provider_name}' is not supported. Make sure the module exists."
        )


def call_provider_api(
    provider_name: str,
    config: LaskConfig,
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Union[str, Iterator[str]]:
    """
    Call the appropriate provider API based on the provider name.

    Args:
        provider_name (str): The name of the provider
        config (LaskConfig): Configuration object
        prompt (str): The user prompt
        conversation_history (Optional[List[Dict[str, str]]]): List of conversation messages
                                                             in the format {"role": "...", "content": "..."}
                                                             If provided, uses this for context.

    Returns:
        Union[str, Iterator[str]]: The response from the provider, either as a full string
                                  or as a stream of string chunks

    Raises:
        ImportError: If the provider is not supported
    """
    provider_module = get_provider_module(provider_name)
    return provider_module.call_api(config, prompt, conversation_history)
