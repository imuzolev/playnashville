"""
AWS Bedrock provider module for lask
"""

import sys
import json
from typing import Dict, Any, cast, Union, Iterator, List, Optional

from src.config import LaskConfig


def call_api(
    config: LaskConfig,
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Union[str, Iterator[str]]:
    """
    Call the AWS Bedrock API with the given prompt.

    Args:
        config (LaskConfig): Configuration object
        prompt (str): The user prompt
        conversation_history (Optional[List[Dict[str, str]]]): List of conversation messages
                                                             for multi-turn dialogues

    Returns:
        Union[str, Iterator[str]]: The response from the AWS Bedrock API,
                                  either full text or a stream iterator

    Raises:
        ImportError: If boto3 is not installed
        Exception: If there's an error calling the AWS Bedrock API
    """
    # We import boto3 only when needed to avoid requiring it for users who don't use AWS
    try:
        import boto3  # type: ignore
    except ImportError:
        print("Error: boto3 is required for AWS Bedrock.")
        print("Install it with: pip install boto3")
        print("Or install lask with AWS support: pip install lask[aws]")
        sys.exit(1)

    # Get provider-specific config
    aws_config = config.get_provider_config("aws")

    # Get the model ID
    model_id: str = aws_config.model_id or "anthropic.claude-3-sonnet-20240229-v1:0"
    region: str = aws_config.region or "us-east-1"

    # Check if streaming is enabled (default to True)
    streaming: bool = aws_config.get("streaming", True)

    # Create a Bedrock Runtime client
    bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)

    # Prepare the request body based on the model provider
    body: Dict[str, Any] = {}

    # Get system prompts
    provider_system_prompt = aws_config.system_prompt
    default_system_prompt = config.system_prompt

    if "anthropic" in model_id:
        # If conversation history is provided, use that instead of building new messages
        if conversation_history is not None:
            messages = conversation_history
        else:
            messages = []

            # Add system prompt if available
            if provider_system_prompt is not None:
                messages.append({"role": "system", "content": provider_system_prompt})
            elif default_system_prompt is not None:
                messages.append({"role": "system", "content": default_system_prompt})

            # Add user message
            messages.append({"role": "user", "content": prompt})

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": aws_config.max_tokens or 4096,
            "messages": messages,
            "stream": streaming,
        }
        if aws_config.temperature is not None:
            body["temperature"] = aws_config.temperature
    elif "amazon" in model_id:
        # For Amazon models, we might need to prepend the system prompt to the input text
        input_text = prompt
        if provider_system_prompt is not None:
            input_text = f"System: {provider_system_prompt}\n\nUser: {prompt}"
        elif default_system_prompt is not None:
            input_text = f"System: {default_system_prompt}\n\nUser: {prompt}"

        body = {
            "inputText": input_text,
            "textGenerationConfig": {"maxTokenCount": aws_config.max_tokens or 4096},
        }
        if aws_config.temperature is not None:
            body["textGenerationConfig"]["temperature"] = aws_config.temperature
    else:
        # Default format for other models
        system_text = ""
        if provider_system_prompt is not None:
            system_text = f"System: {provider_system_prompt}\n\n"
        elif default_system_prompt is not None:
            system_text = f"System: {default_system_prompt}\n\n"

        body = {
            "prompt": f"{system_text}User: {prompt}",
            "max_tokens": aws_config.max_tokens or 4096,
        }
        if aws_config.temperature is not None:
            body["temperature"] = aws_config.temperature

    # Only print the prompt in one-off mode, not in conversation mode to avoid clutter
    if conversation_history is None:
        print(f"Prompting AWS Bedrock with model {model_id}: {prompt}\n")

    if streaming and "anthropic" in model_id:
        return stream_aws_response(bedrock, model_id, body)
    else:
        return non_streaming_aws_response(bedrock, model_id, body)


def stream_aws_response(bedrock, model_id: str, body: Dict[str, Any]) -> Iterator[str]:
    """
    Stream the response from AWS Bedrock API.

    Args:
        bedrock: The boto3 bedrock-runtime client
        model_id (str): The model ID to use
        body (Dict[str, Any]): Request body

    Yields:
        str: Chunks of the response as they arrive
    """
    try:
        # Ensure streaming is enabled
        body["stream"] = True

        response = bedrock.invoke_model_with_response_stream(
            modelId=model_id, body=json.dumps(body)
        )

        stream_body = response.get("body")
        if stream_body:
            for event in stream_body:
                chunk_data = event.get("chunk", {})
                if chunk_data and "bytes" in chunk_data:
                    chunk = json.loads(chunk_data["bytes"])

                    # Extract content based on model provider
                    if "anthropic" in model_id:
                        if chunk.get("type") == "content_block_delta":
                            if "delta" in chunk and "text" in chunk["delta"]:
                                yield chunk["delta"]["text"]
                # Add support for other model types as needed

    except Exception as e:
        print(f"Error streaming from AWS Bedrock: {str(e)}")
        sys.exit(1)


def non_streaming_aws_response(bedrock, model_id: str, body: Dict[str, Any]) -> str:
    """
    Get a non-streaming response from AWS Bedrock API.

    Args:
        bedrock: The boto3 bedrock-runtime client
        model_id (str): The model ID to use
        body (Dict[str, Any]): Request body

    Returns:
        str: The full response
    """
    try:
        # Ensure streaming is disabled for non-streaming request
        if "stream" in body:
            body["stream"] = False

        response = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
        response_body_stream = response.get("body")
        if not response_body_stream:
            raise Exception("Empty response from AWS Bedrock")

        response_body: Dict[str, Any] = json.loads(response_body_stream.read())

        # Extract the content based on the model provider
        if "anthropic" in model_id:
            content = response_body.get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "")
            return ""
        elif "amazon" in model_id:
            results = response_body.get("results", [])
            if results and len(results) > 0:
                return results[0].get("outputText", "")
            return ""
        else:
            return cast(
                str,
                response_body.get(
                    "completion",
                    response_body.get("generated_text", str(response_body)),
                ),
            )
    except Exception as e:
        print(f"Error calling AWS Bedrock: {str(e)}")
        sys.exit(1)
