"""Utility functions for OpenBB Pydantic AI UI adapter."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..models import LlmClientFunctionCallResultMessage

# Constants for special tool names
GET_WIDGET_DATA_TOOL_NAME = "get_widget_data"


def hash_tool_call(function: str, input_arguments: dict[str, Any]) -> str:
    """Generate a deterministic hash-based ID for a tool call.

    This creates a unique identifier by hashing the function name and arguments,
    ensuring consistent tool call IDs across message history and deferred results.

    Parameters
    ----------
    function : str
        The name of the function/tool being called
    input_arguments : dict[str, Any]
        The arguments passed to the tool

    Returns
    -------
    str
        A string combining the function name with a 16-character hash digest
    """
    payload = json.dumps(
        {"function": function, "input_arguments": input_arguments},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{function}_{digest[:16]}"


def serialize_result_content(
    message: LlmClientFunctionCallResultMessage,
) -> dict[str, Any]:
    """Serialize a function call result message into a content dictionary.

    Parameters
    ----------
    message : LlmClientFunctionCallResultMessage
        The function call result message to serialize

    Returns
    -------
    dict[str, Any]
        A dictionary containing input_arguments, data, and optionally extra_state
    """
    data: list[Any] = []
    for item in message.data:
        if hasattr(item, "model_dump"):
            data.append(item.model_dump(mode="json", exclude_none=True))
        else:
            data.append(item)

    content = {
        "input_arguments": message.input_arguments,
        "data": data,
    }
    if message.extra_state:
        content["extra_state"] = message.extra_state
    return content


def normalize_args(args: Any) -> dict[str, Any]:
    """Normalize tool call arguments to a dictionary."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    return {}


def parse_json_content(raw_content: str) -> Any:
    """Parse JSON content, returning the original string if parsing fails."""
    try:
        return json.loads(raw_content)
    except ValueError:
        return raw_content
