from typing import Literal, Any
from .models import (
    DataSourceRequest,
    MessageChunkSSE,
    MessageChunkSSEData,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    Widget,
)

def reasoning_step(
    event_type: Literal["INFO", "WARNING", "ERROR"],
    message: str,
    details: dict[str, Any] | None = None,
) -> StatusUpdateSSE:
    return StatusUpdateSSE(
        data=StatusUpdateSSEData(
            eventType=event_type,
            message=message,
            details=[details] if details else [],
        )
    )

def message_chunk(text: str) -> MessageChunkSSE:
    return MessageChunkSSE(data=MessageChunkSSEData(delta=text))

def get_remote_data(
    widget: Widget,
    input_arguments: dict[str, Any],
) -> DataSourceRequest:
    return DataSourceRequest(
        widget_uuid=str(widget.uuid),
        origin=widget.origin,
        id=widget.widget_id,
        input_args=input_arguments,
    )