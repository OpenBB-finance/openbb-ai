from typing import Any, Literal

from .models import (
    DataSourceRequest,
    FunctionCallSSE,
    FunctionCallSSEData,
    MessageChunkSSE,
    MessageChunkSSEData,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    WidgetRequest,
)


def reasoning_step(
    event_type: Literal["INFO", "WARNING", "ERROR"],
    message: str,
    details: dict[str, Any] | None = None,
) -> StatusUpdateSSE:
    """Create a reasoning step (also known as a status update) SSE.

    This SSE is used to communicate the status of the agent, or any additional
    information as part of the agent's execution to the client.

    This Server-Sent Event (SSE) is typically `yield`ed to the client.

    Parameters
    ----------
    event_type: Literal["INFO", "WARNING", "ERROR"]
        The type of event to create.
    message: str
        The message to display.
    details: dict[str, Any] | None = None
        Additional details to display.

    Returns
    -------
    StatusUpdateSSE
        The status update SSE.
    """
    return StatusUpdateSSE(
        data=StatusUpdateSSEData(
            eventType=event_type,
            message=message,
            details=[details] if details else [],
        )
    )


def message_chunk(text: str) -> MessageChunkSSE:
    """Create a message chunk SSE.

    This SSE is used to stream back chunks of text to the client, typically from
    the agent's streamed response.

    This Server-Sent Event (SSE) is typically `yield`ed to the client.

    Parameters
    ----------
    text: str
        The text chunk to stream to the client.

    Returns
    -------
    MessageChunkSSE
        The message chunk SSE.
    """
    return MessageChunkSSE(data=MessageChunkSSEData(delta=text))


def get_widget_data(widget_requests: list[WidgetRequest]) -> FunctionCallSSE:
    """Create a function call that retrieve data for a widget on the OpenBB Workspace

    The function call is typically `yield`ed to the client. After yielding this
    event, you must immediately close the connection and wait for the follow-up
    request from the client.

    Parameters
    ----------
    widget_requests: list[WidgetRequest]
        A list of widget requests, where each request contains:
        - widget: A Widget instance defining the widget configuration
        - input_arguments: A dictionary of input parameters required by the widget

    Returns
    -------
    FunctionCallSSE
        The function call SSE.
    """

    data_sources: list[DataSourceRequest] = []
    for widget_request in widget_requests:
        data_sources.append(
            DataSourceRequest(
                widget_uuid=str(widget_request.widget.uuid),
                origin=widget_request.widget.origin,
                id=widget_request.widget.widget_id,
                input_args=widget_request.input_arguments,
            )
        )

    return FunctionCallSSE(
        data=FunctionCallSSEData(
            function="get_widget_data",
            input_arguments={"data_sources": data_sources},
        )
    )
