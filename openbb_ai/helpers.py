import uuid
from typing import Any, Literal

from .models import (
    BarChartParameters,
    ChartParameters,
    Citation,
    CitationCollection,
    CitationCollectionSSE,
    ClientArtifact,
    DataSourceRequest,
    DonutChartParameters,
    FunctionCallSSE,
    FunctionCallSSEData,
    LineChartParameters,
    MessageArtifactSSE,
    MessageChunkSSE,
    MessageChunkSSEData,
    PieChartParameters,
    ScatterChartParameters,
    SourceInfo,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    Widget,
    WidgetRequest,
)


def reasoning_step(
    message: str,
    event_type: Literal["INFO", "WARNING", "ERROR"] = "INFO",
    details: dict[str, Any] | str | None = None,
) -> StatusUpdateSSE:
    """Create a reasoning step (also known as a status update) SSE.

    This SSE is used to communicate the status of the agent, or any additional
    information as part of the agent's execution to the client.

    This Server-Sent Event (SSE) is typically `yield`ed to the client.

    Parameters
    ----------
    message: str
        The message to display.
    event_type: Literal["INFO", "WARNING", "ERROR"]
        The type of event to create.
        Default is "INFO".
    details: dict[str, Any] | str | None
        Additional details to display.
        Default is None.

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
    function call, you must immediately close the connection and wait for the
    follow-up function call.

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


def cite(
    widget: Widget,
    input_arguments: dict[str, Any],
    extra_details: dict[str, Any] | None = None,
) -> Citation:
    """Create a citation for a widget.

    Parameters
    ----------
    widget: Widget
        The widget to cite. Typically retrieved from the `QueryRequest` object.
    input_arguments: dict[str, Any]
        The input arguments used to retrieve data from the widget.
    extra_details: dict[str, Any] | None
        Extra details to display in the citation.
        Takes key-value pairs of the form `{"key": "value"}`.
        Default is None.

    Returns
    -------
    Citation
        The citation.
        Typically used as input to the `citations` function to be returned to
        the client.
    """
    return Citation(
        source_info=SourceInfo(
            type="widget",
            origin=widget.origin,
            widget_id=widget.widget_id,
            metadata={
                "input_args": input_arguments,
            },
        ),
        details=[extra_details] if extra_details else None,
    )


def citations(citations: list[Citation]) -> CitationCollectionSSE:
    """Create a citation collection SSE.

    This SSE is used to stream back citations to the client.

    This Server-Sent Event (SSE) is typically `yield`ed to the client.

    Parameters
    ----------
    citations: list[Citation]
        The citations to display.

    Returns
    -------
    CitationCollectionSSE
        The citation collection SSE.
    """
    return CitationCollectionSSE(data=CitationCollection(citations=citations))


def table(
    data: list[dict],
    name: str | None = None,
    description: str | None = None,
) -> MessageArtifactSSE:
    """Create a table message artifact SSE.

    This function constructs a table artifact, which can be `yield`ed to the
    client to display a table as streamed in-line agent output in OpenBB
    Workspace.

    Parameters
    ----------
    data: list[dict]
        The data to be visualized in the table. Each dictionary represents a
        row of the table, where keys represent the "columns" of the data.
    name: str | None
        The name of the table. Optional, but recommended.
        If set, must be unique within the context of the chat.
    description: str | None
        A description of the table. Optional, but recommended.

    Examples
    --------
    >>> # Create a table
    >>> table(
    ...     data=[
    ...         {"x": 1, "y": 2, "z": 3},
    ...         {"x": 2, "y": 3, "z": 4},
    ...         {"x": 3, "y": 4, "z": 5},
    ...         {"x": 4, "y": 5, "z": 6},
    ...     ],
    ...     name="My Table",
    ...     description="This is a table of the data",
    ... )

    Returns
    -------
    MessageArtifactSSE
        The table artifact to be sent to the client.
    """

    return MessageArtifactSSE(
        data=ClientArtifact(
            type="table",
            name=name or f"Table_{uuid.uuid4().hex[:4]}",
            description=description or "A table of data",
            content=data,
        )
    )


def chart(
    type: Literal["line", "bar", "scatter", "pie", "donut"],
    data: list[dict],
    x_key: str | None = None,
    y_keys: list[str] | None = None,
    angle_key: str | None = None,
    callout_label_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> MessageArtifactSSE:
    """
    Create a chart message artifact SSE.

    This function constructs a chart artifact, which can be `yield`ed to the
    client to display various types of charts (line, bar, scatter, pie, donut)
    as streamed in-line agent output in OpenBB Workspace.

    Parameters
    ----------
    type : Literal["line", "bar", "scatter", "pie", "donut"]
        The type of chart to create.
    data : list[dict]
        The data to be visualized in the chart. Each dictionary represents a
        data point, where keys represent the "columns" of the data.
    x_key : str | None
        The key in the data dictionaries to use for the x-axis (for line, bar,
        scatter charts).
    y_keys : list[str] | None
        The keys in the data dictionaries to use for the y-axis (for line, bar,
        scatter charts).
    angle_key : str | None
        The key in the data dictionaries to use for the angle of each sector
        (for pie, donut charts).
    callout_label_key : str | None
        The key in the data dictionaries to use for the callout labels (for pie,
        donut charts).
    name : str | None
        The name of the chart. Optional, but recommended.
    description : str | None
        A description of the chart. Optional, but recommended.

    Examples
    --------
    >>> # Create a line chart
    >>> chart(
    ...     type="line",
    ...     data=[
    ...         {"x": 1, "y": 2},
    ...         {"x": 2, "y": 3},
    ...         {"x": 3, "y": 4},
    ...         {"x": 4, "y": 5},
    ...     ],
    ...     x_key="x",
    ...     y_keys=["y"],
    ...     name="My Chart",
    ...     description="This is a chart of the data",
    ... )

    >>> # Create a pie chart
    >>> chart(
    ...     type="pie",
    ...     data=[
    ...         {"amount": 1, "category": "A"},
    ...         {"amount": 2, "category": "B"},
    ...         {"amount": 3, "category": "C"},
    ...         {"amount": 4, "category": "D"},
    ...     ],
    ...     angle_key="amount",
    ...     callout_label_key="category",
    ...     name="My Chart",
    ...     description="This is a chart of the data",
    ... )

    Returns
    -------
    MessageArtifactSSE
        The chart artifact to be sent to the client.
    """

    parameters: ChartParameters | None = None
    match type:
        case "line":
            parameters = LineChartParameters(
                chartType=type,
                xKey=x_key,
                yKey=y_keys,
            )
        case "bar":
            parameters = BarChartParameters(
                chartType=type,
                xKey=x_key,
                yKey=y_keys,
            )
        case "scatter":
            parameters = ScatterChartParameters(
                chartType=type,
                xKey=x_key,
                yKey=y_keys,
            )
        case "pie":
            parameters = PieChartParameters(
                chartType=type,
                angleKey=angle_key,
                calloutLabelKey=callout_label_key,
            )
        case "donut":
            parameters = DonutChartParameters(
                chartType=type,
                angleKey=angle_key,
                calloutLabelKey=callout_label_key,
            )
        case _:
            raise ValueError(f"Invalid chart type: {type}")

    return MessageArtifactSSE(
        data=ClientArtifact(
            type="chart",
            name=name or f"{type} chart",
            description=description or f"A {type} chart of data",
            content=data,
            chart_params=parameters,
        )
    )
