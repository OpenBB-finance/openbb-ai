from openbb_ai.helpers import chart
from openbb_ai.models import ClientArtifact, MessageArtifactSSE


def test_chart_line():
    result = chart(
        type="line",
        data=[{"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 3, "y": 4}, {"x": 4, "y": 5}],
        x_key="x",
        y_keys=["y"],
        name="My Line Chart",
        description="This is a line chart of the data",
    )

    assert isinstance(result, MessageArtifactSSE)
    assert isinstance(result.data, ClientArtifact)
    assert result.data.type == "chart"
    assert result.data.chart_params.chartType == "line"
    assert result.data.chart_params.xKey == "x"
    assert result.data.chart_params.yKey == ["y"]
    assert result.data.name == "My Line Chart"
    assert result.data.description == "This is a line chart of the data"


def test_chart_bar():
    result = chart(
        type="bar",
        data=[{"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 3, "y": 4}, {"x": 4, "y": 5}],
        x_key="x",
        y_keys=["y"],
        name="My Bar Chart",
        description="This is a bar chart of the data",
    )

    assert isinstance(result, MessageArtifactSSE)
    assert isinstance(result.data, ClientArtifact)
    assert result.data.type == "chart"
    assert result.data.chart_params.chartType == "bar"
    assert result.data.chart_params.xKey == "x"
    assert result.data.chart_params.yKey == ["y"]
    assert result.data.name == "My Bar Chart"
    assert result.data.description == "This is a bar chart of the data"


def test_chart_scatter():
    result = chart(
        type="scatter",
        data=[{"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 3, "y": 4}, {"x": 4, "y": 5}],
        x_key="x",
        y_keys=["y"],
        name="My Scatter Chart",
        description="This is a scatter chart of the data",
    )

    assert isinstance(result, MessageArtifactSSE)
    assert isinstance(result.data, ClientArtifact)
    assert result.data.type == "chart"
    assert result.data.chart_params.chartType == "scatter"
    assert result.data.chart_params.xKey == "x"
    assert result.data.chart_params.yKey == ["y"]
    assert result.data.name == "My Scatter Chart"
    assert result.data.description == "This is a scatter chart of the data"


def test_chart_pie():
    result = chart(
        type="pie",
        data=[
            {"x": 1, "y": "A"},
            {"x": 2, "y": "B"},
            {"x": 3, "y": "C"},
            {"x": 4, "y": "D"},
        ],
        angle_key="x",
        callout_label_key="y",
        name="My Pie Chart",
        description="This is a pie chart of the data",
    )

    assert isinstance(result, MessageArtifactSSE)
    assert isinstance(result.data, ClientArtifact)
    assert result.data.type == "chart"
    assert result.data.chart_params.chartType == "pie"
    assert result.data.chart_params.angleKey == "x"
    assert result.data.chart_params.calloutLabelKey == "y"
    assert result.data.name == "My Pie Chart"
    assert result.data.description == "This is a pie chart of the data"


def test_chart_donut():
    result = chart(
        type="donut",
        data=[
            {"x": 1, "y": "A"},
            {"x": 2, "y": "B"},
            {"x": 3, "y": "C"},
            {"x": 4, "y": "D"},
        ],
        angle_key="x",
        callout_label_key="y",
        name="My Donut Chart",
        description="This is a donut chart of the data",
    )

    assert isinstance(result, MessageArtifactSSE)
    assert isinstance(result.data, ClientArtifact)
    assert result.data.type == "chart"
    assert result.data.chart_params.chartType == "donut"
    assert result.data.chart_params.angleKey == "x"
    assert result.data.chart_params.calloutLabelKey == "y"
    assert result.data.name == "My Donut Chart"
    assert result.data.description == "This is a donut chart of the data"
