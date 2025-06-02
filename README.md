# OpenBB Custom Agent SDK

This package provides a set of pydantic models, tools and helpers to build
custom agents that are compatible with OpenBB Workspace.

For some example agents that demonstrate the full usage of the SDK, see the
[example agents repository](https://github.com/OpenBB-finance/agents-for-openbb).

## Features

- [Streaming Conversations](#streaming-conversations)
- [Reasoning steps / status updates](#reasoning_step)
- [Retrieve widget data from OpenBB Workspace](#get_widget_data)
- [Citations](#cite-and-citations)
- [Display tables](#table)
- [Create charts](#chart)
- [Widget priorities](#widget-priority)

## Usage

All helper functions return Server-Sent Event (SSE) messages that should be streamed back to the OpenBB Workspace from your agent's execution loop. For example, using FastAPI with `EventSourceResponse`:

```python
from fastapi import FastAPI
from starlette.responses import EventSourceResponse
from openbb_ai import (
    reasoning_step,
    message_chunk,
    get_widget_data,
    cite,
    citations,
    table,
    chart,
)

app = FastAPI()

@app.get("/query")
async def stream(request: QueryRequest):
    async def event_generator():
        yield reasoning_step("Starting agent", event_type="INFO")
        yield message_chunk("Hello, world!")
    return EventSourceResponse(event_generator())
```


### `QueryRequest`

`QueryRequest` is the most important Pydantic model and entrypoint for all
requests to the agent. It should be used as the request body for FastAPI
endpoints (if you're using FastAPI).

Refer to the `QueryRequest` model definition (`openbb_ai.models.QueryRequest`) for full details on its fields and validation.

**Agent backends are stateless**: full conversation history (messages), widget
definitions, context, URLs, and any other state will be included in each
`QueryRequest`. Each request to the agent can / should be handled independently
with all necessary data provided upfront.

Key fields:
- `messages`: List of messages to submit to the agent. Supports both chat (`LlmClientMessage`) and function call result (`LlmClientFunctionCallResultMessage`) messages.
- `widgets`: Optional `WidgetCollection` organizing widgets into `primary`, `secondary`, and `extra` groups.
- `context`: Optional additional context items (`RawContext`) to supplement processing. Yielded `table` and `chart` artifacts are automatically added to this list by OpenBB Workspace.
- `urls`: Optional list of URLs (up to 4) to retrieve and include as context.
- ... and more.

### `message_chunk`

Create a message chunk SSE to stream back chunks of text to OpenBB Workspace,
typically from the agent's streamed response.

```python
from openbb_ai.helpers import message_chunk

yield message_chunk("Hello, world!")
```

### `reasoning_step`

Create a reasoning step (also known as a status update) SSE to communicate the
status of the agent, or any additional information as part of the agent's
execution to OpenBB Workspace.

```python
from openbb_ai.helpers import reasoning_step

yield reasoning_step(
    message="Processing data",
    event_type="INFO",
    details={"step": 1},
)
```

### `get_widget_data`

Create a function call SSE that retrieves data from widgets on the OpenBB
Workspace.

```python
from openbb_ai.helpers import get_widget_data
from openbb_ai.models import WidgetRequest

widget_requests = [WidgetRequest(widget=..., input_arguments={...})]

yield get_widget_data(widget_requests)
```

### `cite` and `citations`

Create citations for widgets to display on OpenBB Workspace. Use `cite` to
construct a `Citation` for a widget and `citations` to stream a collection of
citations as an SSE to the client.

```python
from openbb_ai.helpers import cite, citations

citation = cite(
    widget=widget,
    input_arguments={"param1": "value1", "param2": 123},
    extra_details={"note": "Optional extra details"},
)

yield citations([citation])
```

### `table`

Create a table message artifact SSE to display a table as streamed in-line
agent output in OpenBB Workspace.

```python
from openbb_ai.helpers import table

yield table(
    data=[
        {"x": 1, "y": 2, "z": 3},
        {"x": 2, "y": 3, "z": 4},
        {"x": 3, "y": 4, "z": 5},
        {"x": 4, "y": 5, "z": 6},
    ],
    name="My Table",
    description="This is a table of the data",
)
```

### `chart`

Create a chart message artifact SSE to display various types of charts
(line, bar, scatter, pie, donut) as streamed in-line agent output in OpenBB
Workspace.

```python
from openbb_ai.helpers import chart

yield chart(
    type="line",
    data=[
        {"x": 1, "y": 2},
        {"x": 2, "y": 3},
        {"x": 3, "y": 4},
        {"x": 4, "y": 5},
    ],
    x_key="x",
    y_keys=["y"],
    name="My Chart",
    description="This is a chart of the data",
)

yield chart(
    type="pie",
    data=[
        {"amount": 1, "category": "A"},
        {"amount": 2, "category": "B"},
        {"amount": 3, "category": "C"},
        {"amount": 4, "category": "D"},
    ],
    angle_key="amount",
    callout_label_key="category",
    name="My Chart",
    description="This is a chart of the data",
)
```

### Widget Priority
Custom agents receive three widget types via the `QueryRequest.widgets` field:

- **Primary widgets**: Explicitly added by the user to the context
- **Secondary widgets**: Present on the active dashboard but not explicitly added
- **Extra widgets**: Any widgets added to OpenBB Workspace (visible or not)

Currently, only primary and secondary widgets are accessible to custom agents, with extra widget support coming soon.

The dashboard below shows a Management Team widget (primary/priority) and a Historical Stock Price widget (secondary):

<img width="1526" alt="example dashboard" src="https://github.com/user-attachments/assets/9f579a2a-7240-41f5-8aa3-5ffd8a6ed7ba" />

If we inspect the `request.widgets` attribute of the `QueryRequest` object, we
can see the following was sent through to the custom agent:

```python
>>> request.widgets
WidgetCollection(
    primary=[
        Widget(
            uuid=UUID('68ab6973-ed1a-45aa-ab20-efd3e016dd48'),
            origin='OpenBB API',
            widget_id='management_team',
            name='Management Team',
            description='Details about the management team of a company, including name, title, and compensation.',
            params=[
                WidgetParam(
                    name='symbol',
                    type='ticker',
                    description='The symbol of the asset, e.g. AAPL,GOOGL,MSFT',
                    default_value=None,
                    current_value='AAPL',
                    options=[]
                )
            ],
            metadata={
                'source': 'Financial Modelling Prep',
                'lastUpdated': 1746177646279
            }
        )
    ],
    secondary=[
        Widget(
            uuid=UUID('bfa0aaaf-0b63-49b9-bb48-b13ef9db514b'),
            origin='OpenBB API',
            widget_id='eod_price',
            name='Historical Stock Price',
            description='Historical stock price data, including open, high, low, close, volume, etc.',
            params=[
                WidgetParam(
                    name='symbol',
                    type='ticker',
                    description='The symbol of the asset, e.g. AAPL,GOOGL,MSFT',
                    default_value=None,
                    current_value='AAPL',
                    options=[]
                ),
                WidgetParam(
                    name='start_date',
                    type='date',
                    description='The start date of the historical data',
                    default_value='2023-05-02',
                    current_value='2023-05-02',
                    options=[]
                )
            ],
            metadata={
                'source': 'Financial Modelling Prep',
                'lastUpdated': 1746177655947
            }
        )
    ],
    extra=[]
)
```

You can also see the parameter information of each widget in the `params` field
of the `Widget` object.
