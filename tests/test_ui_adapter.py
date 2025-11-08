from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import (
    TextPart,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from openbb_ai.models import (
    ClientCommandResult,
    DataContent,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    MessageChunkSSE,
    QueryRequest,
    RawContext,
    RoleEnum,
    SingleDataContent,
    Widget,
    WidgetCollection,
)
from openbb_ai.pydantic_ui import (
    OpenBBAIAdapter,
    OpenBBAIEventStream,
    build_widget_tool_name,
)
from openbb_ai.pydantic_ui._dependencies import build_deps_from_request


def _sample_widget() -> Widget:
    return Widget(
        origin="OpenBB API",
        widget_id="sample_widget",
        name="Sample Widget",
        description="Widget used for testing.",
        params=[],
        metadata={},
    )


def _simple_request(
    messages: list[LlmClientMessage | LlmClientFunctionCallResultMessage],
) -> QueryRequest:
    return QueryRequest(messages=messages)


def test_build_deps_from_request() -> None:
    """Test dependency building from QueryRequest."""
    raw_context = RawContext(
        uuid=uuid4(),
        name="Test Context",
        description="Context description",
        data=DataContent(items=[SingleDataContent(content="{}")]),
    )
    request = QueryRequest(
        messages=[LlmClientMessage(role=RoleEnum.human, content="Hello")],
        context=[raw_context],
        urls=["https://example.com"],
        timezone="America/New_York",
    )

    deps = build_deps_from_request(request)
    assert deps.timezone == "America/New_York"
    assert deps.urls == ["https://example.com"]
    assert deps.context and deps.context[0].name == "Test Context"


def test_adapter_generates_system_prompt_with_context() -> None:
    """Test that adapter injects system prompts with workspace context."""
    raw_context = RawContext(
        uuid=uuid4(),
        name="Test Context",
        description="Context description",
        data=DataContent(items=[SingleDataContent(content="{}")]),
    )
    request = QueryRequest(
        messages=[LlmClientMessage(role=RoleEnum.human, content="Hello")],
        context=[raw_context],
        urls=["https://example.com"],
    )

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)
    system_parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if part.__class__.__name__ == "SystemPromptPart"
    ]
    assert system_parts, "Expected a system prompt with context"
    assert "Test Context" in system_parts[0].content
    assert "https://example.com" in system_parts[0].content


def test_adapter_tool_call_mapping_and_deferred_results() -> None:
    widget = _sample_widget()
    tool_name = build_widget_tool_name(widget)
    widgets = WidgetCollection(primary=[widget])

    call_message = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function=tool_name, input_arguments={"symbol": "AAPL"}
        ),
    )
    result_message = LlmClientFunctionCallResultMessage(
        function=tool_name,
        input_arguments={"symbol": "AAPL"},
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={"tool_call_id": "tool-123"},
    )
    request = QueryRequest(messages=[call_message, result_message], widgets=widgets)

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)
    tool_parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if part.__class__.__name__ == "ToolCallPart"
    ]
    assert tool_parts, "Expected a tool call part in message history"
    assert tool_parts[0].tool_call_id == "tool-123"

    deferred = adapter.deferred_tool_results
    assert deferred is not None
    assert deferred.calls["tool-123"]["input_arguments"]["symbol"] == "AAPL"


def test_event_stream_emits_function_call_and_citations() -> None:
    async def _run() -> None:
        widget = _sample_widget()
        tool_name = build_widget_tool_name(widget)
        request = _simple_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
        stream = OpenBBAIEventStream(
            run_input=request,
            widget_lookup={tool_name: widget},
        )

        deferred = DeferredToolRequests()
        deferred.calls.append(
            ToolCallPart(
                tool_name=tool_name, tool_call_id="call-1", args={"symbol": "AAPL"}
            )
        )
        run_result_event = SimpleNamespace(result=SimpleNamespace(output=deferred))

        events = [event async for event in stream.handle_run_result(run_result_event)]
        assert events and events[0].event == "copilotStatusUpdate"
        assert "Sample Widget" in events[0].data.message
        # Check that args are in details
        assert events[0].data.details
        assert any("symbol" in str(d) for d in events[0].data.details)
        assert events[1].event == "copilotFunctionCall"
        assert "call-1" in stream._pending_tool_calls  # type: ignore[attr-defined]

        tool_result_event = SimpleNamespace(
            result=ToolReturnPart(
                tool_name=tool_name,
                tool_call_id="call-1",
                content={
                    "data": [
                        {"items": [{"content": json.dumps([{"col": 1}, {"col": 2}])}]}
                    ]
                },
            )
        )
        tool_events = [
            event
            async for event in stream.handle_function_tool_result(tool_result_event)
        ]
        assert tool_events
        # Citations are no longer emitted here - they're collected for later
        # Artifacts are in reasoning steps now
        assert tool_events[0].event == "copilotStatusUpdate"
        assert tool_events[0].data.artifacts is not None
        assert len(tool_events[0].data.artifacts) > 0
        assert tool_events[0].data.artifacts[0].type == "table"

        # Citations should be emitted at the end
        after_events = [event async for event in stream.after_stream()]
        citation_events = [
            e for e in after_events if e.event == "copilotCitationCollection"
        ]
        assert citation_events, "Expected citation event at the end"

    asyncio.run(_run())


def test_artifact_detection_for_table() -> None:
    request = _simple_request(
        [LlmClientMessage(role=RoleEnum.human, content="Data please")]
    )
    stream = OpenBBAIEventStream(run_input=request, widget_lookup={})

    artifact = stream._artifact_from_output([{"col": 1}, {"col": 2}])
    assert artifact is not None
    assert artifact.event == "copilotMessageArtifact"
    assert artifact.data.type == "table"


@pytest.mark.parametrize(
    ("has_streamed_text", "output_text", "expected_in_after"),
    [
        (False, "Hello", True),  # No streaming - emit in after_stream
        (True, "Bob", False),  # Already streamed - skip duplicate
    ],
)
def test_final_output_handling(
    has_streamed_text: bool, output_text: str, expected_in_after: bool
) -> None:
    """Test that final output is emitted only if not already streamed."""

    async def _run() -> None:
        request = _simple_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
        stream = OpenBBAIEventStream(run_input=request, widget_lookup={})

        # Simulate text streaming if needed
        if has_streamed_text:
            text_events = [
                event
                async for event in stream.handle_text_start(
                    TextPart(content=output_text)
                )
            ]
            assert text_events and stream._has_streamed_text

        # Handle final output
        run_result_event = SimpleNamespace(result=SimpleNamespace(output=output_text))
        events = [event async for event in stream.handle_run_result(run_result_event)]
        assert events == []

        # Check if output is emitted in after_stream
        after_events = [event async for event in stream.after_stream()]
        if expected_in_after:
            assert after_events and isinstance(after_events[0], MessageChunkSSE)
            assert after_events[0].data.delta == output_text
        else:
            assert after_events == []

    asyncio.run(_run())


def test_thinking_events_emit_single_reasoning_step() -> None:
    async def _run() -> None:
        request = _simple_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
        stream = OpenBBAIEventStream(run_input=request, widget_lookup={})

        start_part = ThinkingPart(content="We must respond")
        start_events = [
            event async for event in stream.handle_thinking_start(start_part)
        ]
        assert start_events == []

        delta = ThinkingPartDelta(content_delta=" quickly.")
        delta_events = [event async for event in stream.handle_thinking_delta(delta)]
        assert delta_events == []

        end_part = ThinkingPart(content="We must respond quickly.")
        end_events = [event async for event in stream.handle_thinking_end(end_part)]
        assert end_events and end_events[0].event == "copilotStatusUpdate"
        assert end_events[0].data.message == "We must respond quickly."

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("function_name", "input_args", "test_desc"),
    [
        (
            "direct_widget",
            {"symbol": "AAPL"},
            "direct widget call",
        ),
        (
            "get_widget_data",
            {
                "data_sources": [
                    {
                        "widget_uuid": "placeholder",
                        "origin": "OpenBB API",
                        "id": "sample_widget",
                        "input_args": {"symbol": "TSLA"},
                    }
                ]
            },
            "get_widget_data wrapper",
        ),
    ],
)
def test_deferred_results_emit_artifacts_and_citations(
    function_name: str, input_args: dict, test_desc: str
) -> None:
    """
    Test deferred results processing for both direct widget calls and get_widget_data.
    """

    async def _run() -> None:
        widget = _sample_widget()
        tool_name = build_widget_tool_name(widget)

        # Handle both direct widget and get_widget_data cases
        if function_name == "direct_widget":
            func = tool_name
        else:
            func = function_name
            # Replace placeholder with actual widget UUID
            input_args["data_sources"][0]["widget_uuid"] = str(widget.uuid)

        result_message = LlmClientFunctionCallResultMessage(
            function=func,
            input_arguments=input_args,
            data=[DataContent(items=[SingleDataContent(content='[{"price": 150.0}]')])],
        )

        request = _simple_request([LlmClientMessage(role=RoleEnum.human, content="Hi")])
        stream = OpenBBAIEventStream(
            run_input=request,
            widget_lookup={tool_name: widget},
            pending_results=[result_message],
        )

        # Verify artifacts in before_stream
        before_events = [event async for event in stream.before_stream()]
        status_events = [e for e in before_events if e.event == "copilotStatusUpdate"]
        artifact_events = [e for e in status_events if e.data.artifacts]
        assert artifact_events, f"Expected artifacts for {test_desc}"
        assert artifact_events[0].data.artifacts is not None
        assert artifact_events[0].data.artifacts[0].type == "table"

        # Verify citations emitted at end
        after_events = [event async for event in stream.after_stream()]
        citation_events = [
            e for e in after_events if e.event == "copilotCitationCollection"
        ]
        assert citation_events, f"Expected citations for {test_desc}"

    asyncio.run(_run())
