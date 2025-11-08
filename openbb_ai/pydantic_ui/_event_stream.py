"""Event stream transformer for OpenBB Workspace SSE protocol."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)
from pydantic_ai.output import OutputDataT
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.ui import UIEventStream

from ..helpers import (
    chart,
    citations,
    cite,
    get_widget_data,
    message_chunk,
    reasoning_step,
    table,
)
from ..models import (
    SSE,
    Citation,
    ClientArtifact,
    LlmClientFunctionCallResultMessage,
    MessageArtifactSSE,
    MessageChunkSSE,
    QueryRequest,
    StatusUpdateSSE,
    Widget,
    WidgetRequest,
)
from ._utils import GET_WIDGET_DATA_TOOL_NAME, normalize_args, parse_json_content


def _encode_sse(event: SSE) -> str:
    payload = event.model_dump()
    return f"event: {payload['event']}\ndata: {payload['data']}\n\n"


@dataclass
class OpenBBAIEventStream(UIEventStream[QueryRequest, SSE, AgentDepsT, OutputDataT]):
    """Transform native Pydantic AI events into OpenBB SSE events."""

    widget_lookup: Mapping[str, Widget] = field(default_factory=dict)
    pending_results: list[LlmClientFunctionCallResultMessage] = field(
        default_factory=list
    )

    _pending_tool_calls: dict[str, tuple[Widget, dict[str, Any]]] = field(
        init=False, default_factory=dict
    )
    _has_streamed_text: bool = field(init=False, default=False)
    _final_output_pending: str | None = field(init=False, default=None)
    _thinking_buffer: list[str] = field(init=False, default_factory=list)
    _deferred_results_emitted: bool = field(init=False, default=False)
    _pending_citations: list[Citation] = field(init=False, default_factory=list)

    def encode_event(self, event: SSE) -> str:
        return _encode_sse(event)

    async def before_stream(self) -> AsyncIterator[SSE]:
        """Emit tool results for any deferred results provided upfront."""
        if self._deferred_results_emitted:
            return

        self._deferred_results_emitted = True

        # Process any pending deferred tool results from previous requests
        for result_message in self.pending_results:
            async for event in self._process_deferred_result(result_message):
                yield event

    async def _process_deferred_result(
        self, result_message: LlmClientFunctionCallResultMessage
    ) -> AsyncIterator[SSE]:
        """Process a single deferred result message and yield SSE events."""
        widget = self._find_widget_for_result(result_message)

        if widget is None:
            return

        # Collect citation for later emission
        widget_args = self._extract_widget_args(result_message)
        citation = cite(widget, widget_args)
        self._pending_citations.append(citation)

        # Serialize and emit tool result events
        content = self._serialize_result_message(result_message)
        for event in self._tool_result_events_from_content(content):
            yield event

    def _find_widget_for_result(
        self, result_message: LlmClientFunctionCallResultMessage
    ) -> Widget | None:
        """Find the widget associated with a result message."""
        # Check if this is a direct widget tool result
        widget = self.widget_lookup.get(result_message.function)
        if widget is not None:
            return widget

        # Check if it's a get_widget_data call with data_sources
        if result_message.function == GET_WIDGET_DATA_TOOL_NAME:
            data_sources = result_message.input_arguments.get("data_sources", [])
            if data_sources:
                data_source = data_sources[0]
                widget_uuid = data_source.get("widget_uuid")
                # Find widget by UUID in the lookup
                for w in self.widget_lookup.values():
                    if str(w.uuid) == widget_uuid:
                        return w

        return None

    def _extract_widget_args(
        self, result_message: LlmClientFunctionCallResultMessage
    ) -> dict[str, Any]:
        """Extract widget arguments from a result message."""
        if result_message.function == GET_WIDGET_DATA_TOOL_NAME:
            data_sources = result_message.input_arguments.get("data_sources", [])
            if data_sources:
                return data_sources[0].get("input_args", {})
        return result_message.input_arguments

    def _serialize_result_message(
        self, result_message: LlmClientFunctionCallResultMessage
    ) -> dict[str, Any]:
        """Serialize a result message into a content dictionary."""
        content = {
            "input_arguments": result_message.input_arguments,
            "data": [
                item.model_dump(mode="json", exclude_none=True)
                if hasattr(item, "model_dump")
                else item
                for item in result_message.data
            ],
        }
        if result_message.extra_state:
            content["extra_state"] = result_message.extra_state
        return content

    async def on_error(self, error: Exception) -> AsyncIterator[SSE]:
        yield reasoning_step(str(error), event_type="ERROR")

    async def handle_text_start(
        self, part: TextPart, follows_text: bool = False
    ) -> AsyncIterator[SSE]:
        if part.content:
            self._has_streamed_text = True
            yield message_chunk(part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[SSE]:
        if delta.content_delta:
            self._has_streamed_text = True
            yield message_chunk(delta.content_delta)

    async def handle_thinking_start(
        self,
        part: ThinkingPart,
        follows_thinking: bool = False,
    ) -> AsyncIterator[SSE]:
        self._thinking_buffer = []
        if part.content:
            self._thinking_buffer.append(part.content)
        return
        yield  # pragma: no cover

    async def handle_thinking_delta(
        self,
        delta: ThinkingPartDelta,
    ) -> AsyncIterator[SSE]:
        if delta.content_delta:
            self._thinking_buffer.append(delta.content_delta)
        return
        yield  # pragma: no cover

    async def handle_thinking_end(
        self,
        part: ThinkingPart,
        followed_by_thinking: bool = False,
    ) -> AsyncIterator[SSE]:
        content = part.content or "".join(self._thinking_buffer)
        if not content and self._thinking_buffer:
            content = "".join(self._thinking_buffer)

        if content:
            yield reasoning_step(content)

        self._thinking_buffer.clear()

    async def handle_run_result(
        self, event: AgentRunResultEvent[OutputDataT]
    ) -> AsyncIterator[SSE]:
        """Handle agent run result events, including deferred tool requests."""
        result = event.result
        output = getattr(result, "output", None)

        if isinstance(output, DeferredToolRequests):
            async for sse_event in self._handle_deferred_tool_requests(output):
                yield sse_event
            return

        artifact = self._artifact_from_output(output)
        if artifact is not None:
            yield artifact
            return

        if isinstance(output, str) and output and not self._has_streamed_text:
            self._final_output_pending = output

    async def _handle_deferred_tool_requests(
        self, output: DeferredToolRequests
    ) -> AsyncIterator[SSE]:
        """Process deferred tool requests and yield widget request events."""
        widget_requests: list[WidgetRequest] = []
        tool_call_ids: list[dict[str, Any]] = []

        for call in output.calls:
            widget = self.widget_lookup.get(call.tool_name)
            if widget is None:
                continue

            args = normalize_args(call.args)
            widget_requests.append(WidgetRequest(widget=widget, input_arguments=args))
            self._pending_tool_calls[call.tool_call_id] = (widget, args)
            tool_call_ids.append(
                {
                    "tool_call_id": call.tool_call_id,
                    "widget_uuid": str(widget.uuid),
                    "widget_id": widget.widget_id,
                }
            )

            # Create details dict with widget info and arguments for display
            details = {
                "Origin": widget.origin,
                "Widget Id": widget.widget_id,
                **args,
            }
            yield reasoning_step(
                f"Requesting widget '{widget.name}'",
                details=details,
            )

        if widget_requests:
            sse = get_widget_data(widget_requests)
            sse.data.extra_state = {"tool_calls": tool_call_ids}
            yield sse

    async def handle_function_tool_result(
        self, event: FunctionToolResultEvent
    ) -> AsyncIterator[SSE]:
        result_part = event.result

        if isinstance(result_part, RetryPromptPart):
            if result_part.content:
                yield reasoning_step(result_part.content, event_type="ERROR")
            return

        if not isinstance(result_part, ToolReturnPart):
            return

        tool_call_id = result_part.tool_call_id
        if not tool_call_id:
            return

        if isinstance(
            result_part.content, (MessageArtifactSSE, MessageChunkSSE, StatusUpdateSSE)
        ):
            yield result_part.content
            return

        widget_info = self._pending_tool_calls.pop(tool_call_id, None)
        if widget_info is None:
            return

        widget, args = widget_info
        # Collect citation for later emission (at the end)
        citation = cite(widget, args)
        self._pending_citations.append(citation)

        for event in self._tool_result_events_from_content(result_part.content):
            yield event

    async def after_stream(self) -> AsyncIterator[SSE]:
        if self._thinking_buffer:
            content = "".join(self._thinking_buffer)
            if content:
                yield reasoning_step(content)
            self._thinking_buffer.clear()

        if self._final_output_pending and not self._has_streamed_text:
            yield message_chunk(self._final_output_pending)

        self._final_output_pending = None

        # Emit all citations at the end
        if self._pending_citations:
            yield citations(self._pending_citations)
            self._pending_citations.clear()

        return
        yield  # pragma: no cover

    def _tool_result_events_from_content(self, content: Any) -> list[SSE]:
        """Transform tool result content into SSE events."""
        if not isinstance(content, dict):
            return []

        data_entries = content.get("data") or []
        if not isinstance(data_entries, list):
            return []

        events: list[SSE] = []
        artifacts: list[ClientArtifact] = []

        for entry in data_entries:
            if not isinstance(entry, dict):
                continue

            # Process command results (status messages)
            command_event = self._process_command_result(entry)
            if command_event:
                events.append(command_event)

            # Process data items
            entry_artifacts, entry_events = self._process_data_items(entry)
            artifacts.extend(entry_artifacts)
            events.extend(entry_events)

        # Emit collected artifacts as a reasoning step
        if artifacts:
            events.append(reasoning_step("Data retrieved", artifacts=artifacts))

        return events

    def _process_command_result(self, entry: dict[str, Any]) -> SSE | None:
        """Process command result status messages."""
        status = entry.get("status")
        message = entry.get("message")
        if status and message:
            return reasoning_step(f"[{status}] {message}")
        return None

    def _process_data_items(
        self, entry: dict[str, Any]
    ) -> tuple[list[ClientArtifact], list[SSE]]:
        """Process data items and return artifacts and events."""
        items = entry.get("items")
        if not isinstance(items, list):
            return [], []

        artifacts: list[ClientArtifact] = []
        events: list[SSE] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            raw_content = item.get("content")
            if not isinstance(raw_content, str):
                continue

            parsed = parse_json_content(raw_content)

            # Check if it's tabular data (list of dicts)
            if (
                isinstance(parsed, list)
                and parsed
                and all(isinstance(row, dict) for row in parsed)
            ):
                artifacts.append(
                    ClientArtifact(
                        type="table",
                        name=item.get("name") or f"Table_{uuid4().hex[:4]}",
                        description=item.get("description") or "Widget data",
                        content=parsed,
                    )
                )
            elif isinstance(parsed, dict):
                # Emit as message chunk
                self._has_streamed_text = True
                events.append(message_chunk(json.dumps(parsed)))
            else:
                # Emit as raw text message chunk
                self._has_streamed_text = True
                events.append(message_chunk(raw_content))

        return artifacts, events

    def _artifact_from_output(self, output: Any) -> SSE | None:
        """Create an artifact (chart or table) from agent output if possible."""
        if isinstance(output, dict):
            chart_type = output.get("type")
            data = output.get("data")

            if isinstance(chart_type, str) and chart_type in {
                "line",
                "bar",
                "scatter",
                "pie",
                "donut",
            }:
                if isinstance(data, list):
                    chart_type_literal = cast(
                        Literal["line", "bar", "scatter", "pie", "donut"], chart_type
                    )
                    return chart(
                        type=chart_type_literal,
                        data=data,
                        x_key=output.get("x_key"),
                        y_keys=output.get("y_keys"),
                        angle_key=output.get("angle_key"),
                        callout_label_key=output.get("callout_label_key"),
                        name=output.get("name"),
                        description=output.get("description"),
                    )

            table_data = None
            if isinstance(output.get("table"), list):
                table_data = output["table"]
            elif isinstance(data, list) and all(
                isinstance(item, dict) for item in data
            ):
                table_data = data

            if table_data:
                return table(
                    data=table_data,
                    name=output.get("name"),
                    description=output.get("description"),
                )

        if (
            isinstance(output, list)
            and output
            and all(isinstance(item, dict) for item in output)
        ):
            return table(data=output, name=None, description=None)

        return None
