from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

from pydantic_ai import DeferredToolResults
from pydantic_ai.messages import (
    ModelMessage,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset
from pydantic_ai.ui import MessagesBuilder, UIAdapter

from ..models import (
    SSE,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    LlmMessage,
    QueryRequest,
    RoleEnum,
    Widget,
)
from ._dependencies import OpenBBDeps, build_deps_from_request
from ._event_stream import OpenBBAIEventStream
from ._toolsets import WidgetToolset, build_widget_toolsets
from ._utils import hash_tool_call, serialize_result_content


@dataclass(slots=True)
class OpenBBAIAdapter(UIAdapter[QueryRequest, LlmMessage, SSE, OpenBBDeps, Any]):
    """UI adapter that bridges OpenBB Workspace requests with Pydantic AI."""

    # Cache of tool call overrides initialised in __post_init__
    _tool_call_id_overrides: dict[str, str] = field(init=False, default_factory=dict)
    _base_messages: list[LlmMessage] = field(init=False, default_factory=list)
    _pending_results: list[LlmClientFunctionCallResultMessage] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        base, pending = self._split_messages(self.run_input.messages)
        self._base_messages = base
        self._pending_results = pending
        self._tool_call_id_overrides = {}

        for message in self._base_messages:
            if isinstance(message, LlmClientFunctionCallResultMessage):
                key = hash_tool_call(message.function, message.input_arguments)
                tool_call_id = self._tool_call_id_from_result(message)
                self._tool_call_id_overrides[key] = tool_call_id

        for message in self._pending_results:
            key = hash_tool_call(message.function, message.input_arguments)
            self._tool_call_id_overrides.setdefault(
                key,
                self._tool_call_id_from_result(message),
            )

    @classmethod
    def build_run_input(cls, body: bytes) -> QueryRequest:
        return QueryRequest.model_validate_json(body)

    @classmethod
    def load_messages(cls, messages: Sequence[LlmMessage]) -> list[ModelMessage]:
        """Convert OpenBB messages to Pydantic AI messages."""
        builder = MessagesBuilder()
        for message in messages:
            if isinstance(message, LlmClientMessage):
                cls._add_client_message(builder, message)
            elif isinstance(message, LlmClientFunctionCallResultMessage):
                tool_call_id = hash_tool_call(message.function, message.input_arguments)
                cls._add_function_call_result(
                    builder,
                    message,
                    tool_call_id=tool_call_id,
                )
        return builder.messages

    @staticmethod
    def _add_client_message(
        builder: MessagesBuilder,
        message: LlmClientMessage,
        tool_call_id_overrides: dict[str, str] | None = None,
    ) -> None:
        """Add a client message to the builder with optional tool call ID overrides."""
        content = message.content
        if isinstance(content, LlmClientFunctionCall):
            # Generate base tool call ID from hash
            base_id = hash_tool_call(content.function, content.input_arguments)
            # Use override if provided, otherwise use base ID
            tool_call_id = (
                tool_call_id_overrides.get(base_id, base_id)
                if tool_call_id_overrides
                else base_id
            )
            builder.add(
                ToolCallPart(
                    tool_name=content.function,
                    tool_call_id=tool_call_id,
                    args=content.input_arguments,
                )
            )
            return

        if isinstance(content, str):
            if message.role == RoleEnum.human:
                builder.add(UserPromptPart(content=content))
            elif message.role == RoleEnum.ai:
                builder.add(TextPart(content=content))
            else:
                builder.add(TextPart(content=content))

    @classmethod
    def _add_function_call_result(
        cls,
        builder: MessagesBuilder,
        message: LlmClientFunctionCallResultMessage,
        *,
        tool_call_id: str,
    ) -> None:
        """Add a function call result to the builder."""
        builder.add(
            ToolReturnPart(
                tool_name=message.function,
                tool_call_id=tool_call_id,
                content=serialize_result_content(message),
            )
        )

    @staticmethod
    def _split_messages(
        messages: Sequence[LlmMessage],
    ) -> tuple[list[LlmMessage], list[LlmClientFunctionCallResultMessage]]:
        base = list(messages)
        pending: list[LlmClientFunctionCallResultMessage] = []
        while base and isinstance(base[-1], LlmClientFunctionCallResultMessage):
            result_msg = base.pop()
            pending.insert(0, cast(LlmClientFunctionCallResultMessage, result_msg))
        return base, pending

    def _tool_call_id_from_result(
        self, message: LlmClientFunctionCallResultMessage
    ) -> str:
        """Extract or generate a tool call ID from a result message."""
        extra_id = (
            message.extra_state.get("tool_call_id") if message.extra_state else None
        )
        if isinstance(extra_id, str):
            return extra_id
        return hash_tool_call(message.function, message.input_arguments)

    @cached_property
    def deps(self) -> OpenBBDeps:
        return build_deps_from_request(self.run_input)

    @cached_property
    def deferred_tool_results(self) -> DeferredToolResults | None:
        """Build deferred tool results from pending result messages."""
        if not self._pending_results:
            return None

        results = DeferredToolResults()
        for message in self._pending_results:
            actual_id = self._tool_call_id_from_result(message)
            serialized = serialize_result_content(message)
            results.calls[actual_id] = serialized
        return results

    @cached_property
    def _widget_toolsets(self) -> list[WidgetToolset]:
        return build_widget_toolsets(self.run_input.widgets)

    def build_event_stream(self) -> OpenBBAIEventStream:
        return OpenBBAIEventStream(
            self.run_input,
            accept=self.accept,
            widget_lookup=self.widget_lookup,
            pending_results=self._pending_results,
        )

    @cached_property
    def widget_lookup(self) -> dict[str, Widget]:
        lookup: dict[str, Widget] = {}
        for toolset in self._widget_toolsets:
            widgets = getattr(toolset, "widgets_by_tool", None)
            if widgets:
                lookup.update(widgets)
        return lookup

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Build message history with context prompts and tool call ID overrides."""
        builder = MessagesBuilder()
        self._add_context_prompts(builder)
        for message in self._base_messages:
            if isinstance(message, LlmClientMessage):
                # Use the unified method with overrides
                self._add_client_message(
                    builder,
                    message,
                    tool_call_id_overrides=self._tool_call_id_overrides,
                )
            elif isinstance(message, LlmClientFunctionCallResultMessage):
                self._add_function_call_result(
                    builder,
                    message,
                    tool_call_id=self._tool_call_id_from_result(message),
                )
        return builder.messages

    def _add_context_prompts(self, builder: MessagesBuilder) -> None:
        """Add system prompts with workspace context, URLs, and dashboard info."""
        lines: list[str] = []

        if self.deps.context:
            lines.append("Workspace context:")
            for ctx in self.deps.context:
                row_count = len(ctx.data.items) if ctx.data and ctx.data.items else 0
                summary = f"- {ctx.name} ({row_count} rows): {ctx.description}"
                lines.append(summary)

        if self.deps.urls:
            joined = ", ".join(self.deps.urls)
            lines.append(f"Relevant URLs: {joined}")

        workspace_state = self.deps.workspace_state
        if workspace_state and workspace_state.current_dashboard_info:
            dashboard = workspace_state.current_dashboard_info
            lines.append(
                f"Active dashboard: {dashboard.name} (tab {dashboard.current_tab_id})"
            )

        if lines:
            builder.add(SystemPromptPart(content="\n".join(lines)))

    @cached_property
    def toolset(self) -> AbstractToolset | None:
        """Build combined toolset from widget toolsets."""
        if not self._widget_toolsets:
            return None
        if len(self._widget_toolsets) == 1:
            return self._widget_toolsets[0]
        return CombinedToolset(self._widget_toolsets)

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Extract workspace state as a dictionary."""
        if self.run_input.workspace_state is None:
            return None
        return self.run_input.workspace_state.model_dump(exclude_none=True)

    def run_stream_native(
        self,
        *,
        output_type=None,
        message_history=None,
        deferred_tool_results=None,
        model=None,
        deps=None,
        model_settings=None,
        usage_limits=None,
        usage=None,
        infer_name=True,
        toolsets=None,
        builtin_tools=None,
    ):
        """
        Run the agent with OpenBB-specific defaults for
        deps, messages, and deferred results.
        """
        deps = deps or self.deps  # type: ignore[assignment]
        deferred_tool_results = deferred_tool_results or self.deferred_tool_results
        message_history = message_history or self.messages

        return super().run_stream_native(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        )

    def run_stream(
        self,
        *,
        output_type=None,
        message_history=None,
        deferred_tool_results=None,
        model=None,
        deps=None,
        model_settings=None,
        usage_limits=None,
        usage=None,
        infer_name=True,
        toolsets=None,
        builtin_tools=None,
        on_complete=None,
    ):
        """Run the agent and stream protocol-specific events with OpenBB defaults."""
        deps = deps or self.deps  # type: ignore[assignment]
        deferred_tool_results = deferred_tool_results or self.deferred_tool_results
        message_history = message_history or self.messages

        return super().run_stream(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            on_complete=on_complete,
        )
