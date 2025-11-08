"""Toolset implementations for OpenBB widgets and visualization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic_ai import CallDeferred, Tool
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from ..helpers import chart, table
from ..models import Undefined, Widget, WidgetCollection, WidgetParam
from ._dependencies import OpenBBDeps


def _base_param_schema(param: WidgetParam) -> dict[str, Any]:
    """Build the base JSON schema for a widget parameter."""
    type_mapping: dict[str, dict[str, Any]] = {
        "string": {"type": "string"},
        "text": {"type": "string"},
        "number": {"type": "number"},
        "integer": {"type": "integer"},
        "boolean": {"type": "boolean"},
        "date": {"type": "string", "format": "date"},
        "ticker": {"type": "string"},
        "endpoint": {"type": "string"},
    }

    schema = type_mapping.get(param.type, {"type": "string"})
    schema = dict(schema)  # copy
    schema["description"] = param.description

    if param.options:
        schema["enum"] = list(param.options)

    if param.get_options:
        schema.setdefault(
            "description",
            param.description + " (options retrieved dynamically)",
        )

    if param.default_value is not Undefined.UNDEFINED:
        schema["default"] = param.default_value

    if param.current_value is not None and param.multi_select is False:
        schema.setdefault("examples", []).append(param.current_value)

    return schema


def _param_schema(param: WidgetParam) -> tuple[dict[str, Any], bool]:
    """Return the schema for a parameter and whether it's required."""
    schema = _base_param_schema(param)

    if param.multi_select:
        schema = {
            "type": "array",
            "items": schema,
            "description": schema.get("description"),
        }

    is_required = param.default_value is Undefined.UNDEFINED
    return schema, is_required


def _widget_schema(widget: Widget) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in widget.params:
        schema, is_required = _param_schema(param)
        properties[param.name] = schema
        if is_required:
            required.append(param.name)

    widget_schema: dict[str, Any] = {
        "type": "object",
        "title": widget.name,
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        widget_schema["required"] = required

    return widget_schema


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return slug.lower() or "value"


def build_widget_tool_name(widget: Widget) -> str:
    """Generate a deterministic tool name for a widget.

    The tool name is constructed as: openbb_widget_{origin}_{widget_id}
    where both origin and widget_id are slugified.

    Parameters
    ----------
    widget : Widget
        The widget to generate a tool name for

    Returns
    -------
    str
        A unique, deterministic tool name string
    """
    origin_slug = _slugify(widget.origin)
    widget_slug = _slugify(widget.widget_id)
    return f"openbb_widget_{origin_slug}_{widget_slug}"


def build_widget_tool(widget: Widget) -> Tool:
    """Create a deferred tool for a widget.

    This creates a Pydantic AI tool that will be called by the LLM but
    executed by the OpenBB Workspace frontend (deferred execution).

    Parameters
    ----------
    widget : Widget
        The widget to create a tool for

    Returns
    -------
    Tool
        A Tool configured for deferred execution
    """
    tool_name = build_widget_tool_name(widget)
    schema = _widget_schema(widget)
    description = widget.description or widget.name

    async def _call_widget(ctx: RunContext[OpenBBDeps], **input_arguments: Any) -> None:
        # Ensure we have a tool call id for deferred execution
        if ctx.tool_call_id is None:
            raise RuntimeError("Deferred widget tools require a tool call id.")
        raise CallDeferred

    _call_widget.__name__ = f"call_widget_{widget.uuid}"

    return Tool.from_schema(
        function=_call_widget,
        name=tool_name,
        description=description,
        json_schema=schema,
        takes_ctx=True,
    )


class WidgetToolset(FunctionToolset):
    """Toolset that exposes widgets as deferred tools."""

    def __init__(self, widgets: Sequence[Widget]):
        super().__init__()
        self._widgets_by_tool: dict[str, Widget] = {}

        for widget in widgets:
            tool = build_widget_tool(widget)
            self.add_tool(tool)
            self._widgets_by_tool[tool.name] = widget

    @property
    def widgets_by_tool(self) -> Mapping[str, Widget]:
        return self._widgets_by_tool


class VisualizationToolset(FunctionToolset):
    """Toolset exposing helper utilities for charts and tables."""

    def __init__(self) -> None:
        super().__init__()

        def _create_table(
            data: list[dict[str, Any]],
            name: str | None = None,
            description: str | None = None,
        ):
            """Create a table artifact to display in OpenBB Workspace."""

            return table(data=data, name=name, description=description)

        def _create_chart(
            type: Literal["line", "bar", "scatter", "pie", "donut"],
            data: list[dict[str, Any]],
            x_key: str | None = None,
            y_keys: list[str] | None = None,
            angle_key: str | None = None,
            callout_label_key: str | None = None,
            name: str | None = None,
            description: str | None = None,
        ):
            """Create a chart artifact (line, bar, scatter, pie, donut).

            Raises
            ------
            ValueError
                If required parameters for the given chart ``type`` are missing.
            """

            if type in {"line", "bar", "scatter"}:
                if not x_key:
                    raise ValueError(
                        "x_key is required for line, bar, and scatter charts"
                    )
                if not y_keys:
                    raise ValueError(
                        "y_keys is required for line, bar, and scatter charts"
                    )
            elif type in {"pie", "donut"}:
                if not angle_key:
                    raise ValueError("angle_key is required for pie and donut charts")
                if not callout_label_key:
                    raise ValueError(
                        "callout_label_key is required for pie and donut charts"
                    )

            return chart(
                type=type,
                data=data,
                x_key=x_key,
                y_keys=y_keys,
                angle_key=angle_key,
                callout_label_key=callout_label_key,
                name=name,
                description=description,
            )

        self.add_function(_create_table, name="openbb_create_table")
        self.add_function(_create_chart, name="openbb_create_chart")


def build_widget_toolsets(collection: WidgetCollection | None) -> list[FunctionToolset]:
    """Create toolsets for each widget priority group plus visualization tools.

    Widgets are organized into separate toolsets by priority (primary, secondary, extra)
    to allow control over tool selection. The visualization toolset is always
    included for creating charts and tables.

    Parameters
    ----------
    collection : WidgetCollection | None
        Widget collection with priority groups, or None

    Returns
    -------
    list[FunctionToolset]
        List of toolsets including widget toolsets and visualization toolset
    """
    if collection is None:
        return [VisualizationToolset()]

    toolsets: list[FunctionToolset] = []
    for widgets in (collection.primary, collection.secondary, collection.extra):
        if widgets:
            toolsets.append(WidgetToolset(widgets))

    toolsets.append(VisualizationToolset())

    return toolsets
