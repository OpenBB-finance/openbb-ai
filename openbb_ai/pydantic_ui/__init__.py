"""Pydantic AI UI adapter for OpenBB Workspace."""

try:
    from ._adapter import OpenBBAIAdapter
    from ._dependencies import OpenBBDeps, build_deps_from_request
    from ._event_stream import OpenBBAIEventStream
    from ._toolsets import (
        WidgetToolset,
        build_widget_tool,
        build_widget_tool_name,
        build_widget_toolsets,
    )
    from ._utils import GET_WIDGET_DATA_TOOL_NAME
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "`openbb_ai.pydantic_ui` requires the optional Pydantic AI dependency. "
        'Install it via `pip install "openbb-ai[pydantic_ui]"`.'
    ) from exc

__all__ = [
    "OpenBBAIAdapter",
    "OpenBBAIEventStream",
    "OpenBBDeps",
    "build_deps_from_request",
    "WidgetToolset",
    "build_widget_tool",
    "build_widget_tool_name",
    "build_widget_toolsets",
    "GET_WIDGET_DATA_TOOL_NAME",
]
