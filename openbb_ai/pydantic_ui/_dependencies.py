"""Dependency injection container for OpenBB workspace context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..models import QueryRequest, RawContext, Widget, WidgetCollection, WorkspaceState


@dataclass(slots=True)
class OpenBBDeps:
    """Dependency container passed to Pydantic AI runs.

    The dependency bundle exposes OpenBB Workspace specific context so that
    system prompts, tools, and output validators can access widget metadata or
    other request scoped information via ``RunContext[OpenBBDeps]``.

    Attributes:
        widgets: Collection of available widgets organized by priority
        context: Workspace context data (datasets, documents, etc.)
        urls: Relevant URLs for the current request
        workspace_state: Current workspace state including dashboard info
        timezone: User's timezone (defaults to UTC)
        state: Serialized workspace state as dictionary
    """

    widgets: WidgetCollection | None = None
    context: list[RawContext] | None = None
    urls: list[str] | None = None
    workspace_state: WorkspaceState | None = None
    timezone: str = "UTC"
    state: dict[str, Any] = field(default_factory=dict)

    def iter_widgets(self) -> Iterable[Widget]:
        """Yield all widgets across priority groups (primary, secondary, extra)."""
        if not self.widgets:
            return

        for group in (self.widgets.primary, self.widgets.secondary, self.widgets.extra):
            yield from group

    def get_widget_by_uuid(self, widget_uuid: str) -> Widget | None:
        """Find a widget by its UUID string."""
        for widget in self.iter_widgets():
            if str(widget.uuid) == widget_uuid:
                return widget
        return None


def build_deps_from_request(request: QueryRequest) -> OpenBBDeps:
    """Create an OpenBBDeps instance from an incoming QueryRequest."""
    context: Sequence[RawContext] | None = request.context
    urls: Sequence[str] | None = request.urls

    workspace_state = request.workspace_state

    return OpenBBDeps(
        widgets=request.widgets,
        context=list(context) if context is not None else None,
        urls=list(urls) if urls is not None else None,
        workspace_state=workspace_state,
        timezone=request.timezone,
        state=workspace_state.model_dump(exclude_none=True)
        if workspace_state is not None
        else {},
    )
