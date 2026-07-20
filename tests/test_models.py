import uuid

from openbb_ai.models import (
    AgentFeatureOption,
    Citation,
    CitationHighlightBoundingBox,
    LlmClientMessage,
    LlmClientSummaryMessage,
    QueryRequest,
    SourceInfo,
    WorkspaceAgent,
)


def test_query_request_parses_summary_message():
    request = QueryRequest.model_validate(
        {
            "messages": [
                {"role": "human", "content": "hello", "message_id": "m1"},
                {
                    "role": "summary",
                    "content": "Earlier turns summarized.",
                    "covered_through_message_id": "m1",
                },
                {"role": "human", "content": "next question"},
            ]
        }
    )

    assert isinstance(request.messages[0], LlmClientMessage)
    assert request.messages[0].message_id == "m1"
    assert isinstance(request.messages[1], LlmClientSummaryMessage)
    assert request.messages[1].covered_through_message_id == "m1"
    assert isinstance(request.messages[2], LlmClientMessage)
    assert request.messages[2].message_id is None


def test_message_id_is_optional_and_round_trips():
    message = LlmClientMessage(role="ai", content="hi")
    assert message.message_id is None
    dumped = LlmClientMessage(role="ai", content="hi", message_id="m9").model_dump()
    assert dumped["message_id"] == "m9"


def test_workspace_agent_supports_feature_option_metadata():
    agent = WorkspaceAgent(
        id="openbb_ada",
        name="OpenBB Copilot",
        features={
            "streaming": True,
            "prompt-suggestions": {
                "label": "Follow-up Suggestions",
                "default": True,
                "description": "Show follow-up prompt suggestions after each response.",
            },
        },
    )

    feature = agent.features["prompt-suggestions"]
    assert isinstance(feature, AgentFeatureOption)
    assert feature.label == "Follow-up Suggestions"
    assert feature.default is True


def test_citation_eq():
    # Identical source_info, details, and quote_bounding_boxes
    reference_source_info = SourceInfo(
        type="widget",
        uuid=uuid.uuid4(),
        origin="originA",
        widget_id="widget1",
        name="Widget Name",
        description="desc",
        metadata={"input_args": {"a": 1, "b": 2}, "other": 123},
        citable=True,
    )
    reference_details = [{"page": 1, "note": "foo"}]
    reference_bbox = CitationHighlightBoundingBox(
        text="abc", page=1, x0=0.0, top=0.0, x1=1.0, bottom=1.0
    )
    reference_quote_bounding_boxes = [[reference_bbox]]

    citation_1 = Citation(
        source_info=reference_source_info,
        details=reference_details,
        quote_bounding_boxes=reference_quote_bounding_boxes,
    )
    citation_2 = Citation(
        source_info=reference_source_info,
        details=reference_details,
        quote_bounding_boxes=reference_quote_bounding_boxes,
    )
    assert citation_1 == citation_2  # All fields equal

    # Different details
    citation_3 = Citation(
        source_info=reference_source_info,
        details=[{"page": 2, "note": "bar"}],
        quote_bounding_boxes=reference_quote_bounding_boxes,
    )
    assert citation_1 != citation_3

    # Different quote_bounding_boxes
    bbox_2 = CitationHighlightBoundingBox(
        text="xyz", page=2, x0=0.0, top=0.0, x1=1.0, bottom=1.0
    )
    citation_4 = Citation(
        source_info=reference_source_info,
        details=reference_details,
        quote_bounding_boxes=[[bbox_2]],
    )
    assert citation_1 != citation_4

    # Different source_info
    source_info_2 = SourceInfo(
        type="widget",
        uuid=uuid.uuid4(),
        origin="originB",
        widget_id="widget2",
        name="Widget Name 2",
        description="desc2",
        metadata={"input_args": {"a": 1, "b": 2}},
        citable=True,
    )
    citation_5 = Citation(
        source_info=source_info_2,
        details=reference_details,
        quote_bounding_boxes=reference_quote_bounding_boxes,
    )
    assert citation_1 != citation_5

    # Different type
    assert citation_1 != "not a citation"
