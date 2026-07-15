import uuid

from openbb_ai.models import (
    AgentFeatureOption,
    Citation,
    CitationHighlightBoundingBox,
    DataContent,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    LlmClientSummaryMessage,
    QueryRequest,
    SingleDataContent,
    SourceInfo,
    WorkspaceAgent,
    conversation_source_hash,
)


def _human(content: str, message_id: str | None = None) -> LlmClientMessage:
    return LlmClientMessage(role="human", content=content, message_id=message_id)


def _ai(content: str, message_id: str | None = None) -> LlmClientMessage:
    return LlmClientMessage(role="ai", content=content, message_id=message_id)


def _tool_result(
    function: str = "get_widget_data",
    message_id: str | None = None,
) -> LlmClientFunctionCallResultMessage:
    return LlmClientFunctionCallResultMessage(
        function=function,
        input_arguments={"data_sources": []},
        data=[DataContent(items=[SingleDataContent(content="some data")])],
        extra_state={"round_trip": 1},
        message_id=message_id,
    )


def _summary(covered_through: str = "m1") -> LlmClientSummaryMessage:
    return LlmClientSummaryMessage(
        content="Earlier turns summarized.",
        covered_through_message_id=covered_through,
        source_hash="deadbeef",
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
                    "source_hash": "deadbeef",
                },
                {"role": "human", "content": "next question"},
            ]
        }
    )

    assert isinstance(request.messages[0], LlmClientMessage)
    assert request.messages[0].message_id == "m1"
    assert isinstance(request.messages[1], LlmClientSummaryMessage)
    assert request.messages[1].covered_through_message_id == "m1"
    assert request.messages[1].source_hash == "deadbeef"
    assert isinstance(request.messages[2], LlmClientMessage)
    assert request.messages[2].message_id is None


def test_message_id_is_optional_and_round_trips():
    message = LlmClientMessage(role="ai", content="hi")
    assert message.message_id is None
    dumped = LlmClientMessage(role="ai", content="hi", message_id="m9").model_dump()
    assert dumped["message_id"] == "m9"


def test_conversation_source_hash_is_deterministic():
    messages = [_human("q1", "m1"), _ai("a1", "m2"), _tool_result(message_id="m3")]
    identical = [_human("q1", "m1"), _ai("a1", "m2"), _tool_result(message_id="m3")]
    assert conversation_source_hash(messages) == conversation_source_hash(identical)


def test_conversation_source_hash_ignores_volatile_fields():
    messages = [_human("q1", "m1"), _tool_result(message_id="m2")]
    reference = conversation_source_hash(messages)

    mutated = [_human("q1", "m1"), _tool_result(message_id="m2")]
    mutated[1].data = [DataContent(items=[SingleDataContent(content="renewed URL")])]
    mutated[1].extra_state = {"round_trip": 2, "deferred_function_calls": ["x"]}
    assert conversation_source_hash(mutated) == reference


def test_conversation_source_hash_detects_edits():
    messages = [_human("q1", "m1"), _ai("a1", "m2")]
    reference = conversation_source_hash(messages)

    edited_content = [_human("q1 edited", "m1"), _ai("a1", "m2")]
    assert conversation_source_hash(edited_content) != reference

    reordered = [_ai("a1", "m2"), _human("q1", "m1")]
    assert conversation_source_hash(reordered) != reference

    different_id = [_human("q1", "m1"), _ai("a1", "m99")]
    assert conversation_source_hash(different_id) != reference

    removed = [_human("q1", "m1")]
    assert conversation_source_hash(removed) != reference


def test_conversation_source_hash_excludes_summaries():
    messages = [_human("q1", "m1"), _ai("a1", "m2")]
    with_summary = [_human("q1", "m1"), _summary("m1"), _ai("a1", "m2")]
    assert conversation_source_hash(with_summary) == conversation_source_hash(messages)


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
