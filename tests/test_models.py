import uuid

from openbb_ai.models import Citation, CitationHighlightBoundingBox, SourceInfo


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
