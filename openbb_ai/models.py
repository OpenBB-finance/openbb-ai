import json
import uuid
from enum import Enum
from typing import Annotated, Any, AsyncGenerator, Callable, Literal
from uuid import UUID

import xxhash
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

EXCLUDE_CITATION_DETAILS_FIELDS = [
    "lastupdated",
    "source",
    "id",
    "uuid",
    "storedfileuuid",
    "datakey",
    "originalfilename",
    "extension",
    "category",
    "subcategory",
    "transcript_url",
]

EXCLUDE_STATUS_UPDATE_DETAILS_FIELDS = [
    "lastupdated",
    "source",
    "id",
    "uuid",
    "storedfileuuid",
    "url",
    "datakey",
    "originalfilename",
    "extension",
    "category",
    "subcategory",
    "transcript_url",
]


class UserAPIKeys(BaseModel):
    openai_api_key: str | None = Field(
        default=None, description="Use a custom OpenAI API key for the request."
    )


class Pdf(BaseModel):
    filename: str
    content: bytes


class RoleEnum(str, Enum):
    ai = "ai"
    human = "human"
    tool = "tool"


class LineChartParameters(BaseModel):
    chartType: Literal["line"]
    xKey: str = Field(description="The key of the x-axis variable.")
    yKey: list[str] = Field(description="The key (or keys) of the y-axis variables.")


class BarChartParameters(BaseModel):
    chartType: Literal["bar"]
    xKey: str = Field(description="The key of the x-axis variable.")
    yKey: list[str] = Field(description="The key (or keys) of the y-axis variables.")


class ScatterChartParameters(BaseModel):
    chartType: Literal["scatter"]
    xKey: str = Field(
        description="The key of the x-axis variable. Only numerical variables can be provided for a scatter plot."  # noqa: E501
    )
    yKey: list[str] = Field(
        description="The key (or keys) of the y-axis variables. Only numerical variables can be provided for a scatter plot."  # noqa: E501
    )


class PieChartParameters(BaseModel):
    chartType: Literal["pie"]
    angleKey: str = Field(description="Angle of each pie sector.")
    calloutLabelKey: str = Field(
        description="Names of the variable used for the callout labels."
    )


class DonutChartParameters(BaseModel):
    chartType: Literal["donut"]
    angleKey: str = Field(description="Angle of each pie sector.")
    calloutLabelKey: str = Field(
        description="Names of the variable used for the callout labels."
    )


ChartParameters = (
    LineChartParameters
    | BarChartParameters
    | ScatterChartParameters
    | PieChartParameters
    | DonutChartParameters
)


class RawObjectDataFormat(BaseModel):
    data_type: Literal["object"] = "object"
    parse_as: Literal["text", "table", "chart"] = "table"
    chart_params: ChartParameters | None = None

    @model_validator(mode="after")
    def validate_chart_params(cls, values):
        if values.parse_as == "chart" and not values.chart_params:
            raise ValueError("chart_params is required when parse_as is 'chart'")
        if values.parse_as != "chart" and values.chart_params:
            raise ValueError("chart_params is only allowed when parse_as is 'chart'")
        return values


class PdfDataFormat(BaseModel):
    data_type: Literal["pdf"]
    filename: str


class ImageDataFormat(BaseModel):
    data_type: Literal["jpg", "jpeg", "png"]
    filename: str


class SpreadsheetDataFormat(BaseModel):
    data_type: Literal["xlsx", "xls", "csv"]
    parse_as: Literal["text", "table"] = "table"
    filename: str


class PlaintextDataFormat(BaseModel):
    data_type: Literal["txt", "md"]
    parse_as: Literal["text"] = "text"
    filename: str


class DocxDataFormat(BaseModel):
    data_type: Literal["docx"]
    filename: str


DataFileFormat = Annotated[
    PdfDataFormat
    | ImageDataFormat
    | SpreadsheetDataFormat
    | PlaintextDataFormat
    | DocxDataFormat,
    Field(discriminator="data_type"),
]

# Discriminated union of data formats
DataFormat = Annotated[
    RawObjectDataFormat | DataFileFormat,
    Field(discriminator="data_type", default_factory=RawObjectDataFormat),
]


class SourceInfo(BaseModel):
    type: Literal["widget", "direct retrieval", "web", "artifact"]
    uuid: UUID | None = Field(default=None)
    origin: str | None = Field(default=None)
    widget_id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (eg. the selected ticker, endpoint used, etc.).",  # noqa: E501
    )
    citable: bool = Field(
        default=True,
        description="Whether the source is citable.",
    )

    # Make faux immutable
    model_config = {"frozen": True}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SourceInfo):
            return False

        # We only want to compare the input args of the metadata
        reduced_metadata = {"input_args": self.metadata.get("input_args")}
        this = self.model_dump(exclude_none=True)
        this["metadata"] = reduced_metadata

        # Do the same for the `other` SourceInfo object
        other_reduced_metadata = {"input_args": other.metadata.get("input_args")}
        other = other.model_dump(exclude_none=True)
        other["metadata"] = other_reduced_metadata

        return this == other


class CitationHighlightBoundingBox(BaseModel):
    text: str
    page: int
    x0: float
    top: float
    x1: float
    bottom: float


class Citation(BaseModel):
    source_info: SourceInfo
    details: list[dict[str, Any]] | None = Field(
        default=None,
        description="Extra detail to add to the citation, eg. Page numbers.",
    )
    quote_bounding_boxes: list[list[CitationHighlightBoundingBox]] | None = Field(
        default=None,
        description="Bounding boxes for the highlights in the citation.",
    )

    def __hash__(self) -> int:
        return hash((str(self.source_info), str(self.details)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Citation):
            return False

        if self.source_info != other.source_info:
            return False

        if self.details != other.details:
            return False

        if self.quote_bounding_boxes != other.quote_bounding_boxes:
            return False

        return True

    @model_validator(mode="before")
    @classmethod
    def exclude_fields(cls, values):
        # Exclude these fields from being in the "details" field.  (since this
        # pollutes the JSON output)
        _exclude_fields = EXCLUDE_CITATION_DETAILS_FIELDS

        if details := values.get("details"):
            for detail in details:
                for key in list(detail.keys()):
                    if key.lower() in _exclude_fields:
                        detail.pop(key, None)
        return values


class Undefined(str, Enum):
    UNDEFINED = "<<UNDEFINED>>"


class OptionsEndpointParam(BaseModel):
    type: str | None = Field(default=None, description="Type of the options parameter.")
    name: str = Field(description="Name of the options parameter.")
    description: str | None = Field(
        default=None, description="Description of the options parameter."
    )
    # For now, we MUST inherit option param values from user parameters.
    inherit_value_from: str = Field(
        description="Name of the user parameter that this parameter inherits its values from."  # noqa: E501
    )

    @model_validator(mode="after")
    @classmethod
    def validate_type_given_inherit_value_from(cls, data: Any):
        # We set the type of the options parameter to the type of the user
        # parameter that it inherits from if inherit_value_from is set.
        if data.type is None and data.inherit_value_from is not None:
            data.type = data.inherit_value_from
            raise ValueError("Type must be set if inherit_value_from is not set.")
        return data


class WidgetParam(BaseModel):
    name: str = Field(description="Name of the parameter.")
    type: Literal[
        "string", "text", "number", "integer", "boolean", "date", "ticker", "endpoint"
    ] = Field(description="Type of the parameter.")
    description: str = Field(description="Description of the parameter.")
    default_value: Any | None = Field(
        default=None, description="Default value of the parameter."
    )
    current_value: Any | None = Field(
        default=None,
        description="Current value of the parameter. Must not be set for 'extra' widgets.",  # noqa: E501
    )
    multi_select: bool = Field(
        default=False,
        description="Set True to allow multiple values for the parameter.",
    )
    split_param_on_citation: bool = Field(
        default=False,
        description="Set True to split the each parameter value into a separate citation. Only works if `multi_select` is True.",  # noqa: E501
    )
    options: list[Any] | None = Field(
        default=None, description="Optional list of values for enumerations."
    )
    get_options: bool = Field(
        default=False,
        description="Set True to get options for the parameter dynamically. Requires an `optionsEndpoint` definition for the data source.",  # noqa: E501
    )
    options_params: list[OptionsEndpointParam] = Field(
        default_factory=list,
        description="A list of parameters to pass to the options endpoint.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_default_value(cls, data: dict):
        # We want to distinguish between a missing default value and an
        # explicitly set default value of None.  There is a difference between
        # my_function(param=None) and my_function(param).
        if "default_value" not in data:
            data["default_value"] = Undefined.UNDEFINED
        return data


class WidgetParamOption(BaseModel):
    label: str
    value: str


class WidgetParamOptions(BaseModel):
    widget_origin: str
    widget_id: str
    param_name: str
    options: list[WidgetParamOption] = Field(default_factory=list)


class Widget(BaseModel):
    uuid: UUID = Field(
        description="UUID of the widget. Used to identify widgets present on the dashboard. If an `extra` widget, this will be generated.",  # noqa: E501
        default_factory=uuid.uuid4,
    )
    origin: str = Field(description="Origin of the widget.")
    widget_id: str = Field(description="Endpoint ID of the widget.")
    name: str = Field(description="Name of the widget.")
    description: str = Field(description="Description of the widget.")
    params: list[WidgetParam] = Field(description="List of parameters for the widget.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for the widget, must not overlap with current_params.",
    )

    @staticmethod
    def _generate_uuid(origin: str, widget_id: str) -> UUID:
        """Generate a UUID for the widget based on its origin and widget_id."""
        seed = f"origin={origin}&widget_id={widget_id}"
        hash_value = xxhash.xxh64(seed.encode()).hexdigest()
        # Multiply by 2 because xxh64 returns
        # 64 bits -> 8 bytes -> 16 hexadecimal digits
        # and UUID hex requires 32 hexadecimal digits
        namespace = hash_value[:16] * 2
        return UUID(hex=namespace)

    @computed_field  # type: ignore[misc]
    @property
    def split_param(self) -> WidgetParam | None:
        for param in self.params:
            if param.split_param_on_citation:
                return param
        return None

    @model_validator(mode="before")
    @classmethod
    def generate_deterministic_uuid_if_none(cls, data: dict) -> dict:
        if data.get("uuid") is None:
            origin = data.get("origin")
            widget_id = data.get("widget_id")
            if origin and widget_id:
                data["uuid"] = cls._generate_uuid(origin, widget_id)
                return data
        return data

    @model_validator(mode="after")
    @classmethod
    def check_params_are_unique(cls, data: Any):
        param_names = [p.name for p in data.params]
        if len(param_names) != len(set(param_names)):
            raise ValidationError("Parameter names must be unique.")
        return data

    @model_validator(mode="after")
    @classmethod
    def check_only_one_split_param_on_citation(cls, data: Any):
        is_split_param_on_citation_set = False
        for widget_param in data.params:
            if widget_param.split_param_on_citation:
                if is_split_param_on_citation_set:
                    raise ValidationError(
                        "Only one parameter can be split on citation."
                    )
                is_split_param_on_citation_set = True
        return data

    @model_validator(mode="after")
    @classmethod
    def handle_inherit_value_from_options_params(cls, data: Any):
        for widget_param in data.params:
            if widget_param.get_options:
                for options_param in widget_param.options_params:
                    if options_param.inherit_value_from not in [
                        p.name for p in data.params
                    ]:
                        raise ValidationError(
                            f"Parameter {options_param.inherit_value_from} not found in options, but {widget_param.name}'s options endpoint depends on it."  # noqa: E501
                        )
                    else:
                        # Find the referenced param and set the type to match
                        referenced_param = next(
                            p
                            for p in data.params
                            if p.name == options_param.inherit_value_from
                        )
                        options_param.type = referenced_param.type
        return data


class WidgetCollection(BaseModel):
    primary: list[Widget] = Field(
        default_factory=list, description="Explicitly-added widgets with top priority."
    )
    secondary: list[Widget] = Field(
        default_factory=list,
        description="Dashboard widgets with second-highest priority.",
    )
    extra: list[Widget] = Field(
        default_factory=list, description="Extra data sources or custom backends."
    )


class LlmClientFunctionCall(BaseModel):
    function: str
    input_arguments: dict[str, Any]


class LlmClientMessage(BaseModel):
    role: RoleEnum = Field(
        description="The role of the entity that is creating the message"
    )
    content: str | LlmClientFunctionCall = Field(
        description="The content of the message or the result of a function call."
    )

    @field_validator("content", mode="before", check_fields=False)
    def parse_content(cls, v):
        if isinstance(v, str):
            try:
                parsed_content = json.loads(v)
                if isinstance(parsed_content, str):
                    # Sometimes we need a second decode if the content is
                    # escaped and string-encoded
                    parsed_content = json.loads(parsed_content)
                return LlmClientFunctionCall(**parsed_content)
            except (json.JSONDecodeError, TypeError, ValueError):
                return v
        return v


class SingleFileReference(BaseModel):
    url: HttpUrl = Field(
        description="The file reference to the data file. A URL to a file."  # noqa: E501
    )
    data_format: DataFormat = Field(
        description="Optional, but recommended. How the data should be parsed. If not provided, a best-effort attempt will be made to automatically determine the data format.",  # noqa: E501
    )
    citable: bool = Field(
        default=True,
        description="Whether to cite derivatives of the data source.",
    )


class DataFileReferences(BaseModel):
    items: list[SingleFileReference] = Field(description="A list of file references.")
    extra_citations: list[Citation] = Field(
        default_factory=list,
        description="The citations for the data content.",
    )


class SingleDataContent(BaseModel):
    content: str = Field(
        description="The data content, either as a raw string, JSON string, or as a base64 encoded string."  # noqa: E501
    )
    data_format: DataFormat = Field(
        default_factory=RawObjectDataFormat,
        description="How the data should be parsed and handled.",
    )
    citable: bool = Field(
        default=True,
        description="Whether to cite derivatives of the data source.",
    )


class DataContent(BaseModel):
    items: list[SingleDataContent] = Field(description="A list of data content items.")
    extra_citations: list[Citation] = Field(
        default_factory=list,
        description="The citations for the data content.",
    )


class ClientFunctionCallError(BaseModel):
    # TODO: Turn the error_type into an enum when we have more types of errors
    error_type: str = Field(description="The type of error that occurred.")
    content: str = Field(description="The error message of the function call.")


class LlmClientFunctionCallResultMessage(BaseModel):
    """Contains the result of a function call made against a client."""

    role: RoleEnum = RoleEnum.tool
    function: str = Field(description="The name of the called function.")
    input_arguments: dict[str, Any] = Field(
        default_factory=dict, description="The input arguments passed to the function"
    )
    data: list[DataContent | DataFileReferences | ClientFunctionCallError] = Field(
        description="The content of the function call. Each element corresponds to the result of a different data source."  # noqa: E501
    )
    extra_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra state to be passed between the client and this service.",
    )


class RawContext(BaseModel):
    uuid: UUID = Field(description="The UUID of the widget.")
    name: str = Field(description="The name of the widget.")
    description: str = Field(
        description="A description of the data contained in the widget"
    )
    data: DataContent = Field(description="The data content of the widget")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional widget metadata (eg. the selected ticker, etc)",
    )


LlmMessage = LlmClientFunctionCallResultMessage | LlmClientMessage


class DataSourceRequestPayload(BaseModel):
    widget_uuid: str
    origin: str
    id: str
    input_args: dict[str, Any]


class DataSourceParamOptionsRequestPayload(BaseModel):
    origin: str = Field(description="The origin of the data source.")
    id: str = Field(description="The widget id of the data source.")
    param: str = Field(description="The parameter to get options for.")
    options_endpoint_input_args: dict[str, Any] = Field(
        description="A dictionary of input arguments to pass to the options endpoint."
    )


class QueryRequest(BaseModel):
    messages: list[LlmClientFunctionCallResultMessage | LlmClientMessage] = Field(
        description="A list of messages to submit to the copilot."
    )
    context: list[RawContext] | None = Field(
        default=None, description="Additional context."
    )
    widgets: WidgetCollection | None = Field(
        default=None,
        description="A dictionary containing primary, secondary, and extra widgets.",
    )
    urls: list[str] | None = Field(
        default=None,
        description="URLs to retrieve and use as context. Limited to 4 URLs.",
    )
    api_keys: UserAPIKeys | None = Field(
        default=None, description="Use custom API keys for the request."
    )
    force_web_search: bool | None = Field(
        default=None,
        description="Set True to force a web search.",
    )
    timezone: str = Field(
        default="UTC",
        description="The timezone to use for the request.",
        examples=["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"],
    )

    @field_validator("messages", mode="before", check_fields=False)
    def check_messages_not_empty(cls, value):
        if not value:
            raise ValueError("messages list cannot be empty.")
        return value

    @field_validator("urls", mode="before", check_fields=False)
    def check_num_urls_within_limit(cls, value):
        if value and len(value) > 4:
            raise ValueError("urls list cannot have more than 4 elements.")
        return value


class DataSourceRequest(BaseModel):
    widget_uuid: str
    origin: str
    id: str
    input_args: dict[str, Any]


class FunctionCallResponse(BaseModel):
    function: str = Field(description="The name of the function to call.")
    input_arguments: dict | None = Field(
        default=None, description="The input arguments to the function."
    )
    extra_state: dict | None = Field(
        default=None,
        description="Extra state to be passed between the client and this service.",
    )


class ClientArtifact(BaseModel):
    """A piece of output data that is returned to the client."""

    type: Literal["text", "table", "chart"]
    name: str
    description: str
    uuid: UUID = Field(default_factory=uuid.uuid4)
    content: str | list[dict]
    chart_params: ChartParameters | None = None

    @model_validator(mode="after")
    def check_chart_params(cls, values):
        if values.type == "chart" and not values.chart_params:
            raise ValueError("chart_params is required for type 'chart'")
        if values.type != "chart" and values.chart_params:
            raise ValueError("chart_params is only allowed for type 'chart'")
        return values


class BaseSSE(BaseModel):
    event: Any
    data: Any

    def model_dump(self, *args, **kwargs) -> dict:
        return {
            "event": self.event,
            "data": self.data.model_dump_json(exclude_none=True),
        }


class MessageChunkSSEData(BaseModel):
    delta: str


class MessageChunkSSE(BaseSSE):
    event: Literal["copilotMessageChunk"] = "copilotMessageChunk"
    data: MessageChunkSSEData


class MessageArtifactSSE(BaseSSE):
    event: Literal["copilotMessageArtifact"] = "copilotMessageArtifact"
    data: ClientArtifact


class FunctionCallSSEData(BaseModel):
    function: Literal["get_widget_data", "get_extra_widget_data", "get_params_options"]
    input_arguments: dict
    extra_state: dict | None = Field(
        default=None,
        description="Extra state to be passed between the client and this service.",
    )


class FunctionCallSSE(BaseSSE):
    event: Literal["copilotFunctionCall"] = "copilotFunctionCall"
    data: FunctionCallSSEData


class CitationCollection(BaseModel):
    citations: list[Citation]


class CitationCollectionSSE(BaseSSE):
    event: Literal["copilotCitationCollection"] = "copilotCitationCollection"
    # We use a CitationCollection instead of a list because a pydantic model has
    # a .model_dump_json()
    data: CitationCollection


class StatusUpdateSSEData(BaseModel):
    eventType: Literal["INFO", "WARNING", "ERROR"]
    message: str
    group: Literal["reasoning"] = "reasoning"
    details: list[dict[str, Any] | str] | None = None
    artifacts: list[ClientArtifact] | None = None
    hidden: bool = False

    @model_validator(mode="before")
    @classmethod
    def exclude_fields(cls, values):
        # Exclude these fields from being in the "details" field.  (since this
        # pollutes the JSON output)
        _exclude_fields = EXCLUDE_STATUS_UPDATE_DETAILS_FIELDS
        if details := values.get("details"):
            if isinstance(details, list):
                for detail in details:
                    if isinstance(detail, dict):
                        for key in list(detail.keys()):
                            if key.lower() in _exclude_fields:
                                detail.pop(key, None)
        return values


class StatusUpdateSSE(BaseSSE):
    event: Literal["copilotStatusUpdate"] = "copilotStatusUpdate"
    data: StatusUpdateSSEData


SSE = (
    MessageChunkSSE
    | MessageArtifactSSE
    | FunctionCallSSE
    | StatusUpdateSSE
    | CitationCollectionSSE
)


class LocalFunctionCall:
    def __init__(self, function: Callable, **kwargs):
        self.function = function
        self.kwargs = kwargs

    async def __call__(
        self,
    ) -> AsyncGenerator[FunctionCallSSE | StatusUpdateSSE | str, None]:
        async for event in self.function(**self.kwargs):
            yield event


class StreamedText:
    def __init__(self, stream: AsyncGenerator[str, None]):
        self.stream = stream
        self.cached_stream = ""

    async def __aiter__(self):
        async for chunk in self.stream:
            self.cached_stream += chunk
            yield chunk


class WidgetRequest(BaseModel):
    widget: Widget
    input_arguments: dict[str, Any]
