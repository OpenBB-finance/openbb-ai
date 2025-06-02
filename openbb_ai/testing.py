import json
from ast import literal_eval

from pydantic import BaseModel


class CopilotEvent(BaseModel):
    event_type: str
    content: str | dict


class CopilotResponse:
    def __init__(self, event_stream: str):
        self.events: list = []
        self.index = 0
        self.event_stream = event_stream
        self.parse_event_stream()

    def parse_event_stream(self):
        event_name = ""
        lines = self.event_stream.split("\n")
        copilot_message_event_index = None
        for line in lines:
            if line.startswith("event:"):
                event_type = line.split("event:")[1].strip()
            if event_type == "copilotMessageChunk" and line.startswith("data:"):
                if (
                    copilot_message_event_index is None
                    or copilot_message_event_index < len(self.events) - 1
                ):
                    self.events.append(
                        CopilotEvent(event_type="copilotMessage", content="")
                    )
                    copilot_message_event_index = len(self.events) - 1
                data_payload = line.split("data:")[1].strip()
                data_dict_ = literal_eval(data_payload)
                self.events[copilot_message_event_index].content += data_dict_["delta"]
            elif event_type == "copilotMessageArtifact" and line.startswith("data:"):
                event_name = "copilotMessageArtifact"
                data_payload = line.split("data:")[1].strip()
                data_dict_ = json.loads(data_payload)
                self.events.append(
                    CopilotEvent(event_type=event_name, content=data_dict_)
                )
            elif event_type == "copilotFunctionCall" and line.startswith("data:"):
                event_name = "copilotFunctionCall"
                data_payload = line.split("data:")[1].strip()
                data_dict_ = literal_eval(data_payload)
                self.events.append(
                    CopilotEvent(event_type=event_name, content=data_dict_)
                )
            elif event_type == "copilotStatusUpdate" and line.startswith("data:"):
                event_name = "copilotStatusUpdate"
                data_payload = line.split("data:")[1].strip()
                data_dict_ = json.loads(data_payload)
                self.events.append(
                    CopilotEvent(event_type=event_name, content=data_dict_)
                )
            elif event_type == "copilotCitationCollection" and line.startswith("data:"):
                event_name = "copilotCitationCollection"
                data_payload = line.split("data:")[1].strip()
                data_dict_ = json.loads(data_payload)
                self.events.append(
                    CopilotEvent(event_type=event_name, content=data_dict_)
                )

    @property
    def text(self) -> str:
        return "".join(
            str(event.content)
            for event in self.events
            if event.event_type == "copilotMessage"
        )

    @property
    def function_calls(self) -> list[CopilotEvent]:
        return [
            event for event in self.events if event.event_type == "copilotFunctionCall"
        ]

    @property
    def citations(self) -> list[CopilotEvent]:
        return [
            event
            for event in self.events
            if event.event_type == "copilotCitationCollection"
        ]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.events):
            event = self.events[self.index]
            self.index += 1
            return event
        else:
            raise StopIteration

    def _contains(self, event, content_contains: str | dict, ignore_case: bool = True):
        try:
            self._assert_contains(event, content_contains, ignore_case)
            return True
        except AssertionError:
            return False

    def _assert_contains(
        self, event, content_contains: str | dict, ignore_case: bool = True
    ):
        if isinstance(content_contains, str):
            if ignore_case:
                assert content_contains.lower() in str(event.content).lower()
            else:
                assert content_contains in str(event.content)
        elif isinstance(content_contains, dict):
            for key, value in content_contains.items():
                assert key in event.content
                assert event.content[key] == value
        else:
            raise ValueError(f"Invalid content_contains type: {type(content_contains)}")

    def starts(self, event_type: str):
        self.index = 0
        assert self.events[self.index].event_type == event_type
        return self

    def with_(self, content_contains: str | dict, ignore_case: bool = True):
        self._assert_contains(self.events[self.index], content_contains, ignore_case)
        return self

    def then(self, event_type: str):
        self.index += 1
        assert self.events[self.index].event_type == event_type
        return self

    def and_(self, content_contains: str):
        # assert content_contains in str(self.events[self.index].content)
        self._assert_contains(self.events[self.index], content_contains)
        return self

    def ends(self, event_type: str):
        self.index = len(self.events) - 1
        assert self.events[self.index].event_type == event_type
        return self

    def has_any(
        self, event_type: str, content_contains: str | dict, ignore_case: bool = True
    ):
        assert any(
            event_type == event.event_type
            and self._contains(event, content_contains, ignore_case)
            for event in self.events
        ), (
            f"Event type {event_type} with content {content_contains} not found in events.\n"  # noqa: E501
            f"Events found:\n"
            + "\n".join(f"{e.event_type}: {e.content}" for e in self.events)
        )
        return self

    def has_all(self, copilot_events: list[CopilotEvent]):
        assert all(
            copilot_event.event_type == event.event_type
            and copilot_event.content == event.content
            for copilot_event in copilot_events
            for event in self.events
        )
        return self


def capture_stream_response(event_stream: str) -> tuple[str, str]:
    if "copilotFunctionCall" in event_stream:
        event_name = "copilotFunctionCall"
        lines = event_stream.split("\n")
        data_payload = lines[1].split("data:")[-1].strip()
        return event_name, data_payload

    captured_stream = ""
    event_name = ""
    event_type = ""  # Initialize event_type
    lines = event_stream.split("\n")
    for line in lines:
        if line.startswith("event:"):
            event_type = line.split("event:")[1].strip()
        if event_type == "copilotMessageChunk" and line.startswith("data:"):
            event_name = "copilotMessageChunk"
            data_payload = line.split("data:")[1].strip()
            data_dict_ = literal_eval(data_payload)
            captured_stream += data_dict_["delta"]
    return event_name, captured_stream
