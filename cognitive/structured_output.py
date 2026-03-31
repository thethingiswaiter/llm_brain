import json
from typing import Any, Callable


class StructuredOutputError(Exception):
    pass


class StructuredOutputFormatError(StructuredOutputError):
    pass


class StructuredOutputSchemaError(StructuredOutputError):
    pass


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_value(text: str) -> Any:
    candidate = _strip_code_fence(text)
    decoder = json.JSONDecoder()
    start_positions = [index for index, char in enumerate(candidate) if char in "[{"]

    if candidate and candidate[0] in "[{":
        start_positions = [0] + [index for index in start_positions if index != 0]

    for start_index in start_positions:
        try:
            value, _ = decoder.raw_decode(candidate[start_index:])
            return value
        except json.JSONDecodeError:
            continue

    raise StructuredOutputFormatError("No valid JSON object or array found in model output.")


def parse_json_object(
    text: str,
    required_fields: dict[str, type | tuple[type, ...]] | None = None,
    field_validators: dict[str, Callable[[Any], bool]] | None = None,
) -> dict[str, Any]:
    value = extract_json_value(text)
    if not isinstance(value, dict):
        raise StructuredOutputSchemaError("Expected a JSON object.")

    for field_name, expected_type in (required_fields or {}).items():
        if field_name not in value:
            raise StructuredOutputSchemaError(f"Missing required field: {field_name}")
        if not isinstance(value[field_name], expected_type):
            raise StructuredOutputSchemaError(
                f"Field {field_name} expected {expected_type}, got {type(value[field_name])}."
            )

    for field_name, validator in (field_validators or {}).items():
        if field_name in value and not validator(value[field_name]):
            raise StructuredOutputSchemaError(f"Field {field_name} failed validation.")

    return value


def parse_json_array(text: str) -> list[Any]:
    value = extract_json_value(text)
    if not isinstance(value, list):
        raise StructuredOutputSchemaError("Expected a JSON array.")
    return value