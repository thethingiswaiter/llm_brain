import json
from typing import Any, Callable

from core.llm.manager import llm_manager


class StructuredOutputError(Exception):
    pass


class StructuredOutputFormatError(StructuredOutputError):
    pass


class StructuredOutputSchemaError(StructuredOutputError):
    pass


class StructuredOutputFunctionCallError(StructuredOutputError):
    pass


def _strip_code_fence(text: str) -> str:
    stripped = str(text or "").replace("\ufeff", "").replace("\x00", "").strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_value(text: str) -> Any:
    if isinstance(text, (dict, list)):
        return text

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


def _coerce_tool_call_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise StructuredOutputFunctionCallError(f"Tool call arguments are not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise StructuredOutputFunctionCallError("Tool call arguments must decode to a JSON object.")
        return value
    raise StructuredOutputFunctionCallError("Tool call arguments must be a dict or JSON string.")


def extract_tool_call_arguments(response: Any, expected_name: str | None = None) -> dict[str, Any]:
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        additional_kwargs = getattr(response, "additional_kwargs", None) or {}
        tool_calls = additional_kwargs.get("tool_calls", []) or []
    if not tool_calls:
        raise StructuredOutputFunctionCallError("Model response does not contain any tool calls.")

    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("function", {}).get("name") or "").strip()
        if expected_name and name != expected_name:
            continue
        arguments = item.get("args")
        if arguments is None:
            arguments = item.get("arguments")
        if arguments is None:
            arguments = item.get("function", {}).get("arguments")
        return _coerce_tool_call_arguments(arguments)

    raise StructuredOutputFunctionCallError(
        f"Model returned tool calls, but none matched expected function: {expected_name or '<any>'}."
    )


def invoke_function_call(
    prompt: str,
    function_name: str,
    function_description: str,
    parameters_schema: dict[str, Any],
    source: str,
    fallback_content_parser: Callable[[Any], Any] | None = None,
    fallback_content_adapter: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    llm = llm_manager.get_llm()
    if not hasattr(llm, "bind_tools"):
        raise StructuredOutputFunctionCallError("Current LLM does not support bind_tools for function calling.")

    function_schema = {
        "type": "function",
        "function": {
            "name": function_name,
            "description": function_description,
            "parameters": parameters_schema,
        },
    }
    try:
        bound_llm = llm.bind_tools(
            [function_schema],
            tool_choice={"type": "function", "function": {"name": function_name}},
        )
    except TypeError:
        bound_llm = llm.bind_tools([function_schema])
    except Exception as exc:
        raise StructuredOutputFunctionCallError(f"Failed to bind function schema: {exc}") from exc

    response = llm_manager.invoke(prompt, source=source, llm=bound_llm)
    try:
        return extract_tool_call_arguments(response, expected_name=function_name)
    except StructuredOutputFunctionCallError as exc:
        if fallback_content_parser is None:
            raise exc

        raw_payload = getattr(response, "content", response)
        fallback_candidates = [raw_payload]
        stringify = getattr(llm_manager, "_stringify_response", None)
        if callable(stringify):
            try:
                stringified = stringify(response)
            except Exception:
                stringified = ""
            if stringified and stringified != raw_payload:
                fallback_candidates.append(stringified)

        for candidate in fallback_candidates:
            try:
                parsed = fallback_content_parser(candidate)
                if fallback_content_adapter is not None:
                    parsed = fallback_content_adapter(parsed)
                if not isinstance(parsed, dict):
                    raise StructuredOutputSchemaError("Fallback structured content must resolve to a JSON object.")
                return parsed
            except StructuredOutputError:
                continue

        raise exc