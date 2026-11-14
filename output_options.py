import json
from pathlib import Path
from string import Template
from typing import Any, Literal

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field
from referencing import Registry
from referencing.exceptions import Unresolvable

from adapters import GatewayError, Message


class PromptReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    variables: dict[str, str] = Field(default_factory=dict)

    def render(self, templates: dict[str, dict[str, str]]) -> Message:
        source = templates.get(self.id, {}).get(str(self.version))
        if source is None:
            raise GatewayError(404, "prompt_not_found", "The requested prompt version was not found.")
        try:
            content = Template(source).substitute(self.variables)
        except KeyError as exc:
            raise GatewayError(422, "missing_prompt_variable", f"Missing prompt variable: {exc.args[0]}.") from exc
        return Message(role="system", content=content)


def load_templates() -> dict[str, dict[str, str]]:
    path = Path(__file__).with_name("prompts.json")
    return json.loads(path.read_text(encoding="utf-8"))


class JsonObjectFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class SchemaDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    schema_document: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class JsonSchemaFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: SchemaDefinition


def output_constraint(
    response_format: JsonObjectFormat | JsonSchemaFormat,
) -> tuple[dict[str, Any], Draft202012Validator | None]:
    if isinstance(response_format, JsonObjectFormat):
        return {"type": "json_object"}, None
    definition = response_format.json_schema
    schema = definition.schema_document
    # Keep schema evaluation local. Reject references outside this document.
    pending = [schema]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef", "$recursiveRef") and (
                    not isinstance(value, str) or not value.startswith("#")
                ):
                    raise GatewayError(422, "invalid_response_schema", "Use only local JSON Schema references.")
                pending.append(value)
        elif isinstance(node, list):
            pending.extend(node)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise GatewayError(422, "invalid_response_schema", "The JSON Schema is invalid.") from exc
    return {"type": "json_schema", **definition.model_dump(by_alias=True)}, Draft202012Validator(
        schema, registry=Registry()
    )


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def validate_output(text: str, validator: Draft202012Validator | None) -> None:
    try:
        parsed = json.loads(text, parse_constant=reject_json_constant)
    except (ValueError, RecursionError) as exc:
        raise GatewayError(502, "invalid_json_output", "The model did not return valid JSON.") from exc
    if validator is None:
        if not isinstance(parsed, dict):
            raise GatewayError(502, "invalid_json_output", "The model did not return a JSON object.")
        return
    try:
        valid = validator.is_valid(parsed)
    except (Unresolvable, RecursionError) as exc:
        raise GatewayError(422, "invalid_response_schema", "The JSON Schema could not be evaluated.") from exc
    if not valid:
        raise GatewayError(502, "schema_validation_failed", "The model output does not match the JSON Schema.")
