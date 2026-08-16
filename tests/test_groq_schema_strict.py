import json
import sys
import types

from app.classification.provider import (
    GROQ_API_KEY_ENV_VAR,
    GroqLLMClient,
    _classification_schema_object,
    _classification_schema_object_for_groq,
    _with_additional_properties_false,
)


def _object_nodes(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _object_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _object_nodes(item)


def _install_fake_groq(monkeypatch, response_text):
    captured = []

    class FakeCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=response_text)
                )]
            )

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    fake = types.ModuleType("groq")
    fake.Groq = FakeGroq
    monkeypatch.setitem(sys.modules, "groq", fake)
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    return captured


def test_groq_schema_additional_properties_false_on_every_object_node():
    schema = _classification_schema_object_for_groq(["marketing", "security"])
    nodes = list(_object_nodes(schema))
    assert nodes
    assert all(node.get("additionalProperties") is False for node in nodes)


def test_groq_schema_preserves_shared_fields_required_and_taxonomy():
    base = _classification_schema_object(["marketing", "security"])
    groq = _classification_schema_object_for_groq(["marketing", "security"])
    assert groq["properties"] == base["properties"]
    assert groq["required"] == base["required"]
    assert groq["properties"]["category"]["enum"] == ["marketing", "security"]
    assert groq["additionalProperties"] is False


def test_additional_properties_transform_handles_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
            },
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"x": {"type": "number"}}},
            },
        },
    }
    transformed = _with_additional_properties_false(schema)
    assert all(node.get("additionalProperties") is False for node in _object_nodes(transformed))
    assert "additionalProperties" not in schema["properties"]["outer"]


def test_groq_request_uses_strict_schema(monkeypatch):
    captured = _install_fake_groq(
        monkeypatch,
        json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"}),
    )
    GroqLLMClient().classify_raw("system", "user", ["marketing"])
    schema = captured[0]["response_format"]["json_schema"]["schema"]
    assert captured[0]["response_format"]["type"] == "json_schema"
    assert captured[0]["response_format"]["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert all(node.get("additionalProperties") is False for node in _object_nodes(schema))
