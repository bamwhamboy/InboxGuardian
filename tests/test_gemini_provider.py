import json
import sys
import types

import pytest

from app.classification.llm_classifier import ClassificationError, LLMEmailClassifier
from app.classification.provider import (
    DEFAULT_PROVIDER,
    GEMINI_API_KEY_ENV_VAR,
    GEMINI_DEFAULT_MODEL,
    GeminiLLMClient,
    LLMConfigurationError,
    PROVIDER_ENV_VAR,
    default_llm_client,
)
from app.schemas.email import Email


def make_email() -> Email:
    return Email(
        id="E-GEMINI-1",
        sender="Example",
        subject="Promotional course offer",
        body="Limited-time discount.",
        attachment_names=[],
    )


class FakeResponse:
    def __init__(self, text):
        self.text = text


def install_fake_genai(monkeypatch, responses):
    responses = list(responses)
    captured_configs = []

    class FakeModels:
        def generate_content(self, **kwargs):
            captured_configs.append(kwargs.get("config"))
            return FakeResponse(responses.pop(0))

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAutomaticFunctionCallingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_types = types.ModuleType("google.genai.types")
    fake_types.GenerateContentConfig = FakeConfig
    fake_types.AutomaticFunctionCallingConfig = FakeAutomaticFunctionCallingConfig

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    return captured_configs


def test_default_provider_is_groq_when_no_provider_is_set():
    assert DEFAULT_PROVIDER == "groq"


def test_gemini_remains_selectable_as_explicit_alternative(monkeypatch):
    monkeypatch.setenv(PROVIDER_ENV_VAR, "gemini")
    assert isinstance(default_llm_client(), GeminiLLMClient)


def test_gemini_requires_api_key(monkeypatch):
    monkeypatch.delenv(GEMINI_API_KEY_ENV_VAR, raising=False)
    with pytest.raises(LLMConfigurationError):
        GeminiLLMClient().classify_raw("system", "user", ["marketing"])


def test_gemini_default_model():
    assert GeminiLLMClient().model == GEMINI_DEFAULT_MODEL


def test_gemini_parses_structured_response(monkeypatch):
    monkeypatch.setenv(GEMINI_API_KEY_ENV_VAR, "test-key")
    install_fake_genai(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.98, "rationale": "Promotion."})],
    )
    result = GeminiLLMClient().classify_raw("system", "user", ["marketing"])
    assert result["category"] == "marketing"
    assert result["confidence"] == pytest.approx(0.98)


def test_gemini_malformed_json_is_rejected(monkeypatch):
    monkeypatch.setenv(GEMINI_API_KEY_ENV_VAR, "test-key")
    install_fake_genai(monkeypatch, ["not valid json"])
    with pytest.raises(LLMConfigurationError):
        GeminiLLMClient().classify_raw("system", "user", ["marketing"])


def test_gemini_extra_field_is_rejected_by_pydantic_and_retried(monkeypatch):
    monkeypatch.setenv(GEMINI_API_KEY_ENV_VAR, "test-key")
    install_fake_genai(
        monkeypatch,
        [
            json.dumps({"category": "marketing", "confidence": 0.99, "rationale": "x", "protected": True}),
            json.dumps({"category": "marketing", "confidence": 0.99, "rationale": "x"}),
        ],
    )
    result = LLMEmailClassifier(llm_client=GeminiLLMClient()).classify(make_email())
    assert result.category.value == "marketing"
    assert "protected" not in result.model_dump()


def test_gemini_client_works_through_classifier(monkeypatch):
    monkeypatch.setenv(GEMINI_API_KEY_ENV_VAR, "test-key")
    install_fake_genai(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.98, "rationale": "Promotion."})],
    )
    result = LLMEmailClassifier(llm_client=GeminiLLMClient()).classify(make_email())
    assert result.category.value == "marketing"
    assert result.confidence == pytest.approx(0.98)


def test_gemini_explicitly_disables_automatic_function_calling(monkeypatch):
    monkeypatch.setenv(GEMINI_API_KEY_ENV_VAR, "test-key")
    captured_configs = install_fake_genai(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )

    GeminiLLMClient().classify_raw("system", "user", ["marketing"])

    assert len(captured_configs) == 1
    config = captured_configs[0]
    afc_config = config.kwargs["automatic_function_calling"]
    assert afc_config.kwargs["disable"] is True
