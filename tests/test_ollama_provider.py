"""Tests for the Ollama (local) provider. All Ollama SDK calls are mocked
via a fake `ollama` module injected into sys.modules -- no local Ollama
service, model pull, or network access is required to run these.
"""

import json
import sys
import types

import pytest

from app.classification.llm_classifier import ClassificationError, LLMEmailClassifier
from app.classification.provider import (
    DEFAULT_PROVIDER,
    LLMConfigurationError,
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_HOST_ENV_VAR,
    OLLAMA_MODEL_ENV_VAR,
    OllamaLLMClient,
    PROVIDER_ENV_VAR,
    default_llm_client,
)
from app.schemas.email import Email


def make_email() -> Email:
    return Email(
        id="E-OLLAMA-1",
        sender="Example",
        subject="Promotional course offer",
        body="Limited-time discount.",
        attachment_names=[],
    )


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChatResponse:
    def __init__(self, content):
        self.message = FakeMessage(content) if content is not None else None


def install_fake_ollama(monkeypatch, responses, raise_exc=None):
    responses = list(responses)
    captured_calls = []
    captured_hosts = []

    class FakeClient:
        def __init__(self, host=None):
            captured_hosts.append(host)
            self.host = host

        def chat(self, **kwargs):
            captured_calls.append(kwargs)
            if raise_exc is not None:
                raise raise_exc
            return FakeChatResponse(responses.pop(0))

    fake_ollama = types.ModuleType("ollama")
    fake_ollama.Client = FakeClient

    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

    return captured_calls, captured_hosts


# --- Provider selection (must not break Gemini/Groq/Anthropic) -------------


def test_default_provider_is_still_groq_ollama_does_not_change_default():
    assert DEFAULT_PROVIDER == "groq"


def test_ollama_selectable_explicitly(monkeypatch):
    monkeypatch.setenv(PROVIDER_ENV_VAR, "ollama")
    assert isinstance(default_llm_client(), OllamaLLMClient)


def test_groq_still_selectable_after_adding_ollama(monkeypatch):
    from app.classification.provider import GroqLLMClient

    monkeypatch.delenv(PROVIDER_ENV_VAR, raising=False)
    assert isinstance(default_llm_client(), GroqLLMClient)


def test_gemini_still_selectable_after_adding_ollama(monkeypatch):
    from app.classification.provider import GeminiLLMClient

    monkeypatch.setenv(PROVIDER_ENV_VAR, "gemini")
    assert isinstance(default_llm_client(), GeminiLLMClient)


def test_anthropic_still_selectable_after_adding_ollama(monkeypatch):
    from app.classification.provider import AnthropicLLMClient

    monkeypatch.setenv(PROVIDER_ENV_VAR, "anthropic")
    assert isinstance(default_llm_client(), AnthropicLLMClient)


def test_unknown_provider_still_raises_with_ollama_listed(monkeypatch):
    monkeypatch.setenv(PROVIDER_ENV_VAR, "not-a-real-provider")
    with pytest.raises(LLMConfigurationError) as exc_info:
        default_llm_client()
    assert "ollama" in str(exc_info.value)


# --- Provider initialization / default model / no API key needed -----------


def test_ollama_client_initializes_without_touching_network():
    # Constructing the client must not require any API key, env var, or
    # network/service call.
    client = OllamaLLMClient()
    assert client.model == OLLAMA_DEFAULT_MODEL


def test_ollama_default_model_is_qwen3_8b():
    assert OllamaLLMClient().model == OLLAMA_DEFAULT_MODEL == "qwen3:8b"


def test_ollama_model_override_via_constructor():
    assert OllamaLLMClient(model="qwen3:4b").model == "qwen3:4b"


def test_ollama_model_override_via_ollama_specific_env_var(monkeypatch):
    monkeypatch.delenv("INBOXGUARDIAN_LLM_MODEL", raising=False)
    monkeypatch.setenv(OLLAMA_MODEL_ENV_VAR, "qwen3:4b")
    assert OllamaLLMClient().model == "qwen3:4b"


def test_ollama_does_not_inherit_stale_model_from_generic_env_var(monkeypatch):
    # Same protection as Groq: a leftover INBOXGUARDIAN_LLM_MODEL value from
    # a different provider must NOT silently become Ollama's model.
    monkeypatch.delenv(OLLAMA_MODEL_ENV_VAR, raising=False)
    monkeypatch.setenv("INBOXGUARDIAN_LLM_MODEL", "gemini-3.5-flash")

    client = OllamaLLMClient()

    assert client.model != "gemini-3.5-flash"
    assert client.model == OLLAMA_DEFAULT_MODEL == "qwen3:8b"


def test_ollama_requires_no_api_key(monkeypatch):
    # Ollama is local-only: no API key environment variable of any kind
    # should be required to construct or use the client.
    for env_var in ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    install_fake_ollama(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )
    # Should not raise for any missing API key.
    OllamaLLMClient().classify_raw("system", "user", ["marketing"])


def test_ollama_host_override_via_env_var(monkeypatch):
    monkeypatch.setenv(OLLAMA_HOST_ENV_VAR, "http://127.0.0.1:9999")
    _, hosts = install_fake_ollama(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )
    OllamaLLMClient().classify_raw("system", "user", ["marketing"])
    assert hosts == ["http://127.0.0.1:9999"]


def test_ollama_uses_default_host_when_unset(monkeypatch):
    monkeypatch.delenv(OLLAMA_HOST_ENV_VAR, raising=False)
    _, hosts = install_fake_ollama(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )
    OllamaLLMClient().classify_raw("system", "user", ["marketing"])
    # host=None lets the ollama.Client() fall back to its own default
    # rather than us hard-coding the URL.
    assert hosts == [None]


def test_ollama_never_hardcodes_a_url_or_key():
    import inspect

    from app.classification import provider

    source = inspect.getsource(provider.OllamaLLMClient)
    # The default host URL must never appear as an actual assigned/passed
    # value in the client code (mentioning it in an explanatory comment
    # about ollama.Client()'s own default is fine and expected).
    assert 'host = "http://localhost:11434"' not in source
    assert "host='http://localhost:11434'" not in source
    assert "os.environ.get(OLLAMA_MODEL_ENV_VAR)" in source


# --- Structured JSON Schema request -----------------------------------------


def test_ollama_uses_structured_format_schema(monkeypatch):
    captured_calls, _ = install_fake_ollama(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )

    OllamaLLMClient().classify_raw("system", "user", ["marketing", "security"])

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["model"] == OLLAMA_DEFAULT_MODEL

    schema = call["format"]
    # Reuses the exact shared schema/taxonomy -- never redefined here.
    assert schema["type"] == "object"
    assert schema["properties"]["category"]["enum"] == ["marketing", "security"]
    assert set(schema["required"]) == {"category", "confidence", "rationale"}
    assert "protected" not in schema["properties"]


def test_ollama_sends_system_and_user_messages(monkeypatch):
    captured_calls, _ = install_fake_ollama(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )

    OllamaLLMClient().classify_raw("SYSTEM PROMPT", "USER PROMPT", ["marketing"])

    messages = captured_calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert messages[1] == {"role": "user", "content": "USER PROMPT"}


def test_ollama_disables_thinking(monkeypatch):
    # qwen3:8b supports extended thinking; it must be disabled so reasoning
    # tokens never leak into the classification response.
    captured_calls, _ = install_fake_ollama(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )

    OllamaLLMClient().classify_raw("system", "user", ["marketing"])

    assert captured_calls[0]["think"] is False


# --- Parsing a valid classification -----------------------------------------


def test_ollama_parses_structured_response(monkeypatch):
    install_fake_ollama(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.98, "rationale": "Promotion."})],
    )
    result = OllamaLLMClient().classify_raw("system", "user", ["marketing"])
    assert result["category"] == "marketing"
    assert result["confidence"] == pytest.approx(0.98)
    assert result["rationale"] == "Promotion."


def test_ollama_client_works_through_classifier(monkeypatch):
    install_fake_ollama(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.98, "rationale": "Promotion."})],
    )
    result = LLMEmailClassifier(llm_client=OllamaLLMClient()).classify(make_email())
    assert result.category.value == "marketing"
    assert result.confidence == pytest.approx(0.98)


def test_ollama_extra_field_is_rejected_by_pydantic_and_retried(monkeypatch):
    install_fake_ollama(
        monkeypatch,
        [
            json.dumps({"category": "marketing", "confidence": 0.99, "rationale": "x", "protected": True}),
            json.dumps({"category": "marketing", "confidence": 0.99, "rationale": "x"}),
        ],
    )
    result = LLMEmailClassifier(llm_client=OllamaLLMClient()).classify(make_email())
    assert result.category.value == "marketing"
    assert "protected" not in result.model_dump()


# --- Malformed/invalid response handling ------------------------------------


def test_ollama_malformed_json_is_rejected(monkeypatch):
    install_fake_ollama(monkeypatch, ["not valid json"])
    with pytest.raises(LLMConfigurationError):
        OllamaLLMClient().classify_raw("system", "user", ["marketing"])


def test_ollama_empty_response_is_rejected(monkeypatch):
    install_fake_ollama(monkeypatch, [None])
    with pytest.raises(LLMConfigurationError):
        OllamaLLMClient().classify_raw("system", "user", ["marketing"])


def test_ollama_non_object_json_is_rejected(monkeypatch):
    install_fake_ollama(monkeypatch, [json.dumps(["not", "an", "object"])])
    with pytest.raises(LLMConfigurationError):
        OllamaLLMClient().classify_raw("system", "user", ["marketing"])


def test_ollama_invalid_category_retries_then_fails(monkeypatch):
    install_fake_ollama(
        monkeypatch,
        [
            json.dumps({"category": "not_a_real_category", "confidence": 0.9, "rationale": "x"}),
            json.dumps({"category": "not_a_real_category", "confidence": 0.9, "rationale": "x"}),
        ],
    )
    with pytest.raises(ClassificationError):
        LLMEmailClassifier(llm_client=OllamaLLMClient(), max_attempts=2).classify(make_email())


# --- Missing/unavailable local Ollama service -------------------------------


def test_ollama_service_unavailable_is_wrapped_in_configuration_error(monkeypatch):
    install_fake_ollama(monkeypatch, [], raise_exc=ConnectionError("connection refused"))
    with pytest.raises(LLMConfigurationError):
        OllamaLLMClient().classify_raw("system", "user", ["marketing"])


def test_ollama_service_unavailable_is_wrapped_in_classification_error_via_classifier(monkeypatch):
    install_fake_ollama(monkeypatch, [], raise_exc=ConnectionError("connection refused"))
    with pytest.raises(ClassificationError):
        LLMEmailClassifier(llm_client=OllamaLLMClient(), max_attempts=2).classify(make_email())


def test_ollama_missing_package_raises_configuration_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "ollama", None)  # simulate ImportError on `import ollama`
    with pytest.raises(LLMConfigurationError):
        OllamaLLMClient().classify_raw("system", "user", ["marketing"])


# --- Classification failure routes to REVIEW, never DELETE -----------------


def test_ollama_classification_failure_routes_to_review_not_delete(monkeypatch):
    from app.evaluators.dataset import load_eval_cases
    from app.evaluators.runner import evaluate_case

    install_fake_ollama(monkeypatch, [], raise_exc=ConnectionError("connection refused"))
    classifier = LLMEmailClassifier(llm_client=OllamaLLMClient(), max_attempts=1)
    case = load_eval_cases(ids=["E001"])[0]

    result = evaluate_case(case, classifier)

    assert result.error is True
    assert result.confidence is None
    assert result.predicted_category is None
    assert result.proposed_action == "review"
    assert result.proposed_action != "delete"
    assert result.human_review_required is True
