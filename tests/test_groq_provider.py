"""Tests for the Groq provider. All Groq SDK calls are mocked via a fake
`groq` module injected into sys.modules -- no GROQ_API_KEY or network
access is required to run these.
"""

import json
import sys
import types

import pytest

from app.classification.llm_classifier import ClassificationError, LLMEmailClassifier
from app.classification.provider import (
    DEFAULT_PROVIDER,
    GROQ_API_KEY_ENV_VAR,
    GROQ_DEFAULT_MODEL,
    GroqLLMClient,
    LLMConfigurationError,
    PROVIDER_ENV_VAR,
    default_llm_client,
)
from app.schemas.email import Email


def make_email() -> Email:
    return Email(
        id="E-GROQ-1",
        sender="Example",
        subject="Promotional course offer",
        body="Limited-time discount.",
        attachment_names=[],
    )


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeChatCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)] if content is not None else []


def install_fake_groq(monkeypatch, responses, raise_exc=None):
    responses = list(responses)
    captured_calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            captured_calls.append(kwargs)
            if raise_exc is not None:
                raise raise_exc
            return FakeChatCompletion(responses.pop(0))

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeGroqClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = FakeChat()

    fake_groq = types.ModuleType("groq")
    fake_groq.Groq = FakeGroqClient

    monkeypatch.setitem(sys.modules, "groq", fake_groq)

    return captured_calls


# --- Provider selection (Groq is now default) -------------------------------


def test_default_provider_is_groq():
    assert DEFAULT_PROVIDER == "groq"


def test_default_client_is_groq(monkeypatch):
    monkeypatch.delenv(PROVIDER_ENV_VAR, raising=False)
    assert isinstance(default_llm_client(), GroqLLMClient)


def test_groq_selectable_explicitly(monkeypatch):
    monkeypatch.setenv(PROVIDER_ENV_VAR, "groq")
    assert isinstance(default_llm_client(), GroqLLMClient)


# --- Provider initialization / GROQ_API_KEY loading / model ----------------


def test_groq_client_initializes_without_touching_network():
    # Constructing the client must not require an API key or make any call.
    client = GroqLLMClient()
    assert client.model == GROQ_DEFAULT_MODEL


def test_groq_default_model_is_gpt_oss_120b():
    assert GroqLLMClient().model == GROQ_DEFAULT_MODEL == "openai/gpt-oss-120b"


def test_groq_model_override_via_constructor():
    assert GroqLLMClient(model="openai/gpt-oss-20b").model == "openai/gpt-oss-20b"


def test_groq_model_override_via_groq_specific_env_var(monkeypatch):
    monkeypatch.delenv("INBOXGUARDIAN_LLM_MODEL", raising=False)
    monkeypatch.setenv("INBOXGUARDIAN_GROQ_MODEL", "openai/gpt-oss-20b")
    assert GroqLLMClient().model == "openai/gpt-oss-20b"


def test_groq_does_not_inherit_stale_model_from_generic_env_var(monkeypatch):
    # Regression test: INBOXGUARDIAN_LLM_MODEL is shared by other providers
    # (e.g. Gemini). A leftover value there -- such as a Gemini model name
    # from a previous provider configuration -- must NOT silently become
    # Groq's model. Groq only honors GROQ_MODEL_ENV_VAR or an explicit
    # constructor argument.
    monkeypatch.delenv("INBOXGUARDIAN_GROQ_MODEL", raising=False)
    monkeypatch.setenv("INBOXGUARDIAN_LLM_MODEL", "gemini-3.5-flash")

    client = GroqLLMClient()

    assert client.model != "gemini-3.5-flash"
    assert client.model == GROQ_DEFAULT_MODEL == "openai/gpt-oss-120b"


def test_groq_specific_env_var_wins_over_stale_generic_one(monkeypatch):
    # Even with a stale generic value present, the Groq-specific override
    # still works correctly and takes priority over the default.
    monkeypatch.setenv("INBOXGUARDIAN_LLM_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("INBOXGUARDIAN_GROQ_MODEL", "openai/gpt-oss-20b")

    assert GroqLLMClient().model == "openai/gpt-oss-20b"


def test_groq_constructor_arg_wins_over_groq_specific_env_var(monkeypatch):
    monkeypatch.setenv("INBOXGUARDIAN_GROQ_MODEL", "openai/gpt-oss-20b")
    assert GroqLLMClient(model="openai/gpt-oss-120b").model == "openai/gpt-oss-120b"


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv(GROQ_API_KEY_ENV_VAR, raising=False)
    with pytest.raises(LLMConfigurationError):
        GroqLLMClient().classify_raw("system", "user", ["marketing"])


def test_groq_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )
    # Should not raise -- the key is present and picked up from the environment.
    GroqLLMClient().classify_raw("system", "user", ["marketing"])


def test_groq_never_hardcodes_api_key():
    import inspect

    from app.classification import provider

    source = inspect.getsource(provider)
    assert "gsk_" not in source  # common Groq API key prefix; must never appear literally
    assert "os.environ.get(GROQ_API_KEY_ENV_VAR)" in source


# --- Structured JSON Schema request -----------------------------------------


def test_groq_uses_json_schema_response_format(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    captured_calls = install_fake_groq(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )

    GroqLLMClient().classify_raw("system", "user", ["marketing", "security"])

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["model"] == GROQ_DEFAULT_MODEL

    response_format = call["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "classification"

    schema = response_format["json_schema"]["schema"]
    # Reuses the exact shared schema/taxonomy -- never redefined here.
    assert schema["properties"]["category"]["enum"] == ["marketing", "security"]
    assert set(schema["required"]) == {"category", "confidence", "rationale"}
    assert "protected" not in schema["properties"]


def test_groq_sends_system_and_user_messages(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    captured_calls = install_fake_groq(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )

    GroqLLMClient().classify_raw("SYSTEM PROMPT", "USER PROMPT", ["marketing"])

    messages = captured_calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert messages[1] == {"role": "user", "content": "USER PROMPT"}


def test_groq_reasoning_effort_is_low(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    captured_calls = install_fake_groq(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )

    GroqLLMClient().classify_raw("system", "user", ["marketing"])

    call = captured_calls[0]
    assert call["reasoning_effort"] == "low"


def test_groq_include_reasoning_is_false(monkeypatch):
    # Reasoning tokens must never be exposed in the classifier's output.
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    captured_calls = install_fake_groq(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )

    GroqLLMClient().classify_raw("system", "user", ["marketing"])

    call = captured_calls[0]
    assert call["include_reasoning"] is False


def test_groq_does_not_send_reasoning_format():
    # reasoning_format is not supported by openai/gpt-oss-120b and must
    # never be sent as an actual keyword argument on the request (mentioning
    # it in an explanatory comment is fine).
    import inspect

    from app.classification import provider

    source = inspect.getsource(provider.GroqLLMClient.classify_raw)
    assert "reasoning_format=" not in source


def test_groq_reasoning_format_key_is_absent_from_request(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    captured_calls = install_fake_groq(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.9, "rationale": "x"})],
    )

    GroqLLMClient().classify_raw("system", "user", ["marketing"])

    call = captured_calls[0]
    assert "reasoning_format" not in call


# --- Parsing a valid classification -----------------------------------------


def test_groq_parses_structured_response(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.98, "rationale": "Promotion."})],
    )
    result = GroqLLMClient().classify_raw("system", "user", ["marketing"])
    assert result["category"] == "marketing"
    assert result["confidence"] == pytest.approx(0.98)
    assert result["rationale"] == "Promotion."


def test_groq_client_works_through_classifier(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(
        monkeypatch,
        [json.dumps({"category": "marketing", "confidence": 0.98, "rationale": "Promotion."})],
    )
    result = LLMEmailClassifier(llm_client=GroqLLMClient()).classify(make_email())
    assert result.category.value == "marketing"
    assert result.confidence == pytest.approx(0.98)


def test_groq_extra_field_is_rejected_by_pydantic_and_retried(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(
        monkeypatch,
        [
            json.dumps({"category": "marketing", "confidence": 0.99, "rationale": "x", "protected": True}),
            json.dumps({"category": "marketing", "confidence": 0.99, "rationale": "x"}),
        ],
    )
    result = LLMEmailClassifier(llm_client=GroqLLMClient()).classify(make_email())
    assert result.category.value == "marketing"
    assert "protected" not in result.model_dump()


# --- Malformed response handling --------------------------------------------


def test_groq_malformed_json_is_rejected(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(monkeypatch, ["not valid json"])
    with pytest.raises(LLMConfigurationError):
        GroqLLMClient().classify_raw("system", "user", ["marketing"])


def test_groq_empty_response_is_rejected(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(monkeypatch, [None])
    with pytest.raises(LLMConfigurationError):
        GroqLLMClient().classify_raw("system", "user", ["marketing"])


def test_groq_non_object_json_is_rejected(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(monkeypatch, [json.dumps(["not", "an", "object"])])
    with pytest.raises(LLMConfigurationError):
        GroqLLMClient().classify_raw("system", "user", ["marketing"])


def test_groq_invalid_category_retries_then_fails(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(
        monkeypatch,
        [
            json.dumps({"category": "not_a_real_category", "confidence": 0.9, "rationale": "x"}),
            json.dumps({"category": "not_a_real_category", "confidence": 0.9, "rationale": "x"}),
        ],
    )
    with pytest.raises(ClassificationError):
        LLMEmailClassifier(llm_client=GroqLLMClient(), max_attempts=2).classify(make_email())


# --- API failure handling ----------------------------------------------------


def test_groq_api_failure_is_wrapped_in_classification_error(monkeypatch):
    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(monkeypatch, [], raise_exc=RuntimeError("groq api outage"))
    with pytest.raises(ClassificationError):
        LLMEmailClassifier(llm_client=GroqLLMClient(), max_attempts=2).classify(make_email())


def test_groq_missing_package_raises_configuration_error(monkeypatch):
    import sys as sys_module

    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    monkeypatch.setitem(sys_module.modules, "groq", None)  # simulate ImportError on `from groq import Groq`
    with pytest.raises(LLMConfigurationError):
        GroqLLMClient().classify_raw("system", "user", ["marketing"])


# --- Classification failure routes to REVIEW, never DELETE -----------------


def test_groq_classification_failure_routes_to_review_not_delete(monkeypatch):
    from app.evaluators.dataset import load_eval_cases
    from app.evaluators.runner import evaluate_case

    monkeypatch.setenv(GROQ_API_KEY_ENV_VAR, "test-key")
    install_fake_groq(monkeypatch, [], raise_exc=RuntimeError("groq api outage"))
    classifier = LLMEmailClassifier(llm_client=GroqLLMClient(), max_attempts=1)
    case = load_eval_cases(ids=["E001"])[0]

    result = evaluate_case(case, classifier)

    assert result.error is True
    assert result.confidence is None
    assert result.predicted_category is None
    assert result.proposed_action == "review"
    assert result.proposed_action != "delete"
    assert result.human_review_required is True
