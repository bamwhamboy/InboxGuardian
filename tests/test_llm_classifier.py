import pytest
from pydantic import ValidationError

from app.classification.llm_classifier import ClassificationError, LLMEmailClassifier
from app.classification.pipeline import evaluate_email
from app.classification.prompts import valid_categories
from app.classification.provider import AnthropicLLMClient, LLMConfigurationError, _classification_tool_schema
from app.schemas.email import Classification, Decision, Email, EmailCategory, RiskLevel


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def classify_raw(self, system_prompt, user_prompt, category_values):
        self.calls.append((system_prompt, user_prompt, category_values))
        return self.responses.pop(0)


def email(**kwargs):
    values = {"id": "E001", "sender": "Example", "subject": "Example", "body": "Example"}
    values.update(kwargs)
    return Email(**values)


def test_valid_classification():
    fake = FakeLLMClient([{"category": "marketing", "confidence": 0.97, "rationale": "Promotional email."}])
    result = LLMEmailClassifier(fake).classify(email())
    assert result.category == EmailCategory.MARKETING
    assert result.confidence == pytest.approx(0.97)


def test_invalid_category_retries():
    fake = FakeLLMClient([
        {"category": "bogus", "confidence": 0.9, "rationale": "bad"},
        {"category": "newsletter", "confidence": 0.9, "rationale": "Recurring digest."},
    ])
    result = LLMEmailClassifier(fake).classify(email())
    assert result.category == EmailCategory.NEWSLETTER
    assert len(fake.calls) == 2


def test_invalid_confidence_retries():
    fake = FakeLLMClient([
        {"category": "newsletter", "confidence": 1.5, "rationale": "bad"},
        {"category": "newsletter", "confidence": 0.7, "rationale": "Digest."},
    ])
    assert LLMEmailClassifier(fake).classify(email()).confidence == pytest.approx(0.7)


def test_extra_fields_fail_closed():
    fake = FakeLLMClient([{"category": "marketing", "confidence": 0.9, "rationale": "x", "protected": True}])
    with pytest.raises(ClassificationError):
        LLMEmailClassifier(fake, max_attempts=1).classify(email())


def test_missing_field_fails():
    fake = FakeLLMClient([{"category": "marketing", "confidence": 0.9}])
    with pytest.raises(ClassificationError):
        LLMEmailClassifier(fake, max_attempts=1).classify(email())


def test_persistently_invalid_output_fails():
    fake = FakeLLMClient([{"category": "bogus", "confidence": 0.5, "rationale": "x"}] * 2)
    with pytest.raises(ClassificationError):
        LLMEmailClassifier(fake, max_attempts=2).classify(email())


def test_pipeline_protected_category_wins():
    fake = FakeLLMClient([{"category": "salary", "confidence": 0.55, "rationale": "Payslip."}])
    result = evaluate_email(email(subject="Your payslip"), LLMEmailClassifier(fake))
    assert result.proposed_action == Decision.KEEP
    assert result.risk == RiskLevel.CRITICAL


def test_pipeline_high_confidence_junk_is_delete_candidate():
    fake = FakeLLMClient([{"category": "marketing", "confidence": 0.97, "rationale": "Promotion."}])
    result = evaluate_email(email(subject="Flash sale"), LLMEmailClassifier(fake))
    assert result.proposed_action == Decision.DELETE


def test_pipeline_low_confidence_junk_requires_review():
    fake = FakeLLMClient([{"category": "newsletter", "confidence": 0.6, "rationale": "Possibly a digest."}])
    result = evaluate_email(email(), LLMEmailClassifier(fake))
    assert result.proposed_action == Decision.REVIEW
    assert result.human_review_required is True


def test_controlled_taxonomy_is_limited():
    offered = set(valid_categories())
    assert "marketing" in offered
    assert "security" in offered
    assert "banking" not in offered
    assert "payment" not in offered


def test_classification_schema_forbids_extra_fields():
    assert Classification.model_config.get("extra") == "forbid"
    with pytest.raises(ValidationError):
        Classification.model_validate({"category": "marketing", "confidence": 0.9, "rationale": "x", "protected": True})


def test_provider_tool_schema_has_no_protected_field():
    schema = _classification_tool_schema(["marketing", "security"])
    assert "protected" not in schema["input_schema"]["properties"]
    assert schema["input_schema"]["additionalProperties"] is False


def test_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError):
        AnthropicLLMClient().classify_raw("system", "user", ["marketing"])


def test_email_content_is_passed_to_model():
    fake = FakeLLMClient([{"category": "marketing", "confidence": 0.8, "rationale": "Promotion."}])
    LLMEmailClassifier(fake).classify(email(id="E999", sender="Acme", subject="50% off", body="Buy now"))
    _, prompt, _ = fake.calls[0]
    assert "E999" in prompt and "Acme" in prompt and "50% off" in prompt and "Buy now" in prompt
