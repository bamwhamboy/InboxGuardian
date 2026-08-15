import json
from pathlib import Path

from app.guardrails.policy import apply_policy
from app.schemas.email import Decision, EmailCategory, RiskLevel


PROTECTED = [
    EmailCategory.JOB_OFFER,
    EmailCategory.EMPLOYMENT_DOCUMENT,
    EmailCategory.SALARY,
    EmailCategory.INVESTMENT,
    EmailCategory.INSURANCE,
    EmailCategory.TAX,
    EmailCategory.LEGAL,
    EmailCategory.PERSONAL,
    EmailCategory.WORK,
    EmailCategory.TRANSACTION,
    EmailCategory.SECURITY,
]


DISPOSABLE = [
    EmailCategory.COURSE_PROMOTION,
    EmailCategory.MARKETING,
    EmailCategory.NEWSLETTER,
    EmailCategory.JOB_ALERT,
    EmailCategory.INSURANCE_MARKETING,
    EmailCategory.SOCIAL,
]


def test_protected_categories_always_keep():
    for category in PROTECTED:
        decision, risk, reasons, human_review = apply_policy(category, confidence=1.0)
        assert decision == Decision.KEEP
        assert risk == RiskLevel.CRITICAL
        assert reasons
        assert human_review is False


def test_high_confidence_disposable_categories_can_delete():
    for category in DISPOSABLE:
        decision, risk, reasons, human_review = apply_policy(category, confidence=0.99)
        assert decision == Decision.DELETE
        assert risk == RiskLevel.LOW
        assert human_review is False


def test_ambiguous_disposable_case_requires_human_review():
    for category in DISPOSABLE:
        decision, risk, reasons, human_review = apply_policy(category, confidence=0.80)
        assert decision == Decision.REVIEW
        assert risk == RiskLevel.MEDIUM
        assert human_review is True
        assert reasons


def test_unknown_category_requires_human_review():
    decision, risk, reasons, human_review = apply_policy(EmailCategory.OTHER, confidence=0.99)
    assert decision == Decision.REVIEW
    assert risk == RiskLevel.MEDIUM
    assert human_review is True
    assert reasons


def test_dataset_contains_100_cases():
    dataset_path = Path(__file__).parents[1] / "data" / "eval" / "email_dataset.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8"))

    assert len(data) == 100
    assert len({item["id"] for item in data}) == 100


def test_dataset_protected_cases_are_not_delete():
    dataset_path = Path(__file__).parents[1] / "data" / "eval" / "email_dataset.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8"))

    for item in data:
        if item["protected"]:
            assert item["expected_action"] == "keep"
