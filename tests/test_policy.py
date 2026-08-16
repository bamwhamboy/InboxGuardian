import json
from pathlib import Path

from app.guardrails.policy import apply_policy
from app.schemas.email import Decision, EmailCategory, RiskLevel


PROTECTED = [
    EmailCategory.JOB_OFFER,
    EmailCategory.EMPLOYMENT_DOCUMENT,
    EmailCategory.SALARY,
    EmailCategory.INVESTMENT_RECORD,
    EmailCategory.INSURANCE,
    EmailCategory.TAX,
    EmailCategory.LEGAL,
    EmailCategory.PERSONAL,
    EmailCategory.WORK,
    EmailCategory.TRANSACTION,
    EmailCategory.SECURITY,
    EmailCategory.CREDENTIAL_RECORD,
]

DISPOSABLE = [
    EmailCategory.COURSE_PROMOTION,
    EmailCategory.MARKETING,
    EmailCategory.NEWSLETTER,
    EmailCategory.JOB_ALERT,
    EmailCategory.INSURANCE_MARKETING,
    EmailCategory.INVESTMENT_MARKETING,
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


def test_investment_record_is_protected_but_marketing_is_disposable():
    record = apply_policy(EmailCategory.INVESTMENT_RECORD, confidence=0.99)
    marketing = apply_policy(EmailCategory.INVESTMENT_MARKETING, confidence=0.99)
    assert record[0] == Decision.KEEP
    assert record[1] == RiskLevel.CRITICAL
    assert marketing[0] == Decision.DELETE
    assert marketing[1] == RiskLevel.LOW


def test_credential_record_is_protected_and_distinct_from_course_and_employment():
    decision, risk, reasons, human_review = apply_policy(EmailCategory.CREDENTIAL_RECORD, confidence=0.5)
    assert decision == Decision.KEEP
    assert risk == RiskLevel.CRITICAL
    assert human_review is False
    assert reasons
    assert EmailCategory.CREDENTIAL_RECORD != EmailCategory.COURSE_PROMOTION
    assert EmailCategory.CREDENTIAL_RECORD != EmailCategory.EMPLOYMENT_DOCUMENT


def test_e082_credential_record_correction():
    corrections_path = Path(__file__).parents[1] / "data" / "eval" / "ground_truth_corrections.json"
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    correction = corrections["E082"]
    assert correction["category"] == "credential_record"
    assert correction["expected_action"] == "keep"
    assert correction["ground_truth"] == "not_spam"
    assert correction["protected"] is True


def test_e030_and_e037_use_investment_record_correction():
    corrections_path = Path(__file__).parents[1] / "data" / "eval" / "ground_truth_corrections.json"
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    for email_id in ("E030", "E037"):
        correction = corrections[email_id]
        assert correction["category"] == "investment_record"
        assert correction["expected_action"] == "keep"
        assert correction["ground_truth"] == "not_spam"
        assert correction["protected"] is True


def test_no_evaluation_case_uses_the_obsolete_investment_category():
    from app.evaluators.dataset import load_eval_cases

    cases = load_eval_cases()
    assert len(cases) == 100
    literal_investment = [c.id for c in cases if c.expected_category == "investment"]
    assert literal_investment == []


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
