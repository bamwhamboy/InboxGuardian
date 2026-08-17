"""Regression tests for V3 classification-guidance improvements.

These tests lock the prompt/category guidance and safety invariants. They do not
claim to prove real-model accuracy; that is measured by the 100-case evaluator.
"""

import re

from app.classification.prompts import CATEGORY_GUIDE, SYSTEM_PROMPT, valid_categories
from app.guardrails.policy import AUTO_DELETE_CONFIDENCE, DISPOSABLE_CATEGORIES, PROTECTED_CATEGORIES
from app.schemas.email import Classification, EmailCategory


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def test_social_is_offered_to_model_but_remains_disposable():
    assert "social" in valid_categories()
    assert EmailCategory.SOCIAL in DISPOSABLE_CATEGORIES
    assert EmailCategory.SOCIAL not in PROTECTED_CATEGORIES


def test_financial_marketing_boundaries_are_documented():
    guide = _normalized(CATEGORY_GUIDE)
    assert "promotes a specific named investment product, fund, or curated set of" in guide
    assert "promotes a specific named insurance product or plan type with concrete" in guide
    assert "being from a broker or fund house is not by itself enough" in guide
    assert "no specific product or pricing is marketing instead" in guide


def test_newsletter_marketing_boundary_is_documented():
    guide = _normalized(CATEGORY_GUIDE)
    assert "primary purpose is to inform rather than sell" in guide
    assert "prefer newsletter over marketing" in guide
    assert "informational product/service announcement as marketing" in guide


def test_social_newsletter_boundary_is_documented():
    guide = _normalized(CATEGORY_GUIDE)
    assert "an event about the user's own activity on a social platform" in guide
    assert "curated content digest" in guide
    assert "is newsletter" in guide


def test_transaction_security_boundary_is_documented():
    guide = _normalized(CATEGORY_GUIDE)
    assert "a financial/account event that already occurred" in guide
    assert "an authentication or account-protection event" in guide
    assert "a bank or financial sender does not automatically imply security" in guide


def test_transaction_includes_owned_loyalty_points_and_balances():
    guide = _normalized(CATEGORY_GUIDE)
    assert "loyalty/rewards points summary" in guide
    assert "any summary, balance, or statement of an owned account metric" in guide


def test_job_offer_requires_actual_hiring_organization_and_concrete_terms():
    guide = _normalized(CATEGORY_GUIDE)
    assert "actual hiring organization" in guide
    assert "concrete role/compensation/next-step details" in guide
    assert "hypothetical or rhetorical" in guide


def test_institutional_sender_alone_does_not_imply_protected():
    guide = _normalized(CATEGORY_GUIDE)
    assert "institutional sender alone" in guide
    assert "must not imply a protected category" in guide
    assert "only when both are present" in guide
    assert "explicit ownership/account-state signal" in guide


def test_category_selection_hierarchy_is_documented():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "first identify the email's primary intent" in prompt
    assert "prefer the broader applicable category" in prompt
    assert "rather than over-specializing" in prompt
    assert "financial-sounding sender or topic alone is not strong evidence" in prompt


def test_confidence_keeps_original_category_correctness_meaning():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "it always means the same thing regardless of category" in prompt
    assert "confidence that the predicted category itself is correct" in prompt
    assert "confidence that this email is safely disposable" not in prompt
    assert "disposition confidence" not in prompt


def test_disposable_selection_requires_protected_record_check():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "when selecting a disposable category, first verify that there is no evidence of an owned/" in prompt
    assert "protected record" in prompt
    assert "do not increase your confidence merely because an email looks promotional" in prompt
    assert "absence of an ownership signal is not by itself evidence" in prompt
    assert "too sparse or ambiguous" in prompt


def test_owned_records_prefer_protected_category_over_other():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "prefer the corresponding protected record category" in prompt
    assert "loyalty/points balance" in prompt


def test_credential_record_guidance_is_preserved():
    guide = _normalized(CATEGORY_GUIDE)
    prompt = _normalized(SYSTEM_PROMPT)
    assert "evidence a course/qualification was already completed" in guide
    assert "distinct from course_promotion" in guide
    assert "completed course or certification is credential_record" in prompt


def test_no_case_specific_email_ids_in_prompt():
    combined = SYSTEM_PROMPT + CATEGORY_GUIDE
    assert not re.search(r"\bE\d{3}\b", combined)


def test_no_literal_evaluation_subjects_in_prompt():
    combined = _normalized(SYSTEM_PROMPT + CATEGORY_GUIDE)
    banned = [
        "top 5 funds to invest in 2024",
        "open demat account & get",
        "hot stocks for today",
        "term insurance starting at",
        "what if your offer letter arrives in 14 days",
    ]
    for phrase in banned:
        assert phrase not in combined


def test_controlled_categories_match_existing_taxonomy():
    expected = {
        "course_promotion", "job_alert", "newsletter", "marketing",
        "insurance_marketing", "investment_marketing", "social", "job_offer",
        "employment_document", "salary", "investment_record", "insurance", "tax",
        "legal", "personal", "work", "transaction", "security",
        "credential_record", "other",
    }
    assert set(valid_categories()) == expected


def test_classification_schema_is_unchanged():
    assert set(Classification.model_fields) == {"category", "confidence", "rationale"}


def test_auto_delete_threshold_is_unchanged():
    assert AUTO_DELETE_CONFIDENCE == 0.95
