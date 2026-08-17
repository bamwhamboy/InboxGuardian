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


# ---------------------------------------------------------------------------
# V4: symmetric, evidence-strength-calibrated confidence (fixes V3's
# over-conservative suppression of confidence on clearly disposable email).
# Confidence semantics themselves are unchanged (still "confidence the
# predicted category is correct" for every category) -- only the guidance
# on how to calibrate that number changed.
# ---------------------------------------------------------------------------


def test_strong_disposable_evidence_should_get_high_confidence():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "strong disposable evidence -> disposable category + high confidence" in prompt
    assert "be willing to report high" in prompt
    assert "do not hold it back out of general caution" in prompt


def test_strong_disposable_evidence_examples_are_present_and_generic():
    prompt = _normalized(SYSTEM_PROMPT)
    for example in (
        "an explicit invitation to enroll in a course",
        "an obvious promotional offer or sales pitch",
        "a recurring newsletter/editorial digest",
        "an automated job-board alert",
        "a social-platform activity notification",
    ):
        assert example in prompt


def test_legitimate_sender_does_not_justify_lower_confidence():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "do not lower confidence merely because the sender is a legitimate" in prompt
    assert "a clear sales pitch from a real bank or broker is still a clear sales pitch" in prompt


def test_strong_protected_evidence_should_get_high_confidence():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "strong protected evidence -> protected category + high confidence" in prompt
    for example in (
        "a transaction/order/payment event",
        "an account security event",
        "an insurance policy/claim/maturity event",
        "an investment statement/holding/transaction",
        "a credential/course-completion record",
        "an employment/hr document",
        "a tax/legal record",
    ):
        assert example in prompt


def test_ambiguous_evidence_still_routes_to_low_confidence_and_other():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "ambiguous or insufficient evidence -> low confidence + other" in prompt
    assert "an empty/generic bank notification with no meaningful content" in prompt
    assert "these should get low confidence and route to review" in prompt


def test_absence_of_protected_evidence_is_still_not_positive_disposable_evidence():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "do not treat the mere absence of protected evidence as positive evidence" in prompt
    assert "disposable confidence must come from evidence of disposable intent" in prompt


def test_goal_is_calibration_not_a_global_confidence_increase():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "the goal is not to raise confidence across the board" in prompt
    assert "track the actual strength of the evidence" in prompt


def test_v4_confidence_guidance_has_no_literal_dataset_phrases_or_email_ids():
    prompt = SYSTEM_PROMPT
    assert not re.search(r"\bE\d{3}\b", prompt)
    for banned in (
        "Top 5 Funds to Invest in 2024",
        "Open Demat Account",
        "Hot stocks for today",
        "Term Insurance starting at",
        "What if your offer letter arrives",
    ):
        assert banned.lower() not in prompt.lower()


def test_v3_phase2_category_boundaries_are_unchanged_by_v4():
    guide = _normalized(CATEGORY_GUIDE)
    assert "promotes a specific named investment product, fund, or curated set of" in guide
    assert "promotes a specific named insurance product or plan type with concrete" in guide
    assert "an event about the user's own activity on a social platform" in guide
    assert "any summary, balance, or statement of an owned account metric" in guide
    assert "an actual job offer, interview invite or recruiting conversation from the actual" in guide
    assert "an institutional sender alone" in guide
    assert "must not imply a protected category" in guide


def test_auto_delete_confidence_threshold_still_unchanged_v4():
    assert AUTO_DELETE_CONFIDENCE == 0.95


# ---------------------------------------------------------------------------
# V4 clarification: empty/sparse body does not by itself require low
# confidence -- confidence should track strength/specificity of available
# evidence (sender, subject, metadata), not body length.
# ---------------------------------------------------------------------------


def test_empty_body_does_not_by_itself_require_low_confidence():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "an empty or sparse body does not by itself require low confidence" in prompt
    assert "the classifier may assign high confidence" in prompt
    assert "even when the body is empty" in prompt


def test_confidence_reflects_evidence_strength_not_body_length():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "confidence should reflect the strength and specificity of the" in prompt
    assert "not the amount of body text" in prompt


def test_empty_body_examples_are_generic_and_cover_both_dispositions():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "a clear promotional sale subject from a retailer can be high-confidence marketing" in prompt
    assert "a clear course enrollment invitation can be high-confidence course_promotion" in prompt
    assert "a clear security-code/password-reset subject can be high-confidence security" in prompt
    assert "a vague bank notification with an empty body" in prompt
    assert "remains ambiguous and should" in prompt
    assert "remain low-confidence and route to review" in prompt


def test_empty_body_clarification_does_not_manufacture_evidence():
    prompt = _normalized(SYSTEM_PROMPT)
    assert "it does not manufacture evidence that isn't there" in prompt


def test_empty_body_clarification_has_no_literal_dataset_phrases_or_ids():
    prompt = SYSTEM_PROMPT
    assert not re.search(r"\bE\d{3}\b", prompt)
    for banned in (
        "Top 5 Funds to Invest in 2024",
        "Open Demat Account",
        "Hot stocks for today",
        "Term Insurance starting at",
        "What if your offer letter arrives",
    ):
        assert banned.lower() not in prompt.lower()
