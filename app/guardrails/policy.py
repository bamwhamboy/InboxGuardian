import os

from app.schemas.email import Decision, EmailCategory, RiskLevel


# Hard-protection policy. These categories must never be autonomously deleted.
PROTECTED_CATEGORIES = {
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
    EmailCategory.BANKING,
    EmailCategory.PAYMENT,
    EmailCategory.TRAVEL,
    EmailCategory.ENTERTAINMENT,
    EmailCategory.E_COMMERCE,
}

DISPOSABLE_CATEGORIES = {
    EmailCategory.COURSE_PROMOTION,
    EmailCategory.MARKETING,
    EmailCategory.NEWSLETTER,
    EmailCategory.JOB_ALERT,
    EmailCategory.INSURANCE_MARKETING,
    EmailCategory.SOCIAL,
    EmailCategory.TELECOM,
    EmailCategory.FOOD_DELIVERY,
}

REVIEW_CATEGORIES = {
    EmailCategory.SERVICE_UPDATE,
    EmailCategory.EDUCATION,
    EmailCategory.OTHER,
}

# Configurable safety threshold, not a calibrated probability.
AUTO_DELETE_CONFIDENCE = float(
    os.environ.get("INBOXGUARDIAN_AUTO_DELETE_CONFIDENCE", "0.95")
)

# TODO(calibration): use data/eval/email_dataset.json and real model outputs
# to calibrate reported confidence before enabling autonomous deletion.


def is_protected_category(category: EmailCategory) -> bool:
    """Deterministic protection check; the LLM cannot override this."""
    return category in PROTECTED_CATEGORIES


def apply_policy(
    category: EmailCategory, confidence: float
) -> tuple[Decision, RiskLevel, list[str], bool]:
    """Apply deterministic safety policy after model classification."""
    if is_protected_category(category):
        return (
            Decision.KEEP,
            RiskLevel.CRITICAL,
            [f"Protected category: {category.value}"],
            False,
        )

    if category in DISPOSABLE_CATEGORIES and confidence >= AUTO_DELETE_CONFIDENCE:
        return Decision.DELETE, RiskLevel.LOW, [], False

    if category in DISPOSABLE_CATEGORIES:
        return (
            Decision.REVIEW,
            RiskLevel.MEDIUM,
            ["Disposable category but confidence is below auto-action threshold"],
            True,
        )

    if category in REVIEW_CATEGORIES:
        return (
            Decision.REVIEW,
            RiskLevel.MEDIUM,
            [f"Context-dependent category: {category.value}"],
            True,
        )

    return (
        Decision.REVIEW,
        RiskLevel.MEDIUM,
        ["Category is not approved for autonomous deletion"],
        True,
    )
