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
}

# User-specific disposable categories. These are safe deletion candidates only
# when confidence is high and no protected signal is present.
DISPOSABLE_CATEGORIES = {
    EmailCategory.COURSE_PROMOTION,
    EmailCategory.MARKETING,
    EmailCategory.NEWSLETTER,
    EmailCategory.JOB_ALERT,
    EmailCategory.INSURANCE_MARKETING,
    EmailCategory.SOCIAL,
}

AUTO_DELETE_CONFIDENCE = 0.95


def apply_policy(category: EmailCategory, confidence: float) -> tuple[Decision, RiskLevel, list[str], bool]:
    """Apply deterministic safety policy after model classification.

    The model never has authority to override protected categories.
    Ambiguous disposable cases are routed to human review rather than deleted.
    """
    if category in PROTECTED_CATEGORIES:
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

    # Unknown or unclassified content is never autonomously deleted.
    return (
        Decision.REVIEW,
        RiskLevel.MEDIUM,
        ["Category is not approved for autonomous deletion"],
        True,
    )
