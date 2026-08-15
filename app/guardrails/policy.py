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

DISPOSABLE_CATEGORIES = {
    EmailCategory.COURSE_PROMOTION,
    EmailCategory.MARKETING,
    EmailCategory.NEWSLETTER,
}


def apply_policy(category: EmailCategory, confidence: float) -> tuple[Decision, RiskLevel, list[str], bool]:
    """Apply deterministic safety policy after model classification.

    The model never has authority to override protected categories.
    """
    if category in PROTECTED_CATEGORIES:
        return (
            Decision.KEEP,
            RiskLevel.CRITICAL,
            [f"Protected category: {category.value}"],
            False,
        )

    if category in DISPOSABLE_CATEGORIES and confidence >= 0.95:
        return Decision.DELETE, RiskLevel.LOW, [], False

    if category in DISPOSABLE_CATEGORIES:
        return Decision.REVIEW, RiskLevel.MEDIUM, ["Disposable category but confidence is below auto-action threshold"], True

    return Decision.KEEP, RiskLevel.MEDIUM, ["Category is not approved for autonomous deletion"], False
