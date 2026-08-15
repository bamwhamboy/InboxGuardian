"""Prompt construction for the Sprint 1 email classifier."""

from app.schemas.email import Email, EmailCategory

CONTROLLED_CATEGORIES = [
    EmailCategory.COURSE_PROMOTION,
    EmailCategory.JOB_ALERT,
    EmailCategory.NEWSLETTER,
    EmailCategory.MARKETING,
    EmailCategory.INSURANCE_MARKETING,
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
    EmailCategory.OTHER,
]

CATEGORY_GUIDE = """
Disposable / low-value:
- course_promotion: courses, cohorts, bootcamps and learning offers.
- marketing: general promotions, discounts and sales pitches.
- newsletter: recurring editorial/content digests.
- job_alert: automated job-board alerts, NOT a direct offer.
- insurance_marketing: insurance sales pitches, generic renewal reminders or cross-sells,
  including comparison sites. An insurer's renewal email without evidence that the user
  actually owns a policy is insurance_marketing.

Protected / high-value:
- job_offer: actual job offer, interview invite or recruiting conversation directed at the user.
- employment_document: offer letters, contracts and HR paperwork.
- salary: payroll, compensation and payslips.
- investment: brokerage/investment statements and confirmations.
- insurance: an actual owned-policy record such as a premium receipt, policy document or claim,
  with concrete evidence such as a policy number, premium amount or claim/reference number.
- tax: tax filings, tax documents and tax-authority correspondence.
- legal: contracts, notices and legal correspondence.
- personal: correspondence from people the user knows.
- work: substantive work-related correspondence.
- transaction: purchase/order confirmations, receipts and shipping notices.
- security: account security alerts, password resets, login alerts and 2FA codes.

Fallback:
- other: anything that does not clearly fit. When uncertainty could result in deletion of
  valuable information, use lower confidence and/or other rather than guessing.
""".strip()

SYSTEM_PROMPT = f"""You are an email triage classifier for InboxGuardian.

Classify one email into exactly one allowed category and return confidence and a concise rationale.
You do NOT decide protection, risk or deletion; a deterministic downstream policy does that.

Confidence is a coarse model-reported signal, NOT a calibrated probability. Do not inflate it.
Use high confidence only when evidence is clear. When uncertain, express uncertainty with lower
confidence and/or other. Do not guess a protected or disposable category at high confidence solely
to avoid uncertainty.

Insurance is especially important: if the email does not show evidence that the user actually
owns a policy, classify a sales/renewal pitch as insurance_marketing at low confidence when needed.
That low-confidence result is routed to human review by the policy layer.

Allowed categories:
{CATEGORY_GUIDE}

Respond only through the emit_classification tool.
"""


def build_user_prompt(email: Email) -> str:
    attachments = ", ".join(email.attachment_names) if email.attachment_names else "none"
    body = email.body.strip() or "(empty body)"
    if len(body) > 4000:
        body = body[:4000] + "... [truncated]"
    return (
        f"Email id: {email.id}\n"
        f"Sender: {email.sender}\n"
        f"Subject: {email.subject}\n"
        f"Attachments: {attachments}\n"
        f"Body:\n{body}"
    )


def valid_categories() -> list[str]:
    return [category.value for category in CONTROLLED_CATEGORIES]
