"""Prompt construction for the email classifier."""

from app.schemas.email import Email, EmailCategory

CONTROLLED_CATEGORIES = [
    EmailCategory.COURSE_PROMOTION,
    EmailCategory.JOB_ALERT,
    EmailCategory.NEWSLETTER,
    EmailCategory.MARKETING,
    EmailCategory.INSURANCE_MARKETING,
    EmailCategory.INVESTMENT_MARKETING,
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
- investment_marketing: investment promotions, recommendations, offers and campaigns that
  entice or invite the user to invest. Being about investing does NOT make an email protected.

Protected / high-value:
- job_offer: actual job offer, interview invite or recruiting conversation directed at the user.
- employment_document: offer letters, contracts and HR paperwork.
- salary: payroll, compensation and payslips.
- investment_record: genuine owned investment records such as statements, SIP/transaction
  confirmations, dividend statements, or brokerage/retirement documents evidencing a holding.
  A generic invitation or recommendation to invest is investment_marketing, not a record.
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

Topic is not the same as disposition. Do not classify an email as protected merely because its
topic is investment, insurance, banking or employment. Determine whether it is a genuine
record/transaction/document or primarily promotional/marketing content.

Insurance and investment are intent-sensitive. Actual owned records are protected; generic
sales pitches and invitations to buy/invest are disposable candidates.

Allowed categories:
{CATEGORY_GUIDE}

Respond only through the emit_classification tool.
"""


def build_user_prompt(email: Email) -> str:
    attachments = ", ".join(email.attachment_names) if email.attachment_names else "none"
    body = email.body.strip() or "(empty body)"
    if len(body) > 4000:
        body = body[:4000] + "... [truncated]"
    return f"Email id: {email.id}\nSender: {email.sender}\nSubject: {email.subject}\nAttachments: {attachments}\nBody:\n{body}"


def valid_categories() -> list[str]:
    return [category.value for category in CONTROLLED_CATEGORIES]
