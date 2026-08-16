from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EmailCategory(str, Enum):
    # Disposable / low-value categories
    COURSE_PROMOTION = "course_promotion"
    MARKETING = "marketing"
    NEWSLETTER = "newsletter"
    JOB_ALERT = "job_alert"
    SOCIAL = "social"
    INSURANCE_MARKETING = "insurance_marketing"
    INVESTMENT_MARKETING = "investment_marketing"
    EDUCATION = "education"
    TELECOM = "telecom"
    FOOD_DELIVERY = "food_delivery"

    # Potentially important / context-dependent categories
    SERVICE_UPDATE = "service_update"

    # Protected / high-value categories
    JOB_OFFER = "job_offer"
    EMPLOYMENT_DOCUMENT = "employment_document"
    SALARY = "salary"
    INVESTMENT_RECORD = "investment_record"
    # Legacy alias retained for compatibility.
    INVESTMENT = "investment_record"
    INSURANCE = "insurance"
    TAX = "tax"
    LEGAL = "legal"
    PERSONAL = "personal"
    WORK = "work"
    TRANSACTION = "transaction"
    SECURITY = "security"
    BANKING = "banking"
    PAYMENT = "payment"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    E_COMMERCE = "e_commerce"
    # Certificates, certifications, course-completion records and similar
    # evidence of completed learning/qualification. Distinct from
    # course_promotion (an invitation to enroll) and employment_document
    # (offer letters/HR paperwork, not general certificates).
    CREDENTIAL_RECORD = "credential_record"

    OTHER = "other"


class Decision(str, Enum):
    KEEP = "keep"
    REVIEW = "review"
    DELETE = "delete"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str = ""
    attachment_names: list[str] = Field(default_factory=list)


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: EmailCategory
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class EmailDecision(BaseModel):
    email_id: str
    classification: Classification
    risk: RiskLevel
    proposed_action: Decision
    guardrail_reasons: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    final_action: Optional[Decision] = None
