from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EmailCategory(str, Enum):
    COURSE_PROMOTION = "course_promotion"
    MARKETING = "marketing"
    NEWSLETTER = "newsletter"
    JOB_OFFER = "job_offer"
    EMPLOYMENT_DOCUMENT = "employment_document"
    SALARY = "salary"
    INVESTMENT = "investment"
    INSURANCE = "insurance"
    TAX = "tax"
    LEGAL = "legal"
    PERSONAL = "personal"
    WORK = "work"
    TRANSACTION = "transaction"
    SECURITY = "security"
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
    category: EmailCategory
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    protected: bool


class EmailDecision(BaseModel):
    email_id: str
    classification: Classification
    risk: RiskLevel
    proposed_action: Decision
    guardrail_reasons: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    final_action: Optional[Decision] = None
