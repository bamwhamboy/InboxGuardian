"""Compose LLM classification with deterministic guardrails."""

from __future__ import annotations

from app.classification.llm_classifier import LLMEmailClassifier
from app.guardrails.policy import apply_policy
from app.schemas.email import Email, EmailDecision


def evaluate_email(
    email: Email, classifier: LLMEmailClassifier | None = None
) -> EmailDecision:
    active_classifier = classifier or LLMEmailClassifier()
    classification = active_classifier.classify(email)
    decision, risk, reasons, human_review_required = apply_policy(
        classification.category, classification.confidence
    )
    return EmailDecision(
        email_id=email.id,
        classification=classification,
        risk=risk,
        proposed_action=decision,
        guardrail_reasons=reasons,
        human_review_required=human_review_required,
        final_action=None,
    )
