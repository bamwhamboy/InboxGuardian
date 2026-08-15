"""LLM-based email classifier.

Sprint 1 / Step 1: classify an Email into a validated Classification. The
classifier does not decide keep/delete/review; deterministic guardrails do.
"""

from __future__ import annotations

from pydantic import ValidationError

from app.classification.prompts import SYSTEM_PROMPT, build_user_prompt, valid_categories
from app.classification.provider import LLMClient, default_llm_client
from app.schemas.email import Classification, Email

MAX_ATTEMPTS = 2


class ClassificationError(RuntimeError):
    """Raised when valid structured output cannot be obtained after retries."""


class LLMEmailClassifier:
    def __init__(
        self, llm_client: LLMClient | None = None, max_attempts: int = MAX_ATTEMPTS
    ) -> None:
        self.llm_client = llm_client or default_llm_client()
        self.max_attempts = max_attempts

    def classify(self, email: Email) -> Classification:
        system_prompt = SYSTEM_PROMPT
        user_prompt = build_user_prompt(email)
        categories = valid_categories()
        last_error: Exception | None = None

        for _ in range(self.max_attempts):
            try:
                raw = self.llm_client.classify_raw(
                    system_prompt, user_prompt, categories
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

            try:
                return Classification.model_validate(raw)
            except ValidationError as exc:
                last_error = exc
                user_prompt = (
                    build_user_prompt(email)
                    + "\n\nYour previous response was invalid: "
                    + str(exc)
                    + "\nRespond again, strictly following the emit_classification schema."
                )

        raise ClassificationError(
            f"Failed to obtain a valid classification for email {email.id!r} "
            f"after {self.max_attempts} attempt(s): {last_error}"
        ) from last_error
