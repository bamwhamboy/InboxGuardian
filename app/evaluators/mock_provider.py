from __future__ import annotations

from typing import Any


class GroundTruthEchoLLMClient:
    """Offline stub used only for deterministic evaluator tests."""

    def __init__(self, id_to_category: dict[str, str], confidence: float = 1.0):
        self.id_to_category = id_to_category
        self.confidence = confidence

    def classify_raw(self, system_prompt: str, user_prompt: str, category_values: list[str]) -> dict[str, Any]:
        email_id = next(line.split(": ", 1)[1] for line in user_prompt.splitlines() if line.startswith("Email id:"))
        return {
            "category": self.id_to_category[email_id],
            "confidence": self.confidence,
            "rationale": "Deterministic evaluation stub.",
        }
