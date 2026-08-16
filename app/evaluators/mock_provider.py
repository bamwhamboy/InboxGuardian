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


class MalformedLLMClient:
    """Offline stub that always returns invalid structured output (an
    unrecognized category), so classification fails validation on every
    attempt. Used to test the evaluator's handling of real-world
    classification failures (e.g. a blocked/malformed Gemini response)
    without needing an API key or network access."""

    def __init__(self) -> None:
        self.calls = 0

    def classify_raw(self, system_prompt: str, user_prompt: str, category_values: list[str]) -> dict[str, Any]:
        self.calls += 1
        return {"category": "not_a_real_category", "confidence": 0.99, "rationale": "malformed"}
