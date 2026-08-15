"""Provider abstraction for the Sprint 1 LLM classifier."""

from __future__ import annotations

import os
from typing import Any, Protocol

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
MODEL_ENV_VAR = "INBOXGUARDIAN_LLM_MODEL"
DEFAULT_MODEL = "claude-sonnet-5"
CLASSIFICATION_TOOL_NAME = "emit_classification"


def _classification_tool_schema(category_values: list[str]) -> dict[str, Any]:
    return {
        "name": CLASSIFICATION_TOOL_NAME,
        "description": "Emit the structured classification result for one email.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": category_values},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale": {"type": "string"},
            },
            "required": ["category", "confidence", "rationale"],
            "additionalProperties": False,
        },
    }


class LLMClient(Protocol):
    def classify_raw(
        self, system_prompt: str, user_prompt: str, category_values: list[str]
    ) -> dict[str, Any]: ...


class LLMConfigurationError(RuntimeError):
    pass


class AnthropicLLMClient:
    def __init__(self, model: str | None = None, max_tokens: int = 512) -> None:
        self.model = model or os.environ.get(MODEL_ENV_VAR, DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            raise LLMConfigurationError(f"Missing {API_KEY_ENV_VAR} environment variable.")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMConfigurationError(
                "Install the 'anthropic' package to use AnthropicLLMClient."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def classify_raw(
        self, system_prompt: str, user_prompt: str, category_values: list[str]
    ) -> dict[str, Any]:
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            tools=[_classification_tool_schema(category_values)],
            tool_choice={"type": "tool", "name": CLASSIFICATION_TOOL_NAME},
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == CLASSIFICATION_TOOL_NAME:
                return dict(block.input)
        raise LLMConfigurationError("Model response did not include the expected tool call.")


def default_llm_client() -> LLMClient:
    return AnthropicLLMClient()
