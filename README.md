# InboxGuardian

A risk-aware, human-in-the-loop AI agent for safe autonomous email management.

## Project status

Early development — Sprint 1: offline LLM classification and deterministic protection policy.

## Initial objective

Identify unwanted promotional email while protecting career, financial, investment, insurance, tax/legal, personal, work, transaction, and security-related email.

No autonomous Gmail deletion in the initial sprint.

## Sprint 1 / Step 1

The current implementation adds an LLM classifier that produces only `category`, `confidence`, and `rationale`. Protection and action decisions remain exclusively in the deterministic guardrail layer.

The classifier uses a controlled taxonomy for the current evaluation dataset and validates model output with strict Pydantic schemas. Unexpected fields are rejected and trigger the classifier retry path.

Insurance is intent-sensitive: actual owned-policy records such as premium receipts are protected, while generic insurance sales/renewal pitches are disposable candidates. Ambiguous cases are routed to human review by the downstream policy.

The model-reported confidence is a coarse signal, not a calibrated probability. `AUTO_DELETE_CONFIDENCE` defaults to 0.95 and is configurable via `INBOXGUARDIAN_AUTO_DELETE_CONFIDENCE`; future work will calibrate this threshold against the evaluation dataset.

Out of scope for this step: Gmail integration, deletion, autonomous action-taking, memory/state persistence, multi-agent orchestration, and production deployment.

### Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Classifier tests use a mocked `LLMClient` and never call a real LLM API or require an API key.
