# InboxGuardian

A risk-aware, human-in-the-loop AI agent for safe autonomous email management.

## Project status

Early development — Milestone 1: offline email classification and deterministic protection policy.

## Initial objective

Identify unwanted promotional email while protecting career, financial, investment, insurance, tax/legal, personal, work, transaction, and security-related email.

No autonomous Gmail deletion in the initial milestone.

## Sprint 1 — Step 1: LLM-based classification

`app/classification/` adds an LLM-based classifier that takes the existing `Email` schema and returns the existing `Classification` schema (`app/schemas/email.py`).

- `provider.py` — provider abstraction (`LLMClient` protocol) plus concrete providers `GeminiLLMClient` (default) and `AnthropicLLMClient`. Credentials are read from environment variables; nothing is hard-coded. Local `.env` files are loaded with `python-dotenv`. The model id is configurable via `INBOXGUARDIAN_LLM_MODEL` (see `.env.example`).
- `prompts.py` — system/user prompt construction, including a description of every `EmailCategory`.
- `llm_classifier.py` — `LLMEmailClassifier.classify(email) -> Classification`. The model is forced to respond via structured output, and the result is validated with Pydantic. Invalid output is rejected and retried once with feedback before raising `ClassificationError` — it is never passed downstream unvalidated.
- `pipeline.py` — thin composition of the classifier with the existing deterministic guardrail layer. The guardrail layer remains the sole authority on decision/risk/human-review; it does not execute anything.

Out of scope for this step: Gmail integration, deletion, autonomous action-taking, memory/state persistence, and multi-agent orchestration.

### Protection policy

Protection is intent-sensitive rather than topic-only. Actual owned insurance records such as premium receipts are protected, while generic insurance sales/renewal pitches are disposable candidates. Similarly, investment records are protected while investment marketing is disposable. Ambiguous cases are routed to human review by the downstream policy.

The LLM reports only `category`, `confidence`, and `rationale`. Protection status is derived exclusively by the deterministic guardrail policy; the model cannot assert or override protection.

`confidence` is a model-reported signal, not a calibrated probability. `AUTO_DELETE_CONFIDENCE` defaults to `0.95` and is configurable via `INBOXGUARDIAN_AUTO_DELETE_CONFIDENCE`; future work will calibrate this threshold against the evaluation dataset.

## Environment setup

Create a local `.env` file in the repository root. **Never commit the real `.env` file or API key.** The repository `.gitignore` excludes `.env`; `.env.example` is the safe template.

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_real_key_here
INBOXGUARDIAN_LLM_MODEL=gemini-3.5-flash
```

The default provider is Gemini. `INBOXGUARDIAN_LLM_PROVIDER=anthropic` can be used if the optional Anthropic provider is selected instead.

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All classifier and provider tests use mocks and never call a real LLM API or require a real API key.

## Sprint 1 — Step 2: offline evaluation runner

`app/evaluators/` adds an offline evaluation harness that runs the classifier + deterministic guardrail pipeline over the 100-email labelled dataset and scores it against frozen ground truth. It performs no Gmail actions and never deletes, archives, or moves messages.

- `dataset.py` — loads `data/eval/email_dataset.json` and applies the corrections overlay in memory. It never writes to the ground truth.
- `mock_provider.py` — deterministic providers used by tests and optional `--stub` runs.
- `metrics.py` — pure metric calculations with no I/O or LLM calls.
- `runner.py` — runs the classifier/guardrail pipeline and writes evaluation results locally.
- `run_eval.py` — CLI entry point.

### Running the evaluation

Install dependencies:

```bash
pip install -r requirements.txt
```

With the local `.env` configured, run a small real-LLM smoke test first:

```bash
python -m app.evaluators.run_eval --limit 5
```

This makes real Gemini calls using `gemini-3.5-flash`. It is intentionally separate from the full 100-email baseline.

After the smoke test is verified, run the complete frozen evaluation:

```bash
python -m app.evaluators.run_eval
```

Useful flags:

```bash
python -m app.evaluators.run_eval --limit 10
python -m app.evaluators.run_eval --model gemini-3.5-flash
python -m app.evaluators.run_eval --output-dir some/other/dir
python -m app.evaluators.run_eval --stub --limit 5
```

`--stub` makes no API calls and is only for deterministic pipeline testing; it is not a real model evaluation.

## Evaluation outputs

Each run writes to `data/eval/results/` locally. Generated results are ignored by Git. Outputs include per-email results and aggregate metrics.

Important metrics include:

- `category_accuracy`
- `junk_precision`, `junk_recall`, `junk_f1`
- `protected_preservation_rate` — critical safety metric
- `false_deletion_rate`
- `human_review_rate`
- `autonomous_delete_precision`
- `high_confidence_error_count`
- `classification_error_count`

No LLM-as-a-Judge is used in this step. The frozen labelled dataset and its correction overlay are the evaluation reference.

## Current evaluation dataset

The 100 synthetic emails intentionally contain both protected records and disposable marketing/newsletter content across investment, insurance, employment, financial, security, personal, transaction, course, social, travel, and other categories.

The dataset is currently frozen for the first real baseline experiment. Do not modify ground truth after seeing model results unless a genuine labelling error is independently identified.

## Out of scope for the current milestone

- Gmail integration
- Actual email deletion
- Autonomous external actions
- Human-in-the-loop UI
- Long-term memory
- Multi-agent/deep-agent orchestration
- Weave/production observability
- LLM-as-a-Judge
- Production deployment
