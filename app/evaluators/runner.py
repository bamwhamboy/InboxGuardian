from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from app.classification.llm_classifier import LLMEmailClassifier
from app.classification.pipeline import evaluate_email
from app.evaluators.dataset import EvalCase, load_eval_cases
from app.evaluators.metrics import (
    compute_confusion_matrix,
    compute_metrics,
    list_false_deletions,
    list_high_confidence_errors,
    list_protected_sent_to_review,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "eval" / "results"


@dataclass
class EvalResult:
    email_id: str
    predicted_category: str | None
    confidence: float | None
    rationale: str | None
    proposed_action: str
    expected_category: str
    expected_action: str
    ground_truth: str
    protected: bool
    human_review_required: bool
    correct_category: bool
    correct_action: bool
    error: bool = False

    @property
    def expected_junk(self) -> bool:
        return self.ground_truth.lower() in {"spam", "junk"}

    def to_dict(self):
        return asdict(self)


@dataclass
class EvalRun:
    run_id: str
    results: list[EvalResult]
    metrics: dict
    confusion_matrix: dict
    false_deletions: list[dict]
    protected_sent_to_review: list[dict]
    high_confidence_errors: list[dict]


def evaluate_case(case: EvalCase, classifier: LLMEmailClassifier) -> EvalResult:
    try:
        decision = evaluate_email(case.to_email(), classifier=classifier)
        predicted = decision.classification.category.value
        confidence = decision.classification.confidence
        rationale = decision.classification.rationale
        action = decision.proposed_action.value
        return EvalResult(
            email_id=case.id,
            predicted_category=predicted,
            confidence=confidence,
            rationale=rationale,
            proposed_action=action,
            expected_category=case.expected_category,
            expected_action=case.expected_action,
            ground_truth=case.ground_truth,
            protected=case.protected,
            human_review_required=decision.human_review_required,
            correct_category=predicted == case.expected_category,
            correct_action=action == case.expected_action,
        )
    except Exception as exc:
        return EvalResult(
            email_id=case.id,
            predicted_category=None,
            confidence=None,
            rationale=f"Classification error: {exc}",
            proposed_action="review",
            expected_category=case.expected_category,
            expected_action=case.expected_action,
            ground_truth=case.ground_truth,
            protected=case.protected,
            human_review_required=True,
            correct_category=False,
            correct_action=case.expected_action == "review",
            error=True,
        )


def run_evaluation(classifier: LLMEmailClassifier, ids=None, limit=None) -> EvalRun:
    cases = load_eval_cases(ids=ids)
    if limit is not None:
        cases = cases[:limit]
    results = [evaluate_case(case, classifier) for case in cases]
    threshold = float(__import__("os").environ.get("INBOXGUARDIAN_AUTO_DELETE_CONFIDENCE", "0.95"))
    return EvalRun(
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        results=results,
        metrics=compute_metrics(results, threshold),
        confusion_matrix=compute_confusion_matrix(results),
        false_deletions=list_false_deletions(results),
        protected_sent_to_review=list_protected_sent_to_review(results),
        high_confidence_errors=list_high_confidence_errors(results, threshold),
    )


def write_results(run: EvalRun, output_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"run_{run.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    per_email = run_dir / "per_email_results.json"
    metrics = run_dir / "metrics.json"
    per_email.write_text(json.dumps([r.to_dict() for r in run.results], indent=2), encoding="utf-8")
    metrics.write_text(json.dumps({"metrics": run.metrics, "confusion_matrix": run.confusion_matrix, "false_deletions": run.false_deletions, "protected_sent_to_review": run.protected_sent_to_review, "high_confidence_errors": run.high_confidence_errors}, indent=2), encoding="utf-8")
    latest_per_email = output_dir / "latest_per_email_results.json"
    latest_metrics = output_dir / "latest_metrics.json"
    latest_per_email.write_text(per_email.read_text(encoding="utf-8"), encoding="utf-8")
    latest_metrics.write_text(metrics.read_text(encoding="utf-8"), encoding="utf-8")
    return {"per_email_results.json": per_email, "metrics.json": metrics, "latest_per_email_results.json": latest_per_email, "latest_metrics.json": latest_metrics}
