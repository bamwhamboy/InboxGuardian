"""Regression tests for the evaluator robustness fix: a classification
failure produces confidence=None, and metrics calculations must not crash
on that -- while the failure itself must remain visible in results/metrics
and must never resolve to an autonomous DELETE.
"""

import pytest

from app.classification.llm_classifier import LLMEmailClassifier
from app.evaluators.dataset import load_eval_cases
from app.evaluators.metrics import compute_metrics, list_high_confidence_errors
from app.evaluators.mock_provider import GroundTruthEchoLLMClient, MalformedLLMClient
from app.evaluators.runner import EvalResult, evaluate_case, run_evaluation


def make_result(**overrides) -> EvalResult:
    defaults = dict(
        email_id="X",
        predicted_category="marketing",
        confidence=0.97,
        rationale="stub",
        proposed_action="delete",
        expected_category="marketing",
        expected_action="delete",
        ground_truth="spam",
        protected=False,
        human_review_required=False,
        correct_category=True,
        correct_action=True,
        error=False,
    )
    defaults.update(overrides)
    return EvalResult(**defaults)


# --- (a) confidence=None must not crash metric calculations -----------------


def test_compute_metrics_does_not_crash_when_confidence_is_none():
    results = [
        make_result(email_id="A", confidence=None, predicted_category=None, error=True,
                    proposed_action="review", correct_category=False, human_review_required=True),
        make_result(email_id="B"),
    ]
    metrics = compute_metrics(results, auto_delete_confidence=0.95)
    assert metrics["total_emails"] == 2
    assert metrics["classification_error_count"] == 1


def test_high_confidence_error_count_excludes_none_confidence_results():
    results = [
        make_result(email_id="A", confidence=None, predicted_category=None, error=True,
                    proposed_action="review", correct_category=False, human_review_required=True),
        make_result(email_id="B", confidence=0.99, correct_category=False, predicted_category="newsletter"),
        make_result(email_id="C", confidence=0.99, correct_category=True),
    ]
    metrics = compute_metrics(results, auto_delete_confidence=0.95)
    # Only B is a genuine high-confidence category error; A's None confidence
    # must be excluded rather than crashing or being coerced into a count.
    assert metrics["high_confidence_error_count"] == 1


def test_list_high_confidence_errors_does_not_crash_on_none_confidence():
    results = [
        make_result(email_id="A", confidence=None, predicted_category=None, error=True,
                    proposed_action="review", correct_category=False, human_review_required=True),
    ]
    # Must not raise TypeError comparing None >= float.
    flagged = list_high_confidence_errors(results, threshold=0.95)
    assert flagged == []


# --- (b) malformed real classification produces confidence=None safely -----


def test_evaluate_case_handles_malformed_classification():
    case = load_eval_cases(ids=["E001"])[0]
    classifier = LLMEmailClassifier(llm_client=MalformedLLMClient(), max_attempts=2)

    result = evaluate_case(case, classifier)

    assert result.error is True
    assert result.confidence is None
    assert result.predicted_category is None
    assert result.correct_category is False


# --- (c) one failure does not stop the rest of the evaluation --------------


class _SingleFailureClient:
    """Fails classification for exactly one target email id; classifies
    everything else correctly via ground-truth echo."""

    def __init__(self, id_to_category: dict[str, str], failing_id: str) -> None:
        self._echo = GroundTruthEchoLLMClient(id_to_category)
        self._failing_id = failing_id

    def classify_raw(self, system_prompt: str, user_prompt: str, category_values: list[str]):
        if f"Email id: {self._failing_id}" in user_prompt:
            return {"category": "not_a_real_category", "confidence": 0.9, "rationale": "malformed"}
        return self._echo.classify_raw(system_prompt, user_prompt, category_values)


def test_run_evaluation_continues_after_one_classification_failure():
    ids = ["E001", "E033", "E034", "E035", "E082"]
    cases = load_eval_cases(ids=ids)
    id_to_category = {c.id: c.expected_category for c in cases}
    classifier = LLMEmailClassifier(llm_client=_SingleFailureClient(id_to_category, failing_id="E035"), max_attempts=2)

    # Must not raise -- the rest of the batch is still evaluated.
    run = run_evaluation(classifier=classifier, ids=ids)

    assert len(run.results) == 5
    failing = next(r for r in run.results if r.email_id == "E035")
    others = [r for r in run.results if r.email_id != "E035"]

    assert failing.error is True
    assert failing.confidence is None
    assert all(r.error is False for r in others)
    assert all(r.correct_category for r in others)
    # Computing metrics over the mixed batch must not crash either.
    assert run.metrics["total_emails"] == 5
    assert run.metrics["classification_error_count"] == 1


# --- (d) a failed classification can never become an autonomous DELETE -----


def test_failed_classification_is_never_delete():
    case = load_eval_cases(ids=["E082"])[0]  # protected in ground truth
    classifier = LLMEmailClassifier(llm_client=MalformedLLMClient(), max_attempts=2)

    result = evaluate_case(case, classifier)

    assert result.proposed_action != "delete"
    assert result.proposed_action == "review"
    assert result.human_review_required is True


def test_failed_classification_never_counted_as_false_deletion():
    from app.evaluators.metrics import list_false_deletions

    results = [
        make_result(
            email_id="E082",
            confidence=None,
            predicted_category=None,
            error=True,
            proposed_action="review",
            correct_category=False,
            human_review_required=True,
            protected=True,
        )
    ]
    metrics = compute_metrics(results, auto_delete_confidence=0.95)
    assert metrics["false_deletion_count"] == 0
    assert metrics["protected_preservation_rate"] == pytest.approx(1.0)
    assert list_false_deletions(results) == []


# --- (e) explicit malformed/failed classification visibility ---------------


def test_classification_error_field_and_metric_reflect_the_failure():
    # Requirement 7: reuse the existing `error` field / classification_error_count
    # metric rather than inventing a new schema field.
    results = [
        make_result(email_id="A", confidence=None, predicted_category=None, error=True,
                    proposed_action="review", correct_category=False, human_review_required=True),
        make_result(email_id="B"),
        make_result(email_id="C"),
    ]
    metrics = compute_metrics(results, auto_delete_confidence=0.95)
    assert metrics["classification_error_count"] == 1
    assert metrics["total_emails"] == 3
