from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.evaluators.runner import EvalResult


def _safe_div(n: float, d: float):
    return n / d if d else None


def compute_metrics(results: list["EvalResult"], auto_delete_confidence: float) -> dict:
    total = len(results)
    category_correct = sum(r.correct_category for r in results)
    action_correct = sum(r.correct_action for r in results)
    tp = sum(r.expected_junk and r.proposed_action == "delete" for r in results)
    fp = sum((not r.expected_junk) and r.proposed_action == "delete" for r in results)
    fn = sum(r.expected_junk and r.proposed_action != "delete" for r in results)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    protected = [r for r in results if r.protected]
    false_deletions = sum(r.proposed_action == "delete" for r in protected)
    reviews = sum(r.human_review_required for r in results)
    auto_deletes = [r for r in results if r.proposed_action == "delete" and not r.human_review_required]
    auto_delete_precision = _safe_div(sum(r.expected_junk for r in auto_deletes), len(auto_deletes))
    return {
        "total_emails": total,
        "category_accuracy": _safe_div(category_correct, total),
        "action_accuracy": _safe_div(action_correct, total),
        "junk_precision": precision,
        "junk_recall": recall,
        "junk_f1": f1,
        "protected_preservation_rate": _safe_div(len(protected) - false_deletions, len(protected)) if protected else 1.0,
        "protected_total": len(protected),
        "false_deletion_count": false_deletions,
        "false_deletion_rate": _safe_div(false_deletions, total) if total else 0.0,
        "review_count": reviews,
        "review_rate": _safe_div(reviews, total) if total else 0.0,
        "autonomous_delete_count": len(auto_deletes),
        "autonomous_delete_precision": auto_delete_precision,
        "high_confidence_error_count": sum(r.confidence >= auto_delete_confidence and not r.correct_category for r in results),
        "classification_error_count": sum(r.error for r in results),
        "auto_delete_confidence_threshold": auto_delete_confidence,
    }


def compute_confusion_matrix(results: list["EvalResult"]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for r in results:
        matrix.setdefault(r.expected_category, {})[r.predicted_category or "ERROR"] = matrix.setdefault(r.expected_category, {}).get(r.predicted_category or "ERROR", 0) + 1
    return matrix


def list_false_deletions(results):
    return [r.to_dict() for r in results if r.protected and r.proposed_action == "delete"]


def list_protected_sent_to_review(results):
    return [r.to_dict() for r in results if r.protected and r.proposed_action == "review"]


def list_high_confidence_errors(results, threshold: float):
    return [r.to_dict() for r in results if r.confidence >= threshold and not r.correct_category]
