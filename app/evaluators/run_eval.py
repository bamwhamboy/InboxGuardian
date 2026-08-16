from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.classification.llm_classifier import LLMEmailClassifier
from app.classification.provider import default_llm_client
from app.evaluators.mock_provider import GroundTruthEchoLLMClient
from app.evaluators.dataset import load_eval_cases
from app.evaluators.runner import DEFAULT_RESULTS_DIR, run_evaluation, write_results


def build_arg_parser():
    p = argparse.ArgumentParser(description="Evaluate InboxGuardian classification and guardrails.")
    p.add_argument("--stub", action="store_true", help="Use deterministic ground-truth echo provider; no API calls.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases.")
    p.add_argument("--ids", nargs="+", help="Evaluate specific email IDs.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return p


def _build_classifier(args):
    if args.stub:
        cases = load_eval_cases(ids=args.ids)
        if args.limit is not None:
            cases = cases[:args.limit]
        mapping = {c.id: c.expected_category for c in cases}
        return LLMEmailClassifier(llm_client=GroundTruthEchoLLMClient(mapping))
    return LLMEmailClassifier(llm_client=default_llm_client())


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.stub:
        print("Running with --stub: deterministic ground-truth-echo provider, NOT a real LLM.")
    classifier = _build_classifier(args)
    run = run_evaluation(classifier=classifier, ids=args.ids, limit=args.limit)
    print(f"Evaluated {len(run.results)} case(s). Run id: {run.run_id}")
    print("\nAggregate metrics:")
    for key, value in run.metrics.items():
        print(f"  {key}: {value}")
    print("\nConfusion matrix (rows = expected category, columns = predicted category):")
    for expected, columns in run.confusion_matrix.items():
        print(f"  {expected}: " + ", ".join(f"{k}={v}" for k, v in columns.items()))
    print(f"\nFalse deletions (protected email proposed for delete) — {len(run.false_deletions)}:")
    print(json.dumps(run.false_deletions, indent=2))
    print(f"\nProtected emails sent to REVIEW instead of KEEP — {len(run.protected_sent_to_review)}:")
    print(json.dumps(run.protected_sent_to_review, indent=2))
    print(f"\nHigh-confidence errors (confidence >= {run.metrics['auto_delete_confidence_threshold']}) — {len(run.high_confidence_errors)}:")
    print(json.dumps(run.high_confidence_errors, indent=2))
    paths = write_results(run, output_dir=args.output_dir)
    print("\nResults written to:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
