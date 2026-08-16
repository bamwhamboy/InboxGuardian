"""Loads the frozen evaluation dataset. Never writes to it.

Reads two files as ground truth and never modifies either:

- data/eval/email_dataset.json — the frozen 100-email labelled dataset.
- data/eval/ground_truth_corrections.json — a small, explicit overlay of
  per-email corrections, applied here in memory. This is how a labelling
  mistake gets fixed without ever editing the frozen dataset file itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.email import Email

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "eval" / "email_dataset.json"
DEFAULT_CORRECTIONS_PATH = REPO_ROOT / "data" / "eval" / "ground_truth_corrections.json"

JUNK_LABELS = {"spam", "junk"}


def is_junk_label(ground_truth: str) -> bool:
    return ground_truth.strip().lower() in JUNK_LABELS


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    sender: str
    subject: str
    body: str = ""
    attachment_names: list[str] = Field(default_factory=list)
    expected_category: str
    expected_action: str
    ground_truth: str
    protected: bool
    correction_applied: bool = False
    correction_reason: Optional[str] = None

    def to_email(self) -> Email:
        return Email(
            id=self.id,
            sender=self.sender,
            subject=self.subject,
            body=self.body,
            attachment_names=self.attachment_names,
        )

    @property
    def expected_junk(self) -> bool:
        return is_junk_label(self.ground_truth)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_cases(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    corrections_path: Path = DEFAULT_CORRECTIONS_PATH,
    ids: Optional[list[str]] = None,
) -> list[EvalCase]:
    raw_dataset = _load_json(dataset_path)
    corrections: dict = _load_json(corrections_path) if corrections_path.exists() else {}

    cases: list[EvalCase] = []
    for item in raw_dataset:
        correction = corrections.get(item["id"])
        category = item["category"]
        expected_action = item["expected_action"]
        ground_truth = item["ground_truth"]
        protected = item["protected"]
        correction_reason = None
        correction_applied = False

        if correction:
            category = correction.get("category", category)
            expected_action = correction.get("expected_action", expected_action)
            ground_truth = correction.get("ground_truth", ground_truth)
            protected = correction.get("protected", protected)
            correction_reason = correction.get("reason")
            correction_applied = True

        cases.append(
            EvalCase(
                id=item["id"],
                sender=item["sender"],
                subject=item["subject"],
                body=item.get("body", ""),
                attachment_names=item.get("attachment_names", []),
                expected_category=category,
                expected_action=expected_action,
                ground_truth=ground_truth,
                protected=protected,
                correction_applied=correction_applied,
                correction_reason=correction_reason,
            )
        )

    if ids is not None:
        wanted = list(dict.fromkeys(ids))
        by_id = {c.id: c for c in cases}
        missing = [eid for eid in wanted if eid not in by_id]
        if missing:
            raise ValueError(f"Unknown email id(s) requested via --ids: {missing}")
        wanted_set = set(wanted)
        cases = [c for c in cases if c.id in wanted_set]

    return cases
