"""Model evaluation, deployment and prediction (改造开发方案.md 17 / 20 / 21).

The important constraint here is that **evaluation runs the model**. The old
endpoint accepted caller-supplied ``y_true`` / ``y_pred``, which made a perfect
report reachable by typing the right numbers in. This module takes a model
version plus the real ``test.csv`` and produces predictions itself.

Metrics that were not measured are ``None``. Format compliance in particular is
computed from what the model actually emitted, not from the fact that predictions
were supplied at all.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from son_evaluator import (
    EffectReport,
    LatencySample,
    build_effect_report,
    build_performance_report,
    percentile,
)

from son_contracts import Decision, PredictRequest, PredictResponse


class InferenceUnavailable(RuntimeError):
    """No engine could be started for this model."""


class DeploymentNotReady(RuntimeError):
    """The deployment is not in a state that can serve requests."""


@dataclass
class PredictionRecord:
    """One evaluated row, written to predictions.jsonl."""

    account: str
    y_true: str
    y_pred: str
    correct: bool
    score: float | None = None
    confidence: float | None = None
    reason: str = ""
    latency_ms: float | None = None
    schema_ok: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "correct": self.correct,
            "score": self.score,
            "confidence": self.confidence,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "schema_ok": self.schema_ok,
            "error": self.error,
        }


@dataclass
class EvaluationOutcome:
    """Everything a run's evaluation produced."""

    predictions_path: Path
    error_cases_path: Path
    report_path: Path
    effect: dict[str, Any]
    performance: dict[str, Any]
    top_factors: list[tuple[str, float]]
    notes: list[str] = field(default_factory=list)


def read_test_rows(path: Path) -> list[dict[str, str]]:
    """Read the test CSV. This is the only input the evaluator takes."""
    if not path.exists():
        raise FileNotFoundError(f"测试集不存在：{path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run_evaluation(
    test_csv: Path,
    output_dir: Path,
    engine: Any,
    *,
    limit: int | None = None,
    skip_label: str = "label",
) -> EvaluationOutcome:
    """Predict every test row, then score the predictions.

    ``engine`` is anything with ``predict(request) -> Prediction``. That keeps the
    evaluator independent of whether the model is served by llama.cpp, loaded
    locally, or stubbed in a test -- while the product path always passes the
    real one.
    """
    rows = read_test_rows(test_csv)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("测试集为空，无法评测")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"

    records: list[PredictionRecord] = []
    samples: list[LatencySample] = []

    for row in rows:
        account = row.get("account") or row.get("id") or ""
        features = {
            k: v for k, v in row.items() if k not in (skip_label, "origin", "account", "id")
        }
        started = time.perf_counter()
        try:
            prediction = engine.predict(PredictRequest(**features))
            response = prediction.response
            elapsed = (time.perf_counter() - started) * 1000
            record = PredictionRecord(
                account=account,
                y_true=str(row.get(skip_label, "")).strip().lower(),
                y_pred=response.decision.value,
                correct=response.decision.value == str(row.get(skip_label, "")).strip().lower(),
                score=response.score,
                confidence=response.confidence,
                reason=response.reason,
                latency_ms=elapsed,
            )
            samples.append(LatencySample(total_ms=elapsed, ttft_ms=prediction.latency.ttft_ms))
        except Exception as exc:
            record = PredictionRecord(
                account=account,
                y_true=str(row.get(skip_label, "")).strip().lower(),
                y_pred="",
                correct=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                schema_ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        records.append(record)

    with predictions_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    with errors_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.error or not record.correct:
                handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    # Score only the rows the model actually answered. A row that failed to
    # produce a parseable answer is a format-compliance failure, not a wrong
    # prediction, and folding it into y_pred would misreport both.
    scored = [r for r in records if r.schema_ok]
    notes: list[str] = []
    failed = len(records) - len(scored)
    if failed:
        notes.append(f"{failed} 行未能生成合规输出，已单独计入格式合规率")

    y_true = tuple(_as_decision(r.y_true) for r in scored)
    y_pred = tuple(_as_decision(r.y_pred) for r in scored)
    scores = tuple(r.score for r in scored if r.score is not None)

    effect: EffectReport = build_effect_report(
        y_true,
        y_pred,
        scores=scores if len(scores) == len(scored) else None,
        format_compliant=len(scored),
        format_total=len(records),
    )

    performance = build_performance_report(samples) if samples else build_performance_report([])

    report: dict[str, Any] = {
        "effect": {
            "accuracy": effect.accuracy,
            "macro_f1": effect.macro_f1,
            "recall": effect.recall,
            "false_kill_rate": effect.false_kill_rate,
            "auc_roc": effect.auc_roc,
            "format_compliance": effect.format_compliance,
            "per_label": {k: v.as_dict() for k, v in effect.per_label.items()},
            "targets": effect.target_check(),
        },
        "performance": {
            "sample_count": performance.sample_count,
            "p50_ms": performance.p50_ms,
            "p95_ms": performance.p95_ms,
            "p99_ms": performance.p99_ms,
            "ttft_p95_ms": performance.ttft_p95_ms,
            "targets": performance.targets,
        },
        "rows": len(records),
        "notes": [*effect.notes, *performance.notes, *notes],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return EvaluationOutcome(
        predictions_path=predictions_path,
        error_cases_path=errors_path,
        report_path=report_path,
        effect=report["effect"],
        performance=report["performance"],
        top_factors=[],
        notes=report["notes"],
    )


def _as_decision(value: str) -> Decision:
    try:
        return Decision(value.strip().lower())
    except ValueError:
        # An unrecognised ground truth is itself a finding; black keeps the row
        # scorable and the confusion matrix makes the damage visible.
        return Decision.BLACK


def summarize_latency(samples: list[float]) -> dict[str, float | None]:
    """Latency summary over real samples. Empty input yields all-None."""
    if not samples:
        return {"p50": None, "p95": None, "p99": None, "min": None, "max": None, "n": 0}
    return {
        "p50": percentile(list(samples), 50),
        "p95": percentile(list(samples), 95),
        "p99": percentile(list(samples), 99),
        "min": min(samples),
        "max": max(samples),
        "n": len(samples),
    }


def build_predict_response(
    raw: dict[str, Any], *, latency_ms: float | None = None
) -> PredictResponse:
    """Coerce a raw engine payload into the response shape.

    ``jev_compatible`` is carried through rather than assumed: with format compat
    off the caller is told they are not holding a JEV-shaped record.
    """
    return PredictResponse(
        decision=Decision(str(raw["decision"]).strip().lower()),
        score=float(raw["score"]),
        confidence=float(raw["confidence"]),
        reason=str(raw["reason"]),
        jev_compatible=bool(raw.get("jev_compatible", True)),
    )
