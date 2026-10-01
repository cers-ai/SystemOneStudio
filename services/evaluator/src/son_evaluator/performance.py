"""Performance metric aggregation (需求方案.txt 9.1, first priority).

Aggregation only. The numbers come from measurements taken by the inference
service; this module turns a list of samples into the percentiles and checks
them against targets.

That separation is deliberate and it is also the honest part: this machine has
no GPU and no llama.cpp, so **no number this module produces has been measured
here**. What *is* verifiable is the aggregation itself -- percentile math,
the target comparison, and the guard against reporting a latency number from a
run that never happened.

OPEN QUESTION Q1 (技术方案.md 1.1): 需求方案.txt requires P95 <= 100ms end to end
while also requiring a 200-character ``reason``, which needs roughly 150-250
decode tokens. That combination is not simultaneously satisfiable on a 3B model.
Until the target's meaning is decided, ``p95_ms`` is reported as-is and the
target check marks it 未定案 rather than passing or failing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from son_contracts import PERFORMANCE_TARGETS

#: Metrics the requirement treats as targets. Q1 puts two of them in dispute.
DISPUTED_TARGETS: frozenset[str] = frozenset({"p95_ms", "p99_ms"})


@dataclass(frozen=True)
class LatencySample:
    """One measured request."""

    total_ms: float
    ttft_ms: float | None = None


@dataclass
class PerformanceReport:
    """The first-priority block."""

    sample_count: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    ttft_p95_ms: float | None
    qps: float | None
    peak_memory_mb: float | None
    quant_level: str | None = None
    notes: tuple[str, ...] = ()
    targets: dict[str, dict[str, float | bool | None]] = field(default_factory=dict)

    def describe(self) -> str:
        if self.p50_ms is None:
            return "性能指标：未测量"
        return f"P50 {self.p50_ms:.0f}ms / P95 {self.p95_ms:.0f}ms / P99 {self.p99_ms:.0f}ms"


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile, matching numpy's default method.

    Written out rather than pulled from numpy so the math is visible: this is
    the number a customer-facing latency claim rests on.
    """
    if not values:
        return None
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def check_target(name: str, value: float | None) -> dict[str, float | bool | None]:
    """Compare one metric against its target.

    Returns ``passed=None`` for the metrics whose target is under dispute (Q1) or
    unmeasured. A caller that needs a strict bool should assert on ``passed is
    None`` first -- that is the point: an unsettled target must not silently
    read as a pass.
    """
    target = PERFORMANCE_TARGETS.get(name)
    if value is None or target is None:
        return {"value": value, "target": target, "passed": None}

    # Lower is better for latency and memory, higher is better for throughput.
    lower_is_better = name in {"p50_ms", "p95_ms", "p99_ms", "ttft_ms", "peak_memory_mb"}
    passed = value <= target if lower_is_better else value >= target
    return {"value": round(value, 3), "target": target, "passed": passed}


def build_performance_report(
    samples: list[LatencySample],
    *,
    quant_level: str | None = None,
    concurrent: int = 1,
    wall_clock_s: float | None = None,
    peak_memory_mb: float | None = None,
) -> PerformanceReport:
    """Aggregate latency samples into the first-priority block.

    ``wall_clock_s`` is required for a QPS figure: deriving throughput from
    ``len(samples) / sum(latency)`` silently reports single-stream speed as if
    it were concurrent throughput, which is a different and more flattering
    number.
    """
    notes: list[str] = []

    if not samples:
        return PerformanceReport(
            sample_count=0,
            p50_ms=None,
            p95_ms=None,
            p99_ms=None,
            ttft_p95_ms=None,
            qps=None,
            peak_memory_mb=peak_memory_mb,
            quant_level=quant_level,
            notes=("尚无实测样本，性能指标未测量",),
        )

    totals = [s.total_ms for s in samples]
    ttfts = [s.ttft_ms for s in samples if s.ttft_ms is not None]

    qps: float | None = None
    if wall_clock_s and wall_clock_s > 0:
        qps = len(samples) / wall_clock_s
    else:
        notes.append("未提供压测时长，吞吐量未计算")

    if peak_memory_mb is None:
        notes.append("未提供进程内存采样，峰值内存未计算")

    if quant_level:
        notes.append(f"量化档位 {quant_level}；评测仅在同档位下可比")

    report = PerformanceReport(
        sample_count=len(samples),
        p50_ms=_round(percentile(totals, 50)),
        p95_ms=_round(percentile(totals, 95)),
        p99_ms=_round(percentile(totals, 99)),
        ttft_p95_ms=_round(percentile(ttfts, 95)) if ttfts else None,
        qps=_round(qps),
        peak_memory_mb=_round(peak_memory_mb),
        quant_level=quant_level,
    )

    targets = {
        "p50_ms": check_target("p50_ms", report.p50_ms),
        "p95_ms": check_target("p95_ms", report.p95_ms),
        "p99_ms": check_target("p99_ms", report.p99_ms),
        "ttft_ms": check_target("ttft_ms", report.ttft_p95_ms),
        "qps": check_target("qps", report.qps),
        "peak_memory_mb": check_target("peak_memory_mb", report.peak_memory_mb),
    }

    disputed = sorted(DISPUTED_TARGETS.intersection(targets))
    if disputed:
        # Marked rather than silently passed: Q1 is unresolved.
        for name in disputed:
            entry = targets[name]
            entry["passed"] = None
        notes.append(
            f"{', '.join(disputed)} 的目标口径未定案（技术方案 Q1：与 reason 200 字符冲突），"
            "不作达标判断"
        )

    report.targets = targets
    report.notes = tuple(notes)
    return report


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)
