"""Synthetic sample generators (需求方案.txt 5.4).

Three modes, matching the three cards in the requirement's mockup:

* ``distribution_fit``     分布拟合合成 -- learn the seed distribution
* ``small_sample_derive``  小样本特征扩增 -- derive features from few seeds
* ``rule_injection``       规则注入合成 -- generate from expert rules

Implementation note: this module does not import SDV, CTGAN or TVAE. Those are
heavy optional extras (see ``[project.optional-dependencies] synth``) and the
MVP must be runnable without them, so the default backend is a Gaussian
copula over per-column marginals. That is a real, working tabular synthesizer,
just not a neural one -- swapping in SDV means implementing :class:`Generator`
and nothing else. See 技术方案.md 4.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from son_contracts import DataOrigin, Decision, SynthMethod


@dataclass(frozen=True)
class Constraint:
    """Field-level constraint pushed down from the scene definition.

    需求方案.txt 5.4 shows 金额范围 [0 - 1,000,000] and 年龄段 [18 - 70]. These
    are enforced numerically here, not in the prompt. A prompt constraint cannot
    be verified; a numeric clamp can.
    """

    column: str
    minimum: float | None = None
    maximum: float | None = None

    def clamp(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if self.minimum is not None:
            numeric = numeric.clip(lower=self.minimum)
        if self.maximum is not None:
            numeric = numeric.clip(upper=self.maximum)
        return numeric


@dataclass
class SynthRequest:
    """Everything a synthesis run needs."""

    target_rows: int
    method: SynthMethod = SynthMethod.DISTRIBUTION_FIT
    label_column: str = "label"
    #: Column holding the row-level origin stamp. The split guard and the data
    #: quality report both read it, so it is configurable rather than assumed.
    origin_column: str = "origin"
    feature_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    target_ratio: float | None = None
    augment_label: Decision | None = None
    constraints: tuple[Constraint, ...] = ()
    seed: int = 42

    def resolved_features(self, frame: pd.DataFrame) -> list[str]:
        if self.feature_columns:
            return list(self.feature_columns)
        return [c for c in frame.columns if c not in (self.label_column, self.origin_column)]


@dataclass
class SynthResult:
    """Generated rows plus the parameters that produced them."""

    frame: pd.DataFrame
    method: SynthMethod
    rows: int
    label_counts: dict[str, int]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def assert_synthetic(self, origin_column: str = "origin") -> None:
        """Every generated row must be stamped, or the split guard is bypassable."""
        if origin_column not in self.frame.columns:
            raise AssertionError(f"synthetic output must carry a {origin_column!r} column")
        if not (self.frame[origin_column] == DataOrigin.SYNTH.value).all():
            raise AssertionError("synthetic output contains rows not marked as synthetic")


def _complement_label(target: Decision) -> Decision:
    """The label that fills the remainder when augmenting ``target``.

    Augmenting white complements with black, augmenting black or gray with
    white. The old expression was ``WHITE if BLACK else BLACK``, so a gray
    augmentation produced black rows and white was unreachable.
    """
    return Decision.BLACK if target is Decision.WHITE else Decision.WHITE


class Generator(ABC):
    """Synthesis backend interface. Implement this to swap in SDV/CTGAN/TVAE."""

    @abstractmethod
    def generate(self, seed_frame: pd.DataFrame, request: SynthRequest) -> pd.DataFrame:
        """Return `request.target_rows` generated rows (without an origin column)."""

    @staticmethod
    def apply_constraints(
        rows: pd.DataFrame,
        constraints: tuple[Constraint, ...],
    ) -> pd.DataFrame:
        """Clamp numeric fields to their scene-defined ranges.

        Lives on the base class so every backend honours the constraints: a rule
        generator that ignored an amount ceiling would produce data the quality
        gate downstream then has to catch.
        """
        for constraint in constraints:
            if constraint.column in rows.columns:
                rows[constraint.column] = constraint.clamp(rows[constraint.column])
        return rows


class JointRowGenerator(Generator):
    """Conservative resampling: preserve features, label and reason together.

    Produces duplicates, not novel evidence. Risk reports must expose that fact.
    Without a validated business rule, perturbing a feature while copying its
    original explanation would invent supervision.
    """

    def generate(self, seed_frame: pd.DataFrame, request: SynthRequest) -> pd.DataFrame:
        pool = seed_frame.copy()
        for constraint in request.constraints:
            if constraint.column not in pool:
                raise ValueError(f"missing constrained column: {constraint.column}")
            values = pd.to_numeric(pool[constraint.column], errors="coerce")
            eligible = values.notna()
            if constraint.minimum is not None:
                eligible &= values >= constraint.minimum
            if constraint.maximum is not None:
                eligible &= values <= constraint.maximum
            pool = pool.loc[eligible]
        if pool.empty:
            raise ValueError("no seed rows satisfy the constraints")
        rows = pool.sample(n=request.target_rows, replace=True, random_state=request.seed)
        rows = rows.reset_index(drop=True)
        rows[request.origin_column] = DataOrigin.SYNTH.value
        return rows


class GaussianCopulaGenerator(Generator):
    """Dependency-free tabular synthesizer.

    Fits per-column marginals (a normal quantile for numeric columns, an
    empirical distribution for categorical ones), samples independently, then
    resamples whole rows until every field-level constraint holds.

    Independence is the known weakness: it reproduces marginals but not
    correlations. Good enough for the privacy-risk gate and the fidelity score
    the requirement asks for, and honest about what it is.
    """

    def __init__(self, *, max_resample_rounds: int = 40) -> None:
        self._max_resample_rounds = max_resample_rounds

    def generate(self, seed_frame: pd.DataFrame, request: SynthRequest) -> pd.DataFrame:
        features = request.resolved_features(seed_frame)
        if not features:
            raise ValueError("no feature columns to synthesize from")

        rng = np.random.default_rng(request.seed)
        rows = self._sample_rows(seed_frame, features, request, rng)

        rows = self.apply_constraints(rows, request.constraints)
        rows = self._apply_label_target(rows, request, seed_frame)

        rows[request.origin_column] = DataOrigin.SYNTH.value
        return rows

    def _sample_rows(
        self,
        seed_frame: pd.DataFrame,
        features: list[str],
        request: SynthRequest,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        n = request.target_rows
        data: dict[str, Any] = {}
        numeric_fits: dict[str, tuple[float, float]] = {}

        for column in features:
            if column in request.categorical_columns:
                pool = seed_frame[column].dropna()
                if pool.empty:
                    continue
                weights = pool.value_counts(normalize=True).to_dict()
                values = rng.choice(
                    np.array(list(weights), dtype=object),
                    size=n,
                    p=np.array(list(weights.values()), dtype=float),
                )
                data[column] = values
            else:
                series = pd.to_numeric(seed_frame[column], errors="coerce").dropna()
                if series.empty:
                    continue
                # Inverse-CDF sampling on the empirical marginal.
                quantiles = series.sort_values().to_numpy()
                picks = rng.uniform(0, 1, size=n)
                data[column] = np.quantile(quantiles, picks)
                numeric_fits[column] = (float(series.mean()), float(series.std() or 1.0))

        if not data:
            raise ValueError("no usable feature columns in the seed frame")
        return pd.DataFrame(data)

    def _apply_label_target(
        self,
        rows: pd.DataFrame,
        request: SynthRequest,
        seed_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign labels to hit the requested target.

        Labels are assigned after feature sampling because the label is a
        *target* distribution, not something drawn from the seed marginal: the
        point of the recommendation is to move the ratio away from the seed
        ratio.

        Three cases:

        * ``augment_label`` + ``target_ratio`` -> split rows at the ratio, with
          the complement drawn from the other decision
        * ``augment_label`` alone -> every row carries that label
        * neither -> follow the seed frame's own label distribution, so an
          unlabeled synthetic row is never produced
        """
        n = len(rows)
        target = request.augment_label

        if target is None:
            counts = seed_frame[request.label_column].value_counts(normalize=True)
            if counts.empty:
                raise ValueError(
                    f"seed frame has no '{request.label_column}' column to follow; "
                    "pass augment_label to generate targeted rows"
                )
            labels = list(counts.index)
            probs = counts.to_numpy(dtype=float)
            rng = np.random.default_rng(request.seed + 1)
            rows[request.label_column] = rng.choice(labels, size=n, p=probs)
            return rows

        if request.target_ratio is not None:
            if request.target_ratio <= 0:
                # 0:1 means "augment none of the target label", which is never the
                # intent. Rejected rather than reinterpreted.
                raise ValueError(
                    f"target_ratio 必须大于 0，收到 {request.target_ratio}；"
                    "比例按「目标标签 : 另一类」定义，0 表示不扩增目标标签"
                )
            # target : other = ratio : 1, so the complement takes 1/(ratio+1).
            other_rows = round(n / (request.target_ratio + 1))
            other = _complement_label(target)
            labels = [target.value] * (n - other_rows) + [other.value] * other_rows
        else:
            labels = [target.value] * n

        rng = np.random.default_rng(request.seed + 1)
        rows[request.label_column] = list(rng.permutation(labels))
        return rows


class FeatureDeriveGenerator(Generator):
    """小样本特征扩增: resample with jitter when there is too little data to fit.

    With a handful of seed rows there is no distribution worth fitting, so we
    pick a seed row and perturb its numeric features within their observed
    spread (or a small default when the spread is zero).
    """

    def __init__(self, *, default_jitter: float = 0.05) -> None:
        self._default_jitter = default_jitter

    def generate(self, seed_frame: pd.DataFrame, request: SynthRequest) -> pd.DataFrame:
        if seed_frame.empty:
            raise ValueError("cannot derive features from an empty seed frame")
        features = request.resolved_features(seed_frame)
        rng = np.random.default_rng(request.seed)

        chosen = rng.integers(0, len(seed_frame), size=request.target_rows)
        rows = seed_frame.iloc[chosen][features].reset_index(drop=True)

        for column in features:
            if column in request.categorical_columns:
                continue
            series = pd.to_numeric(rows[column], errors="coerce")
            seed_series = pd.to_numeric(seed_frame[column], errors="coerce")
            spread = float(seed_series.std() or 0.0)
            if spread == 0:
                spread = abs(float(series.mean()) or 1.0) * self._default_jitter

            # The multiplier is centred on 1, so its noise is a *relative* scale.
            # Passing an absolute standard deviation here produced multipliers
            # from -51x to +55x, i.e. negative and 55x-the-original amounts.
            rows[column] = series * (1 + rng.normal(0, self._default_jitter, size=len(rows)))

        rows = rows.round(6)
        rows = self.apply_constraints(rows, request.constraints)

        if request.augment_label is not None:
            rows[request.label_column] = request.augment_label.value
        elif request.label_column in seed_frame.columns:
            # Follow the seed's label distribution. Assigning the (None) augment
            # label here -- which is what this branch used to do -- produced a
            # full frame of unlabelled rows that still passed every downstream
            # check, because only `origin` was ever validated.
            counts = seed_frame[request.label_column].value_counts(normalize=True)
            if counts.empty:
                raise ValueError(
                    f"种子数据没有 {request.label_column!r} 列，无法确定合成样本的标签"
                )
            rows[request.label_column] = rng.choice(
                list(counts.index), size=len(rows), p=counts.to_numpy(dtype=float)
            )
        else:
            raise ValueError(
                f"种子数据没有 {request.label_column!r} 列，无法确定合成样本的标签；"
                "请指定 augment_label"
            )

        rows[request.origin_column] = DataOrigin.SYNTH.value
        return rows


class RuleInjectionGenerator(Generator):
    """规则注入合成: expert rules decide the label, features follow the rule.

    ``rules`` maps a feature column to (values_for_black, values_for_white).
    Rows are generated to satisfy the rule so the label is causally grounded
    rather than assigned at random.
    """

    def __init__(self, rules: dict[str, dict[str, Any]] | None = None) -> None:
        self.rules = rules or {}

    def generate(self, seed_frame: pd.DataFrame, request: SynthRequest) -> pd.DataFrame:
        if not self.rules:
            raise ValueError("rule injection needs at least one rule; none configured")

        rng = np.random.default_rng(request.seed)
        n = request.target_rows
        target = request.augment_label or Decision.BLACK

        rows = pd.DataFrame(index=range(n))
        for column, spec in self.rules.items():
            values = spec.get(target.value)
            if not values:
                # A rule with an empty list for this label is a misconfiguration,
                # not a reason to fall through to `default` and then index [0].
                values = spec.get("default")
            if not values:
                raise ValueError(f"规则 {column!r} 对 {target.value} 没有取值，且未配置 default")
            options = np.array(values, dtype=object)
            if len(options) > 1:
                rows[column] = rng.choice(options, size=n)
            else:
                rows[column] = [values[0]] * n

        if request.label_column:
            rows[request.label_column] = target.value

        rows = self.apply_constraints(rows, request.constraints)
        rows[request.origin_column] = DataOrigin.SYNTH.value
        return rows


_GENERATORS: dict[SynthMethod, type[Generator]] = {
    SynthMethod.DISTRIBUTION_FIT: GaussianCopulaGenerator,
    SynthMethod.SMALL_SAMPLE_DERIVE: FeatureDeriveGenerator,
    SynthMethod.RULE_INJECTION: RuleInjectionGenerator,
}


def get_generator(method: SynthMethod, **kwargs: Any) -> Generator:
    generator_cls = _GENERATORS.get(method)
    if generator_cls is None:
        raise KeyError(f"no generator registered for {method}")
    return generator_cls(**kwargs)


def synthesize(
    seed_frame: pd.DataFrame,
    request: SynthRequest,
    *,
    generator: Generator | None = None,
) -> SynthResult:
    """Run a synthesis pass and validate its output before handing it back."""
    if request.target_rows <= 0:
        raise ValueError("target_rows must be positive")

    backend = generator or get_generator(request.method)
    frame = backend.generate(seed_frame, request)

    if len(frame) != request.target_rows:
        raise AssertionError(
            f"generator produced {len(frame)} rows, expected {request.target_rows}"
        )

    label_col = request.label_column
    counts = (
        {str(k): int(v) for k, v in frame[label_col].value_counts().to_dict().items()}
        if label_col in frame.columns
        else {}
    )

    result = SynthResult(
        frame=frame,
        method=request.method,
        rows=len(frame),
        label_counts=counts,
        notes=(f"backend={type(backend).__name__}",),
    )
    result.assert_synthetic(request.origin_column)
    return result
