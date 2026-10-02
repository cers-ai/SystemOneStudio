"""Preference pair generation (需求方案.txt 8.2).

DPO needs (prompt, chosen, rejected) triples. The requirement gives two sources:

1. automatic: cold-start with SFT, sample the seed rows repeatedly, then keep
   correct judgements as `chosen` and wrong ones as `rejected`
2. manual: Playground A/B clicks

Only source 1 lives here; source 2 is a write path on ``preference_pairs``.

Two things are deliberate:

* Sampling is restricted to ``origin == 'seed'`` rows. Sampling a synthetic row
  and training on its verdict feeds synthesis artefacts back into the model.
* The `chosen` side is *not* the ground-truth label rendered from the dataset.
  It is a genuine model output that happened to be correct. Training on the
  template string instead would teach formatting, not judgement, and would make
  DPO a no-op next to SFT.
"""

from __future__ import annotations

from dataclasses import dataclass

from son_contracts import DataOrigin, Decision
from son_trainer.backends import TrainSample


@dataclass(frozen=True)
class PreferencePair:
    """One (prompt, chosen, rejected) triple."""

    prompt: str
    chosen: str
    rejected: str
    label: Decision
    origin: str


@dataclass
class PairBuildReport:
    pairs: list[PreferencePair]
    sampled: int
    skipped_non_seed: int
    prompts_without_a_valid_pair: int
    notes: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.pairs)


def parse_decision(output: str) -> Decision | None:
    """Read a decision out of a raw model output.

    Accepts JSON and bare text, because the point of this function is to
    classify whatever the model actually emitted, format compliance or not.
    """
    import json
    import re

    text = output.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            raw = payload.get("decision")
            if raw is not None:
                try:
                    return Decision(str(raw).strip().lower())
                except ValueError:
                    return None
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r"black|white|gray|黑|白|灰", text, flags=re.IGNORECASE)
    if match is None:
        return None
    token = match.group(0).lower()
    try:
        return Decision(token)
    except ValueError:
        mapping = {"黑": Decision.BLACK, "白": Decision.WHITE, "灰": Decision.GRAY}
        return mapping.get(token)


def build_preference_pairs(
    prompts: tuple[tuple[str, Decision, str], ...],
    samples: tuple[str, ...],
    *,
    n_per_prompt: int = 4,
) -> PairBuildReport:
    """Group sampled outputs per prompt into preference pairs.

    ``prompts`` holds (prompt, ground_truth_label, origin) triples. ``samples``
    holds ``n_per_prompt`` outputs per prompt, in prompt order.
    """
    if n_per_prompt <= 0:
        raise ValueError("n_per_prompt must be positive")

    seed_prompts: list[tuple[str, Decision, str]] = []
    skipped = 0
    for entry in prompts:
        if entry[2] == DataOrigin.SEED.value:
            seed_prompts.append(entry)
        else:
            skipped += 1

    pairs: list[PreferencePair] = []
    without_pair = 0

    # `samples` is laid out in the order of `prompts` as passed in, with
    # n_per_prompt entries each. Filtering that list first and then indexing by
    # the *filtered* position hands every prompt after a skipped row somebody
    # else's samples -- and since synthetic rows are always skipped and synthetic
    # rows always land in train, that mis-assignment was the normal case rather
    # than an edge case. Walking the original list and skipping without moving
    # the cursor keeps the offset correct.
    cursor = 0
    for prompt, truth, origin in prompts:
        block = samples[cursor : cursor + n_per_prompt]
        cursor += n_per_prompt

        if origin != DataOrigin.SEED.value:
            continue

        correct = [o for o in block if parse_decision(o) is truth]
        wrong = [o for o in block if parse_decision(o) not in (None, truth)]

        if not correct or not wrong:
            # Nothing to contrast. A pair built from two samples of the same
            # quality teaches noise.
            without_pair += 1
            continue

        pairs.append(
            PreferencePair(
                prompt=prompt,
                # Shortest correct output: same judgement, least token cost.
                chosen=min(correct, key=len),
                # Longest wrong output: the clearest contrast available.
                rejected=max(wrong, key=len),
                label=truth,
                origin=origin,
            )
        )

    unparsed = sum(1 for s in samples if parse_decision(s) is None)

    notes: list[str] = []
    if skipped:
        notes.append(f"跳过 {skipped} 条非种子样本（避免合成数据回流训练）")
    if without_pair:
        notes.append(f"{without_pair} 条样本未生成偏好对（全部判定一致或全部错误）")
    if unparsed:
        notes.append(f"{unparsed} 条采样结果无法解析出判定结果，未参与配对")

    return PairBuildReport(
        pairs=pairs,
        sampled=len(samples),
        skipped_non_seed=skipped,
        prompts_without_a_valid_pair=without_pair,
        notes=tuple(notes),
    )


def to_train_samples(pairs: list[PreferencePair]) -> list[TrainSample]:
    return [
        TrainSample(
            prompt=p.prompt, response="", origin=p.origin, chosen=p.chosen, rejected=p.rejected
        )
        for p in pairs
    ]
