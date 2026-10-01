"""JEV prompt formatting and constrained-decoding grammar (需求方案.txt 6.1).

Two jobs:

* render the JEV input template with a scene sample
* derive a GBNF grammar from the JEV output schema

The grammar matters more than it looks. Validating a JSON string after the fact
and repairing it is how format-compliance ends up at 97% instead of the required
99%+, and it silently distorts ``reason``. Locking the structure during decoding
makes a compliant output the default rather than a cleanup job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from son_contracts import (
    JEV_INPUT_TEMPLATE,
    JEV_OUTPUT_SCHEMA,
    JEV_SPEC_VERSION,
    REASON_MAX_LENGTH,
    JevFlags,
)

#: Plain instruction prompt used when jev_format_compat is disabled. It exists
#: so one adapter serves both switch states; see BaseModelAdapter.
PLAIN_PROMPT_TEMPLATE = (
    "<|im_start|>system\n你是一个专业的决策模型，负责对输入样本进行判定。<|im_end|>\n"
    "<|im_start|>user\n{features}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def render_features(sample: dict[str, object]) -> str:
    """Render a sample for the ``{features}`` slot, with stable key order.

    Stable order matters: training and evaluation must see byte-identical
    prompts or the metrics are not comparable.
    """
    return "\n".join(f"{key}: {sample[key]}" for key in sorted(sample))


def format_jev_prompt(sample: dict[str, object], flags: JevFlags | None = None) -> str:
    """Render the JEV prompt, or the plain one when format compat is off.

    Substitution is an explicit replace, not ``str.format``: the JEV template
    from 需求方案.txt 6.1 contains a literal JSON example with braces, and
    ``format()`` treats ``{"decision": ...}`` as a field reference and raises.
    """
    active = flags if flags is not None else JevFlags()
    features = render_features(sample)
    if not active.jev_format_compat:
        return PLAIN_PROMPT_TEMPLATE.replace("{features}", features)
    return JEV_INPUT_TEMPLATE.replace("{features}", features)


@dataclass(frozen=True)
class GrammarSpec:
    """A GBNF grammar plus the reason cap it encodes."""

    grammar: str
    reason_max_length: int
    schema_version: str = JEV_SPEC_VERSION


def _number_rule() -> str:
    """Definition of the ``number`` rule.

    The range from the schema (0..1) is enforced after decoding in
    ``validate_jev_payload``; a GBNF grammar cannot express a numeric bound, and
    pretending otherwise here would make the grammar look stricter than it is.
    """
    return 'number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)?'


def _string_rule(cap: int) -> str:
    """Definition of the ``string`` rule, with a hard character cap.

    The cap is the enforcement that matters: an over-long reason is the most
    likely way to break format compliance (需求方案.txt 6.1 sets maxLength 200).
    """
    return f'string ::= "\\"" ([^"\\\\] | "\\\\" .) {{0,{cap}}}"\\"'


def build_grammar() -> GrammarSpec:
    """Derive a GBNF grammar from the JEV output schema.

    The structure is read from the schema rather than hand-written, so a change
    to the schema propagates instead of silently diverging.

    Field rules *reference* ``decision`` / ``number`` / ``string`` by name; the
    definitions appear once at the end. Inlining a definition instead produces a
    rule that looks plausible and is not parseable.
    """
    decisions = JEV_OUTPUT_SCHEMA["properties"]["decision"]["enum"]
    reason_cap = int(JEV_OUTPUT_SCHEMA["properties"]["reason"]["maxLength"])

    grammar = "\n".join(
        [
            'root ::= "{" ws decision_field "," ws score_field "," ws confidence_field '
            '"," ws reason_field "}" ws',
            'decision ::= "' + '" | "'.join(decisions) + '"',
            'decision_field ::= "\\"decision\\"" ws ":" ws decision',
            'score_field ::= "\\"score\\"" ws ":" ws number',
            'confidence_field ::= "\\"confidence\\"" ws ":" ws number',
            'reason_field ::= "\\"reason\\"" ws ":" ws string',
            "ws ::= [ \\t\\n]*",
            _number_rule(),
            _string_rule(reason_cap),
        ]
    )
    return GrammarSpec(grammar=grammar, reason_max_length=reason_cap)


def parse_completion(raw: str) -> dict[str, object]:
    """Extract the JSON object a completion contains.

    Models wrap JSON in prose or fences even with a grammar in place, so this
    finds the outermost braces rather than trusting the whole string. It does
    *not* repair the contents: a malformed value has to surface as a compliance
    failure, otherwise the format-compliance metric measures this function
    rather than the model.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Models wrap JSON in fences even with a grammar in place.
        text = text.strip("`")
        if text.startswith("json"):
            text = text[len("json") :]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in completion: {raw[:120]!r}")

    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("completion payload is not a JSON object")
    return payload


def build_grammar_prompt_suffix() -> str:
    """Instruction appended to the prompt when grammar-constrained decoding is on."""
    return (
        "\n\n只输出一个 JSON 对象，字段顺序为 decision、score、confidence、reason，"
        f"reason 不超过 {REASON_MAX_LENGTH} 字符。"
    )
