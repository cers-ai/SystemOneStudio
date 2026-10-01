"""JEV adapter tests.

Grammar structure, prompt formatting, the two independent switches, and the
refusal to claim an unverifiable compatibility level (技术方案.md Q3).
"""

from __future__ import annotations

import pytest
from son_jev import (
    PLAIN_PROMPT_TEMPLATE,
    assert_level_claimable,
    assert_training_methods_allowed,
    build_grammar,
    build_grammar_prompt_suffix,
    format_jev_prompt,
    metric_basis,
    parse_completion,
    render_features,
    training_method_whitelist,
)

from son_contracts import JEV_INPUT_TEMPLATE, JevCompatLevel, JevFlags


class TestGrammarStructure:
    """A grammar that only looks right is worse than none: it fails at runtime."""

    def _rules(self) -> dict[str, str]:
        rules: dict[str, str] = {}
        for line in build_grammar().grammar.strip().splitlines():
            name, _, body = line.partition("::=")
            rules[name.strip()] = body.strip()
        return rules

    def test_every_referenced_rule_is_defined(self) -> None:
        rules = self._rules()
        for name, body in rules.items():
            body = body.strip()
            for token in body.split():
                if token.isidentifier() and not token.startswith('"'):
                    assert token in rules, f"{name} references undefined rule {token}"

    def test_root_requires_all_four_fields(self) -> None:
        root = self._rules()["root"]
        for field in ("decision", "score", "confidence", "reason"):
            assert field in root

    def test_field_order_is_fixed(self) -> None:
        """Order is pinned so the emitted JSON matches the schema's reading order."""
        root = self._rules()["root"]
        assert root.index("decision_field") < root.index("score_field")
        assert root.index("score_field") < root.index("confidence_field")
        assert root.index("confidence_field") < root.index("reason_field")

    def test_no_additional_properties(self) -> None:
        assert "additional" not in self._rules()["root"]

    def test_reason_cap_is_two_hundred(self) -> None:
        assert "{0,200}" in self._rules()["string"]
        assert build_grammar().reason_max_length == 200

    def test_decision_rule_lists_exactly_three_values(self) -> None:
        body = self._rules()["decision"]
        assert body.count('"') == 6  # three values, two quotes each
        assert all(f'"{d}"' in body for d in ("black", "white", "gray"))

    def test_no_definition_is_inlined_into_a_reference(self) -> None:
        """Inlining a definition produces a plausible-looking unparseable rule."""
        for name, body in self._rules().items():
            if name == "root":
                continue
            assert "::=" not in body, f"{name} inlines a rule definition"

    def test_grammar_ends_with_a_complete_rule(self) -> None:
        assert build_grammar().grammar.endswith('"')


class TestPromptFormatting:
    def test_jev_template_is_used_when_format_compat_on(self) -> None:
        prompt = format_jev_prompt({"a": 1}, JevFlags(jev_format_compat=True))
        assert "<system>" in prompt
        assert "a: 1" in prompt

    def test_plain_template_when_format_compat_off(self) -> None:
        prompt = format_jev_prompt({"a": 1}, JevFlags(jev_format_compat=False))
        assert prompt.startswith("<|im_start|>system")
        assert "<system>" not in prompt

    def test_training_compat_does_not_change_the_prompt(self) -> None:
        """The switches are independent: format is decided by format alone."""
        on = format_jev_prompt({"a": 1}, JevFlags(jev_format_compat=True, jev_training_compat=True))
        off = format_jev_prompt(
            {"a": 1}, JevFlags(jev_format_compat=True, jev_training_compat=False)
        )
        assert on == off

    def test_feature_order_is_stable(self) -> None:
        assert render_features({"b": 2, "a": 1}) == render_features({"a": 1, "b": 2})

    def test_suffix_states_the_reason_cap(self) -> None:
        assert "200" in build_grammar_prompt_suffix()

    def test_documented_template_still_has_a_features_slot(self) -> None:
        assert "{features}" in JEV_INPUT_TEMPLATE
        assert "{features}" in PLAIN_PROMPT_TEMPLATE


class TestParseCompletion:
    def test_plain_json(self) -> None:
        assert parse_completion('{"decision": "black"}') == {"decision": "black"}

    def test_surrounding_prose_is_tolerated(self) -> None:
        raw = '好的，判定结果如下：{"decision": "gray"} 以上。'
        assert parse_completion(raw)["decision"] == "gray"

    def test_fenced_json_is_tolerated(self) -> None:
        assert parse_completion('```json\n{"decision": "white"}\n```')["decision"] == "white"

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON object"):
            parse_completion("I cannot decide.")

    def test_malformed_contents_are_not_repaired(self) -> None:
        """Repairing would make the compliance metric measure this function."""
        with pytest.raises(ValueError):
            parse_completion('{"decision": "black", "score": }')

    def test_json_array_is_not_mistaken_for_an_object(self) -> None:
        with pytest.raises(ValueError, match="no JSON object"):
            parse_completion("[1, 2, 3]")


class TestTrainingWhitelist:
    def test_jev_methods_allowed_when_on(self) -> None:
        assert "dpo" in training_method_whitelist(JevFlags())

    def test_jev_methods_refused_when_off(self) -> None:
        assert "dpo" not in training_method_whitelist(JevFlags(jev_training_compat=False))

    def test_adapter_methods_stay_allowed(self) -> None:
        """Turning training compat off does not disable plain LoRA."""
        allowed = training_method_whitelist(JevFlags(jev_training_compat=False))
        assert "lora" in allowed and "qlora" in allowed

    def test_assertion_rejects_a_jev_method_when_off(self) -> None:
        with pytest.raises(ValueError, match="jev_training_compat"):
            assert_training_methods_allowed(("dpo",), JevFlags(jev_training_compat=False))

    def test_assertion_passes_a_jev_method_when_on(self) -> None:
        assert_training_methods_allowed(("dpo",), JevFlags())

    def test_format_compat_does_not_restrict_training(self) -> None:
        """Two independent switches: one being off must not narrow the other."""
        assert_training_methods_allowed(("dpo",), JevFlags(jev_format_compat=False))

    def test_metric_basis_follows_training_switch(self) -> None:
        assert metric_basis(JevFlags()) == "jev_eval_baseline"
        assert metric_basis(JevFlags(jev_training_compat=False)) == "internal"


class TestCompatibilityClaims:
    """Q3: the JEV spec in this repo is a paraphrase, not the upstream document."""

    def test_l1_is_claimable(self) -> None:
        # No return value: the call succeeding is the assertion.
        assert_level_claimable(JevCompatLevel.L1, evidence=None)

    def test_l3_requires_measurement(self) -> None:
        with pytest.raises(ValueError, match="对比测试"):
            assert_level_claimable(JevCompatLevel.L3, evidence="declared")

    def test_l3_rejected_with_no_evidence(self) -> None:
        with pytest.raises(ValueError, match="对比测试"):
            assert_level_claimable(JevCompatLevel.L3, evidence=None)

    def test_l3_allowed_once_measured(self) -> None:
        assert_level_claimable(JevCompatLevel.L3, evidence="measured")

    def test_l2_is_blocked_on_the_missing_benchmark(self) -> None:
        with pytest.raises(ValueError, match="Q3"):
            assert_level_claimable(JevCompatLevel.L2, evidence="measured")
