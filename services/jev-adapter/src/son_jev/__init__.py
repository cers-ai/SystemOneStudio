"""JEV alignment: the two independent switches and their consequences.

需求方案.txt 6.3 splits JEV compatibility into two switches that control
different layers, and this module keeps them separate rather than collapsing
them into one "JEV mode":

* ``jev_format_compat``   -> prompt, output schema, API shape
* ``jev_training_compat`` -> training paradigm, hyperparameters, eval basis

Only L1 is assertable today. The repository holds a paraphrase of the JEV spec,
not the upstream document, so L2 and L3 cannot be claimed (技术方案.md Q3).
"""

from son_contracts import JevCompatLevel, JevFlags
from son_jev.format import (
    PLAIN_PROMPT_TEMPLATE,
    GrammarSpec,
    build_grammar,
    build_grammar_prompt_suffix,
    format_jev_prompt,
    parse_completion,
    render_features,
)


def training_method_whitelist(flags: JevFlags) -> tuple[str, ...]:
    """Methods allowed under a given ``jev_training_compat``.

    With training compat off, the JEV-paradigm-only methods are refused rather
    than run with an implicit caveat: running them and reporting a normal result
    would produce a model the platform cannot describe accurately.
    """
    always = ("lora", "qlora")
    jev_only = ("sft", "dpo", "distillation", "qat", "fewshot")
    return always if not flags.jev_training_compat else always + jev_only


def assert_training_methods_allowed(methods: tuple[str, ...], flags: JevFlags) -> None:
    allowed = set(training_method_whitelist(flags))
    refused = [m for m in methods if m not in allowed]
    if refused:
        raise ValueError(
            f"训练方法 {refused} 属于 JEV 范式，但 jev_training_compat 已关闭；"
            f"当前允许：{sorted(allowed)}"
        )


def metric_basis(flags: JevFlags) -> str:
    """Which evaluation basis applies."""
    return "jev_eval_baseline" if flags.jev_training_compat else "internal"


def assert_level_claimable(level: JevCompatLevel, *, evidence: str | None) -> None:
    """Refuse an unverifiable compatibility claim.

    L3 requires a comparison test against native JEV output (需求方案.txt 6.2).
    L2 requires the JEV benchmark, which is not in this repository. With Q3
    open, both are refused at the boundary rather than noted in a comment.
    """
    if level is JevCompatLevel.L1:
        return
    if level is JevCompatLevel.L3 and evidence != "measured":
        raise ValueError("JEV L3（风格兼容）必须有对比测试报告支撑；声明不算证据")
    if level is JevCompatLevel.L2:
        raise ValueError(
            "JEV L2（格式+评测兼容）需要 JEV 官方评测基准数据集，"
            "该数据集不在仓库中（技术方案 Q3），暂不可声明"
        )


__all__ = [
    "PLAIN_PROMPT_TEMPLATE",
    "GrammarSpec",
    "assert_level_claimable",
    "assert_training_methods_allowed",
    "build_grammar",
    "build_grammar_prompt_suffix",
    "format_jev_prompt",
    "metric_basis",
    "parse_completion",
    "render_features",
    "training_method_whitelist",
]
