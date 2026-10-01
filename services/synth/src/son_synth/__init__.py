"""Synthetic data generation, fidelity scoring and privacy reporting.

Every generated row is stamped ``origin='synth'`` before it leaves this
package. That stamp is what lets
:func:`son_data_pipeline.stratified_split` structurally refuse to put synthetic
rows in the test split (需求方案.txt 5.4, "合成数据不进测试集").
"""

from son_synth.generators import (
    Constraint,
    FeatureDeriveGenerator,
    GaussianCopulaGenerator,
    Generator,
    RuleInjectionGenerator,
    SynthRequest,
    SynthResult,
    get_generator,
    synthesize,
)
from son_synth.privacy import (
    FidelityReport,
    PrivacyReport,
    assess_fidelity,
    assess_privacy,
    find_duplicates,
    nearest_neighbour_distance,
)

__all__ = [
    "Constraint",
    "FeatureDeriveGenerator",
    "FidelityReport",
    "GaussianCopulaGenerator",
    "Generator",
    "PrivacyReport",
    "RuleInjectionGenerator",
    "SynthRequest",
    "SynthResult",
    "assess_fidelity",
    "assess_privacy",
    "find_duplicates",
    "get_generator",
    "nearest_neighbour_distance",
    "synthesize",
]
