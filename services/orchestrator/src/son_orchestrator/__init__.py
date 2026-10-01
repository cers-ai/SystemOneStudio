"""Workflow orchestration: the 7-step state machine and task dispatch.

The state machine is the spine of the platform and is generated from
``son_contracts.StepState``, so it cannot drift from the wizard UI or from any
database ``status`` column derived from the same enum.

Rapid mode ends after step 2 and is enforced here, not just in the UI -- see
``ManualStepError``.
"""

from son_orchestrator.statemachine import (
    AUTOMATED_STEPS,
    USER_STEPS,
    ManualStepError,
    StepRecord,
    TransitionError,
    Workflow,
    advance_through_defaults,
    clone_workflow,
    start_workflow,
)

__all__ = [
    "AUTOMATED_STEPS",
    "USER_STEPS",
    "ManualStepError",
    "StepRecord",
    "TransitionError",
    "Workflow",
    "advance_through_defaults",
    "clone_workflow",
    "start_workflow",
]
