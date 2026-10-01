"""Workflow orchestration: the 7-step state machine and task dispatch.

Plan: M5. The state machine is generated from `son_contracts.StepState` so it
cannot drift from the `status` columns it drives.
"""
