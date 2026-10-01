import { useCallback, useMemo, useState } from 'react';

import {
  isManualInMode,
  nextStep,
  rapidModeSteps,
  stepIndex,
  STEP_IDS,
  userSteps,
  WIZARD_STEPS,
  type StepId,
  type WizardStep,
} from './flow';

export type RunState = 'pending' | 'succeeded' | 'failed';

export interface StepStatus {
  state: RunState;
  /** True when completed with system defaults rather than user configuration. */
  usedDefaults: boolean;
  reason?: string;
}

/**
 * Wizard progress, mirroring `son_orchestrator.statemachine`.
 *
 * The rapid-mode boundary is enforced here, not only in the UI chrome: calling
 * `complete` for a step past the boundary in rapid mode is refused, so a
 * mis-wired control cannot quietly break the "two steps and done" promise.
 */
export interface WizardState {
  mode: 'wizard' | 'rapid';
  current: StepId;
  statuses: Record<StepId, StepStatus>;
}

const INITIAL: Record<StepId, StepStatus> = Object.fromEntries(
  STEP_IDS.map((id) => [id, { state: 'pending', usedDefaults: false }]),
) as Record<StepId, StepStatus>;

export function initialWizardState(mode: WizardState['mode'] = 'wizard'): WizardState {
  return { mode, current: 'scene_selected', statuses: { ...INITIAL } };
}

export function canConfigure(state: WizardState, id: StepId): boolean {
  return isManualInMode(id, state.mode);
}

export function isComplete(state: WizardState, id: StepId): boolean {
  return state.statuses[id].state === 'succeeded';
}

export function completeStep(
  state: WizardState,
  id: StepId,
  options: { usedDefaults?: boolean; byUser?: boolean; reason?: string } = {},
): WizardState {
  const { usedDefaults = false, byUser = true, reason } = options;
  if (byUser && !canConfigure(state, id)) {
    // Refused in rapid mode past step 2, matching ManualStepError server-side.
    return state;
  }
  return {
    ...state,
    current: nextStep(id) ?? id,
    statuses: {
      ...state.statuses,
      [id]: { state: 'succeeded', usedDefaults, reason },
    },
  };
}

export function failStep(state: WizardState, id: StepId, reason: string): WizardState {
  return {
    ...state,
    statuses: { ...state.statuses, [id]: { state: 'failed', usedDefaults: false, reason } },
  };
}

export function skipStep(state: WizardState, id: StepId): WizardState {
  return completeStep(state, id, { usedDefaults: true, reason: '使用默认值' });
}

export function resetFrom(state: WizardState, id: StepId): WizardState {
  const start = stepIndex(id);
  const statuses = { ...state.statuses };
  for (const candidate of STEP_IDS.slice(start)) {
    statuses[candidate] = { state: 'pending', usedDefaults: false };
  }
  return { ...state, current: id, statuses };
}

export function nextPending(state: WizardState): StepId {
  return STEP_IDS.find((id) => !isComplete(state, id)) ?? 'deployed';
}

/**
 * Fill the remaining steps with their defaults, which is what the orchestrator
 * does after the user finishes step 2 in rapid mode.
 *
 * Deployment is excluded: 一键部署 is an explicit user action.
 */
export function applyDefaultsForRemaining(state: WizardState): WizardState {
  let next = state;
  for (const step of WIZARD_STEPS) {
    if (step.id === 'deployed' || isComplete(next, step.id)) continue;
    next = completeStep(next, step.id, {
      usedDefaults: true,
      byUser: false,
      reason: '系统自动采用推荐值',
    });
  }
  return next;
}

export interface WizardController {
  state: WizardState;
  steps: readonly WizardStep[];
  canConfigure: (id: StepId) => boolean;
  isComplete: (id: StepId) => boolean;
  visibleSteps: readonly WizardStep[];
  goTo: (id: StepId) => void;
  complete: (id: StepId) => void;
  skip: (id: StepId) => void;
  fail: (id: StepId, reason: string) => void;
  resetFrom: (id: StepId) => void;
  setMode: (mode: WizardState['mode']) => void;
  progress: number;
}

export function useWizard(mode: WizardState['mode'] = 'wizard'): WizardController {
  const [state, setState] = useState<WizardState>(() => initialWizardState(mode));

  const goTo = useCallback((id: StepId) => setState((s) => ({ ...s, current: id })), []);
  const complete = useCallback(
    (id: StepId) => setState((s) => completeStep(s, id)),
    [],
  );
  const skip = useCallback((id: StepId) => setState((s) => skipStep(s, id)), []);
  const fail = useCallback(
    (id: StepId, reason: string) => setState((s) => failStep(s, id, reason)),
    [],
  );
  const resetFromStep = useCallback((id: StepId) => setState((s) => resetFrom(s, id)), []);
  const setMode = useCallback(
    (next: WizardState['mode']) => setState((s) => ({ ...s, mode: next })),
    [],
  );

  const visibleSteps = useMemo(
    () => (state.mode === 'rapid' ? [...userSteps(), ...WIZARD_STEPS.slice(7)] : WIZARD_STEPS),
    [state.mode],
  );

  const progress = useMemo(() => {
    const done = STEP_IDS.filter((id) => isComplete(state, id)).length;
    return Math.round((done / STEP_IDS.length) * 100);
  }, [state]);

  return {
    state,
    steps: state.mode === 'rapid' ? rapidModeSteps() : userSteps(),
    canConfigure: (id: StepId) => canConfigure(state, id),
    isComplete: (id: StepId) => isComplete(state, id),
    visibleSteps,
    goTo,
    complete,
    skip,
    fail,
    resetFrom: resetFromStep,
    setMode,
    progress,
  };
}
export type { StepId, WizardStep };
