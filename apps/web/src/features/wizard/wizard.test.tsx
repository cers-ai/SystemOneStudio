import { act, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import {
  isManualInMode,
  nextStep,
  rapidModeSteps,
  stepIndex,
  STEP_IDS,
  userSteps,
  WIZARD_STEPS,
  USER_STEP_COUNT,
} from './flow';
import {
  applyDefaultsForRemaining,
  canConfigure,
  completeStep,
  failStep,
  initialWizardState,
  isComplete,
  nextPending,
  resetFrom,
  skipStep,
  useWizard,
  type StepId,
} from './useWizard';

describe('flow definition', () => {
  it('has nine steps: seven user-facing plus evaluate and deploy', () => {
    expect(WIZARD_STEPS).toHaveLength(9);
    expect(USER_STEP_COUNT).toBe(7);
    expect(userSteps()).toHaveLength(7);
  });

  it('step order matches the contract enum', () => {
    const titles = WIZARD_STEPS.slice(0, 7).map((s) => s.id);
    expect(titles).toEqual([
      'scene_selected',
      'seed_uploaded',
      'data_governed',
      'synth_done',
      'base_model_selected',
      'training_configured',
      'trained',
    ]);
  });

  it('rapid mode requires exactly steps one and two', () => {
    const manual = rapidModeSteps();
    expect(manual).toHaveLength(2);
    expect(manual.map((s) => s.id)).toEqual(['scene_selected', 'seed_uploaded']);
  });

  it('every step declares a hint', () => {
    expect(WIZARD_STEPS.every((s) => s.hint.length > 0)).toBe(true);
  });

  it('scene and upload cannot be skipped', () => {
    // Both define the dataset; a default here would be meaningless.
    expect(userSteps().filter((s) => s.skippable).every((s) => stepIndex(s.id) >= 2)).toBe(true);
  });

  it('next step walks the order and stops at the end', () => {
    expect(nextStep('scene_selected')).toBe('seed_uploaded');
    expect(nextStep('deployed')).toBeUndefined();
  });

  it('rapid mode blocks manual configuration past step two', () => {
    expect(isManualInMode('seed_uploaded', 'rapid')).toBe(true);
    expect(isManualInMode('data_governed', 'rapid')).toBe(false);
  });

  it('wizard mode allows all seven user steps', () => {
    expect(userSteps().every((s) => isManualInMode(s.id, 'wizard'))).toBe(true);
  });
});

describe('step completion', () => {
  it('completing advances the cursor', () => {
    const state = completeStep(initialWizardState(), 'scene_selected');
    expect(state.current).toBe('seed_uploaded');
  });

  it('a completed step is recorded', () => {
    expect(isComplete(completeStep(initialWizardState(), 'scene_selected'), 'scene_selected')).toBe(
      true,
    );
  });

  it('defaults are recorded distinctly from configuration', () => {
    // Otherwise an evaluation cannot tell a default run from a configured one.
    const skipped = skipStep(initialWizardState(), 'data_governed');
    expect(skipped.statuses.data_governed.usedDefaults).toBe(true);
    const configured = completeStep(initialWizardState(), 'data_governed');
    expect(configured.statuses.data_governed.usedDefaults).toBe(false);
  });

  it('failure is recorded with its reason', () => {
    const state = failStep(initialWizardState(), 'synth_done', '样本量不足');
    expect(state.statuses.synth_done.state).toBe('failed');
    expect(state.statuses.synth_done.reason).toBe('样本量不足');
  });

  it('next pending skips completed steps', () => {
    let state = completeStep(initialWizardState(), 'scene_selected');
    state = completeStep(state, 'seed_uploaded');
    expect(nextPending(state)).toBe('data_governed');
  });
});

describe('rapid mode boundary', () => {
  it('step one and two are configurable', () => {
    const state = initialWizardState('rapid');
    expect(canConfigure(state, 'scene_selected')).toBe(true);
    expect(canConfigure(state, 'seed_uploaded')).toBe(true);
  });

  it('past step two manual completion is refused', () => {
    const state = initialWizardState('rapid');
    expect(canConfigure(state, 'data_governed')).toBe(false);
    const attempted = completeStep(state, 'data_governed');
    expect(isComplete(attempted, 'data_governed')).toBe(false);
  });

  it('refusal leaves the cursor where it was', () => {
    const state = initialWizardState('rapid');
    expect(completeStep(state, 'data_governed').current).toBe(state.current);
  });

  it('the driver fills the remaining steps automatically', () => {
    let state = initialWizardState('rapid');
    state = completeStep(state, 'scene_selected');
    state = completeStep(state, 'seed_uploaded');
    state = applyDefaultsForRemaining(state);
    expect(isComplete(state, 'training_configured')).toBe(true);
    expect(isComplete(state, 'evaluated')).toBe(true);
  });

  it('deployment stays with the user', () => {
    let state = initialWizardState('rapid');
    state = completeStep(state, 'seed_uploaded');
    state = applyDefaultsForRemaining(state);
    expect(isComplete(state, 'deployed')).toBe(false);
  });

  it('the driver marks automated steps as defaults', () => {
    let state = initialWizardState('rapid');
    state = completeStep(state, 'seed_uploaded');
    state = applyDefaultsForRemaining(state);
    expect(state.statuses.synth_done.usedDefaults).toBe(true);
    expect(state.statuses.synth_done.reason).toContain('系统自动');
  });

  it('the driver does not overwrite configured steps', () => {
    let state = initialWizardState('rapid');
    state = completeStep(state, 'data_governed', { byUser: false, usedDefaults: false, reason: '用户配置' });
    state = applyDefaultsForRemaining(state);
    expect(state.statuses.data_governed.reason).toBe('用户配置');
  });

  it('wizard mode has no such boundary', () => {
    const state = initialWizardState('wizard');
    expect(canConfigure(state, 'training_configured')).toBe(true);
  });
});

describe('reversibility', () => {
  it('reset clears the step and everything downstream', () => {
    let state = initialWizardState();
    for (const id of STEP_IDS) state = completeStep(state, id);
    const reset = resetFrom(state, 'synth_done');
    expect(isComplete(reset, 'seed_uploaded')).toBe(true);
    expect(isComplete(reset, 'synth_done')).toBe(false);
    expect(isComplete(reset, 'deployed')).toBe(false);
  });

  it('reset moves the cursor back', () => {
    let state = initialWizardState();
    for (const id of STEP_IDS) state = completeStep(state, id);
    expect(resetFrom(state, 'synth_done').current).toBe('synth_done');
  });
});

describe('useWizard', () => {
  it('exposes exactly the user-facing steps', () => {
    function Probe() {
      const wizard = useWizard();
      return <ul data-testid="steps">{wizard.steps.map((s) => <li key={s.id}>{s.title}</li>)}</ul>;
    }
    render(<Probe />);
    expect(screen.getAllByRole('listitem')).toHaveLength(7);
  });

  it('shows the seven step titles in order', () => {
    function Probe() {
      const wizard = useWizard();
      return (
        <ol>
          {wizard.steps.map((s) => (
            <li key={s.id}>{s.title}</li>
          ))}
        </ol>
      );
    }
    render(<Probe />);
    const titles = screen.getAllByRole('listitem').map((li) => li.textContent);
    expect(titles[0]).toBe('选择场景模板');
    expect(titles[6]).toBe('模型训练');
  });

  it('progress starts at zero', () => {
    function Probe() {
      const { progress } = useWizard();
      return <span data-testid="progress">{progress}</span>;
    }
    render(<Probe />);
    expect(screen.getByTestId('progress').textContent).toBe('0');
  });

  it('progress advances with completed steps', () => {
    function Probe() {
      const wizard = useWizard();
      return (
        <>
          <span data-testid="progress">{wizard.progress}</span>
          <button type="button" onClick={() => wizard.complete('scene_selected')}>
            next
          </button>
        </>
      );
    }
    render(<Probe />);
    // React 18 batches updates, so a bare .click() does not flush the state
    // change synchronously. Wrapping in act() is what makes this assertion
    // meaningful rather than racing.
    act(() => {
      screen.getByRole('button', { name: 'next' }).click();
    });
    const value = Number(screen.getByTestId('progress').textContent);
    expect(value).toBeGreaterThan(0);
  });

  it('visible steps include deploy even in rapid mode', () => {
    function Probe() {
      const wizard = useWizard('rapid');
      return (
        <ul>
          {wizard.visibleSteps.map((s: WizardStepLike) => (
            <li key={s.id}>{s.title}</li>
          ))}
        </ul>
      );
    }
    render(<Probe />);
    expect(screen.getByText('部署')).toBeInTheDocument();
  });
});

type WizardStepLike = (typeof WIZARD_STEPS)[number] & { id: StepId };