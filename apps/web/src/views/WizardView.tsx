import { StepRail } from '@/features/wizard/StepRail';
import { StepBody } from '@/features/wizard/StepBody';
import { useWizard } from '@/features/wizard/useWizard';

/**
 * Wizard inside the app shell.
 *
 * Owns its own controller so the rail and the body stay in sync without lifting
 * wizard state into the shell -- the shell routes between views, it does not
 * know what a step is.
 */
export function WizardView() {
  const wizard = useWizard();

  return (
    <div style={{ margin: 'calc(var(--sp-8) * -1)', display: 'grid', gridTemplateColumns: 'var(--rail-w) minmax(0, 1fr)' }}>
      <StepRail wizard={wizard} onNavigate={wizard.goTo} />
      <StepBody wizard={wizard} onModeHint="" current={wizard.state.current} />
    </div>
  );
}
