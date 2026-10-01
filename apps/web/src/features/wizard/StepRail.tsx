import type { StepId } from './flow';
import type { WizardController } from './useWizard';

/**
 * Step navigation (需求方案.txt 4.1: 7 steps, one screen each).
 *
 * The rail shows completion *and* whether a step was completed with defaults,
 * which is the distinction 原则 5 depends on: a run where the user configured
 * nothing is not the same run as one where they configured everything, and an
 * evaluation report cannot tell them apart otherwise.
 *
 * Uses the same WIZARD_STEPS definition as the step body. No titles here.
 */
export function StepRail({
  wizard,
  onNavigate,
}: {
  wizard: WizardController;
  onNavigate: (id: StepId) => void;
}) {
  return (
    <nav className="rail" aria-label="向导步骤">
      <div className="rail__heading">训练流程</div>

      <div className="rail__progress">
        <div
          className="rail__progress-track"
          role="progressbar"
          aria-valuenow={wizard.progress}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="整体进度"
        >
          <div className="rail__progress-fill" style={{ width: `${wizard.progress}%` }} />
        </div>
        <div className="rail__progress-label">已完成 {wizard.progress}%</div>
      </div>

      <ol className="rail__list">
        {wizard.visibleSteps.map((step) => {
          const status = wizard.state.statuses[step.id];
          const done = status?.state === 'succeeded';
          const failed = status?.state === 'failed';
          const current = wizard.state.current === step.id;

          const classes = [
            'rail__item',
            done ? 'rail__item--done' : '',
            failed ? 'rail__item--failed' : '',
            done && status?.usedDefaults ? 'rail__item--defaults' : '',
          ]
            .filter(Boolean)
            .join(' ');

          return (
            <li key={step.id} className={classes}>
              <button
                type="button"
                className="rail__button"
                aria-current={current ? 'step' : undefined}
                onClick={() => onNavigate(step.id)}
              >
                <span className="rail__marker" aria-hidden="true">
                  {failed ? '!' : done ? (status?.usedDefaults ? '·' : '✓') : stepIndexLabel(step.id)}
                </span>
                <span className="rail__label">{step.title}</span>
                {done && status?.usedDefaults ? <span className="rail__tag">默认</span> : null}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

function stepIndexLabel(id: StepId): string {
  const order: StepId[] = [
    'scene_selected',
    'seed_uploaded',
    'data_governed',
    'synth_done',
    'base_model_selected',
    'training_configured',
    'trained',
    'evaluated',
    'deployed',
  ];
  return String(order.indexOf(id) + 1);
}