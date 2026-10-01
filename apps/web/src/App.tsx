import { useState } from 'react';

import { HealthPanel } from '@/features/system/HealthPanel';
import { StepRail } from '@/features/wizard/StepRail';
import { StepBody } from '@/features/wizard/StepBody';
import { useWizard } from '@/features/wizard/useWizard';
import './styles/global.css';

type UiMode = 'wizard' | 'rapid';

const MODE_COPY: Record<UiMode, { label: string; blurb: string }> = {
  wizard: {
    label: '向导模式',
    blurb: '七步流程，每步一屏，可随时跳过使用推荐值',
  },
  rapid: {
    label: '极速模式',
    blurb: '只需完成前两步，其余由系统自动采用推荐值',
  },
};

export function App() {
  const wizard = useWizard();
  const [mode, setMode] = useState<UiMode>(wizard.state.mode);
  const [showStatus, setShowStatus] = useState(false);

  const copy = MODE_COPY[mode];
  const current = wizard.state.current;

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__brand">
          <strong>SystemOneStudio</strong>
          <span>决策模型训练平台</span>
        </div>

        <div className="segmented" role="group" aria-label="流程模式">
          {(Object.keys(MODE_COPY) as UiMode[]).map((key) => (
            <button
              key={key}
              type="button"
              className="segmented__btn"
              aria-pressed={mode === key}
              onClick={() => {
                setMode(key);
                wizard.setMode(key);
              }}
            >
              {MODE_COPY[key].label}
            </button>
          ))}
        </div>

        <span className="app__header-spacer" />

        <button
          type="button"
          className="btn btn--ghost"
          aria-expanded={showStatus}
          onClick={() => setShowStatus((v) => !v)}
        >
          平台状态
        </button>
      </header>

      {showStatus ? (
        <div style={{ padding: '16px 24px 0' }}>
          <HealthPanel />
        </div>
      ) : null}

      <div className="app__main">
        <StepRail wizard={wizard} onNavigate={wizard.goTo} />

        <StepBody wizard={wizard} onModeHint={copy.blurb} current={current} />
      </div>
    </div>
  );
}