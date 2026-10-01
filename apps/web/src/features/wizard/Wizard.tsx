import { useState } from 'react';

import { SynthStep, UploadStep } from '@/features/datasets/DatasetSteps';
import { findStep, nextStep } from './flow';
import { useWizard, type StepId } from './useWizard';

/**
 * The 7-step wizard (需求方案.txt 4.1).
 *
 * One screen per step, with a "skip, use defaults" control in the corner of
 * every step that allows it.
 *
 * Two things are deliberately *not* defined here:
 *
 * - the step order, titles and hints all come from `flow.ts`. Restating them in
 *   the component is how the flow definition and the screen drift apart, and
 *   需求方案.txt principle 1 requires all interaction modes to share one engine.
 * - the rapid-mode boundary. `useWizard` refuses a manual completion past step 2
 *   rather than this component hiding the control, so a mis-wired button cannot
 *   quietly break the "two steps and done" promise.
 *
 * Rapid mode renders the same wizard with the post-step-2 controls disabled
 * instead of hidden, so a user can see what the system is doing for them.
 */
export function Wizard() {
  const wizard = useWizard();
  const [file, setFile] = useState<File | null>(null);

  const current = wizard.state.current;
  const step = findStep(current);
  const skippable = step?.skippable ?? false;
  const configurable = wizard.canConfigure(current);
  const following = nextStep(current);

  return (
    <section aria-labelledby="wizard-heading">
      <h2 id="wizard-heading">{step?.title ?? current}</h2>
      <p>{step?.hint ?? ''}</p>

      <div role="group" aria-label="流程模式">
        <button
          type="button"
          aria-pressed={wizard.state.mode === 'wizard'}
          onClick={() => wizard.setMode('wizard')}
        >
          向导模式
        </button>
        <button
          type="button"
          aria-pressed={wizard.state.mode === 'rapid'}
          onClick={() => wizard.setMode('rapid')}
        >
          极速模式
        </button>
      </div>

      <progress value={wizard.progress} max={100} aria-label="整体进度" />

      <nav aria-label="向导步骤">
        <ol>
          {wizard.visibleSteps.map((visible) => (
            <li key={visible.id}>
              <button
                type="button"
                onClick={() => wizard.goTo(visible.id)}
                aria-current={current === visible.id ? 'step' : undefined}
              >
                {markFor(wizard, visible.id)} {visible.title}
                {wizard.state.statuses[visible.id]?.usedDefaults ? '（默认值）' : ''}
              </button>
            </li>
          ))}
        </ol>
      </nav>

      <div>{renderStep(current, file, setFile)}</div>

      <footer>
        <button
          type="button"
          onClick={() => wizard.skip(current)}
          disabled={!skippable || !configurable}
        >
          跳过，用默认值
        </button>
        <button type="button" onClick={() => wizard.complete(current)} disabled={!configurable}>
          {configurable && following
            ? `下一步：${findStep(following)?.title ?? following}`
            : '由系统自动完成'}
        </button>
      </footer>
    </section>
  );
}

function markFor(wizard: ReturnType<typeof useWizard>, id: StepId): string {
  const status = wizard.state.statuses[id];
  if (status?.state === 'succeeded') return '✓';
  if (status?.state === 'failed') return '✗';
  return '·';
}

function renderStep(current: StepId, file: File | null, setFile: (f: File | null) => void) {
  switch (current) {
    case 'scene_selected':
      return (
        <ul>
          <li>反诈账户判定 [推荐]</li>
          <li>支付风控</li>
          <li>合规审核</li>
          <li>营销决策</li>
          <li>信贷审批</li>
          <li>内容安全</li>
        </ul>
      );
    case 'seed_uploaded':
      return (
        <label>
          上传种子数据（支持 CSV）
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
      );
    case 'data_governed':
      return <UploadStep />;
    case 'synth_done':
      return <SynthStep file={file} />;
    case 'base_model_selected':
      return <p>模型货架与显存预估在训练链路接入后展示。</p>;
    case 'training_configured':
      return (
        <div>
          <h3>训练模式</h3>
          <ul>
            <li>快速模式（推荐）</li>
            <li>精细模式</li>
            <li>蒸馏模式</li>
          </ul>
          <details>
            <summary>高级设置</summary>
            <p>学习率、批次大小、迭代步数由系统按数据量与模型规模自动推荐。</p>
          </details>
        </div>
      );
    case 'trained':
      return <p>训练监控（进度、剩余时间、损失曲线）需要 GPU 节点，本机无法运行。</p>;
    case 'evaluated':
      return <p>评测报告在评测链路接入后展示。</p>;
    case 'deployed':
      return <p>一键部署与调用示例在部署链路接入后展示。</p>;
    default:
      return <p>当前步骤：{current}</p>;
  }
}
