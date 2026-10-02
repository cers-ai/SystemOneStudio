import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';

import { platformApi } from '@/api/platform';
import { SynthStep, UploadStep } from '@/features/datasets/DatasetSteps';
import { Term } from '@/components/Term';
import { findStep, nextStep, type StepId } from './flow';
import type { WizardController } from './useWizard';

/**
 * Step body: one screen per step (需求方案.txt 4.1).
 *
 * Titles, hints and the step order all come from `flow.ts`. Screens that need a
 * GPU node are not faked -- they say so, because a screen that renders plausible
 * placeholder numbers is worse than one that admits the capability is absent.
 */
export function StepBody({
  wizard,
  onModeHint,
  current,
}: {
  wizard: WizardController;
  onModeHint: string;
  current: StepId;
}) {
  const step = findStep(current);
  const skippable = step?.skippable ?? false;
  const configurable = wizard.canConfigure(current);
  const following = nextStep(current);
  const automated = !configurable;

  return (
    <section className="step">
      <header className="step__head">
        <div className="step__eyebrow">
          第 {orderOf(current)} 步 / 共 {wizard.visibleSteps.length}
          {wizard.state.mode === 'rapid' ? ' · 极速模式' : ''}
        </div>
        <h1 className="step__title">{step?.title ?? current}</h1>
        <p className="step__hint">{step?.hint ?? ''}</p>
      </header>

      <div className="step__body">{renderScreen(current, wizard)}</div>

      <footer className="step__actions">
        <button
          type="button"
          className="btn"
          disabled={!skippable || automated}
          onClick={() => wizard.skip(current)}
        >
          跳过，用默认值
        </button>

        <button
          type="button"
          className="btn btn--primary"
          disabled={!configurable}
          onClick={() => wizard.complete(current)}
        >
          {configurable && following
            ? `下一步：${findStep(following)?.title ?? following}`
            : '由系统自动完成'}
        </button>

        <span className="step__actions-note">
          {automated ? onModeHint : automatedHint(skippable)}
        </span>
      </footer>
    </section>
  );
}

function automatedHint(skippable: boolean): string {
  if (!skippable) return '这一步需要你确认，不能跳过';
  return '可以跳过，系统会采用推荐值';
}

function renderScreen(current: StepId, wizard: WizardController) {
  switch (current) {
    case 'scene_selected':
      return <ScenePicker />;
    case 'seed_uploaded':
      return <UploadStep />;
    case 'data_governed':
      return <QualityStep />;
    case 'synth_done':
      return <SynthStep file={null} />;
    case 'base_model_selected':
      return <ModelShelf />;
    case 'training_configured':
      return <TrainingConfig />;
    case 'trained':
      return <Blocked
        title="训练监控"
        detail="loss 曲线、进度与预计剩余时间需要 GPU 节点。当前机器没有 NVIDIA GPU，训练链路无法执行，因此这里不显示任何数字。"
      />;
    case 'evaluated':
      return (
        <Blocked
          title="评测报告"
          detail="效果指标可以立即算出，但性能指标必须来自真实推理服务，本机无法产生。另有一项指标口径尚未定案：响应速度目标与「判定依据」的长度上限存在冲突，需产品确认后才能判定是否达标。"
        />
      );
    case 'deployed':
      return (
        <Blocked
          title="一键部署"
          detail="启动推理服务需要具备 GPU 的机器。当前部署接口只返回启动方案与调用示例，并明确标记为未启动，不会显示一个实际上并未运行的服务。"
        />
      );
    default:
      return (
        <div className="note note--info">
          <span className="note__mark">i</span>
          <span>当前步骤：{current}（{wizard.progress}%）</span>
        </div>
      );
  }
}

/* -------------------------------------------------------------------------
   Screens
   ------------------------------------------------------------------------- */

const SCENES = [
  { id: 'fraud', name: '反诈账户判定', recommended: true, tags: ['32 个字段', '黑 / 白 / 灰'] },
  { id: 'payment', name: '支付风控', recommended: false, tags: ['交易特征'] },
  { id: 'compliance', name: '合规审核', recommended: false, tags: ['文本审核'] },
  { id: 'marketing', name: '营销决策', recommended: false, tags: ['用户分层'] },
  { id: 'credit', name: '信贷审批', recommended: false, tags: ['征信字段'] },
  { id: 'content', name: '内容安全', recommended: false, tags: ['文本审核'] },
];

function ScenePicker() {
  return (
    <>
      <p className="card__sub" style={{ marginBottom: 'var(--sp-4)' }}>
        选择一个行业场景后，字段规范、标签体系、推荐模型与核心指标会自动填好，约 80% 的配置无需再改。
      </p>
      <div className="grid grid--3">
        {SCENES.map((scene) => (
          <div className="card" key={scene.id} style={{ cursor: 'pointer' }}>
            <div className="card__title">
              {scene.name}
              {scene.recommended ? <span className="badge badge--info">推荐</span> : null}
            </div>
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {scene.tags.map((t) => (
                <span className="badge badge--neutral" key={t}>
                  {t}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="note note--info" style={{ marginTop: 'var(--sp-4)' }}>
        <span className="note__mark">i</span>
        <span>
          反诈账户判定的 32 个字段目前是占位模板，真实字段规范尚未提供（需求侧待补）。选定后可在第二步直接上传数据验证流程。
        </span>
      </div>
    </>
  );
}

function QualityStep() {
  return <UploadStep />;
}

/**
 * Base model shelf (需求方案.txt 5.5).
 *
 * Reads /api/models rather than carrying its own copy. It used to hardcode four
 * cards with invented VRAM figures and durations, which meant the screen showed
 * numbers the platform had never measured and had drifted from the registry.
 *
 * No duration is shown at all: none has been measured, and an empty cell is
 * more honest than an estimate wearing a data table's typography.
 */
function ModelShelf() {
  const models = useQuery({ queryKey: ['models'], queryFn: platformApi.models });

  if (models.isPending) return <div className="empty">加载中…</div>;
  if (models.isError) {
    return (
      <div className="note note--bad">
        <span className="note__mark">!</span>
        <span>模型列表加载失败：{models.error.message}</span>
      </div>
    );
  }

  const shelf = models.data;
  const [showAll, setShowAll] = useState(false);
  const entries = (shelf?.all ?? []).filter((m) => showAll || m.in_mvp_scope);

  return (
    <>
      <p className="card__sub" style={{ marginBottom: 'var(--sp-4)' }}>
        底座模型全部可插拔。MVP 范围限定在 1.5B–3B，这是为显存不足做的主动取舍，
        不是遗漏；更大的型号会列出但标记为暂不可选。
      </p>

      <label style={{ fontSize: 'var(--fs-sm)', marginBottom: 'var(--sp-3)' }}>
        <input
          type="checkbox"
          checked={showAll}
          onChange={(e) => setShowAll(e.target.checked)}
          style={{ marginRight: 6 }}
        />
        显示 MVP 范围外的型号
      </label>

      <div className="grid grid--2">
        {entries.map((model) => (
          <div className="card" key={model.model_id}>
            <div className="card__title">
              {model.display_name}
              {!model.in_mvp_scope ? (
                <span className="badge badge--neutral">范围外</span>
              ) : null}
            </div>
            <div className="kv">
              <dt>规格</dt>
              <dd>{model.params}</dd>
              <dt>显存需求</dt>
              <dd>
                {model.min_gpu_memory}
                <span className="badge badge--warn" style={{ marginLeft: 6 }}>
                  估算值
                </span>
              </dd>
              <dt>训练时长</dt>
              <dd>
                <span style={{ color: 'var(--c-text-tertiary)' }}>未测量</span>
              </dd>
              <dt>输出规范兼容度</dt>
              <dd>
                {model.jev_compat_level ? (
                  <span className="badge badge--neutral">{model.jev_compat_level}</span>
                ) : (
                  <span className="badge badge--warn">待实测</span>
                )}
              </dd>
              <dt>许可</dt>
              <dd>{model.license}</dd>
            </div>
          </div>
        ))}
      </div>

      {(shelf?.notes ?? []).map((note) => (
        <div className="note note--warn" key={note} style={{ marginTop: 'var(--sp-3)' }}>
          <span className="note__mark">!</span>
          <span>{note}</span>
        </div>
      ))}
    </>
  );
}

const TRAINING_MODES = [
  {
    key: 'rapid',
    name: '快速模式',
    desc: '决策风格适配 + 决策优化训练',
    recommended: true,
  },
  { key: 'fine', name: '精细模式', desc: '全量微调 + 压缩感知训练', recommended: false },
  { key: 'distill', name: '蒸馏模式', desc: '大模型能力迁移到小模型', recommended: false },
] as const;

/**
 * Training configuration (需求方案.txt 5.6).
 *
 * Durations and VRAM are read from /api/models/{id}/plan, which *computes* them
 * from the dataset size and model spec per principle 2. They used to be
 * hardcoded here, which duplicated the registry and invented numbers the
 * platform had never measured.
 */
function TrainingConfig() {
  const plans = useQuery({
    queryKey: ['training-plans'],
    queryFn: async () => {
      const ids = ['qwen2.5-3b-instruct', 'qwen2.5-1.5b-instruct', 'gemma-2-2b-it'];
      const results = await Promise.all(
        ids.map((id) => fetch(`/api/models/${id}/plan?rows=15000`).then((r) => r.json())),
      );
      return results as {
        model_id: string;
        estimated_minutes: number;
        measured: boolean;
        vram_estimate: string;
        hyperparams: Record<string, number | string | boolean>;
      }[];
    },
  });

  return (
    <>
      <div className="card__title" style={{ marginBottom: 'var(--sp-3)' }}>
        训练模式
      </div>
      <div className="grid grid--3">
        {TRAINING_MODES.map((mode) => (
          <div className="card" key={mode.key}>
            <div className="card__title">
              {mode.name}
              {mode.recommended ? <span className="badge badge--info">推荐</span> : null}
            </div>
            <div className="card__sub" style={{ marginBottom: 'var(--sp-3)' }}>
              {mode.desc}
            </div>
            <div className="row-item__meta">
              时长与显存随所选底座模型变化，见下方「推荐配置」
            </div>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginTop: 'var(--sp-4)' }}>
        <div className="card__title">推荐配置</div>
        <div className="card__sub">
          由系统按数据量与模型规格计算（原则 2：能不动就不动）。
          时长为推算估算，非实测。
        </div>

        {plans.isPending ? <div className="empty">计算中…</div> : null}

        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">底座模型</th>
                <th scope="col">显存（估算）</th>
                <th scope="col">时长（估算）</th>
                <th scope="col">学习率</th>
                <th scope="col">批次</th>
                <th scope="col">步数</th>
              </tr>
            </thead>
            <tbody>
              {(plans.data ?? []).map((plan) => (
                <tr key={plan.model_id}>
                  <td>{plan.model_id}</td>
                  <td>{plan.vram_estimate}</td>
                  <td>约 {Math.round(plan.estimated_minutes)} 分钟</td>
                  <td>{String(plan.hyperparams.learning_rate)}</td>
                  <td>{String(plan.hyperparams.batch_size)}</td>
                  <td>{String(plan.hyperparams.max_steps)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <details className="card" style={{ marginTop: 'var(--sp-4)' }}>
        <summary style={{ cursor: 'pointer', fontWeight: 600 }}>
          高级设置（默认折叠，业务人员无需改动）
        </summary>
        <div className="card__sub" style={{ marginTop: 'var(--sp-3)' }}>
          学习率、批次大小与迭代步数由系统按数据量与模型规模计算得出。
        </div>
        <div className="kv">
          <dt>
            <Term term="lora" />
          </dt>
          <dd>已勾选</dd>
          <dt>
            <Term term="dpo" />
          </dt>
          <dd>已勾选</dd>
          <dt>
            <Term term="quant_q4_k_m" />
          </dt>
          <dd>默认导出</dd>
        </div>
      </details>

      <div className="note note--warn" style={{ marginTop: 'var(--sp-4)' }}>
        <span className="note__mark">!</span>
        <span>
          上表时长与显存均为推算估算，未经 GPU 实测。训练需要 NVIDIA GPU，
          本机没有 GPU，因此没有任何一个数字来自真实运行。
        </span>
      </div>
    </>
  );
}

function Blocked({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="card">
      <div className="card__title">{title}</div>
      <div className="note note--warn">
        <span className="note__mark">!</span>
        <span>{detail}</span>
      </div>
    </div>
  );
}

function orderOf(id: StepId): number {
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
  return order.indexOf(id) + 1;
}