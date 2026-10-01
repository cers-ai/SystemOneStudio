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

function ModelShelf() {
  return (
    <>
      <p className="card__sub" style={{ marginBottom: 'var(--sp-4)' }}>
        底座模型全部可插拔。MVP 范围限定在 1.5B–3B，这是为显存不足做的主动取舍，不是遗漏；更大的型号会列出但标记为暂不可选。
      </p>
      <div className="grid grid--2">
        {[
          { id: 'qwen2.5-3b-instruct', name: 'Qwen2.5-3B-Instruct', vram: '12GB', mins: '约 45 分钟', level: '风格兼容待实测' },
          { id: 'qwen2.5-1.5b-instruct', name: 'Qwen2.5-1.5B-Instruct', vram: '8GB', mins: '约 25 分钟', level: '风格兼容待实测' },
          { id: 'gemma-2-2b-it', name: 'Gemma 2 2B-it', vram: '8GB', mins: '约 30 分钟', level: '格式兼容 L1' },
          { id: 'son-tabular-classifier-lt1b', name: '轻量表格分类器 <1B', vram: '4GB', mins: '约 10 分钟', level: '格式兼容 L1' },
        ].map((m) => (
          <div className="card" key={m.id}>
            <div className="card__title">{m.name}</div>
            <div className="kv">
              <dt>显存需求</dt>
              <dd>{m.vram}</dd>
              <dt>预计时长</dt>
              <dd>{m.mins}</dd>
              <dt>输出规范兼容度</dt>
              <dd>{m.level}</dd>
            </div>
          </div>
        ))}
      </div>

      <div className="note note--warn" style={{ marginTop: 'var(--sp-4)' }}>
        <span className="note__mark">!</span>
        <span>
          显存需求与训练时长为经验估算，尚未在 GPU 节点实测；「风格兼容」等级需要对比测试报告才能声明，目前无任何底座已通过该验证。
        </span>
      </div>
    </>
  );
}

function TrainingConfig() {
  return (
    <>
      <div className="card__title" style={{ marginBottom: 'var(--sp-3)' }}>
        训练模式
      </div>
      <div className="grid grid--3">
        {[
          { name: '快速模式', desc: '决策风格适配 + 决策优化训练', mins: '约 45 分钟', vram: '12GB', recommended: true },
          { name: '精细模式', desc: '全量微调 + 压缩感知训练', mins: '约 3 小时', vram: '40GB', recommended: false },
          { name: '蒸馏模式', desc: '大模型能力迁移到小模型', mins: '约 2 小时', vram: '24GB', recommended: false },
        ].map((m) => (
          <div className="card" key={m.name}>
            <div className="card__title">
              {m.name}
              {m.recommended ? <span className="badge badge--info">推荐</span> : null}
            </div>
            <div className="card__sub" style={{ marginBottom: 'var(--sp-3)' }}>
              {m.desc}
            </div>
            <div className="kv">
              <dt>预计时长</dt>
              <dd>{m.mins}</dd>
              <dt>显存</dt>
              <dd>{m.vram}</dd>
            </div>
          </div>
        ))}
      </div>

      <details className="card" style={{ marginTop: 'var(--sp-4)' }}>
        <summary style={{ cursor: 'pointer', fontWeight: 600 }}>
          高级设置（默认折叠，业务人员无需改动）
        </summary>
        <div className="card__sub" style={{ marginTop: 'var(--sp-3)' }}>
          学习率、批次大小与迭代步数由系统按数据量与模型规模计算得出（原则 2：能不动就不动）。
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
          训练需要 NVIDIA GPU。本机没有 GPU，也没有安装训练依赖，因此本步骤的时长与显存均为估算值，未经验证。
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