import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

import { label as termLabel } from '@son/ui-terminology';

import {
  isBusy,
  runApi,
  STEP_LABELS,
  type Capabilities,
  type Run,
} from '@/api/runs';

/**
 * The wizard, as a projection of the backend Run.
 *
 * There is deliberately no local `completedSteps` state. Each control calls a
 * backend action and the response is the new Run; nothing advances locally. That
 * is the whole point of 改造开发方案.md 2.3 -- the old wizard marked steps done
 * because the user clicked, which is how a step could look finished with no
 * dataset behind it.
 *
 * While a job owns the run, the Run is polled every 1.5s rather than opened over
 * a WebSocket (改造开发方案.md 16).
 */
const POLL_MS = 1500;

export function RunWizard({ runId }: { runId: string | null }) {
  const queryClient = useQueryClient();
  const runQuery = useQuery({
    queryKey: ['run', runId],
    queryFn: () => runApi.run(runId as string),
    enabled: runId !== null,
  });

  const capabilities = useQuery({
    queryKey: ['capabilities'],
    queryFn: runApi.capabilities,
  });

  // Poll only while a job is running. 1.5s is enough to feel live without
  // hammering the API while training runs for an hour.
  useEffect(() => {
    const state = runQuery.data?.state;
    if (!runId || !state || !isBusy(state)) return;
    const timer = setInterval(() => {
      void queryClient.invalidateQueries({ queryKey: ['run', runId] });
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [runId, runQuery.data?.state, queryClient]);

  const refresh = (): void => {
    if (runId) void queryClient.invalidateQueries({ queryKey: ['run', runId] });
  };

  const stepAction = useMutation({
    mutationFn: async (kind: string) => {
      switch (kind) {
        case 'scene':
          return runApi.setScene(runId as string, 'fraud_account');
        case 'prepare':
          return runApi.prepareData(runId as string);
        case 'synth':
          return runApi.synth(runId as string, 1000);
        case 'model':
          return runApi.setModel(runId as string, (capabilities.data?.base_models[0] ?? ''));
        case 'config':
          return runApi.setTrainingConfig(runId as string, { method: 'lora' });
        case 'start':
          return runApi.start(runId as string);
        default:
          throw new Error(`未知步骤动作：${kind}`);
      }
    },
    onSuccess: refresh,
  });

  if (!runId) {
    return (
      <div className="empty">
        请先选择一个 Run。可在「项目」页新建，或从下面选择一个已有的 Run。
        <RunPicker onPick={refresh} />
      </div>
    );
  }

  if (runQuery.isPending) return <div className="empty">加载 Run…</div>;
  if (runQuery.isError) {
    return (
      <div className="note note--bad">
        <span className="note__mark">!</span>
        <span>读取 Run 失败：{runQuery.error.message}</span>
      </div>
    );
  }

  const run = runQuery.data;
  const busy = isBusy(run.state);

  return (
    <>
      <StepRail run={run} />

      <div className="step">
        <header className="step__head">
          <div className="step__eyebrow">
            第 {run.current_step} 步 / 共 7 · 状态 {run.state}
          </div>
          <h1 className="step__title">{STEP_LABELS[run.current_step] ?? run.state}</h1>
          <p className="step__hint">
            当前 Run 状态由后端决定。页面上没有"完成"按钮——每一步都调用后端动作，
            资产真正生成后 Run 才会前进。
          </p>
        </header>

        {run.error_message ? (
          <div className="note note--bad" style={{ marginBottom: 'var(--sp-4)' }}>
            <span className="note__mark">!</span>
            <span>{run.error_message}</span>
          </div>
        ) : null}

        <StepPanel
          run={run}
          capabilities={capabilities.data}
          busy={busy}
          pending={stepAction.isPending}
          error={stepAction.isError ? stepAction.error.message : null}
          onScene={() => stepAction.mutate('scene')}
          onPrepare={() => stepAction.mutate('prepare')}
          onSynth={() => stepAction.mutate('synth')}
          onModel={() => stepAction.mutate('model')}
          onConfig={() => stepAction.mutate('config')}
          onStart={() => stepAction.mutate('start')}
          onUpload={() => refresh()}
        />

        {run.lineage ? (
          <div className="card" style={{ marginTop: 'var(--sp-4)' }}>
            <div className="card__title">血缘</div>
            <div className="card__sub">由系统根据各步骤实际产出的资产自动生成，不接受前端填写。</div>
            <div className="kv">
              {Object.entries(run.lineage).map(([key, value]) => (
                <div key={key} style={{ display: 'contents' }}>
                  <dt>{LINEAGE_LABEL[key] ?? key}</dt>
                  <dd className="mono">{value}</dd>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </>
  );
}

const LINEAGE_LABEL: Record<string, string> = {
  run: 'Run',
  scene: '场景',
  dataset: '数据集',
  split: '划分',
  synth: '合成',
  base_model: '基座模型',
  job: '任务',
  model_version: '模型版本',
};

function StepRail({ run }: { run: Run }) {
  return (
    <nav className="rail" aria-label="流程步骤">
      <div className="rail__heading">训练流程</div>
      <div className="rail__progress">
        <div className="rail__progress-track">
          <div
            className="rail__progress-fill"
            style={{ width: `${Math.round(((run.current_step - 1) / 7) * 100)}%` }}
          />
        </div>
        <div className="rail__progress-label">
          第 {run.current_step} 步 · 后端状态 {run.state}
        </div>
      </div>
      <ol className="rail__list">
        {[1, 2, 3, 4, 5, 6, 7].map((step) => {
          const done = run.current_step > step;
          const current = run.current_step === step;
          return (
            <li key={step} className={`rail__item${done ? ' rail__item--done' : ''}`}>
              <span
                className="rail__button"
                aria-current={current ? 'step' : undefined}
              >
                <span className="rail__marker">{done ? '✓' : step}</span>
                <span className="rail__label">{STEP_LABELS[step]}</span>
              </span>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

interface PanelProps {
  run: Run;
  capabilities?: Capabilities;
  busy: boolean;
  pending: boolean;
  error: string | null;
  onScene: () => void;
  onPrepare: () => void;
  onSynth: () => void;
  onModel: () => void;
  onConfig: () => void;
  onStart: () => void;
  onUpload: () => void;
}

function StepPanel(props: PanelProps) {
  const { run, busy, pending } = props;
  const disabled = busy || pending;

  switch (run.state) {
    case 'CREATED':
      return (
        <ActionCard
          title="选择场景"
          detail="反诈账户判定是本阶段唯一打通的场景模板。"
          action="选择「反诈账户判定」"
          onClick={props.onScene}
          disabled={disabled}
          error={props.error}
        />
      );

    case 'SCENE_READY':
      return <UploadCard run={run} onUploaded={props.onUpload} disabled={disabled} />;

    case 'DATASET_READY':
      return (
        <ActionCard
          title="数据治理与划分"
          detail="自动脱敏、质量评分、按 7:1.5:1.5 划分，并断言测试集不含合成数据。"
          action="执行数据治理"
          onClick={props.onPrepare}
          disabled={disabled}
          error={props.error}
        />
      );

    case 'DATA_QUALITY_READY':
      return (
        <>
          <QualityCard run={run} />
          <ActionCard
            title="生成合成数据"
            detail="从训练集扩增样本；测试集永远是原始上传数据。"
            action="开始合成"
            onClick={props.onSynth}
            disabled={disabled}
            error={props.error}
          />
        </>
      );

    case 'TRAIN_SET_READY':
      return (
        <ActionCard
          title="选择基座模型"
          detail={`本阶段支持：${props.capabilities?.base_models.join('、') ?? '加载中'}`}
          action="选择推荐模型"
          onClick={props.onModel}
          disabled={disabled}
          error={props.error}
        />
      );

    case 'MODEL_SELECTED':
      return (
        <ActionCard
          title="配置训练方法"
          detail={`本阶段支持${termLabel('lora')}；低显存快速适配与决策优化训练属于后续阶段。`}
          action="使用推荐配置"
          onClick={props.onConfig}
          disabled={disabled}
          error={props.error}
        />
      );

    case 'TRAINING_CONFIGURED':
      return (
        <ActionCard
          title="开始训练"
          detail="提交后台任务，由 worker 执行训练、合并与压缩导出。此操作不会阻塞页面。"
          action="开始训练"
          onClick={props.onStart}
          disabled={disabled}
          error={props.error}
        />
      );

    case 'QUEUED':
    case 'TRAINING':
    case 'MERGING':
    case 'QUANTIZING':
    case 'EVALUATING':
    case 'DEPLOYING':
      return <JobCard run={run} />;

    case 'MODEL_READY':
      return (
        <div className="card">
          <div className="card__title">模型已就绪</div>
          <div className="card__sub">
            部署需要在具备本地推理运行时的节点上执行。点击部署会真实启动服务并做健康检查，
            通过后才标记为服务中。
          </div>
          <div className="kv">
            <dt>模型版本</dt>
            <dd className="mono">{run.model_version?.code ?? '—'}</dd>
            <dt>适配器</dt>
            <dd className="mono">{run.model_version?.adapter_path ?? '—'}</dd>
          </div>
        </div>
      );

    case 'SERVING':
      return (
        <div className="card">
          <div className="card__title">服务中</div>
          <div className="kv">
            <dt>端口</dt>
            <dd>{run.deployment?.port ?? '—'}</dd>
            <dt>状态</dt>
            <dd>{run.deployment?.status ?? '—'}</dd>
          </div>
        </div>
      );

    case 'FAILED':
      return (
        <div className="card">
          <div className="card__title">运行失败</div>
          <div className="note note--bad">
            <span className="note__mark">!</span>
            <span>{run.error_message ?? '未记录原因'}</span>
          </div>
          {run.job?.log_path ? (
            <p className="card__sub" style={{ marginTop: 'var(--sp-3)' }}>
              日志：<span className="mono">{run.job.log_path}</span>
            </p>
          ) : null}
        </div>
      );

    default:
      return <div className="empty">未知状态：{run.state}</div>;
  }
}

function ActionCard({
  title,
  detail,
  action,
  onClick,
  disabled,
  error,
}: {
  title: string;
  detail: string;
  action: string;
  onClick: () => void;
  disabled?: boolean;
  error?: string | null;
}) {
  return (
    <div className="card">
      <div className="card__title">{title}</div>
      <div className="card__sub">{detail}</div>
      {error ? (
        <div className="note note--bad">
          <span className="note__mark">!</span>
          <span>{error}</span>
        </div>
      ) : null}
      <button type="button" className="btn btn--primary" onClick={onClick} disabled={disabled}>
        {action}
      </button>
    </div>
  );
}

function UploadCard({
  run,
  onUploaded,
  disabled,
}: {
  run: Run;
  onUploaded: () => void;
  disabled: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onFile = async (file: File): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      await runApi.uploadDataset(run.id, file);
      onUploaded();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <div className="card__title">上传种子数据</div>
      <div className="card__sub">
        只需上传一次。之后的治理、划分与合成都引用同一个数据集，不会再要求上传。
      </div>
      {error ? (
        <div className="note note--bad">
          <span className="note__mark">!</span>
          <span>{error}</span>
        </div>
      ) : null}
      <label className="dropzone">
        <div className="dropzone__icon" aria-hidden="true">
          ⬆
        </div>
        <div className="dropzone__title">{busy ? '上传中…' : '点击选择 CSV'}</div>
        <div className="dropzone__hint">需含 black / white / gray 三分类标签列</div>
        <input
          className="dropzone__input"
          type="file"
          accept=".csv,text/csv"
          disabled={disabled || busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void onFile(file);
          }}
        />
      </label>
    </div>
  );
}

function QualityCard({ run }: { run: Run }) {
  const q = run.split?.quality;
  if (!q) return null;
  return (
    <div className="card">
      <div className="card__title">数据质量报告</div>
      <div className="grid grid--4">
        <div className="stat">
          <div className="stat__label">综合评分</div>
          <div className="stat__value">{q.score}</div>
        </div>
        {q.dimensions.map((d) => (
          <div className="stat" key={d.name}>
            <div className="stat__label">{d.name}</div>
            <div className="stat__value" style={{ fontSize: 'var(--fs-xl)' }}>
              {d.value}
            </div>
            <div className="stat__foot">
              <span className={toneBadge(d.verdict)}>{d.verdict}</span>
            </div>
          </div>
        ))}
      </div>
      <div className="kv" style={{ marginTop: 'var(--sp-4)' }}>
        <dt>训练 / 验证 / 测试</dt>
        <dd>
          {run.split?.train_rows} / {run.split?.valid_rows} / {run.split?.test_rows}
        </dd>
        <dt>测试集合成样本</dt>
        <dd>
          {run.split?.test_synth_rows === 0 ? (
            <span className="badge badge--ok">0 条</span>
          ) : (
            <span className="badge badge--bad">{run.split?.test_synth_rows} 条</span>
          )}
        </dd>
      </div>
      {q.masked_fields.length > 0 ? (
        <div className="note note--ok" style={{ marginTop: 'var(--sp-4)' }}>
          <span className="note__mark">✓</span>
          <span>已脱敏：{q.masked_fields.join('、')}（不可还原）</span>
        </div>
      ) : null}
      {q.suggestions.length > 0 ? (
        <ul className="bullets" style={{ marginTop: 'var(--sp-4)' }}>
          {q.suggestions.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function toneBadge(verdict: string): string {
  return ['充足', '良好', '均衡'].includes(verdict) ? 'badge badge--ok' : 'badge badge--warn';
}

function JobCard({ run }: { run: Run }) {
  const job = run.job;
  if (!job) return <div className="empty">任务信息尚未就绪</div>;

  return (
    <div className="card">
      <div className="card__title">
        {job.stage ?? '执行中'}
        <span className={`badge ${job.status === 'FAILED' ? 'badge--bad' : 'badge--info'}`}>
          {job.status}
        </span>
      </div>
      <div className="card__sub">{job.message ?? '正在执行'}</div>

      <div className="rail__progress-track" style={{ marginTop: 'var(--sp-3)' }}>
        <div className="rail__progress-fill" style={{ width: `${job.progress}%` }} />
      </div>
      <div className="stat__foot">进度 {job.progress}%</div>

      {Object.keys(job.metrics).length > 0 ? (
        <div className="kv" style={{ marginTop: 'var(--sp-3)' }}>
          {Object.entries(job.metrics).map(([key, value]) => (
            <div key={key} style={{ display: 'contents' }}>
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </div>
      ) : null}

      <p className="card__sub" style={{ marginTop: 'var(--sp-3)' }}>
        训练由后台 worker 执行，页面每 1.5 秒轮询一次进度。可以离开本页。
      </p>
    </div>
  );
}

function RunPicker({ onPick }: { onPick: () => void }) {
  const runs = useQuery({ queryKey: ['runs'], queryFn: () => runApi.runs() });
  const projects = useQuery({ queryKey: ['projects'], queryFn: runApi.projects });
  const queryClient = useQueryClient();

  const startNew = async (): Promise<void> => {
    let projectId = projects.data?.[0]?.id;
    if (!projectId) {
      const project = await runApi.createProject(`新建项目 ${new Date().toLocaleDateString('zh-CN')}`);
      projectId = project.id;
    }
    await runApi.createRun(projectId);
    await queryClient.invalidateQueries({ queryKey: ['runs'] });
    onPick();
  };

  return (
    <div style={{ marginTop: 'var(--sp-4)', textAlign: 'left' }}>
      <button type="button" className="btn btn--primary" onClick={() => void startNew()}>
        新建 Run
      </button>
      {runs.data && runs.data.length > 0 ? (
        <ul className="row-list" style={{ marginTop: 'var(--sp-4)' }}>
          {runs.data.slice(0, 8).map((run) => (
            <li className="row-item" key={run.id}>
              <div className="row-item__main">
                <div className="row-item__title">第 {run.current_step} 步</div>
                <div className="row-item__meta">
                  {run.state} · {run.id}
                </div>
              </div>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  window.location.hash = `/run/${run.id}`;
                  onPick();
                }}
              >
                打开
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}