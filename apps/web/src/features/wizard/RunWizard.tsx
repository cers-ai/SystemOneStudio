import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { labelOrFallback as termLabel } from '@son/ui-terminology';
import { runApi, isBusy, type Run } from '@/api/runs';
import { navigate } from '@/app/routes';

const STAGES = ['场景', '种子数据', '数据合成', '训练配置', '训练', '测试验证', '导出部署'];
function suggestedStage(run: Run): number {
  if (!run.scene_code) return 0;
  if (!run.split_id) return 1;
  if (!run.synth_id) return 2;
  if (!run.training_config) return 3;
  if (!run.model_version_id) return 4;
  if (!run.evaluation_id) return 5;
  return 6;
}

export function RunWizard({ runId }: { runId: string | null }) {
  const client = useQueryClient();
  const [selected, setSelected] = useState<number | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [target, setTarget] = useState('');
  const [model, setModel] = useState('');
  const [method, setMethod] = useState('lora');
  const runQuery = useQuery({
    queryKey: ['run', runId], queryFn: () => runApi.run(runId as string), enabled: !!runId,
    refetchInterval: (query) => query.state.data && isBusy(query.state.data.state) ? 1500 : false,
  });
  const capabilities = useQuery({ queryKey: ['capabilities'], queryFn: runApi.capabilities });
  const refresh = (): void => { void client.invalidateQueries({ queryKey: ['run', runId] }); };
  const action = useMutation({
    mutationFn: async (kind: string) => {
      const id = runId as string;
      if (kind === 'scene') return runApi.setScene(id, { name, description });
      if (kind === 'prepare') return runApi.prepareData(id);
      if (kind === 'synth' || kind === 'skip') return runApi.synth(id, kind === 'skip' ? 0 : target === '' ? undefined : Number(target));
      if (kind === 'config') {
        if (!runQuery.data?.base_model_id) await runApi.setModel(id, model || capabilities.data?.base_models[0] || '');
        return runApi.setTrainingConfig(id, { method });
      }
      if (kind === 'start') return runApi.start(id);
      throw new Error('未知操作');
    },
    onSuccess: (run) => { client.setQueryData(['run', runId], run); setSelected(null); },
  });
  if (!runId) return <WorkspaceLanding />;
  if (runQuery.isPending) return <div className="empty">正在打开工作区…</div>;
  if (runQuery.isError) return <div className="note note--bad">{runQuery.error.message}</div>;
  const run = runQuery.data;
  const stage = selected ?? suggestedStage(run);
  const pending = action.isPending || isBusy(run.state);
  const button = (kind: string, text: string, enabled: boolean): React.ReactNode =>
    <button className="btn btn--primary" type="button" disabled={pending || !enabled} onClick={() => action.mutate(kind)}>{action.isPending ? '处理中…' : text}</button>;
  return <>
    <div className="workspace__heading"><div><span className="workspace__eyebrow">YOUR DECISION LAB</span><h1>{run.scene?.name ?? '定义你的第一个场景'}</h1><p>{run.scene?.description || '从一份有标注的种子数据开始，建立可验证、可交付的决策模型。'}</p></div><button className="btn btn--ghost" onClick={() => navigate('wizard')} type="button">全部工作区 ↗</button></div>
    <nav className="stagebar" aria-label="流程导航">{STAGES.map((label, index) => <button type="button" key={label} className={stage === index ? 'stagebar__item is-current' : 'stagebar__item'} aria-current={stage === index ? 'step' : undefined} onClick={() => { setSelected(index); action.reset(); }}><span>{index < suggestedStage(run) ? '✓' : String(index + 1).padStart(2, '0')}</span>{label}</button>)}</nav>
    <div className="workspace__grid"><section className="workspace__canvas">
      <div className="workspace__section"><span className="workspace__eyebrow">STEP {String(stage + 1).padStart(2, '0')}</span><h2>{STAGES[stage]}</h2></div>
      {action.isError ? <div className="note note--bad" role="alert">{action.error.message}</div> : null}
      {run.error_message ? <div className="note note--bad" role="alert">{run.error_message}</div> : null}
      {stage === 0 ? <div className="card">{run.scene_code ? <><h3>{run.scene?.name ?? run.scene_code}</h3><p>{run.scene?.description}</p><p className="card__sub">场景已保存。修改目标请新建工作区，保留当前数据与历史。</p></> : <><label className="field"><span className="field__label">场景名称</span><input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="例如：设备质检分级" maxLength={200} /></label><label className="field"><span className="field__label">描述你希望模型判断什么</span><textarea className="input" rows={5} value={description} onChange={(e) => setDescription(e.target.value)} placeholder="说明输入内容，以及三类决策分别代表什么。真实标注以种子文件为准。" maxLength={4000} /></label>{button('scene', '保存场景，开始准备数据 →', !!name.trim())}</>}</div> : null}
      {stage === 1 ? <>{run.dataset ? <div className="card"><h3>{run.dataset.original_filename}</h3><div className="workspace__numbers"><span><strong>{run.dataset.rows}</strong>种子样本</span><span><strong>{run.dataset.cols}</strong>数据字段</span></div>{!run.split ? button('prepare', '检查、脱敏并划分数据 →', true) : <QualityCard run={run} />}</div> : <UploadCard run={run} onUploaded={refresh} disabled={pending || !run.scene_code} />}{!run.scene_code ? <p className="card__sub">先定义场景，即可上传。</p> : null}</> : null}
      {stage === 2 ? <div className="card"><h3>保留判断关系的样本扩增</h3><p className="card__sub">从训练集整行重采样，特征、标签和依据一起保留。此方式会产生重复，不增加新知识；测试集与验证集保持原始种子。</p>{run.synth ? <><div className="workspace__numbers"><span><strong>{run.synth.rows}</strong>扩增样本</span></div>{run.synth.privacy?.notes.map((note) => <p className="card__sub" key={note}>{note}</p>)}</> : <><label className="field"><span className="field__label">扩增数量（留空采用数据建议）</span><input className="input" type="number" min="1" max="200000" value={target} onChange={(e) => setTarget(e.target.value)} /></label><div className="workspace__actions">{button('synth', '生成扩增数据 →', !!run.split_id && (target === '' || Number(target) > 0))}{button('skip', '仅用种子继续', !!run.split_id)}</div>{!run.split_id ? <p className="card__sub">先完成种子检查与划分。</p> : null}</>}</div> : null}
      {stage === 3 ? <div className="card"><h3>选择适合当前任务的训练配置</h3><label className="field"><span className="field__label">基础模型</span><select className="select" value={run.base_model_id ?? model} disabled={!!run.base_model_id} onChange={(e) => setModel(e.target.value)}><option value="">采用推荐模型</option>{capabilities.data?.base_models.map((id) => <option key={id} value={id}>{id}</option>)}</select></label><label className="field"><span className="field__label">训练方法</span><select className="select" value={method} onChange={(e) => setMethod(e.target.value)}>{capabilities.data?.training_methods.map((id) => <option value={id} key={id}>{termLabel(id)}</option>)}</select></label><p className="card__sub">模型与方法已接入代码；实际训练组合仍需 GPU 实测。本机实验先保存配置。</p>{run.training_config ? <p>配置已保存，可进入训练。</p> : button('config', '保存训练配置 →', !!run.synth_id)}</div> : null}
      {stage === 4 ? <div className="card"><h3>准备开始训练</h3><p className="card__sub">任务在服务器执行，数据和模型产物与当前工作区关联。本机数据实验环境没有训练设备，开始前会检查并说明原因。</p>{run.job ? <JobCard run={run} /> : button('start', '检查环境并开始训练 →', !!run.training_config)}{run.model_version ? <p>模型版本：{run.model_version.code}</p> : null}</div> : null}
      {stage === 5 ? <div className="card"><h3>用独立测试集检验模型</h3><p className="card__sub">完成真实训练并加载模型后，才能运行测试验证。当前{run.evaluation ? '有历史评测记录，实际执行接入正在完善。' : '尚无真实模型评测结果。'}</p><QualityCard run={run} /></div> : null}
      {stage === 6 ? <div className="card"><h3>交付你的模型</h3><p className="card__sub">真实模型产物就绪后，可导出模型包或启动预测服务。工作区备份与模型导出分别交付。</p><p>{run.model_version ? `模型版本：${run.model_version.code}，交付能力正在接入。` : '尚未生成可交付的模型。'}</p></div> : null}
    </section><aside className="workspace__context"><span className="workspace__eyebrow">WORKSPACE</span><h3>每一步都有据可查</h3><dl><dt>当前阶段</dt><dd>{STAGES[suggestedStage(run)]}</dd><dt>种子数据</dt><dd>{run.dataset ? `${run.dataset.rows} 条` : '等待上传'}</dd><dt>独立测试数据</dt><dd>{run.split ? `${run.split.test_rows} 条种子` : '等待划分'}</dd><dt>模型产物</dt><dd>{run.model_version?.code ?? '未生成'}</dd></dl><p>顶部步骤可自由查看；执行动作会检查前置数据。刷新后继续当前工作区。</p><details><summary>查看记录标识</summary><code>{run.id}</code>{run.lineage ? Object.entries(run.lineage).map(([key,value]) => <p key={key}><small>{key}</small><br /><code>{value}</code></p>) : null}</details></aside></div>
  </>;
}

function WorkspaceLanding() {
  const [name, setName] = useState('');
  const runs = useQuery({ queryKey: ['runs'], queryFn: () => runApi.runs() });
  const create = useMutation({ mutationFn: async () => {
    const project = await runApi.createProject(name.trim() || '我的决策实验');
    const run = await runApi.createRun(project.id);
    navigate(`run/${run.id}`);
  } });
  return <><section className="workspace__hero"><span className="workspace__eyebrow">FROM DATA TO DECISIONS</span><h1>让你的判断，<br /><em>成为模型的能力。</em></h1><p>定义场景、准备数据、训练验证，在一个工作区完成。</p><div className="workspace__create"><input className="input" aria-label="工作区名称" placeholder="给这次实验起个名字（可选）" value={name} onChange={(e) => setName(e.target.value)} /><button className="btn btn--primary" disabled={create.isPending} onClick={() => create.mutate()} type="button">{create.isPending ? '正在创建…' : '创建工作区 ↗'}</button></div>{create.isError ? <p role="alert">{create.error.message}</p> : null}</section><section className="workspace__recent"><h2>继续你的实验</h2>{runs.isError ? <div role="alert">{runs.error.message}</div> : null}{runs.isPending ? <p>正在读取工作区…</p> : null}{runs.data?.length === 0 ? <p className="card__sub">第一份数据，第一条可验证的判断。从这里开始。</p> : null}<div className="workspace__tiles">{runs.data?.map((run) => <button type="button" className="workspace__tile" key={run.id} onClick={() => navigate(`run/${run.id}`)}><span>↗</span><strong>{run.scene_code ?? '未定义场景'}</strong><small>{run.id}</small></button>)}</div></section></>;
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
        <div className="dropzone__hint">标签使用 black / white / gray 或 黑 / 白 / 灰；训练还需要每条数据的 reason 判定依据</div>
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
        任务在服务器后台执行，关闭页面后仍会继续。
      </p>
    </div>
  );
}
