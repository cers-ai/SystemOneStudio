import { useQuery } from '@tanstack/react-query';

import { platformApi } from '@/api/platform';
import { navigate, type RoutePath } from '@/app/routes';

const METRIC_LABEL: Record<string, string> = {
  accuracy: '决策准确率',
  recall: '召回率',
  false_kill_rate: '误杀率',
  auc_roc: '综合判别能力',
  format_compliance: '输出规范率',
  p95_ms: '响应速度（95% 请求）',
};

export function methodLabel(id: string): string {
  return (
    {
      lora: '快速风格适配',
      dpo: '决策优化训练',
      sft: '基础风格适配',
      qlora: '低显存快速适配',
    }[id] ?? id
  );
}

/**
 * Home: what the platform is, what this deployment can do, and the three things
 * a user actually wants next.
 *
 * The capability block reports the real GPU state rather than a generic
 * "online" badge. On a machine without a GPU the honest answer is that training
 * cannot run here, and hiding that behind a green light is the kind of thing
 * that wastes an afternoon.
 */
export function HomeView({ onNavigate }: { onNavigate: (p: RoutePath) => void }) {
  const status = useQuery({ queryKey: ['system-status'], queryFn: platformApi.systemStatus });
  const templates = useQuery({ queryKey: ['scene-templates'], queryFn: platformApi.templates });
  const projects = useQuery({ queryKey: ['projects'], queryFn: platformApi.projects });

  const gpu = status.data?.gpu_available ?? false;

  return (
    <>
      <section className="hero">
        <h2 className="hero__title">30 分钟训出可用的决策模型</h2>
        <p className="hero__lead">
          上传样本数据、选一个场景模板，平台自动完成数据治理、样本扩增、模型训练、
          压缩导出与部署，输出带判定依据的决策模型，并向业务系统提供统一接口。
          不需要算法团队介入。
        </p>
        <div className="hero__actions">
          <button
            type="button"
            className="btn btn--primary btn--lg"
            onClick={() => onNavigate('scenes')}
          >
            从场景模板开始
          </button>
          <button type="button" className="btn btn--lg" onClick={() => onNavigate('wizard')}>
            直接进入训练流程
          </button>
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <h2 className="section__title">本节点能力</h2>
          <span className="section__hint">
            数据治理与效果评测已就绪；训练与推理取决于 GPU
          </span>
        </div>

        <div className="grid grid--3">
          <div className="stat">
            <div className="stat__label">图形处理器</div>
            <div className="stat__value" style={{ fontSize: 'var(--fs-lg)' }}>
              {gpu ? '已就绪' : '未检测到'}
            </div>
            <div className="stat__foot">
              <span className={gpu ? 'badge badge--ok' : 'badge badge--warn'}>
                {gpu ? '可训练' : '本节点不可训练'}
              </span>
              <div style={{ marginTop: '6px' }}>{status.data?.gpu_detail ?? '检测中…'}</div>
            </div>
          </div>

          <div className="stat">
            <div className="stat__label">底座模型</div>
            <div className="stat__value">
              {status.data?.in_scope_model_count ?? '—'}
              <span className="stat__unit"> 个在范围内</span>
            </div>
            <div className="stat__foot">
              共 {status.data?.model_count ?? '—'} 个已登记，更大规格标记为暂不可选
            </div>
          </div>

          <div className="stat">
            <div className="stat__label">场景模板</div>
            <div className="stat__value">
              {templates.data?.length ?? '—'}
              <span className="stat__unit"> 个</span>
            </div>
            <div className="stat__foot">覆盖反诈、风控、合规、营销、信贷、内容安全</div>
          </div>
        </div>

        {!gpu ? (
          <div className="note note--warn" style={{ marginTop: 'var(--sp-4)' }}>
            <span className="note__mark">!</span>
            <span>
              本节点没有图形处理器。数据治理、样本扩增与效果评测可以正常进行；
              训练、压缩导出、推理与响应速度指标需要在具备 GPU 的节点上执行，
              相关能力已实现但未在此验证。
            </span>
          </div>
        ) : null}
      </section>

      <section className="section">
        <div className="section__head">
          <h2 className="section__title">最近项目</h2>
          <button type="button" className="btn btn--ghost" onClick={() => onNavigate('projects')}>
            查看全部
          </button>
        </div>

        {projects.isPending ? <div className="empty">加载中…</div> : null}

        {projects.data && projects.data.length === 0 ? (
          <div className="empty">
            还没有项目。先从场景展廊选一个模板，再新建项目即可开始。
          </div>
        ) : null}

        <div className="row-list">
          {(projects.data ?? []).slice(0, 5).map((project) => (
            <div className="row-item" key={project.id}>
              <div className="row-item__main">
                <div className="row-item__title">{project.name}</div>
                <div className="row-item__meta">
                  {project.code}
                  {project.scene_code ? ` · 场景 ${project.scene_code}` : ' · 未绑定场景'} ·
                  模式 {modeLabel(project.mode)}
                </div>
              </div>
              <button
                type="button"
                className="btn"
                onClick={() => onNavigate('wizard')}
              >
                继续
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <h2 className="section__title">平台遵循的核心指标</h2>
          <span className="section__hint">性能指标优先于效果指标</span>
        </div>
        <div className="grid grid--3">
          {[
            { k: 'p95_ms', target: '≤ 100ms' },
            { k: 'accuracy', target: '≥ 90%' },
            { k: 'false_kill_rate', target: '≤ 5%' },
          ].map((metric) => (
            <div className="stat" key={metric.k}>
              <div className="stat__label">{METRIC_LABEL[metric.k]}</div>
              <div className="stat__value" style={{ fontSize: 'var(--fs-xl)' }}>
                {metric.target}
              </div>
            </div>
          ))}
        </div>
        <p className="card__sub" style={{ marginTop: 'var(--sp-3)' }}>
          响应速度目标与「判定依据」的长度上限存在冲突，口径尚未定案，
          因此评测不会对该项给出达标结论。
        </p>
      </section>
    </>
  );
}

export function modeLabel(mode: string): string {
  return (
    { wizard: '向导模式', rapid: '极速模式', canvas: '画布模式', expert: '专家模式' }[mode] ??
    mode
  );
}

export { navigate };
