import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

import { platformApi, type SceneTemplate } from '@/api/platform';
import { navigate } from '@/app/routes';
import { methodLabel } from './HomeView';

const LABEL_LABEL: Record<string, string> = {
  black: '黑样本（涉诈）',
  white: '白样本（正常）',
  gray: '灰样本（待定）',
};

/**
 * Scene gallery (需求方案.txt 5.1: 选择一个场景，或从零开始).
 *
 * The requirement's own mockup shows six templates plus a "from scratch" card,
 * and states that picking a template auto-fills 80% of the configuration. The
 * card therefore leads with what gets filled in -- fields, labels, recommended
 * base model and methods -- rather than a name and a summary.
 */
export function ScenesView() {
  const templates = useQuery({ queryKey: ['scene-templates'], queryFn: platformApi.templates });
  const scenes = useQuery({ queryKey: ['scenes'], queryFn: platformApi.scenes });
  const queryClient = useQueryClient();

  const [dialog, setDialog] = useState<
    { kind: 'template'; template: SceneTemplate } | { kind: 'custom' } | null
  >(null);

  const create = useMutation({
    mutationFn: platformApi.createScene,
    onSuccess: () => {
      setDialog(null);
      void queryClient.invalidateQueries({ queryKey: ['scenes'] });
    },
  });

  return (
    <>
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">场景展廊</h2>
          <span className="section__hint">
            选择模板后字段规范、标签体系、推荐模型与核心指标会自动填好
          </span>
        </div>

        {templates.isPending ? <div className="empty">加载中…</div> : null}
        {create.isError ? (
          <div className="note note--bad" style={{ marginBottom: 'var(--sp-4)' }}>
            <span className="note__mark">!</span>
            <span>创建失败：{create.error.message}</span>
          </div>
        ) : null}

        <div className="gallery">
          {(templates.data ?? []).map((template) => (
            <article className="scene-card" key={template.id}>
              <div className="scene-card__head">
                <span className="scene-card__name">{template.name}</span>
                {template.recommended ? <span className="badge badge--info">推荐</span> : null}
              </div>

              <p className="scene-card__summary">{template.summary}</p>

              <div className="scene-card__tags">
                <span className="badge badge--neutral">{template.field_count} 个字段</span>
                <span className="badge badge--neutral">黑 / 白 / 灰 三分类</span>
                {template.example_dataset ? (
                  <span className="badge badge--ok">含示例数据集</span>
                ) : null}
              </div>

              <div className="scene-card__meta">
                推荐模型 {template.recommended_base_model}
                <br />
                推荐方法 {template.recommended_methods.map(methodLabel).join(' + ')}
                <br />
                核心指标{' '}
                {template.core_metrics
                  .map((m) => ({ accuracy: '准确率', recall: '召回率', false_kill_rate: '误杀率', auc_roc: '判别能力', format_compliance: '规范率', p95_ms: '响应速度' }[m] ?? m))
                  .join(' / ')}
              </div>

              <div className="scene-card__foot">
                <span className="scene-card__meta">创建后可修改</span>
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={() => setDialog({ kind: 'template', template })}
                >
                  使用此场景
                </button>
              </div>
            </article>
          ))}

          <article
            className="scene-card"
            style={{ borderStyle: 'dashed', alignItems: 'flex-start', justifyContent: 'center' }}
          >
            <span className="scene-card__name">从零创建场景</span>
            <p className="scene-card__summary">
              自行定义字段规范、标签体系与核心指标。仅支持黑 / 白 / 灰三分类——
              这是产品的硬性约束，二分类会让召回率与待定样本的处理失去意义。
            </p>
            <div className="scene-card__foot">
              <span className="scene-card__meta">空白起点</span>
              <button type="button" className="btn" onClick={() => setDialog({ kind: 'custom' })}>
                创建
              </button>
            </div>
          </article>
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <h2 className="section__title">已创建场景</h2>
          <span className="section__hint">全链路版本关联从场景编号开始</span>
        </div>

        {scenes.data && scenes.data.length === 0 ? (
          <div className="empty">还没有创建任何场景。</div>
        ) : null}

        <div className="row-list">
          {(scenes.data ?? []).map((scene) => (
            <div className="row-item" key={scene.id}>
              <div className="row-item__main">
                <div className="row-item__title">
                  {scene.name}
                  <span className="badge badge--neutral" style={{ marginLeft: 8 }}>
                    {scene.source === 'template' ? '来自模板' : '自建'}
                  </span>
                </div>
                <div className="row-item__meta">
                  {scene.code} · {scene.field_count} 个字段 ·{' '}
                  {scene.labels.map((l) => LABEL_LABEL[l] ?? l).join(' / ')}
                </div>
              </div>
              <span className="badge badge--neutral">
                输出对齐 {scene.jev_format_compat ? '开' : '关'}
              </span>
              <button type="button" className="btn btn--primary" onClick={() => navigate('projects')}>
                新建项目
              </button>
            </div>
          ))}
        </div>
      </section>

      {dialog ? (
        <CreateSceneDialog
          dialog={dialog}
          pending={create.isPending}
          error={create.isError ? create.error.message : null}
          onCancel={() => setDialog(null)}
          onConfirm={(body) => create.mutate(body)}
        />
      ) : null}
    </>
  );
}

function CreateSceneDialog({
  dialog,
  pending,
  error,
  onCancel,
  onConfirm,
}: {
  dialog: { kind: 'template'; template: SceneTemplate } | { kind: 'custom' };
  pending: boolean;
  error: string | null;
  onCancel: () => void;
  onConfirm: (body: Parameters<typeof platformApi.createScene>[0]) => void;
}) {
  const template = dialog.kind === 'template' ? dialog.template : null;
  const [name, setName] = useState(template?.name ?? '');
  const [formatCompat, setFormatCompat] = useState(true);
  const [trainingCompat, setTrainingCompat] = useState(true);

  const needsName = !template || name.trim() === '';

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="创建场景">
      <div className="modal">
        <div className="modal__head">
          <div>
            <h2 className="modal__title">{template ? `使用「${template.name}」` : '从零创建场景'}</h2>
            {template ? <p className="card__sub">{template.summary}</p> : null}
          </div>
          <button type="button" className="btn btn--ghost" onClick={onCancel} aria-label="关闭">
            ✕
          </button>
        </div>

        <label className="field">
          <span className="field__label">场景名称</span>
          <input
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={template ? template.name : '例如：商户风险判定'}
          />
        </label>

        <div className="field">
          <span className="field__label">输出格式对齐</span>
          <div className="switch-row">
            <input
              type="checkbox"
              id="jev-format"
              checked={formatCompat}
              onChange={(e) => setFormatCompat(e.target.checked)}
            />
            <label className="switch-row__text" htmlFor="jev-format">
              <span className="switch-row__title">开启（推荐）</span>
              <span className="switch-row__desc">
                输入输出结构对齐标准格式，调用方拿到的是一致的字段集
              </span>
            </label>
          </div>

          <div className="switch-row">
            <input
              type="checkbox"
              id="jev-training"
              checked={trainingCompat}
              onChange={(e) => setTrainingCompat(e.target.checked)}
            />
            <label className="switch-row__text" htmlFor="jev-training">
              <span className="switch-row__title">训练范式对齐</span>
              <span className="switch-row__desc">
                训练方法、超参与评测口径对齐标准范式。两个开关互相独立，可分别开启
              </span>
            </label>
          </div>
        </div>

        <div className="field">
          <span className="field__label">标签体系</span>
          <div className="scene-card__tags">
            <span className="badge badge--neutral">黑样本（涉诈）</span>
            <span className="badge badge--neutral">白样本（正常）</span>
            <span className="badge badge--neutral">灰样本（待定）</span>
          </div>
          <span className="field__hint">三分类为硬性约束，不支持二分类</span>
        </div>

        {error ? (
          <div className="note note--bad">
            <span className="note__mark">!</span>
            <span>{error}</span>
          </div>
        ) : null}

        <div className="modal__actions">
          <button type="button" className="btn" onClick={onCancel}>
            取消
          </button>
          <button
            type="button"
            className="btn btn--primary"
            disabled={pending || needsName}
            onClick={() =>
              onConfirm({
                template_id: template?.id,
                name: name.trim() || template?.name,
                jev_format_compat: formatCompat,
                jev_training_compat: trainingCompat,
              })
            }
          >
            {pending ? '创建中…' : '创建场景'}
          </button>
        </div>
      </div>
    </div>
  );
}
