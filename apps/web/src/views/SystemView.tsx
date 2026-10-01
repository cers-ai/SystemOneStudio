import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';

import { platformApi, type SceneSummary } from '@/api/platform';

interface ModelCard {
  model_id: string;
  display_name: string;
  family: string;
  params: string;
  min_gpu_memory: string;
  jev_compat_level: string | null;
  license: string;
  in_mvp_scope: boolean;
}

interface Shelf {
  recommended: ModelCard[];
  all: ModelCard[];
  notes: string[];
}

/**
 * 系统管理 (需求方案.txt 10.1 系统基础).
 *
 * Three things an operator needs: whether this node can actually run the heavy
 * steps, what is in the base model registry including unverified claims, and who
 * changed what.
 */
export function SystemView() {
  const status = useQuery({ queryKey: ['system-status'], queryFn: platformApi.systemStatus });
  const audit = useQuery({ queryKey: ['audit'], queryFn: platformApi.audit });
  const models = useQuery({ queryKey: ['models'], queryFn: platformApi.models });

  const [showOutOfScope, setShowOutOfScope] = useState(false);

  const shelf = models.data as Shelf | undefined;
  const visibleModels = (shelf?.all ?? []).filter((m) => showOutOfScope || m.in_mvp_scope);

  return (
    <>
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">运行状态</h2>
          <span className="section__hint">决定哪些能力在本节点可用</span>
        </div>

        <div className="grid grid--3">
          <div className="stat">
            <div className="stat__label">图形处理器</div>
            <div className="stat__value" style={{ fontSize: 'var(--fs-lg)' }}>
              {status.data?.gpu_available ? '已就绪' : '未检测到'}
            </div>
            <div className="stat__foot">{status.data?.gpu_detail ?? '检测中…'}</div>
          </div>

          <div className="stat">
            <div className="stat__label">数据持久化</div>
            <div className="stat__value" style={{ fontSize: 'var(--fs-lg)' }}>
              {status.data?.persistence === 'in-memory' ? '进程内存' : '已接数据库'}
            </div>
            <div className="stat__foot">重启后未落库的数据会丢失</div>
          </div>

          <div className="stat">
            <div className="stat__label">服务</div>
            <div className="stat__value" style={{ fontSize: 'var(--fs-lg)' }}>
              {(status.data?.services ?? []).filter((s) => s.ok).length} /{' '}
              {status.data?.services.length ?? 0}
              <span className="stat__unit"> 可用</span>
            </div>
            <div className="stat__foot">
              {(status.data?.services ?? []).map((s) => (
                <div key={s.name} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <span className={s.ok ? 'badge badge--ok' : 'badge badge--warn'}>
                    {s.ok ? '就绪' : '未就绪'}
                  </span>
                  {s.name}
                </div>
              ))}
            </div>
          </div>
        </div>

        {(status.data?.notes ?? []).map((note) => (
          <div className="note note--info" key={note} style={{ marginTop: 'var(--sp-3)' }}>
            <span className="note__mark">i</span>
            <span>{note}</span>
          </div>
        ))}
      </section>

      <section className="section">
        <div className="section__head">
          <h2 className="section__title">底座模型注册表</h2>
          <label style={{ fontSize: 'var(--fs-sm)' }}>
            <input
              type="checkbox"
              checked={showOutOfScope}
              onChange={(e) => setShowOutOfScope(e.target.checked)}
              style={{ marginRight: 6 }}
            />
            显示 MVP 范围外的型号
          </label>
        </div>

        {models.isPending ? <div className="empty">加载中…</div> : null}

        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">模型</th>
                <th scope="col">家族</th>
                <th scope="col">规格</th>
                <th scope="col">显存</th>
                <th scope="col">输出规范兼容度</th>
                <th scope="col">许可</th>
                <th scope="col">范围</th>
              </tr>
            </thead>
            <tbody>
              {visibleModels.map((model) => (
                <tr key={model.model_id}>
                  <td>{model.display_name}</td>
                  <td>{model.family}</td>
                  <td>{model.params}</td>
                  <td>{model.min_gpu_memory}</td>
                  <td>
                    {model.jev_compat_level ? (
                      <span className="badge badge--neutral">{model.jev_compat_level}</span>
                    ) : (
                      <span className="badge badge--warn">待实测</span>
                    )}
                  </td>
                  <td>{model.license}</td>
                  <td>
                    {model.in_mvp_scope ? (
                      <span className="badge badge--ok">在范围内</span>
                    ) : (
                      <span className="badge badge--neutral">范围外</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {(shelf?.notes ?? []).map((note) => (
          <div className="note note--warn" key={note} style={{ marginTop: 'var(--sp-3)' }}>
            <span className="note__mark">!</span>
            <span>{note}</span>
          </div>
        ))}

        <p className="card__sub" style={{ marginTop: 'var(--sp-3)' }}>
          「风格兼容」（L3）等级必须有对比测试报告才能写入。当前没有任何型号通过该验证，
          校验器会直接拒绝未经实测的声明。
        </p>
      </section>

      <section className="section">
        <div className="section__head">
          <h2 className="section__title">审计日志</h2>
          <span className="section__hint">项目、场景与兼容等级的变更记录</span>
        </div>

        {audit.data && audit.data.length === 0 ? (
          <div className="empty">暂无变更记录。创建项目或场景后会出现在这里。</div>
        ) : null}

        <div className="audit-list">
          {(audit.data ?? []).map((entry) => (
            <div className="audit-item" key={entry.id}>
              <span className="audit-item__time">{formatTime(entry.ts)}</span>
              <span>
                <span className="badge badge--neutral">{actionLabel(entry.action)}</span>
              </span>
              <span>
                <strong>{entry.target}</strong> · {entry.detail}
              </span>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

function actionLabel(action: string): string {
  return (
    {
      create_scene: '创建场景',
      update_jev_flags: '调整对齐',
      create_project: '新建项目',
    }[action] ?? action
  );
}

function formatTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString('zh-CN', { hour12: false });
}

export type { SceneSummary };
