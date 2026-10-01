import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

import { platformApi } from '@/api/platform';
import { navigate } from '@/app/routes';
import { modeLabel } from './HomeView';

/** 项目管理 (需求方案.txt 10.1 P0: 新建、模板、模式). */
export function ProjectsView() {
  const projects = useQuery({ queryKey: ['projects'], queryFn: platformApi.projects });
  const scenes = useQuery({ queryKey: ['scenes'], queryFn: platformApi.scenes });
  const queryClient = useQueryClient();

  const [name, setName] = useState('');
  const [sceneId, setSceneId] = useState('');
  const [mode, setMode] = useState('wizard');

  const create = useMutation({
    mutationFn: platformApi.createProject,
    onSuccess: () => {
      setName('');
      setSceneId('');
      void queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });

  return (
    <>
      <section className="section">
        <div className="section__head">
          <h2 className="section__title">新建项目</h2>
          <span className="section__hint">
            极速模式只需完成前两步，其余由系统自动采用推荐值
          </span>
        </div>

        <div className="card">
          <div className="grid grid--3">
            <label className="field">
              <span className="field__label">项目名称</span>
              <input
                className="input"
                value={name}
                placeholder="例如：Q3 反诈账户判定"
                onChange={(e) => setName(e.target.value)}
              />
            </label>

            <label className="field">
              <span className="field__label">绑定场景</span>
              <select
                className="select"
                value={sceneId}
                onChange={(e) => setSceneId(e.target.value)}
              >
                <option value="">暂不绑定</option>
                {(scenes.data ?? []).map((scene) => (
                  <option key={scene.id} value={scene.id}>
                    {scene.name}（{scene.code}）
                  </option>
                ))}
              </select>
              {(scenes.data ?? []).length === 0 ? (
                <span className="field__hint">还没有场景，请先到场景展廊创建</span>
              ) : null}
            </label>

            <label className="field">
              <span className="field__label">交互模式</span>
              <select className="select" value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="wizard">向导模式（七步）</option>
                <option value="rapid">极速模式（两步）</option>
                <option value="expert">专家模式</option>
              </select>
              <span className="field__hint">模式可随时切换，底层流程一致</span>
            </label>
          </div>

          {create.isError ? (
            <div className="note note--bad">
              <span className="note__mark">!</span>
              <span>创建失败：{create.error.message}</span>
            </div>
          ) : null}

          <div className="hero__actions">
            <button
              type="button"
              className="btn btn--primary"
              disabled={!name.trim() || create.isPending}
              onClick={() =>
                create.mutate({ name: name.trim(), scene_id: sceneId || undefined, mode })
              }
            >
              {create.isPending ? '创建中…' : '创建项目'}
            </button>
            <button type="button" className="btn" onClick={() => navigate('scenes')}>
              去场景展廊
            </button>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <h2 className="section__title">全部项目</h2>
          <span className="section__hint">
            数据当前保存在进程内存中，重启后会清空
          </span>
        </div>

        {projects.isPending ? <div className="empty">加载中…</div> : null}
        {projects.data && projects.data.length === 0 ? (
          <div className="empty">还没有项目。新建一个即可开始。</div>
        ) : null}

        <div className="row-list">
          {(projects.data ?? []).map((project) => (
            <div className="row-item" key={project.id}>
              <div className="row-item__main">
                <div className="row-item__title">{project.name}</div>
                <div className="row-item__meta">
                  {project.code}
                  {project.scene_code ? ` · 场景 ${project.scene_code}` : ' · 未绑定场景'} ·{' '}
                  {modeLabel(project.mode)} · 创建于 {formatTime(project.created_at)}
                </div>
              </div>
              <span className="badge badge--neutral">{modeLabel(project.mode)}</span>
              <button type="button" className="btn" onClick={() => navigate('wizard')}>
                进入流程
              </button>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

function formatTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString('zh-CN', { hour12: false });
}
