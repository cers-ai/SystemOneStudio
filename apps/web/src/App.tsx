import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

import { assistantApi } from '@/api/assistant';
import { AssistantPanel } from '@/features/assistant/AssistantPanel';
import { currentPath, navigate, routeTitle, ROUTES, type RoutePath } from '@/app/routes';
import { HomeView } from '@/views/HomeView';
import { ProjectsView } from '@/views/ProjectsView';
import { ScenesView } from '@/views/ScenesView';
import { SystemView } from '@/views/SystemView';
import { WizardView } from '@/views/WizardView';
import './styles/global.css';
import './styles/app-shell.css';
import './styles/assistant.css';

export function App() {
  const [path, setPath] = useState<RoutePath>(() => currentPath());
  const [showStatus, setShowStatus] = useState(false);
  const [assistantOpen, setAssistantOpen] = useState(true);

  useEffect(() => {
    const onChange = (): void => setPath(currentPath());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  // A bare path (#/ or #) should land on the home view rather than a blank
  // shell, since that is where a first-time visitor is expected to start.
  useEffect(() => {
    if (path === '' && window.location.hash !== '#/') navigate('');
  }, [path]);

  const assistantSettings = useQuery({
    queryKey: ['assistant-settings'],
    queryFn: assistantApi.settings,
  });

  // Hidden entirely when the operator disabled it, rather than showing a panel
  // that cannot answer. Not shown on 系统管理 either: that is where you
  // configure the assistant, and a chat box next to its own settings is noise.
  const assistantVisible =
    assistantOpen && assistantSettings.data?.enabled !== false && path !== 'system';

  return (
    <div className="shell">
      <aside className="shell__nav">
        <div className="shell__brand">
          <strong>SystemOneStudio</strong>
          <span>决策模型训练平台</span>
        </div>

        <nav className="shell__links" aria-label="主导航">
          {ROUTES.map((route) => (
            <button
              key={route.path}
              type="button"
              className="shell__link"
              aria-current={path === route.path ? 'page' : undefined}
              onClick={() => navigate(route.path)}
            >
              <span className="shell__link-icon" aria-hidden="true">
                {route.icon}
              </span>
              {route.label}
            </button>
          ))}
        </nav>

        <div className="shell__nav-foot">
          <p className="shell__nav-note">
            极速模式下只需完成前两步，其余由系统自动采用推荐值。
          </p>
        </div>
      </aside>

      <div className="shell__content">
        <header className="shell__header">
          <h1 className="shell__title">{routeTitle(path)}</h1>
          <span className="app__header-spacer" />
          <button
            type="button"
            className="btn btn--ghost"
            onClick={() => setAssistantOpen((v) => !v)}
            aria-pressed={assistantVisible}
          >
            {assistantVisible ? '收起助手' : '展开助手'}
          </button>
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
          <div style={{ padding: 'var(--sp-5) var(--sp-8) 0' }}>
            <SystemStatusInline />
          </div>
        ) : null}

        <div className={assistantVisible ? 'shell__view with-assistant' : 'shell__view'}>
          <div>
            {path === '' ? <HomeView onNavigate={navigate} /> : null}
            {path === 'projects' ? <ProjectsView /> : null}
            {path === 'scenes' ? <ScenesView /> : null}
            {path === 'wizard' ? <WizardView /> : null}
            {path === 'system' ? <SystemView /> : null}
          </div>

          {assistantVisible ? (
            <AssistantPanel
              currentStep={path === 'wizard' ? 'training_configured' : undefined}
              onClose={() => setAssistantOpen(false)}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}

function SystemStatusInline() {
  const status = useQuery({
    queryKey: ['system-status'],
    queryFn: () => fetch('/api/system/status').then((r) => r.json()),
  });
  const gpu = (status.data as { gpu_available?: boolean } | undefined)?.gpu_available;

  return (
    <div className="note note--info">
      <span className="note__mark">i</span>
      <span>
        {gpu
          ? '本节点具备图形处理器，训练与推理可执行。'
          : '本节点没有图形处理器：数据治理、样本扩增与效果评测可用，训练与推理需在具备 GPU 的节点执行。'}
      </span>
    </div>
  );
}
