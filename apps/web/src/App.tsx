import { useEffect, useState } from 'react';

import { HomeView } from '@/views/HomeView';
import { ProjectsView } from '@/views/ProjectsView';
import { ScenesView } from '@/views/ScenesView';
import { SystemView } from '@/views/SystemView';
import { WizardView } from '@/views/WizardView';
import { currentPath, navigate, routeTitle, ROUTES, type RoutePath } from '@/app/routes';
import './styles/global.css';
import './styles/app-shell.css';

export function App() {
  const [path, setPath] = useState<RoutePath>(() => currentPath());

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
        </header>
        <div className="shell__view">
          {path === '' ? <HomeView onNavigate={navigate} /> : null}
          {path === 'projects' ? <ProjectsView /> : null}
          {path === 'scenes' ? <ScenesView /> : null}
          {path === 'wizard' ? <WizardView /> : null}
          {path === 'system' ? <SystemView /> : null}
        </div>
      </div>
    </div>
  );
}
