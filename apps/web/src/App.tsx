import { useEffect, useState } from 'react';
import { RunWizard } from '@/features/wizard/RunWizard';
import { currentPath, navigate, type RoutePath } from '@/app/routes';
import './styles/global.css';
import './styles/app-shell.css';
import './styles/workspace.css';

export function App() {
  const [path, setPath] = useState<RoutePath>(() => currentPath());
  useEffect(() => {
    const change = (): void => setPath(currentPath());
    window.addEventListener('hashchange', change);
    return () => window.removeEventListener('hashchange', change);
  }, []);
  const runId = path.startsWith('run/') ? path.slice(4) : null;
  return <div className="studio">
    <header className="studio__header">
      <button className="studio__brand" onClick={() => navigate('wizard')} type="button">
        <span className="studio__logo">S</span><span>SystemOne<span className="studio__muted">Studio</span></span>
      </button>
      <span className="studio__tag">决策模型工作台</span>
      <a className="studio__docs" href="/docs" target="_blank" rel="noreferrer">接口文档 ↗</a>
    </header>
    <main className="studio__main"><RunWizard key={runId ?? 'new'} runId={runId} /></main>
  </div>;
}
