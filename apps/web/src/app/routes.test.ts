import { afterEach, describe, expect, it } from 'vitest';
import { currentPath, navigate, routeTitle } from './routes';

afterEach(() => { window.location.hash = ''; });

describe('workspace routes', () => {
  it('preserves a selected workspace after navigation and reload parsing', () => {
    navigate('run/run_abc123');
    expect(currentPath()).toBe('run/run_abc123');
    expect(routeTitle(currentPath())).toBe('训练工作区');
    window.location.hash = '#/run/run_other';
    expect(currentPath()).toBe('run/run_other');
  });

  it.each(['#/run/', '#/run/a/b', '#/run/../../secret', '#/unknown'])(
    'rejects invalid workspace paths: %s', (hash) => {
      window.location.hash = hash;
      expect(currentPath()).toBe('');
    },
  );
});
