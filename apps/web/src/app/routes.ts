/**
 * Hash-based routing.
 *
 * Hash rather than history API for two reasons: the web image is static files
 * behind nginx, so there is no server to rewrite deep paths, and the product has
 * a fixed, shallow set of sections where a router dependency would cost more
 * than it returns. Hash URLs are refresh-safe and shareable under any static
 * host.
 */

export const ROUTES = [
  { path: '', label: '首页', icon: '◈' },
  { path: 'projects', label: '项目', icon: '▤' },
  { path: 'scenes', label: '场景展廊', icon: '◫' },
  { path: 'wizard', label: '训练流程', icon: '▷' },
  { path: 'system', label: '系统管理', icon: '⚙' },
] as const;

export type RoutePath = (typeof ROUTES)[number]['path'] | `run/${string}`;

export function currentPath(): RoutePath {
  const raw = window.location.hash.replace(/^#\/?/, '');
  if (/^run\/[A-Za-z0-9_-]+$/.test(raw)) return raw as `run/${string}`;
  const known = ROUTES.find((r) => r.path === raw);
  return known ? known.path : '';
}

export function navigate(path: RoutePath): void {
  window.location.hash = path === '' ? '/' : `/${path}`;
}

export function routeTitle(path: RoutePath): string {
  if (path.startsWith('run/')) return '训练工作区';
  return ROUTES.find((r) => r.path === path)?.label ?? '首页';
}
