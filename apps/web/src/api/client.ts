/**
 * Thin fetch wrapper around the control-plane API.
 *
 * The web client calls the API through same-origin paths; the dev server
 * proxies them (see vite.config.ts). Keep it that way rather than introducing
 * a base URL: it avoids CORS entirely and keeps cookies first-party.
 */

import type {
  Healthresponse,
  Jevspecresponse,
  Predictresponse,
} from './generated/types';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly path: string,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(response.status, path, body || response.statusText);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Healthresponse>('/health'),
  jevSpec: () => request<Jevspecresponse>('/meta/jev-spec'),
  echoPrediction: (sample: Record<string, unknown>) =>
    request<Predictresponse>('/meta/echo-prediction', {
      method: 'POST',
      body: JSON.stringify(sample),
    }),
};
