import type { EChartsOption } from 'echarts';

/**
 * API client for the platform endpoints.
 *
 * Typed shapes mirror the generated OpenAPI document. Regenerate with
 * `uv run python tools/gen_openapi_types.py` rather than editing by hand.
 */

export interface SceneTemplate {
  id: string;
  name: string;
  summary: string;
  recommended: boolean;
  field_count: number;
  labels: string[];
  recommended_base_model: string;
  recommended_methods: string[];
  core_metrics: string[];
  example_dataset: string | null;
}

export interface SceneSummary {
  id: string;
  code: string;
  name: string;
  summary: string;
  field_count: number;
  labels: string[];
  recommended_base_model: string;
  recommended_methods: string[];
  core_metrics: string[];
  jev_format_compat: boolean;
  jev_training_compat: boolean;
  source: string;
}

export interface Project {
  id: string;
  code: string;
  name: string;
  mode: string;
  scene_code: string | null;
  created_at: string;
}

export interface ServiceStatus {
  name: string;
  ok: boolean;
  detail: string;
}

export interface SystemStatus {
  gpu_available: boolean;
  gpu_detail: string;
  services: ServiceStatus[];
  persistence: string;
  model_count: number;
  in_scope_model_count: number;
  notes: string[];
}

export interface AuditEntry {
  id: string;
  actor: string;
  action: string;
  target: string;
  detail: string;
  ts: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text().catch(() => response.statusText);
    throw new Error(extractDetail(body) || `请求失败（${response.status}）`);
  }
  return (await response.json()) as T;
}

/** FastAPI returns {"detail": ...}; surface the message rather than the JSON. */
function extractDetail(body: string): string {
  try {
    const parsed: unknown = JSON.parse(body);
    if (parsed && typeof parsed === 'object' && 'detail' in parsed) {
      const detail = (parsed as { detail: unknown }).detail;
      if (typeof detail === 'string') return detail;
      if (Array.isArray(detail)) {
        return detail
          .map((d) =>
            d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : String(d),
          )
          .join('；');
      }
    }
  } catch {
    /* not JSON; fall through to the raw body */
  }
  return body.slice(0, 200);
}

export interface ModelCard {
  model_id: string;
  display_name: string;
  family: string;
  params: string;
  min_gpu_memory: string;
  jev_compat_level: string | null;
  license: string;
  in_mvp_scope: boolean;
}

export const platformApi = {
  templates: () => request<SceneTemplate[]>('/api/scenes/templates'),
  scenes: () => request<SceneSummary[]>('/api/scenes'),
  createScene: (body: {
    template_id?: string;
    name?: string;
    summary?: string;
    jev_format_compat?: boolean;
    jev_training_compat?: boolean;
  }) => request<SceneSummary>('/api/scenes', { method: 'POST', body: JSON.stringify(body) }),
  updateJev: (sceneId: string, flags: { jev_format_compat: boolean; jev_training_compat: boolean }) =>
    request<SceneSummary>(`/api/scenes/${sceneId}/jev`, {
      method: 'PATCH',
      body: JSON.stringify(flags),
    }),
  projects: () => request<Project[]>('/api/projects'),
  createProject: (body: { name: string; scene_id?: string; mode?: string }) =>
    request<Project>('/api/projects', { method: 'POST', body: JSON.stringify(body) }),
  systemStatus: () => request<SystemStatus>('/api/system/status'),
  audit: () => request<AuditEntry[]>('/api/system/audit'),
  models: () =>
    request<{
      recommended: ModelCard[];
      all: ModelCard[];
      notes: string[];
    }>('/api/models'),
};

/** Bar chart option for a scene template's field/metric counts. */
export function templateChartOption(counts: { name: string; value: number }[]): EChartsOption {
  return {
    grid: { left: 8, right: 16, top: 8, bottom: 8, containLabel: true },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'value',
      axisLabel: { fontSize: 11 },
      splitLine: { lineStyle: { color: '#eef0f3' } },
    },
    yAxis: {
      type: 'category',
      data: counts.map((c) => c.name),
      axisLabel: { fontSize: 11 },
      axisLine: { lineStyle: { color: '#cdd2d9' } },
    },
    series: [
      {
        type: 'bar',
        data: counts.map((c) => c.value),
        itemStyle: { color: '#2563eb', borderRadius: [0, 4, 4, 0] },
        barMaxWidth: 18,
      },
    ],
  };
}
