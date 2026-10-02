/**
 * Run API client.
 *
 * The frontend holds no workflow state of its own. Every step is a backend
 * action, and after each one the client re-reads the Run -- which is the single
 * source of truth (改造开发方案.md 2.3 / 25).
 */

export interface RunDataset {
  id: string;
  code: string;
  original_filename: string;
  rows: number;
  cols: number;
  label_column: string | null;
  checksum: string;
}

export interface RunSplit {
  id: string;
  code: string;
  train_rows: number;
  valid_rows: number;
  test_rows: number;
  test_synth_rows: number;
  quality: {
    score: number;
    label_distribution: Record<string, number>;
    dimensions: { name: string; value: string; verdict: string }[];
    anomalies: string[];
    suggestions: string[];
    masked_fields: string[];
  } | null;
}

export interface RunSynth {
  id: string;
  code: string;
  method: string;
  rows: number;
  fidelity_score: number | null;
  privacy: {
    reversible_risk: string;
    lines: string[];
    notes: string[];
  } | null;
}

export interface RunJob {
  id: string;
  type: string;
  status: string;
  progress: number;
  stage: string | null;
  message: string | null;
  metrics: Record<string, number>;
  result: Record<string, unknown>;
  error_message: string | null;
  log_path: string | null;
}

export interface RunModelVersion {
  id: string;
  code: string;
  base_model_id: string;
  training_method: string;
  adapter_path: string | null;
  merged_model_path: string | null;
}

export interface RunDeployment {
  id: string;
  runtime: string;
  port: number;
  status: string;
  pid: number | null;
}

export interface Run {
  id: string;
  project_id: string;
  state: string;
  mode: string;
  current_step: number;
  scene_code: string | null;
  base_model_id: string | null;
  dataset_id: string | null;
  split_id: string | null;
  synth_id: string | null;
  job_id: string | null;
  model_version_id: string | null;
  evaluation_id: string | null;
  deployment_id: string | null;
  error_message: string | null;
  available_actions: string[];
  created_at: string;
  updated_at: string;
  dataset: RunDataset | null;
  split: RunSplit | null;
  synth: RunSynth | null;
  training_config: Record<string, unknown> | null;
  job: RunJob | null;
  model_version: RunModelVersion | null;
  evaluation: Record<string, unknown> | null;
  deployment: RunDeployment | null;
  lineage: Record<string, string> | null;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
}

export interface Capabilities {
  scene: string;
  base_models: string[];
  training_methods: string[];
  quant: string[];
  runtime: string;
  jev_level: string;
  output_schema: Record<string, unknown>;
  reason_max_length: number;
  unsupported: {
    note: string;
    base_models: string[];
    training_methods: string[];
    quant: string[];
    runtimes: string[];
    jev_levels: string[];
  };
}

export class RunApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'RunApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) {
    throw new RunApiError(response.status, extractDetail(await response.text()));
  }
  return (await response.json()) as T;
}

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
    /* not JSON */
  }
  return body.slice(0, 300);
}

async function upload<T>(path: string, file: File): Promise<T> {
  const body = new FormData();
  body.append('upload', file);
  return request<T>(path, { method: 'POST', body });
}

/** The seven steps, keyed by the state the backend reports. */
export const STEP_LABELS: Record<number, string> = {
  1: '选择场景模板',
  2: '上传种子数据',
  3: '数据治理与质量检查',
  4: '数据合成',
  5: '选择基座模型',
  6: '配置训练方法',
  7: '训练、量化与部署',
};

export const runApi = {
  capabilities: () => request<Capabilities>('/api/runtime/capabilities'),

  createProject: (name: string) =>
    request<Project>('/api/projects', { method: 'POST', body: JSON.stringify({ name }) }),
  projects: () => request<Project[]>('/api/projects'),

  createRun: (projectId: string) => request<Run>(`/api/projects/${projectId}/runs`, { method: 'POST' }),
  run: (runId: string) => request<Run>(`/api/runs/${runId}`),
  runs: (projectId?: string) =>
    request<Run[]>(`/api/runs${projectId ? `?project_id=${projectId}` : ''}`),

  setScene: (runId: string, sceneCode: string) =>
    request<Run>(`/api/runs/${runId}/scene`, {
      method: 'PATCH',
      body: JSON.stringify({ scene_code: sceneCode }),
    }),
  uploadDataset: (runId: string, file: File) =>
    upload<Run>(`/api/runs/${runId}/dataset`, file),
  prepareData: (runId: string) => request<Run>(`/api/runs/${runId}/prepare-data`, { method: 'POST' }),
  synth: (runId: string, targetRows: number) =>
    request<Run>(`/api/runs/${runId}/synth`, {
      method: 'POST',
      body: JSON.stringify({ target_rows: targetRows }),
    }),
  setModel: (runId: string, modelId: string) =>
    request<Run>(`/api/runs/${runId}/model`, {
      method: 'PATCH',
      body: JSON.stringify({ model_id: modelId }),
    }),
  setTrainingConfig: (runId: string, config: Record<string, unknown>) =>
    request<Run>(`/api/runs/${runId}/training-config`, {
      method: 'PATCH',
      body: JSON.stringify(config),
    }),
  start: (runId: string) => request<Run>(`/api/runs/${runId}/start`, { method: 'POST' }),

  job: (jobId: string) => request<RunJob>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) => request<RunJob>(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),
};

/** States in which a job owns the run and the UI should poll the job. */
export const BUSY_STATES = new Set([
  'QUEUED',
  'TRAINING',
  'MERGING',
  'QUANTIZING',
  'EVALUATING',
  'DEPLOYING',
]);

export function isBusy(state: string): boolean {
  return BUSY_STATES.has(state);
}

export function isFailed(state: string): boolean {
  return state === 'FAILED';
}