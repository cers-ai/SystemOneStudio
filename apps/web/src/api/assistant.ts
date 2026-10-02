/** Assistant API client. The key is write-only; nothing here ever reads it back. */

export interface AssistantSettings {
  provider: string;
  base_url: string;
  model: string;
  temperature: number;
  max_tokens: number;
  timeout_s: number;
  supports_tools: boolean;
  enabled: boolean;
  configured: boolean;
  has_api_key: boolean;
  masked_key: string;
}

export interface ProvidersView {
  providers: Record<string, string>;
  needs_api_key: string[];
}

export interface ToolInvocation {
  name: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown>;
}

export interface AskResponse {
  text: string;
  tool_calls: ToolInvocation[];
  rounds: number;
  elapsed_ms: number;
  truncated: boolean;
  usage: Record<string, number>;
}

export interface AssistantPreferences {
  ground_with_platform_state: boolean;
  max_tool_rounds: number;
  require_confirmation_for_actions: boolean;
}

export class AssistantError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'AssistantError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text().catch(() => response.statusText);
    throw new AssistantError(response.status, extract(body));
  }
  return (await response.json()) as T;
}

function extract(body: string): string {
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
    /* fall through */
  }
  return body.slice(0, 300);
}

export const assistantApi = {
  providers: () => request<ProvidersView>('/api/assistant/providers'),
  settings: () => request<AssistantSettings>('/api/assistant/settings'),
  saveSettings: (body: Partial<AssistantSettings> & { api_key?: string }) =>
    request<AssistantSettings>('/api/assistant/settings', {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  probe: () =>
    request<{ reachable: boolean; detail: string; model?: string; models?: string[] }>(
      '/api/assistant/settings/probe',
      { method: 'POST' },
    ),
  tools: () =>
    request<{ tools: { name: string; description: string; parameters: unknown }[] }>(
      '/api/assistant/tools',
    ),
  preferences: () => request<AssistantPreferences>('/api/assistant/preferences'),
  savePreferences: (body: Partial<AssistantPreferences>) =>
    request<AssistantPreferences>('/api/assistant/preferences', {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  ask: (body: {
    question: string;
    current_step?: string | null;
    history?: { role: 'user' | 'assistant'; content: string }[];
    quality_report?: Record<string, unknown> | null;
    has_data?: boolean;
    has_quality?: boolean;
  }) => request<AskResponse>('/api/assistant/ask', { method: 'POST', body: JSON.stringify(body) }),
  suggest: (body: { current_step?: string | null; has_data?: boolean; has_quality?: boolean }) =>
    request<{ text: string }>('/api/assistant/suggest', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
};

/**
 * Business-language labels for the settings screen.
 *
 * The operator picking a gateway does need to know which service they are
 * pointing at, so the technical identifier is shown in the field hint rather
 * than as the visible label: a `<select>` cannot carry a tooltip, so the
 * technical name moves to the hint line beneath it.
 */
export const TOOL_LABEL: Record<string, string> = {
  get_node_capabilities: '查询本节点能力',
  get_workflow_state: '查询流程进度',
  get_dataset_quality: '读取数据质量报告',
  explain_hyperparameters: '解释超参数来源',
  diagnose_failure: '定位报错原因',
  explain_product_constraint: '查询产品硬性约束',
};
