import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

import { assistantApi, TOOL_LABEL, type AssistantPreferences } from '@/api/assistant';
import {
  SERVICE_IDS,
  serviceHint,
  serviceLabel,
  serviceNeedsApiKey,
  type ServiceId,
} from '@son/ui-terminology';

/**
 * 智能训练助手 configuration (系统管理).
 *
 * The key field is write-only: the form starts blank with a placeholder saying
 * whether one is saved, and an empty submit means "leave it alone". An operator
 * editing the temperature should never have to re-paste a secret.
 */
export function AssistantSettingsCard() {
  const queryClient = useQueryClient();
  const settings = useQuery({ queryKey: ['assistant-settings'], queryFn: assistantApi.settings });
  const providers = useQuery({ queryKey: ['assistant-providers'], queryFn: assistantApi.providers });
  const tools = useQuery({ queryKey: ['assistant-tools'], queryFn: assistantApi.tools });
  const prefs = useQuery({ queryKey: ['assistant-prefs'], queryFn: assistantApi.preferences });

  const [provider, setProvider] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [model, setModel] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [temperature, setTemperature] = useState(0.2);
  const [enabled, setEnabled] = useState(true);
  const [probeResult, setProbeResult] = useState<{ reachable: boolean; detail: string } | null>(null);

  useEffect(() => {
    if (!settings.data) return;
    setProvider(settings.data.provider);
    setBaseUrl(settings.data.base_url);
    setModel(settings.data.model);
    setTemperature(settings.data.temperature);
    setEnabled(settings.data.enabled);
    setApiKey('');
  }, [settings.data]);

  const save = useMutation({
    mutationFn: () =>
      assistantApi.saveSettings({
        provider,
        base_url: baseUrl,
        model,
        temperature,
        enabled,
        // Only sent when the operator typed something, so an empty field leaves
        // the stored key untouched.
        ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
      }),
    onSuccess: () => {
      setApiKey('');
      setProbeResult(null);
      void queryClient.invalidateQueries({ queryKey: ['assistant-settings'] });
    },
  });

  const probe = useMutation({
    // Save the form first, then probe. It used to PUT an empty body, so the
    // button labelled "保存并测试连通性" tested the *previously stored*
    // endpoint -- reporting success for a configuration the operator had just
    // edited away from.
    mutationFn: async () => {
      await assistantApi.saveSettings({
        provider,
        base_url: baseUrl,
        model,
        temperature,
        enabled,
        ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
      });
      return assistantApi.probe();
    },
    onSuccess: (result) => {
      setApiKey('');
      setProbeResult(result);
      void queryClient.invalidateQueries({ queryKey: ['assistant-settings'] });
    },
  });

  const serviceId = (SERVICE_IDS as readonly string[]).includes(provider)
    ? (provider as ServiceId)
    : 'custom';
  const needsKey = providers.data ? serviceNeedsApiKey(serviceId) : true;
  const known = providers.data?.providers ?? {};

  return (
    <section className="section">
      <div className="section__head">
        <h2 className="section__title">智能训练助手</h2>
        <span className="section__hint">可使用平台自己部署的模型，无需外部服务</span>
      </div>

      <div className="card">
        <div className="card__title">模型服务</div>
        <div className="card__sub">
          助手只读取平台状态，不会代替你执行训练或部署。选择「平台自部署模型」时，
          助手会调用本平台训练出的模型推理服务，数据不离开本部署。
        </div>

        <div className="grid grid--2">
          <label className="field">
            <span className="field__label">服务商</span>
            <select
              className="select"
              value={provider}
              onChange={(e) => {
                setProvider(e.target.value);
                const url = known[e.target.value];
                if (url) setBaseUrl(url);
              }}
            >
              {(Object.keys(known).length ? Object.keys(known) : SERVICE_IDS).map((id) => (
                <option key={id} value={id}>
                  {serviceLabel(id as ServiceId)}
                </option>
              ))}
              {!Object.keys(known).includes(provider) ? (
                <option value={provider}>{provider}</option>
              ) : null}
            </select>
            <span className="field__hint">
              {serviceHint(serviceId)}
              {' · '}
              {known[provider] ?? '自定义地址'}
            </span>
          </label>

          <label className="field">
            <span className="field__label">服务地址</span>
            <input
              className="input"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://…/v1"
            />
          </label>

          <label className="field">
            <span className="field__label">模型名称</span>
            <input
              className="input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="例如 qwen2.5-3b-instruct"
            />
          </label>

          <label className="field">
            <span className="field__label">
              访问密钥
              {needsKey ? '' : '（本地服务无需填写）'}
            </span>
            <input
              className="input"
              type="password"
              autoComplete="off"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={
                settings.data?.has_api_key
                  ? `已保存（${settings.data.masked_key}），留空表示不修改`
                  : '未填写'
              }
            />
            <span className="field__hint">密钥只写入，不回显，也不会出现在任何读取接口里</span>
          </label>

          <label className="field">
            <span className="field__label">回答发散度：{temperature.toFixed(2)}</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
            />
            <span className="field__hint">越低越贴合事实、越高越自由发挥。建议保持较低</span>
          </label>

          <div className="field">
            <span className="field__label">启用</span>
            <div className="switch-row">
              <input
                type="checkbox"
                id="assistant-enabled"
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
              />
              <label className="switch-row__text" htmlFor="assistant-enabled">
                <span className="switch-row__title">在平台上提供助手</span>
                <span className="switch-row__desc">关闭后助手面板隐藏，已保存的配置保留</span>
              </label>
            </div>
          </div>
        </div>

        {save.isError ? (
          <div className="note note--bad">
            <span className="note__mark">!</span>
            <span>保存失败：{(save.error as Error).message}</span>
          </div>
        ) : null}

        {probeResult ? (
          <div className={probeResult.reachable ? 'note note--ok' : 'note note--bad'}>
            <span className="note__mark">{probeResult.reachable ? '✓' : '!'}</span>
            <span>{probeResult.detail || (probeResult.reachable ? '连接正常' : '无法连接')}</span>
          </div>
        ) : null}

        <div className="hero__actions">
          <button
            type="button"
            className="btn btn--primary"
            disabled={save.isPending || !provider || !model}
            onClick={() => save.mutate()}
          >
            {save.isPending ? '保存中…' : '保存配置'}
          </button>
          <button
            type="button"
            className="btn"
            disabled={probe.isPending}
            onClick={() => probe.mutate()}
          >
            {probe.isPending ? '测试中…' : '保存并测试连通性'}
          </button>
          {settings.data ? (
            <span className="step__actions-note">
              当前状态：{settings.data.configured ? '已配置' : '未配置'}
            </span>
          ) : null}
        </div>
      </div>

      <PreferencesCard prefs={prefs.data} />
      <ToolsCard tools={tools.data?.tools ?? []} />
    </section>
  );
}

function PreferencesCard({ prefs }: { prefs?: AssistantPreferences }) {
  const queryClient = useQueryClient();
  const update = useMutation({
    mutationFn: assistantApi.savePreferences,
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['assistant-prefs'] }),
  });

  if (!prefs) return null;

  return (
    <div className="card">
      <div className="card__title">行为偏好</div>
      <div className="card__sub">
        影响助手如何回答，不影响其权限——助手始终只读平台状态。
      </div>

      <div className="switch-row">
        <input
          type="checkbox"
          id="ground"
          checked={prefs.ground_with_platform_state}
          onChange={(e) => update.mutate({ ground_with_platform_state: e.target.checked })}
        />
        <label className="switch-row__text" htmlFor="ground">
          <span className="switch-row__title">注入平台真实状态</span>
          <span className="switch-row__desc">
            让助手知道本节点有没有显卡、你在哪一步。没有它，助手可能建议在本机训练。
          </span>
        </label>
      </div>

      <div className="switch-row">
        <input
          type="checkbox"
          id="confirm"
          checked={prefs.require_confirmation_for_actions}
          onChange={(e) => update.mutate({ require_confirmation_for_actions: e.target.checked })}
        />
        <label className="switch-row__text" htmlFor="confirm">
          <span className="switch-row__title">执行前需确认</span>
          <span className="switch-row__desc">
            当前助手只读取数据、不执行任何操作；此开关为将来开放执行类能力预留。
          </span>
        </label>
      </div>

      <label className="field" style={{ marginTop: 'var(--sp-4)', maxWidth: 280 }}>
        <span className="field__label">单轮最多查询平台 {prefs.max_tool_rounds} 次</span>
        <input
          type="range"
          min={1}
          max={12}
          step={1}
          value={prefs.max_tool_rounds}
          onChange={(e) => update.mutate({ max_tool_rounds: Number(e.target.value) })}
        />
        <span className="field__hint">上限用于防止模型反复查询同一项，失控时产生费用</span>
      </label>
    </div>
  );
}

function ToolsCard({ tools }: { tools: { name: string; description: string }[] }) {
  if (tools.length === 0) return null;
  return (
    <div className="card">
      <div className="card__title">助手可查询的内容</div>
      <div className="card__sub">
        助手只能读取下列信息，无法修改或删除任何数据，也无法代你启动训练。
      </div>
      <div className="row-list">
        {tools.map((tool) => (
          <div className="row-item" key={tool.name}>
            <span className="badge badge--neutral">{TOOL_LABEL[tool.name] ?? tool.name}</span>
            <div className="row-item__main">
              <div className="row-item__meta" style={{ fontSize: 'var(--fs-sm)', color: 'var(--c-text-secondary)' }}>
                {tool.description}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
