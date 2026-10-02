import { useMutation, useQuery } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';

import {
  assistantApi,
  AssistantError,
  TOOL_LABEL,
  type AskResponse,
} from '@/api/assistant';

interface Turn {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  toolCalls?: { name: string; result: Record<string, unknown> }[];
  meta?: string;
  error?: boolean;
}

let counter = 0;
const nextId = (): string => `turn-${++counter}`;

/**
 * The assistant panel.
 *
 * Three states the user has to be able to tell apart, and they are visually
 * distinct on purpose:
 *
 * - not configured  -> a setup prompt, not an empty chat box
 * - provider down   -> the upstream error, surfaced rather than swallowed
 * - refused         -> the assistant declined (tool cap, undecided constraint)
 */
export function AssistantPanel({
  currentStep,
  hasData,
  hasQuality,
  onClose,
}: {
  currentStep?: string;
  hasData?: boolean;
  hasQuality?: boolean;
  onClose: () => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState('');
  const [showTrace, setShowTrace] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const settings = useQuery({ queryKey: ['assistant-settings'], queryFn: assistantApi.settings });
  const suggest = useQuery({
    queryKey: ['assistant-suggest', currentStep, hasData, hasQuality],
    queryFn: () =>
      assistantApi.suggest({
        current_step: currentStep ?? null,
        has_data: Boolean(hasData),
        has_quality: Boolean(hasQuality),
      }),
    enabled: settings.data?.configured === true,
  });

  const ask = useMutation({
    mutationFn: (question: string) =>
      assistantApi.ask({
        question,
        current_step: currentStep ?? null,
        has_data: Boolean(hasData),
        has_quality: Boolean(hasQuality),
        history: turns
          .filter((t) => !t.error)
          .slice(-10)
          .map((t) => ({ role: t.role, content: t.text })),
      }),
    onSuccess: (reply: AskResponse) => {
      setTurns((prev) => [
        ...prev,
        {
          id: nextId(),
          role: 'assistant',
          text: reply.text,
          toolCalls: reply.tool_calls.map((c) => ({ name: c.name, result: c.result })),
          meta: `${reply.rounds} 轮 · ${Math.round(reply.elapsed_ms)}ms${
            reply.usage.total_tokens ? ` · ${reply.usage.total_tokens} tokens` : ''
          }${reply.truncated ? ' · 已达轮次上限' : ''}`,
        },
      ]);
    },
    onError: (error: unknown) => {
      const message =
        error instanceof AssistantError ? error.message : '请求失败，请稍后再试';
      const status = error instanceof AssistantError ? error.status : 0;
      setTurns((prev) => [
        ...prev,
        { id: nextId(), role: 'assistant', text: message, error: true, meta: status ? `HTTP ${status}` : undefined },
      ]);
    },
  });

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns.length, ask.isPending]);

  const send = (): void => {
    const question = input.trim();
    if (!question || ask.isPending) return;
    setTurns((prev) => [...prev, { id: nextId(), role: 'user', text: question }]);
    setInput('');
    ask.mutate(question);
  };

  const configured = settings.data?.configured ?? false;

  return (
    <aside className="assistant" aria-label="智能训练助手">
      <header className="assistant__head">
        <div>
          <div className="assistant__title">智能训练助手</div>
          <div className="assistant__model">
            {settings.data ? `${settings.data.model} · ${settings.data.provider}` : '读取配置中…'}
          </div>
        </div>
        <button type="button" className="btn btn--ghost" onClick={onClose} aria-label="收起助手">
          ✕
        </button>
      </header>

      {!configured && settings.data ? (
        <div className="assistant__empty">
          <div className="note note--warn">
            <span className="note__mark">!</span>
            <span>
              智能训练助手尚未配置可用的大模型。到「系统管理 → 智能训练助手」选择服务商并填写配置；
              若使用本平台自己部署的模型，无需填写密钥。
            </span>
          </div>
        </div>
      ) : null}

      <div className="assistant__log" ref={logRef}>
        {turns.length === 0 ? (
          <div className="assistant__empty">
            <p>{suggest.data?.text ?? '问点什么吧：某一步在做什么、某个参数意味着什么、报错怎么排查。'}</p>
          </div>
        ) : null}

        {turns.map((turn) => (
          <div key={turn.id} className={`bubble bubble--${turn.role}${turn.error ? ' bubble--error' : ''}`}>
            <div className="bubble__text">{turn.text}</div>

            {turn.toolCalls && turn.toolCalls.length > 0 ? (
              <details className="bubble__tools">
                <summary>查看助手查询了 {turn.toolCalls.length} 项平台数据</summary>
                <ul>
                  {turn.toolCalls.map((call, i) => (
                    <li key={i}>
                      <strong>{TOOL_LABEL[call.name] ?? call.name}</strong>
                      <pre>{JSON.stringify(call.result, null, 2)}</pre>
                    </li>
                  ))}
                </ul>
              </details>
            ) : null}

            {turn.meta ? <div className="bubble__meta">{turn.meta}</div> : null}
          </div>
        ))}

        {ask.isPending ? <div className="bubble bubble--assistant bubble--pending">正在思考…</div> : null}
      </div>

      <footer className="assistant__foot">
        <div className="assistant__presets">
          {[
            '这一步在做什么？',
            '刚才的质量报告说明什么？',
            '为什么要选这个配置？',
            '训练报错了，怎么排查？',
          ].map((preset) => (
            <button
              key={preset}
              type="button"
              className="assistant__preset"
              disabled={ask.isPending || !configured}
              onClick={() => {
                setInput(preset);
              }}
            >
              {preset}
            </button>
          ))}
        </div>

        <div className="assistant__input">
          <textarea
            className="textarea"
            rows={2}
            value={input}
            disabled={!configured || ask.isPending}
            placeholder={suggest.data?.text ?? '描述你的问题'}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
          <button
            type="button"
            className="btn btn--primary"
            disabled={!configured || ask.isPending || !input.trim()}
            onClick={send}
          >
            发送
          </button>
        </div>

        <label className="assistant__trace">
          <input
            type="checkbox"
            checked={showTrace}
            onChange={(e) => setShowTrace(e.target.checked)}
            style={{ marginRight: 6 }}
          />
          展开助手查询过程
        </label>
      </footer>
    </aside>
  );
}
