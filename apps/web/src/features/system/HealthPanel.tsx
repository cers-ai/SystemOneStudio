import { label, hint } from '@son/ui-terminology';
import { useQuery } from '@tanstack/react-query';

import { api } from '@/api/client';

/**
 * M0 status panel. Doubles as the check that the dev proxy, the query client
 * and the generated contract types line up end to end.
 */
export function HealthPanel() {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health });
  const spec = useQuery({ queryKey: ['jev-spec'], queryFn: api.jevSpec });

  return (
    <section aria-labelledby="status-heading">
      <h2 id="status-heading">平台状态</h2>

      <p>
        {health.isPending
          ? '正在检查服务状态…'
          : health.isError
            ? `服务不可用：${health.error.message}`
            : `服务正常（${health.data.service}）`}
      </p>

      {spec.data ? (
        <dl>
          <dt>输出规范版本</dt>
          <dd>{spec.data.spec_version}</dd>

          <dt>默认输出格式对齐</dt>
          <dd>
            {spec.data.default_flags.jev_format_compat ? '已开启' : '已关闭'}（
            {label('quant_q4_k_m').replace('（推荐）', '')}，{hint('quant_q4_k_m')}）
          </dd>

          <dt>默认训练范式对齐</dt>
          <dd>{spec.data.default_flags.jev_training_compat ? '已开启' : '已关闭'}</dd>
        </dl>
      ) : null}
    </section>
  );
}
