import { useMutation, useQuery } from '@tanstack/react-query';
import { Suspense, lazy, useRef, useState } from 'react';
import type { EChartsOption } from 'echarts';

import { Term } from '@/components/Term';
import { label } from '@son/ui-terminology';

/**
 * ECharts is ~700KB and is only needed on the synthesis step. Lazy-loading it
 * keeps it out of the initial bundle for the six steps that do not chart
 * anything.
 */
const Chart = lazy(async () => {
  const module = await import('@/components/Chart');
  return { default: module.Chart };
});

/* -------------------------------------------------------------------------
   API types. Mirrors the generated OpenAPI shapes; regenerate with
   `uv run python tools/gen_openapi_types.py` rather than editing by hand.
   ------------------------------------------------------------------------- */

interface PreviewResponse {
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number;
  label_column: string | null;
  label_confidence: string;
  label_distribution: Record<string, number>;
  sensitive_fields: Record<string, string>;
  missing_rate: number;
}

interface Dimension {
  name: string;
  value: string;
  verdict: string;
  detail: string;
}

interface QualityResponse {
  score: number;
  grade: string;
  dimensions: Dimension[];
  anomalies: string[];
  masked_fields: string[];
  suggestions: string[];
}

interface SplitResponse {
  train_rows: number;
  valid_rows: number;
  test_rows: number;
  test_contains_synth: boolean;
  label_distribution: Record<string, Record<string, number>>;
}

interface SynthResponse {
  rows: number;
  label_distribution: Record<string, number>;
  fidelity_score: number;
  fidelity_verdict: string;
  privacy_lines: string[];
  reversible_risk: string;
}

interface RecommendResponse {
  method: string;
  total_rows: number;
  black_white_ratio: number;
  augment_label: string | null;
  augment_rows: number;
  reasons: string[];
  describe: string;
}

/** The dataset arrives as a file, so these endpoints are multipart POSTs. */
async function postFile<T>(path: string, file: File): Promise<T> {
  const body = new FormData();
  body.append('upload', file);
  const response = await fetch(path, { method: 'POST', body });
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(detail || `请求失败（${response.status}）`);
  }
  return (await response.json()) as T;
}

/* -------------------------------------------------------------------------
   Upload (需求方案.txt 5.2)
   ------------------------------------------------------------------------- */

export function UploadStep() {
  const [file, setFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const preview = useQuery({
    queryKey: ['dataset-preview', file?.name, file?.size],
    queryFn: () => postFile<PreviewResponse>('/api/datasets/preview', file as File),
    enabled: file !== null,
  });

  const quality = useQuery({
    queryKey: ['dataset-quality', file?.name, file?.size],
    queryFn: () => postFile<QualityResponse>('/api/datasets/quality', file as File),
    enabled: file !== null,
  });

  const split = useMutation({
    mutationFn: () => postFile<SplitResponse>('/api/datasets/split', file as File),
  });

  const sensitiveCount = Object.keys(preview.data?.sensitive_fields ?? {}).length;

  if (!file) {
    return (
      <div className="dropzone" onClick={() => inputRef.current?.click()}>
        <div className="dropzone__icon" aria-hidden="true">
          ⬆
        </div>
        <div className="dropzone__title">把样本数据拖到这里，或点击选择文件</div>
        <div className="dropzone__hint">支持 CSV、Excel，或从数据库导入</div>
        <input
          ref={inputRef}
          className="dropzone__input"
          type="file"
          accept=".csv,.xlsx,.xls,text/csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </div>
    );
  }

  return (
    <>
      <div className="card">
        <div className="card__title">
          {file.name}
          <span className="badge badge--neutral">
            {preview.data ? `${preview.data.total_rows.toLocaleString()} 行` : '读取中'}
          </span>
          <button
            type="button"
            className="btn btn--ghost"
            style={{ marginLeft: 'auto' }}
            onClick={() => setFile(null)}
          >
            换一个文件
          </button>
        </div>

        {preview.isPending ? <p className="empty">正在识别标签列与敏感字段…</p> : null}
        {preview.isError ? (
          <div className="note note--bad">
            <span className="note__mark">!</span>
            <span>识别失败：{preview.error.message}</span>
          </div>
        ) : null}

        {preview.data ? (
          <>
            <div className="grid grid--3">
              <div className="stat">
                <div className="stat__label">已识别标签列</div>
                <div className="stat__value" style={{ fontSize: 'var(--fs-lg)' }}>
                  {preview.data.label_column ?? '未识别'}
                </div>
                <div className="stat__foot">
                  <span
                    className={
                      preview.data.label_confidence === 'high' ? 'badge badge--ok' : 'badge badge--warn'
                    }
                  >
                    可信度 {preview.data.label_confidence}
                  </span>
                </div>
              </div>

              <div className="stat">
                <div className="stat__label">标签分布</div>
                <div className="stat__value" style={{ fontSize: 'var(--fs-lg)' }}>
                  {describeDistribution(preview.data.label_distribution)}
                </div>
                <div className="stat__foot">
                  缺失率 {(preview.data.missing_rate * 100).toFixed(2)}%
                </div>
              </div>

              <div className="stat">
                <div className="stat__label">将自动脱敏</div>
                <div className="stat__value" style={{ fontSize: 'var(--fs-lg)' }}>
                  {sensitiveCount}
                  <span className="stat__unit"> 个字段</span>
                </div>
                <div className="stat__foot">
                  {sensitiveCount > 0 ? Object.keys(preview.data.sensitive_fields).join('、') : '未发现敏感字段'}
                </div>
              </div>
            </div>

            <div className="table-wrap" style={{ marginTop: '16px' }}>
              <table className="table">
                <thead>
                  <tr>
                    {preview.data.columns.map((c) => (
                      <th key={c} scope="col">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.data.rows.slice(0, 5).map((row, i) => (
                    <tr key={i}>
                      {preview.data.columns.map((c) => (
                        <td key={c}>{String(row[c] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </div>

      {quality.data ? <QualityReportCard quality={quality.data} /> : null}

      <div style={{ display: 'flex', gap: 'var(--sp-3)', marginTop: 'var(--sp-4)' }}>
        <button
          type="button"
          className="btn"
          disabled={split.isPending}
          onClick={() => split.mutate()}
        >
          {split.isPending ? '正在划分…' : '确认划分'}
        </button>
      </div>

      {split.isError ? (
        <div className="note note--bad" style={{ marginTop: 'var(--sp-3)' }}>
          <span className="note__mark">!</span>
          <span>划分失败：{split.error.message}</span>
        </div>
      ) : null}

      {split.data ? <SplitCard split={split.data} /> : null}
    </>
  );
}

/* -------------------------------------------------------------------------
   Quality report (需求方案.txt 5.3)
   ------------------------------------------------------------------------- */

function QualityReportCard({ quality }: { quality: QualityResponse }) {
  const tone = quality.score >= 80 ? 'ok' : quality.score >= 60 ? 'warn' : 'bad';

  return (
    <div className="card" style={{ marginTop: 'var(--sp-4)' }}>
      <div className="card__title">数据质量报告</div>
      <div className="card__sub">
        评分综合样本量、数据完整性与黑白均衡度三个维度。数据量越大、缺失越少、黑白越均衡，分数越高。
      </div>

      <div className="grid grid--4">
        <div className="stat">
          <div className="stat__label">综合评分</div>
          <div className="stat__value">
            {quality.score}
            <span className="stat__unit"> 分</span>
          </div>
          <div className="stat__foot">
            <span className={`badge badge--${tone}`}>{quality.grade}</span>
          </div>
        </div>

        {quality.dimensions.map((d) => (
          <div className="stat" key={d.name}>
            <div className="stat__label">{d.name}</div>
            <div className="stat__value" style={{ fontSize: 'var(--fs-xl)' }}>
              {d.value}
            </div>
            <div className="stat__foot">
              <span
                className={
                  d.verdict === '充足' || d.verdict === '良好' || d.verdict === '均衡'
                    ? 'badge badge--ok'
                    : 'badge badge--warn'
                }
              >
                {d.verdict}
              </span>
            </div>
          </div>
        ))}
      </div>

      {quality.masked_fields.length > 0 ? (
        <div className="note note--ok" style={{ marginTop: 'var(--sp-4)' }}>
          <span className="note__mark">✓</span>
          <span>
            脱敏检查通过：{quality.masked_fields.join('、')} 已脱敏，且不可还原。
          </span>
        </div>
      ) : null}

      {quality.anomalies.length > 0 ? (
        <div className="card" style={{ marginTop: 'var(--sp-4)' }}>
          <div className="card__title">异常检测</div>
          <ul className="bullets">
            {quality.anomalies.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {quality.suggestions.length > 0 ? (
        <div className="card" style={{ marginTop: 'var(--sp-4)' }}>
          <div className="card__title">智能建议</div>
          <div className="card__sub">以下建议可一键应用，不影响你已经调整过的配置。</div>
          <ul className="bullets">
            {quality.suggestions.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function SplitCard({ split }: { split: SplitResponse }) {
  return (
    <div className="note note--ok" style={{ marginTop: 'var(--sp-3)' }}>
      <span className="note__mark">✓</span>
      <span>
        数据已按 7 : 1.5 : 1.5 划分 —— 训练 {split.train_rows} 行 / 验证 {split.valid_rows} 行 / 测试{' '}
        {split.test_rows} 行。
        <br />
        测试集只取自原始上传数据，不含合成样本（业务硬性约束）。
      </span>
    </div>
  );
}

/* -------------------------------------------------------------------------
   Synthesis (需求方案.txt 5.4)
   ------------------------------------------------------------------------- */

export function SynthStep({ file }: { file: File | null }) {
  const recommend = useQuery({
    queryKey: ['synth-recommend', file?.name, file?.size],
    queryFn: () => postFile<RecommendResponse>('/api/datasets/recommend-synth', file as File),
    enabled: file !== null,
  });

  const synth = useMutation({
    mutationFn: () =>
      postFile<SynthResponse>('/api/datasets/synth?target_rows=2000', file as File),
  });

  if (!file) {
    return (
      <div className="note note--info">
        <span className="note__mark">i</span>
        <span>请先在上一步上传种子数据，系统会据此推荐扩增参数。</span>
      </div>
    );
  }

  return (
    <>
      {recommend.data ? (
        <div className="card">
          <div className="card__title">
            <Term term="synth_distribution_fit" />
            <span className="badge badge--info">推荐</span>
          </div>
          <div className="card__sub">{recommend.data.describe}</div>

          <div className="grid grid--4">
            <div className="stat">
              <div className="stat__label">建议合成总量</div>
              <div className="stat__value">
                {recommend.data.total_rows.toLocaleString()}
                <span className="stat__unit"> 条</span>
              </div>
            </div>
            <div className="stat">
              <div className="stat__label">建议黑白比例</div>
              <div className="stat__value">
                {recommend.data.black_white_ratio.toFixed(0)}
                <span className="stat__unit"> : 1</span>
              </div>
            </div>
            <div className="stat">
              <div className="stat__label">重点扩增</div>
              <div className="stat__value" style={{ fontSize: 'var(--fs-xl)' }}>
                {recommend.data.augment_label ?? '无需定向扩增'}
              </div>
            </div>
            <div className="stat">
              <div className="stat__label">预计新增</div>
              <div className="stat__value">
                {recommend.data.augment_rows.toLocaleString()}
                <span className="stat__unit"> 条</span>
              </div>
            </div>
          </div>

          <ul className="bullets" style={{ marginTop: 'var(--sp-4)' }}>
            {recommend.data.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div style={{ display: 'flex', gap: 'var(--sp-3)', marginTop: 'var(--sp-4)' }}>
        <button
          type="button"
          className="btn btn--primary"
          disabled={synth.isPending}
          onClick={() => synth.mutate()}
        >
          {synth.isPending ? '正在扩增…' : `开始扩增 ${label('synth_distribution_fit')}`}
        </button>
      </div>

      {synth.isError ? (
        <div className="note note--bad" style={{ marginTop: 'var(--sp-3)' }}>
          <span className="note__mark">!</span>
          <span>扩增失败：{synth.error.message}</span>
        </div>
      ) : null}

      {synth.data ? (
        <>
          <div className="card" style={{ marginTop: 'var(--sp-4)' }}>
            <div className="card__title">分布对比</div>
            <div className="card__sub">
              原始分布与合成分布的逐类对比。灰色柱为合成数据。
            </div>
            <Suspense fallback={<div className="empty">图表加载中…</div>}>
              <Chart
                option={distributionOption(synth.data.label_distribution)}
                height={220}
                label="原始与合成数据的标签分布对比图"
              />
            </Suspense>
          </div>

          <div className="grid grid--2" style={{ marginTop: 'var(--sp-4)' }}>
            <div className="stat">
              <div className="stat__label">保真度评分</div>
              <div className="stat__value">
                {synth.data.fidelity_score.toFixed(2)}
              </div>
              <div className="stat__foot">
                <span
                  className={
                    synth.data.fidelity_verdict === 'excellent' || synth.data.fidelity_verdict === 'good'
                      ? 'badge badge--ok'
                      : 'badge badge--warn'
                  }
                >
                  {verdictLabel(synth.data.fidelity_verdict)}
                </span>
              </div>
            </div>

            <div className="stat">
              <div className="stat__label">隐私风险报告</div>
              <ul className="bullets" style={{ marginTop: 0 }}>
                {synth.data.privacy_lines.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          </div>
        </>
      ) : null}
    </>
  );
}

/* -------------------------------------------------------------------------
   Helpers
   ------------------------------------------------------------------------- */

const DECISION_ORDER = ['black', 'white', 'gray'] as const;
const DECISION_LABEL: Record<string, string> = {
  black: '黑样本（涉诈）',
  white: '白样本（正常）',
  gray: '灰样本（待定）',
};

function describeDistribution(counts: Record<string, number>): string {
  // `!== undefined`, not truthiness: a class with genuinely zero rows is falsy
  // and would vanish from the summary -- and zero gray is exactly the condition
  // the quality report flags as needing attention.
  const present = DECISION_ORDER.filter((d) => counts[d] !== undefined);
  if (present.length === 0) return '未识别到标签列';

  return present
    .map((d) => {
      const n = counts[d] ?? 0;
      const label = { black: '黑', white: '白', gray: '灰' }[d];
      return n > 0 ? `${n} ${label}` : `0 ${label}`;
    })
    .join(' / ');
}

function verdictLabel(verdict: string): string {
  return (
    { excellent: '优秀', good: '良好', acceptable: '可接受', poor: '偏差' }[verdict] ?? verdict
  );
}

function distributionOption(counts: Record<string, number>): EChartsOption {
  const labels = DECISION_ORDER.filter((d) => counts[d] !== undefined) as readonly string[];
  return {
    grid: { left: 8, right: 8, top: 24, bottom: 8, containLabel: true },
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    xAxis: {
      type: 'category',
      data: labels.map((d) => DECISION_LABEL[d] ?? d),
      axisLabel: { fontSize: 11 },
      axisLine: { lineStyle: { color: '#cdd2d9' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { fontSize: 11 },
      splitLine: { lineStyle: { color: '#eef0f3' } },
    },
    series: [
      {
        name: '合成数据',
        type: 'bar',
        data: labels.map((d) => counts[d] ?? 0),
        itemStyle: { color: '#2563eb', borderRadius: [4, 4, 0, 0] },
        barMaxWidth: 56,
      },
    ],
  };
}