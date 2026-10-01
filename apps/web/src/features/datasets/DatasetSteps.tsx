import { useMutation, useQuery } from '@tanstack/react-query';
import { useState } from 'react';

interface PreviewResponse {
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number;
  label_column: string | null;
  label_confidence: string;
  label_reason: string;
  label_distribution: Record<string, number>;
  sensitive_fields: Record<string, string>;
  missing_rate: number;
}

interface QualityResponse {
  score: number;
  grade: string;
  dimensions: { name: string; value: string; verdict: string; detail: string }[];
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
  fidelity_score: number;
  fidelity_verdict: string;
  privacy_lines: string[];
}

/** Multipart upload helper: the API takes the dataset as a file, not JSON. */
async function postFile<T>(path: string, file: File): Promise<T> {
  const body = new FormData();
  body.append('upload', file);
  const response = await fetch(path, { method: 'POST', body });
  if (!response.ok) {
    throw new Error(`${response.status} ${await response.text()}`);
  }
  return (await response.json()) as T;
}

export function UploadStep() {
  const [file, setFile] = useState<File | null>(null);

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

  if (!file) {
    return (
      <label>
        拖拽文件到此处，或点击上传（支持 CSV）
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </label>
    );
  }

  return (
    <section aria-labelledby="upload-heading">
      <h2 id="upload-heading">已上传：{file.name}</h2>

      {preview.isPending ? <p>正在识别数据…</p> : null}
      {preview.isError ? <p role="alert">识别失败：{preview.error.message}</p> : null}

      {preview.data ? (
        <>
          <p>
            共 {preview.data.total_rows} 行 × {preview.data.columns.length} 列
          </p>
          <p>已识别标签列：{preview.data.label_column ?? '未识别'}</p>
          <p>识别可信度：{preview.data.label_confidence}</p>

          <h3>标签分布</h3>
          <ul>
            {Object.entries(preview.data.label_distribution).map(([label, count]) => (
              <li key={label}>
                {label}：{count}
              </li>
            ))}
          </ul>

          <h3>数据预览（前 5 行）</h3>
          <table>
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
              {preview.data.rows.map((row, i) => (
                <tr key={i}>
                  {preview.data.columns.map((c) => (
                    <td key={c}>{String(row[c] ?? '')}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          {Object.keys(preview.data.sensitive_fields).length > 0 ? (
            <p>将自动脱敏的字段：{Object.keys(preview.data.sensitive_fields).join('、')}</p>
          ) : null}
        </>
      ) : null}

      {quality.data ? (
        <>
          <h3>数据质量报告</h3>
          <p>综合评分：{quality.data.score} 分（{quality.data.grade}）</p>
          <dl>
            {quality.data.dimensions.map((d) => (
              <div key={d.name}>
                <dt>{d.name}</dt>
                <dd>
                  {d.value} — {d.verdict}
                </dd>
              </div>
            ))}
          </dl>
          {quality.data.masked_fields.length > 0 ? (
            <p>脱敏检查：{quality.data.masked_fields.join('已脱敏，')}已脱敏</p>
          ) : null}
          {quality.data.anomalies.length > 0 ? (
            <ul>
              {quality.data.anomalies.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          ) : null}
          {quality.data.suggestions.length > 0 ? (
            <>
              <h4>智能建议</h4>
              <ul>
                {quality.data.suggestions.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </>
          ) : null}
        </>
      ) : null}

      <button
        type="button"
        onClick={() => split.mutate()}
        disabled={file === null || split.isPending}
      >
        {split.isPending ? '正在划分…' : '确认划分'}
      </button>

      {split.isError ? <p role="alert">划分失败：{split.error.message}</p> : null}
      {split.data ? (
        <p>
          训练 {split.data.train_rows} / 验证 {split.data.valid_rows} / 测试{' '}
          {split.data.test_rows}
        </p>
      ) : null}
    </section>
  );
}

export function SynthStep({ file }: { file: File | null }) {
  const synth = useMutation({
    mutationFn: () => postFile<SynthResponse>('/api/datasets/synth?target_rows=1000', file as File),
  });

  if (!file) return <p>请先上传种子数据。</p>;

  return (
    <section aria-labelledby="synth-heading">
      <h2 id="synth-heading">智能样本扩增</h2>
      <button type="button" onClick={() => synth.mutate()} disabled={synth.isPending}>
        {synth.isPending ? '正在扩增…' : '开始扩增'}
      </button>
      {synth.isError ? <p role="alert">扩增失败：{synth.error.message}</p> : null}
      {synth.data ? (
        <>
          <p>
            保真度评分：{synth.data.fidelity_score}（{synth.data.fidelity_verdict}）
          </p>
          <h3>隐私风险报告</h3>
          <ul>
            {synth.data.privacy_lines.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </>
      ) : null}
    </section>
  );
}

/**
 * Endpoint inventory.
 *
 * Mirrors the paths the backend serves so a typo shows up as a failing
 * reference rather than a 404 at the moment a user clicks the button.
 */
export const datasetEndpoints = {
  preview: '/api/datasets/preview',
  quality: '/api/datasets/quality',
  split: '/api/datasets/split',
  recommendSynth: '/api/datasets/recommend-synth',
  synth: '/api/datasets/synth',
  health: '/health',
  jevSpec: '/meta/jev-spec',
} as const;

export type { PreviewResponse, QualityResponse, SplitResponse, SynthResponse };
