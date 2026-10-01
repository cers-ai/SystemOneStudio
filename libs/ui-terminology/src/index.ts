/**
 * Technical term to business language map.
 *
 * 需求方案.txt chapter 2, principle 4: the interface shows business language
 * only; the technical wording lives in a hover hint. That makes this module
 * the single legal source of user-facing copy for training methods,
 * quantization levels and evaluation metrics.
 *
 * Two rules this file exists to enforce:
 *
 * 1. Backend, database and code keep the technical identifiers. The mapping
 *    happens here and nowhere else.
 * 2. `source: 'requirement'` entries are transcribed verbatim from the
 *    requirement's mapping table. Anything added by us is `source: 'design'`
 *    so a reviewer can tell which labels are mandated and which are our call.
 */

/** Technical identifiers used as lookup keys. Values match backend enums. */
export const TECHNICAL_TERMS = [
  'lora',
  'qlora',
  'dpo',
  'distillation',
  'sft',
  'qat',
  'fewshot',
  'quant_q4_k_m',
  'quant_q5_k_m',
  'quant_q8_0',
  'synth_distribution_fit',
  'synth_small_sample_derive',
  'synth_rule_injection',
  'feature_importance',
  'metric_p50',
  'metric_p95',
  'metric_p99',
  'metric_ttft',
  'metric_qps',
  'metric_peak_memory',
  'metric_auc_roc',
  'metric_f1',
  'metric_format_compliance',
  'metric_false_kill_rate',
] as const;

export type TechnicalTerm = (typeof TECHNICAL_TERMS)[number];

export type TermSource = 'requirement' | 'design';

export interface TermCopy {
  /** What the user reads. Must not contain a technical term. */
  readonly label: string;
  /** Hover hint explaining what is actually happening underneath. */
  readonly hint: string;
  readonly source: TermSource;
}

export const TERM_COPY: Readonly<Record<TechnicalTerm, TermCopy>> = {
  // --- Transcribed verbatim from 需求方案.txt chapter 2, principle 4 ---
  lora: {
    label: '快速风格适配',
    hint: '低秩微调：冻结底座权重，只训练少量适配参数，显存占用低、速度快',
    source: 'requirement',
  },
  qlora: {
    label: '低显存快速适配',
    hint: '在 4bit 量化底座上做微调，显存需求约为快速风格适配的一半',
    source: 'requirement',
  },
  dpo: {
    label: '决策优化训练',
    hint: '偏好对齐：用「判定正确」与「判定错误」的样本对直接优化判定倾向',
    source: 'requirement',
  },
  distillation: {
    label: '大模型能力迁移',
    hint: '知识蒸馏：用大模型的判定能力训练小模型，需要教师模型服务',
    source: 'requirement',
  },
  quant_q4_k_m: {
    label: '标准压缩（推荐）',
    hint: 'GGUF Q4_K_M 量化，体积与精度平衡，评测基准与 JEV 基线一致',
    source: 'requirement',
  },
  synth_distribution_fit: {
    label: '智能样本扩增',
    hint: '分布拟合合成：学习种子数据的分布，生成同分布的扩增样本',
    source: 'requirement',
  },
  feature_importance: {
    label: '关键判定因素',
    hint: '各输入字段对判定结果的影响权重排名',
    source: 'requirement',
  },

  // --- Added by us to complete the vocabulary. Not mandated by the table. ---
  sft: {
    label: '基础风格适配',
    hint: '监督微调：在标注数据上直接训练模型输出风格与格式',
    source: 'design',
  },
  qat: {
    label: '压缩感知训练',
    hint: '量化感知训练：训练时就考虑压缩，减少压缩后的精度损失',
    source: 'design',
  },
  fewshot: {
    label: '冷启动适配',
    hint: '少样本迁移：仅有极少样本时用示例引导模型适应任务',
    source: 'design',
  },
  quant_q5_k_m: {
    label: '高质量压缩',
    hint: 'GGUF Q5_K_M 量化，体积更大、精度略高',
    source: 'design',
  },
  quant_q8_0: {
    label: '最高质量压缩',
    hint: 'GGUF Q8_0 量化，接近无损，体积最大',
    source: 'design',
  },
  synth_small_sample_derive: {
    label: '少量样本快速扩增',
    hint: '小样本特征扩增：基于少量种子做特征衍生',
    source: 'design',
  },
  synth_rule_injection: {
    label: '按业务规则扩增',
    hint: '规则注入合成：导入专家规则，生成符合规则的样本',
    source: 'design',
  },
  metric_p50: {
    label: '响应速度（50% 请求）',
    hint: 'P50 端到端延迟',
    source: 'design',
  },
  metric_p95: {
    label: '响应速度（95% 请求）',
    hint: 'P95 端到端延迟，核心性能目标',
    source: 'design',
  },
  metric_p99: {
    label: '响应速度（99% 请求）',
    hint: 'P99 端到端延迟',
    source: 'design',
  },
  metric_ttft: {
    label: '首字响应时间',
    hint: '首 Token 延迟：从发出请求到返回第一个字',
    source: 'design',
  },
  metric_qps: {
    label: '每秒处理量',
    hint: '并发吞吐量',
    source: 'design',
  },
  metric_peak_memory: {
    label: '内存占用峰值',
    hint: '推理进程运行时的峰值内存',
    source: 'design',
  },
  metric_auc_roc: {
    label: '综合判别能力',
    hint: 'AUC-ROC：模型区分黑与白样本的整体能力',
    source: 'design',
  },
  metric_f1: {
    label: '综合准确度',
    hint: 'F1 值：查全率与查准率的调和均值',
    source: 'design',
  },
  metric_format_compliance: {
    label: '输出规范率',
    hint: '模型输出符合 JEV 输出格式的比例',
    source: 'design',
  },
  metric_false_kill_rate: {
    label: '误杀率',
    hint: '白样本被错判为黑样本的比例',
    source: 'design',
  },
} as const;

/**
 * Substrings that must never appear in rendered UI text.
 *
 * The scan test in apps/web greps component source for these. Adding an entry
 * here is how you make a leak impossible to reintroduce silently.
 */
export const FORBIDDEN_IN_UI: readonly string[] = [
  'LoRA',
  'QLoRA',
  'DPO',
  'QAT',
  'SFT',
  'GGUF',
  'Q4_K_M',
  'Q5_K_M',
  'Q8_0',
  'Transformer',
  'PEFT',
  'llama.cpp',
  'vLLM',
  'bitandbytes',
  'bitsandbytes',
  'P50',
  'P95',
  'P99',
  'AUC',
  'AUC-ROC',
  '特征重要性',
  '分布拟合合成',
  '知识蒸馏',
];

/** Business label for a technical term. */
export function label(term: TechnicalTerm): string {
  return TERM_COPY[term].label;
}

/** Hover hint for a technical term. */
export function hint(term: TechnicalTerm): string {
  return TERM_COPY[term].hint;
}

/**
 * Terms that leaked technical wording into user-visible copy.
 *
 * Intended for tests and for a dev-time assertion, not for the render path:
 * failing loudly at build time beats shipping a jargon label.
 */
export function findTechnicalLeaks(text: string): string[] {
  return FORBIDDEN_IN_UI.filter((term) => text.includes(term));
}

export function assertNoTechnicalLeaks(text: string): void {
  const leaks = findTechnicalLeaks(text);
  if (leaks.length > 0) {
    throw new Error(
      `UI copy contains technical terms ${leaks.join(', ')}: "${text}". ` +
        'Use label()/hint() from @son/ui-terminology instead.',
    );
  }
}
