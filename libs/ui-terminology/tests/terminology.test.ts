import { describe, expect, it } from 'vitest';

import {
  assertNoTechnicalLeaks,
  findTechnicalLeaks,
  FORBIDDEN_IN_UI,
  label,
  hint,
  TECHNICAL_TERMS,
  TERM_COPY,
  type TechnicalTerm,
} from '../src/index';

/** The seven terms 需求方案.txt chapter 2, principle 4 mandates verbatim. */
const REQUIRED_FROM_REQUIREMENT: ReadonlyArray<[TechnicalTerm, string]> = [
  ['lora', '快速风格适配'],
  ['qlora', '低显存快速适配'],
  ['dpo', '决策优化训练'],
  ['distillation', '大模型能力迁移'],
  ['quant_q4_k_m', '标准压缩（推荐）'],
  ['synth_distribution_fit', '智能样本扩增'],
  ['feature_importance', '关键判定因素'],
];

describe('requirement-mandated labels', () => {
  it.each(REQUIRED_FROM_REQUIREMENT)('%s maps to %s', (term, expected) => {
    expect(label(term)).toBe(expected);
  });

  it.each(REQUIRED_FROM_REQUIREMENT)('%s is sourced from the requirement', (term) => {
    expect(TERM_COPY[term].source).toBe('requirement');
  });

  it('covers every term listed in the requirement table', () => {
    expect(REQUIRED_FROM_REQUIREMENT).toHaveLength(7);
  });
});

describe('vocabulary completeness', () => {
  it('has an entry for every declared technical term', () => {
    for (const term of TECHNICAL_TERMS) {
      expect(TERM_COPY[term], `missing copy for ${term}`).toBeDefined();
    }
  });

  it('has no orphan copy entries', () => {
    const declared = new Set<string>(TECHNICAL_TERMS);
    for (const key of Object.keys(TERM_COPY)) {
      expect(declared.has(key), `copy declared for undeclared term ${key}`).toBe(true);
    }
  });

  it('gives every term a non-empty hint for the hover tooltip', () => {
    for (const term of TECHNICAL_TERMS) {
      expect(hint(term).length).toBeGreaterThan(0);
    }
  });

  it('marks added vocabulary as design so reviewers can tell them apart', () => {
    const designOnly = TECHNICAL_TERMS.filter((t) => TERM_COPY[t].source === 'design');
    expect(designOnly.length).toBeGreaterThan(0);
  });
});

describe('labels never leak technical wording', () => {
  it.each(TECHNICAL_TERMS)('%s label is jargon-free', (term) => {
    expect(findTechnicalLeaks(label(term))).toEqual([]);
  });
});

describe('leak detector', () => {
  it('flags a raw technical string', () => {
    expect(findTechnicalLeaks('使用 LoRA 训练')).toContain('LoRA');
  });

  it('flags a raw quant level', () => {
    expect(findTechnicalLeaks('导出 Q4_K_M')).toContain('Q4_K_M');
  });

  it('flags the Chinese technical phrasings', () => {
    expect(findTechnicalLeaks('查看特征重要性')).toContain('特征重要性');
  });

  it('does not flag business copy', () => {
    expect(findTechnicalLeaks('标准压缩（推荐）')).toEqual([]);
    expect(findTechnicalLeaks('关键判定因素')).toEqual([]);
  });

  it('assertion throws with an actionable message', () => {
    expect(() => assertNoTechnicalLeaks('use DPO here')).toThrow(/label\(\)\/hint\(\)/);
  });

  it('assertion is silent for clean copy', () => {
    expect(() => assertNoTechnicalLeaks('智能样本扩增')).not.toThrow();
  });

  it('forbidden list has no duplicates', () => {
    expect(new Set(FORBIDDEN_IN_UI).size).toBe(FORBIDDEN_IN_UI.length);
  });
});
