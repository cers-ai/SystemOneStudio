// @vitest-environment node
// Scans the filesystem, so it must not run under jsdom: import.meta.url is an
// http: URL there and fileURLToPath() throws.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { FORBIDDEN_IN_UI, findTechnicalLeaks } from '@son/ui-terminology';

const SRC = fileURLToPath(new URL('../../', import.meta.url)); // src
const REPO_ROOT = fileURLToPath(new URL('../../../../', import.meta.url));

/** Excluded: generated from backend contracts, which may name technical terms. */
const GENERATED_DIR = join(SRC, 'api', 'generated');

/** Excluded: this scanner's own fixtures contain the terms it looks for. */
const isTestFile = (path: string): boolean => /\.(test|spec)\.[jt]sx?$/.test(path);

function walk(dir: string, exts: string[]): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    if (full === GENERATED_DIR) {
      return [];
    }
    if (statSync(full).isDirectory()) {
      return walk(full, exts);
    }
    return exts.some((e) => entry.endsWith(e)) && !isTestFile(full) ? [full] : [];
  });
}

const components = walk(SRC, ['.ts', '.tsx']);

/**
 * 需求方案.txt chapter 2, principle 4: business users must not see technical
 * terminology. Relying on reviewers to catch it does not work, so the whole
 * component tree is scanned on every test run.
 */
describe('no technical terms in UI source', () => {
  it('found components to scan', () => {
    expect(components.length).toBeGreaterThan(0);
  });

  it.each(components)('%s contains no forbidden technical term', (file) => {
    const leaks = findTechnicalLeaks(readFileSync(file, 'utf8'));
    expect(
      leaks,
      `${relative(REPO_ROOT, file)} leaks ${leaks.join(', ')}. ` +
        'Use label()/hint() from @son/ui-terminology.',
    ).toEqual([]);
  });

  it('excluded the generated contract directory', () => {
    expect(components.some((f) => f.startsWith(GENERATED_DIR))).toBe(false);
  });

  it('excluded test files, which hold the fixtures the scanner looks for', () => {
    expect(components.some(isTestFile)).toBe(false);
  });
});

describe('no fabricated measurements in UI source', () => {
  /**
   * The September audit found the model shelf hardcoding `12GB` / `约 45 分钟`
   * while `/api/models` was returning the real registry. A figure written into a
   * component is therefore either fetched or invented, and invented is what this
   * catches.
   */
  const MEASUREMENT = /\d+\s*(GB|MB|毫秒|分钟|小时|秒)/;
  /** Same pattern, anchored to a quoted literal so fetched values do not match. */
  const LITERAL = /['"`][^'"`]*\d+\s*(GB|MB|毫秒|分钟|小时|秒)/;

  /** Comments may cite figures to explain a decision; rendered copy may not. */
  const stripComments = (source: string): string =>
    source
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/(^|[^:])\/\/.*$/gm, '$1');

  const literalLines = (source: string): string[] =>
    stripComments(source)
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => LITERAL.test(line));

  it.each(components)('%s fetches rather than hardcodes measurements', (file) => {
    const literals = literalLines(readFileSync(file, 'utf8'));

    expect(
      literals,
      `${relative(REPO_ROOT, file)} hardcodes a measurement: ${literals.join(' | ')}. ` +
        'Fetch it from the API, and label it an estimate if it is one.',
    ).toEqual([]);
  });

  it('the pattern would catch a real violation', () => {
    // Guard against the regex silently matching nothing.
    const bad = ['<span>{"12GB"}</span>', '<span>{"约 45 分钟"}</span>'].join('\n');
    expect(literalLines(bad)).toHaveLength(2);
  });

  it('does not flag a value pulled off a response', () => {
    expect(literalLines('<td>{model.min_gpu_memory}</td>')).toEqual([]);
  });

  it('measures the units it claims to', () => {
    for (const sample of ['12GB', '45 分钟', '30 分钟', '512MB', '200 毫秒']) {
      expect(MEASUREMENT.test(sample), sample).toBe(true);
    }
  });
});

describe('leak detector is meaningful', () => {
  it('catches a realistic regression', () => {
    expect(findTechnicalLeaks('<button>使用 LoRA 训练</button>')).toContain('LoRA');
  });

  it('catches a raw quant level', () => {
    expect(findTechnicalLeaks('导出 Q4_K_M')).toContain('Q4_K_M');
  });

  it('catches the Chinese technical phrasing from the requirement table', () => {
    expect(findTechnicalLeaks('查看特征重要性')).toEqual(['特征重要性']);
  });

  it('does not flag the business wording that replaced it', () => {
    expect(findTechnicalLeaks('查看关键判定因素')).toEqual([]);
  });

  it('list is non-trivial', () => {
    expect(FORBIDDEN_IN_UI.length).toBeGreaterThan(10);
  });
});
