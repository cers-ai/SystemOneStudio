/**
 * The single flow definition for all four interaction modes.
 *
 * 需求方案.txt principle 1: 向导模式 / 画布模式 / 极速模式 share one engine and
 * can be switched at any time. That is only true if the steps, their defaults,
 * their validation and whether they may be skipped live in one place. Two
 * implementations of the flow is the most likely way that requirement breaks.
 *
 * Derived from the generated OpenAPI path list rather than restated, so a
 * backend endpoint appearing or disappearing surfaces here.
 */

export interface WizardStep {
  readonly id: StepId;
  readonly title: string;
  readonly hint: string;
  /** The single endpoint this step talks to, if any. */
  readonly endpoint: string;
  /** Whether the step can be completed with system defaults (需求方案.txt 4.1). */
  readonly skippable: boolean;
  /** Rapid mode completes steps after this index automatically. */
  readonly manualInRapidMode: boolean;
}

export const STEP_IDS = [
  'scene_selected',
  'seed_uploaded',
  'data_governed',
  'synth_done',
  'base_model_selected',
  'training_configured',
  'trained',
  'evaluated',
  'deployed',
] as const;

export type StepId = (typeof STEP_IDS)[number];

/** The seven steps a user walks through; the rest follow from step 7. */
export const USER_STEP_COUNT = 7;

export const WIZARD_STEPS: readonly WizardStep[] = [
  {
    id: 'scene_selected',
    title: '选择场景模板',
    hint: '选一个行业场景，或从零开始',
    endpoint: '',
    skippable: false,
    manualInRapidMode: true,
  },
  {
    id: 'seed_uploaded',
    title: '上传种子数据',
    hint: '上传样本数据，或使用示例数据集',
    endpoint: '/api/datasets/preview',
    skippable: false,
    manualInRapidMode: true,
  },
  {
    id: 'data_governed',
    title: '数据治理与质量检查',
    hint: '自动脱敏、质量评分、一键应用建议',
    endpoint: '/api/datasets/quality',
    skippable: true,
    manualInRapidMode: false,
  },
  {
    id: 'synth_done',
    title: '智能数据合成',
    hint: '按推荐参数扩增样本，并输出隐私风险报告',
    endpoint: '/api/datasets/recommend-synth',
    skippable: true,
    manualInRapidMode: false,
  },
  {
    id: 'base_model_selected',
    title: '选择基座模型',
    hint: '按场景推荐，显示显存需求与预计时长',
    endpoint: '',
    skippable: true,
    manualInRapidMode: false,
  },
  {
    id: 'training_configured',
    title: '配置训练方法',
    hint: '推荐项已自动勾选，高级设置默认折叠',
    endpoint: '',
    skippable: true,
    manualInRapidMode: false,
  },
  {
    id: 'trained',
    title: '模型训练',
    hint: '训练中可离开页面，完成后站内信通知',
    endpoint: '',
    skippable: true,
    manualInRapidMode: false,
  },
  {
    id: 'evaluated',
    title: '评测',
    hint: '性能指标优先，效果指标其次',
    endpoint: '',
    skippable: true,
    manualInRapidMode: false,
  },
  {
    id: 'deployed',
    title: '部署',
    hint: '一键启动服务并生成调用示例',
    endpoint: '',
    skippable: false,
    manualInRapidMode: false,
  },
];

export function userSteps(): readonly WizardStep[] {
  return WIZARD_STEPS.slice(0, USER_STEP_COUNT);
}

/** Steps a human configures in rapid mode. Mirrors RAPID_MODE_LAST_MANUAL_STEP. */
export function rapidModeSteps(): readonly WizardStep[] {
  return WIZARD_STEPS.filter((s) => s.manualInRapidMode);
}

/** The last step a human configures in rapid mode. */
export const RAPID_MODE_LAST_MANUAL_STEP: StepId =
  rapidModeSteps()[rapidModeSteps().length - 1]?.id ?? 'seed_uploaded';

export function stepIndex(id: StepId): number {
  return STEP_IDS.indexOf(id);
}

export function nextStep(id: StepId): StepId | undefined {
  return STEP_IDS[stepIndex(id) + 1];
}

export function findStep(id: StepId): WizardStep | undefined {
  return WIZARD_STEPS.find((step) => step.id === id);
}

/**
 * Whether a human configures this step in this mode.
 *
 * Only rapid mode has a boundary (需求方案.txt 4.1: the user completes steps 1-2
 * and the rest is automatic). Wizard and expert modes let a person drive every
 * step, including the evaluate and deploy that follow from step 7 -- the
 * backend state machine allows exactly that, and restricting it here locked the
 * primary button on the last two steps with a message reading "由系统自动完成".
 */
export function isManualInMode(id: StepId, mode: 'wizard' | 'rapid'): boolean {
  if (mode === 'rapid') {
    return stepIndex(id) <= stepIndex(RAPID_MODE_LAST_MANUAL_STEP);
  }
  return true;
}
