# AGENTS.md — SystemOneStudio

零代码决策模型训练平台。`需求方案.txt` 是产品需求的唯一权威来源；本文件记录仓库状态、环境事实和不可违背的约束。

## 唯一事实来源

- `需求方案.txt`（923 行）是产品需求的唯一权威来源。架构、技术选型、MVP 边界、指标目标全部出自此文件。
- 改动产品行为前先读该文件对应章节，不要凭记忆推断。
- `技术方案.md`（架构与接口设计）和 `开发计划.md`（里程碑）是实现细节的落点；**若与 `需求方案.txt` 冲突，以 `需求方案.txt` 为准**，并同步修正另两份文档。
- `libs/contracts` 是**跨服务的类型唯一来源**。服务之间只依赖它，禁止互相 import 内部模块。前端类型由 `tools/gen_openapi_types.py` 从 apps/api 的 OpenAPI 生成，不要手写。

## 工具链命令（已实测通过）

两套栈，互相独立。**改动后按顺序跑完整的 verify，不要只跑单个。**

```bash
# 前端 + TypeScript 库
npm install                # 仅首次 / 依赖变更时
npm run verify             # = lint -> typecheck -> test
npm run build              # tsc -b && vite build

# Python（11 个包在 uv workspace 里）
uv sync --all-packages     # 仅首次 / 依赖变更时
npm run verify:py          # = ruff format --check -> ruff check -> mypy -> pytest

# 契约类型重新生成（改了 libs/contracts 或 apps/api 之后）
uv run python tools/gen_openapi_types.py

# 本地依赖服务（PostgreSQL / MinIO / Redis，不含 GPU 服务）
docker compose -f deploy/compose/docker-compose.dev.yml up -d

# 开发服务器
npm run dev:api            # uvicorn :8000
npm run dev                # vite :5173，已配置 /api /meta /v1 /health 代理到 :8000
```

单点验证：

```bash
uv run pytest libs/contracts/tests          # 只跑 Python 某包
uv run pytest -k reason                     # 按名字过滤
uv run mypy libs/contracts                  # 只做类型检查
npm test --workspace libs/ui-terminology    # 只跑某个前端 workspace
```

## 仓库结构

```text
apps/web              React + Vite + TanStack Query（AntV X6 / ECharts 在 M4 接入）
apps/api              FastAPI 控制面
libs/contracts        跨服务 Pydantic 契约 + JEV 常量（src/son_contracts）
libs/model-registry   底座注册表 + BaseModelAdapter 抽象（src/son_model_registry）
libs/ui-terminology   技术术语 → 业务文案映射表（TypeScript）
services/data-pipeline  上传 / 脱敏 / 质量报告 / 7:1.5:1.5 划分 / 参数推荐
services/synth          三种合成模式 + 保真度 + 隐私风险报告
services/orchestrator   7 步流程状态机（M5）
services/trainer        训练编排（M2，需 GPU）
services/quantizer      合并与 GGUF 量化（M2，需 GPU）
services/evaluator      评测（M3，受 Q1/Q3 阻塞）
services/inference      llama.cpp 编排（M3，受 Q1 阻塞）
services/jev-adapter    JEV 格式 / 训练 / 评测对齐（M5）
tools/                codegen 脚本
deploy/compose        本地依赖编排；deploy/gpu 放 GPU 节点（尚空）
```

**Python 导入名不跟目录名一致**：`services/jev-adapter` → `son_jev`，`libs/model-registry` → `son_model_registry`，`services/data-pipeline` → `son_data_pipeline`。mypy 的 `mypy_path` 在根 `pyproject.toml` 里逐条列了 `src` 目录——**新增 Python 包时要同步加进去**，否则 mypy 解析不到。

`libs/ui-terminology` 是 TypeScript 包，**不能**被根 `pyproject.toml` 的 `libs/*` 通配符匹配到；uv workspace 里 `libs/` 是逐条列出的。

## 已落地的核心不变量（改动前先读源码）

这几条都有测试守着，改动前先看 `services/data-pipeline/src/son_data_pipeline/split.py`：

- **合成数据不进测试集**是**结构性**保证：test 从只含 seed 行的池子里抽，再断言一次兜底。全合成数据输入会被显式拒绝。`split.py` 里的 `test_split.py::test_guard_catches_a_leak_if_one_is_injected` 专门证明断言不是死代码。
- **脱敏不可逆**：没有 unmask 路径。`text[-0:]` 是整个字符串不是空串，写脱敏规则时注意 `keep_suffix=0`。
- **异常检测用修正 z-score 不是 IQR**：需求 5.3 的形状（离群占 31%）会让 q3 落进离群块，IQR 直接失效。
- **合成器必带 label**：未指定 `augment_label` 时按种子分布采样，不会产出 `None` 标签。

## 环境事实（已实测，不要重新假设）

| 项 | 状态 |
|---|---|
| Node | v22.22.3，npm 10.9.8，**无 pnpm** |
| Python | 系统 3.14.4；**workspace 固定 3.11.12**（`uv python install 3.11`） |
| Docker | 28.3.2 可用 |
| git | 2.51.0，仓库已 `git init` |
| NVIDIA GPU | **无**，`nvidia-smi` 不存在 |

硬性要求：

- **本机跑不了训练、GGUF 量化、llama.cpp 推理和 P95 延迟验证。** 不要在本地"验证通过"这些环节；一律放到 GPU 机器/容器，或明确标注未验证。
- 训练栈依赖 torch / bitsandbytes / trl，**没有装进主依赖**。它们是重依赖且与推理侧依赖冲突，需要时用 extra 单独装，装完立刻验证 wheel 可用性。
- 前端包管理用 **npm**（机器上没有 pnpm）。
- **PowerShell 控制台是 GBK 代码页**，中文文件名和中文输出会显示成乱码。文件本身是 UTF-8，没坏。要看中文内容用 read 工具或 `Get-Content -Encoding UTF8`，别去改文件编码。

## 不可违背的产品约束

这些是需求文档里写死的硬约束，代码和设计都不能违反：

- **三分类，不是二分类**：`decision` 枚举为 `black / white / gray`。灰样本贯穿合成比例、质量报告、评测细粒度指标。
- **`reason` 字段强制存在**，每条模型输出都要带决策依据，`max_length` 200。这是可解释性与合规的底线，不能做成可选。
- **合成数据不进测试集**。这是隐私与评测有效性的硬约束，必须在数据划分层强制，不能靠调用方自觉。行级 `DataOrigin` 标记就是为此存在的。
- **JEV 兼容是两个独立开关**，不要合并：
  - `jev_format_compat` → 输入输出格式、API 结构
  - `jev_training_compat` → 训练范式、超参、评测口径
  - 默认开启；`jev_format_compat` 关闭时输出 schema 不再保证 JEV 合规。
- **JEV 三级兼容**：L1 格式 / L2 格式+评测 / L3 风格。L3 依赖 Qwen 系底座，靠对比测试验证，不能靠声明——`RegistryModel` 里有校验器强制 `jev_level_evidence="measured"`，**没有对比报告就别想写 L3**。
- **评测优先级不可调换**：性能指标第一优先（P95 ≤ 100ms 端到端、峰值内存 ≤ 2GB），效果指标第二优先（准确率 ≥ 90%、召回率 ≥ 85%、误杀率 ≤ 5%、格式合规率 ≥ 99%）。这个排序是架构选型的依据（P95 100ms 决定了选 llama.cpp GGUF 路线）。
- **MVP 只做 1.5B–3B + LoRA/QLoRA**，是应对显存不足的主动取舍，不要在 MVP 引入更大底座。注册表里 7B 型号标了 `in_mvp_scope=False`。
- **数据划分比例固定 7 : 1.5 : 1.5**，自动完成。
- **默认量化档位 Q4_K_M**，与 JEV `eval_baseline.quant` 对齐。评测只在同量化档位下可比。
- **全链路版本关联是需求**：产物要能回溯到 `场景 sc_* / 数据 ds_* / 合成 syn_* / 基座 / 方法 / 模型版本 mv_*`。命名风格见需求文档 5.7.1，`Lineage` 有正则校验。
- **极速模式只要求用户完成第 1、2 步**，其余全自动——这是"30 分钟训出模型"的核心路径，任何新增配置项都不能强加到这条路径上。边界由 `RAPID_MODE_LAST_MANUAL_STEP` 常量定义，向导、编排层、UI 都读它。

## 界面禁用技术术语

需求文档第 2 章原则 4 规定：界面只显示业务语言，技术说法放在 hover 提示里。UI 文案一律走 `libs/ui-terminology`：

| 技术术语 | 界面文案 |
|---|---|
| LoRA 微调 | 快速风格适配 |
| QLoRA 低显存微调 | 低显存快速适配 |
| DPO 偏好对齐 | 决策优化训练 |
| 知识蒸馏 | 大模型能力迁移 |
| Q4_K_M 量化 | 标准压缩（推荐） |
| 分布拟合合成 | 智能样本扩增 |
| 特征重要性 | 关键判定因素 |

后端接口、数据库、代码里仍然用技术术语——映射只发生在前端文案层。

**这条有测试强制执行**：`apps/web/src/features/system/terminology-leak.test.ts` 每次跑测试都会扫全部组件源码，命中 `FORBIDDEN_IN_UI` 里的词就失败。新增训练方法或量化档位时，除了往 `TERM_COPY` 加映射表条目，**不需要**改这个测试。

## MVP 明确不做

需求文档 10.3：图结构数据合成与训练、联邦学习、多租户与模型市场、智能体编排、自动超参搜索、ONNX / MNN 导出、大规模分布式训练。

技术选型里若出现这些项，属于超出范围，要么砍掉要么显式标注为"阶段二/三"。

## 尚未在仓库中定义的东西

- **JEV 官方规范只有片段**。仓库里的 `jev_spec`（input_template / output_schema / eval_baseline）是需求文档转述的版本，不是官方原文；JEV 风格的完整定义、JEV 评测基准数据集都不在仓库里。涉及 JEV 精确对齐的结论必须外部核实后再写进代码，并在文档里标注来源。`/meta/jev-spec` 端点会在 `note` 字段里主动声明这一点。
- **P95 ≤ 100ms 的口径未定案**（技术方案 Q1）。3B 模型带 200 字符 `reason` 的输出约 150–250 token，100ms 内完成需要 1500–2500 tok/s，不现实。**不要在文档或代码里承诺任何 P95 数字**，也不要声称已达标。
- 目标反诈场景的真实字段规范（32 个字段）只有示意，没有实际 schema。
- 许可合规（Gemma / Llama / DeepSeek 各家许可）只要求"注册表记录、按场景过滤"，没有法务结论。注册表里只记录了 license 字符串。

完整未决问题清单见 `技术方案.md` 第 10 节，阻塞项是 Q1（P95 口径）和 Q3（JEV 官方规范）——**这两项定案前不要开始推理层与 JEV 对齐的编码**。

## 开工顺序

M0–M5 的代码已全部落地并提交（460 Python + 94 TypeScript 测试全绿）。

**尚未验证的环节——改动时不要假设它们已经跑通**：

| 环节 | 状态 | 缺什么 |
|---|---|---|
| 训练 SFT/LoRA/QLoRA/DPO | 代码完成，未验证 | `TrainingBackend` 实现 + torch/PEFT/TRL + GPU |
| 合并 LoRA + GGUF 量化 | 代码完成，未验证 | `LlamaCppTools` 实现 + llama.cpp + GPU |
| llama.cpp 推理 / 延迟 / 内存 | 代码完成，未验证 | `InferenceEngine` 实现 + llama.cpp + GPU |
| 效果指标（准确率/召回/误杀/F1/AUC） | **已本地验证** | — |
| 性能指标聚合（分位数/TTFT/QPS 口径） | **已本地验证** | — |
| JEV L2 / L3 兼容 | **不可声明** | Q3：官方规范与评测基准集不在仓库 |
| 保真度 / 最近邻距离阈值 | 只报告不阻断 | Q2 |

**"代码完成但未验证"是什么意思**：这三个环节的实际执行走 `son_trainer.TrainingBackend`、`son_quantizer.LlamaCppTools`、`son_inference.InferenceEngine` 三个协议。未实现的路径**显式抛错并说明原因**，不返回伪造的成功结果。`/api/deployments` 返回 `status=pending` 加启动命令，不谎称服务已启动。写实现时保持这个约定。

**下一步**：
1. 在 GPU 节点实现三个 Backend，跑通 M2 → M3 → M6，拿到实测数字
2. 落实 Q1（P95 口径）与 Q3（JEV 规范），这两项不定案，M3 性能验收与 L2/L3 无法完成
3. 用真实反诈字段替换 Q4 占位模板

已落地端点：`/health`、`/meta/jev-spec`、`/meta/echo-prediction`、`/api/datasets/{preview,quality,split,recommend-synth,synth}`、`/api/models`、`/api/models/{id}/adapter`、`/api/model-versions`、`/api/evaluations`、`/api/deployments`、`/api/jev/level-claim`。