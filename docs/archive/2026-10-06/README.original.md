# SystemOneStudio

零代码决策模型训练平台。业务人员上传样本数据、选一个场景模板，平台自动完成数据合成、模型训练、量化部署，输出可直接对接业务系统的决策模型。

产品需求见 [`需求方案.txt`](需求方案.txt)（唯一权威来源）。架构见 [`技术方案.md`](技术方案.md)，里程碑见 [`开发计划.md`](开发计划.md)。

> ## ⚠️ 当前状态：代码完成，关键能力未验证
>
> **本机没有 NVIDIA GPU**，训练、GGUF 量化、推理、延迟与内存指标**一次都没有真实执行过**。
> 这些环节的实际运行走三个协议边界（`TrainingBackend` / `LlamaCppTools` / `InferenceEngine`），
> 未实现的路径**显式抛错并说明原因**，不返回伪造的成功结果。
>
> 已完整本地验证的只有**效果指标**（准确率、召回率、误杀率、F1、AUC-ROC、格式合规率、分标签细粒度），
> 因为它们是纯计算。
>
> 另有三项需要产品侧定案，见 [`技术方案.md` 第 10 节](技术方案.md)：
> - **Q1**：`响应速度 P95 ≤ 100ms` 与 `判定依据 200 字符上限`互相冲突（详见下文「已知矛盾」）
> - **Q3**：JEV 官方规范与评测基准集不在仓库内，因此**只能声明 L1 兼容**，不能声明 L2/L3
> - **Q4**：反诈场景 32 个字段的真实 schema 未提供，当前用占位模板

---

## 快速开始

### 容器化部署（推荐）

```bash
npm run up       # build + docker compose up -d
open http://127.0.0.1:9969
```

`9969` 是唯一对外入口：nginx 托管前端并代理 `/api`、`/meta`、`/health`、`/v1`、`/docs`。
API 容器**不映射到宿主机**，避免出现第二条未经代理的路径。

```bash
npm run logs     # 跟踪容器日志
npm run down     # 停止并移除
```

容器编排包含 `nginx` + `api` + `postgres` + `redis`。**不含**训练、量化与推理服务——
它们需要 GPU，镜像也达到 GB 级，在 GPU 节点上单独部署。

### 本地开发

```bash
# Python：11 个包在 uv workspace 中，固定 3.11
uv sync --all-packages
npm run dev:api            # uvicorn :9969

# 前端（npm workspaces；本机 5173 被占用，故用 5174）
npm install
npm run dev                # vite :5174，已配置代理到 :9969
```

---

## 技术栈

| 层 | 选型 |
|---|---|
| 前端 | React 18 + Vite + TypeScript + TanStack Query + ECharts + AntV X6 |
| 后端 | FastAPI + Pydantic + SQLAlchemy |
| 任务调度 | 编排状态机 + Redis 队列（Prefect 接入待办） |
| 训练 | Transformers + PEFT + TRL + bitsandbytes（**optional extra，未安装**） |
| 量化 / 推理 | llama.cpp（**需要 GPU 节点**） |
| 数据合成 | 自研 Gaussian Copula（SDV 为 optional extra） |
| 存储 | PostgreSQL（元数据 + 表格型数据）+ 模型产物卷。**不用 MinIO**，理由见 [`技术方案.md` 3.0](技术方案.md) |
| 部署 | Docker + nginx |

> **关于 MinIO**：需求 11 技术栈表写的是 `MinIO + PostgreSQL`，本项目**不采用**。
> 按需求自己的数字，种子数据与合成数据都在 PostgreSQL 舒适区内且本质关系型；
> 唯一的大件是模型权重（合并权重约 6GB），而它是一次性写入、部署时读取的产物，
> 属于文件系统而非对象存储。这是一个**对权威文档的显式偏离**，记录在 `技术方案.md` 3.0，**需产品确认**。

---

## 仓库结构

```text
apps/web              React 前端：七步向导、画布、训练监控、评测报告、部署
apps/api              FastAPI 控制面
libs/contracts        跨服务 Pydantic 契约 + JEV 常量          ← 类型唯一来源
libs/model-registry   底座注册表 + BaseModelAdapter 抽象
libs/ui-terminology   技术术语 → 业务文案映射表（唯一合法的 UI 文案来源）
services/             能力引擎层
  data-pipeline       治理、质量评分、7:1.5:1.5 划分
  synth               三种合成模式 + 保真度 + 隐私报告
  trainer             SFT / LoRA / QLoRA / DPO 编排      [GPU]
  quantizer           合并 + GGUF 量化                    [GPU]
  inference           推理编排 + schema 强制              [GPU]
  evaluator           性能指标（第一优先）+ 效果指标
  jev-adapter         JEV 格式 / 训练 / 评测对齐
  orchestrator        7 步流程状态机
tools/                codegen
deploy/compose        容器编排
deploy/nginx          网关配置
```

---

## 命令

```bash
# 前端 + TypeScript 库
npm run verify         # lint -> typecheck -> test
npm run build          # tsc -b && vite build

# Python
npm run verify:py      # ruff format -> ruff check -> mypy -> pytest

# 契约类型重新生成（改了 libs/contracts 或 apps/api 之后）
uv run python tools/gen_openapi_types.py

# 单点验证
uv run pytest services/data-pipeline/tests    # 只跑某个包
uv run pytest -k split                        # 按名字过滤
npm test --workspace libs/ui-terminology      # 只跑某个前端 workspace
```

---

## 不可违背的产品约束

这些来自需求文档，由代码与测试强制，改动前请先读 [`AGENTS.md`](AGENTS.md)：

- **三分类，不是二分类**：`decision` 为 `black / white / gray`。灰样本贯穿合成比例、质量报告与评测细粒度。
- **`reason` 强制存在**：每条模型输出都要带决策依据，200 字符上限。这是可解释性与合规的底线。
- **合成数据不进测试集**：结构性保证——测试行从一个只含种子行的池子里抽取。有一个测试故意注入泄漏、证明断言不是死代码。
- **JEV 双开关独立**：`jev_format_compat`（输出格式 / API）与 `jev_training_compat`（训练范式 / 评测口径）不合并。
- **JEV L3 必须实测**：没有对比测试报告，注册表校验器直接拒绝写入 L3。
- **评测优先级不可调换**：性能指标第一优先，效果指标第二优先。这决定了推理层选型。
- **MVP 只做 1.5B–3B + LoRA/QLoRA**：显存不足的主动取舍。
- **数据划分固定 7 : 1.5 : 1.5**。
- **极速模式只要求用户完成前两步**，边界由 `RAPID_MODE_LAST_MANUAL_STEP` 常量定义，编排层与前端读同一个值。
- **界面禁用技术术语**：界面只显示业务语言，技术说法在 hover 提示里。有一条测试每次跑都会扫全部组件源码。

---

## 已知矛盾与未决项

**需求内部矛盾（已标注，未擅自修改）**

1. `P95 ≤ 100ms` 与 `reason` 200 字符不可兼得。200 字符中文约需 130–200 token，
   加上 JSON 包裹约 150–250 token，100ms 内完成需要 1500–2500 tok/s —— 3B 模型上不现实。
   **评测接口因此对 P95/P99 返回 `passed=null` 并注明原因，不静默通过或失败。**
2. 「30 分钟训出模型」与「3B 底座约 45 分钟训练」矛盾。

**未决问题**

| # | 问题 | 影响 |
|---|---|---|
| Q1 | 响应速度 P95 的口径 | 阻塞性能验收 |
| Q2 | 保真度 / 最近邻距离阈值 | 当前只报告不阻断 |
| Q3 | JEV 官方规范与基准集 | 阻塞 L2/L3 声明 |
| Q4 | 反诈场景 32 字段真实 schema | 阻塞场景模板 |
| Q5 | 各底座许可的法务结论 | 注册表只记录 license 字符串 |

完整清单见 [`技术方案.md` 第 10 节](技术方案.md)。

---

## 环境要求

| 项 | 版本 |
|---|---|
| Node | ≥ 22（**不要用 pnpm**，本机未安装） |
| Python | **3.11**（训练依赖在 3.14 上无可验证 wheel） |
| uv | ≥ 0.11 |
| Docker | ≥ 28 |
| GPU | **训练 / 量化 / 推理必需**，本机无 |

---

## 状态

```
tests: 472 Python + 94 TypeScript，两栈全绿
已验证: 效果指标、状态机、契约校验、术语映射、数据治理不变量、容器编排
未验证: 训练、GGUF 量化、推理、延迟、显存
```