# AGENTS.md — SystemOneStudio

## 1. 当前授权与阅读顺序

用户已于 2026-10-06 明确批准开发计划 P3，授权按 W0–W9 实施。后续开发不需要重复请求整体批准。没有 GPU 时继续可独立验证的 CPU 工作；不得宣称 GPU Spike 或完整产品已通过验收。

接手先读 HANDOFF.md → 需求重整.md R3 → 用户 SystemOneStudio_判断模型技术方案_v3.md → 技术方案.md T3 → 开发计划.md P3。SystemOneStudio-demo-v3.html 是视觉基准，里面的数字、训练动画、预设项目和服务状态均为演示。

冲突优先级：用户最新指示 > 用户 V3 技术方案 > 已批准 P3 的实施取舍 > Demo 视觉交互 > 原需求未被替代的约束。归档原需求在 docs/archive/2026-10-06/root-cleanup/需求方案.txt。历史方案、审计与旧测试不能覆盖当前基线。

仓库规范文件实际名称是 AGENTS.md；不要再维护一份内容不同的 AGENT.md。

## 2. 产品与算法约束

- Web 控制面不绑定 Windows；一体发行、内部进程分责。Linux 容器运行，Windows GPU 主机走经实测的 WSL2/Docker 路线；原生 Windows 训练栈不在 P0。
- 自定义任务名称、描述、判断问题与三类业务含义。七步：判断任务 / 决策样本 / 训练集构造 / 训练配方 / 判断模型训练 / 校准验证 / 发布服务。
- 三类逻辑 ID 固定 black/white/gray，业务含义随 schema 版本固化。gray 是类别，REVIEW 是处理动作，不能合并。
- 新判断模型只做一次 forward 的候选概率读出，不生成自由文本 decision/reason JSON。reason/evidence 是可选审计元数据，不能进入 State 或训练目标。
- 默认 Qwen3.5-2B-Base，原始 LM Head 受限读出 + LoRA + CE/Brier 联合损失；0.8B 单独验收。不能复用 Qwen2.5 加载类和 target modules 作为已验证事实。
- 默认 CE weight=1、label smoothing=0.05、Brier weight=0.2；Brier 使用未平滑 one-hot，按 1/K 归一化。双损失都必须有真实梯度证据。
- token map 必须在实际模板边界解析、验证单 token，并与 tokenizer revision 绑定；不硬编码 A/B/C token ID。候选换序必须同步重映射标签，评测映射回业务 ID。
- 序列化、schema、模板、tokenizer 与 token map 必须在训练、校准、评测和线上一致。
- group-aware 划分：默认 train/valid/cal/test=70/10/10/10；小数据显式共享 valid/cal 时为 70/15/15，并报告独立性降低。完整 group 隔离优先于精确比例，不自动拆组或悄悄切模式。
- 测试集只含原始种子，划分后冻结；构造仅作用于 train。训练描述符不得包含 test 路径。真实种子不足 200 条只作链路验证，重复或增强样本不增加独立证据。
- 校准用 held-out 数据拟合 Temperature，保存拟合来源/hash；阈值在 cal/valid 上按经验错误预算扫描，不能用 test 调参。无可行解用明确 review_all，零自动样本 error 未定义。
- 发布包绑定 actual runtime、calibration、policy、evaluation 和血缘。合并模型必须在新进程重新加载。GGUF 单独通过转换/logits/parity 门槛后开放；合并模型报告不能替代量化模型报告。
- 未测量、缺类、零分母、失败均显式区分；不返回 0/1 冒充实测，不用动画登记成功。
- 不承诺硬件无关 P95、训练时长或显存。所有外显量值来自 API/测量，估算需来源与未测量说明。
- P0 不做业务模板、助手、自定义 Head、ModernBERT、RLCD-like、DPO、教师合成、分机代理、多租户/市场、分布式或 ONNX/MNN。JEV L2/L3 不作为首版前置，也不得未经证据声明兼容。

## 3. 当前代码事实（2026-10-06）

已实现 V3 的契约、CSV 字段角色与类型导入、canonical serializer、候选换序重映射、group split、冻结清单和 CPU CLI。位置与用法见 HANDOFF.md 和 docs/implementation/2026-10-06-p3.md。

当前 App、RunWizard、Run API、Worker 的主流程仍是旧生成路线；新增 V3 类型未接产品 OpenAPI。SQLite 仍使用 001/002 SQL 迁移，Alembic 和完整领域持久化未做。校准/策略 package 未实现。默认 compose 只有 CPU studio 服务，没有 worker profile。CPU Dockerfile 引用不存在的 tsconfig.base.json，镜像首次构建有待修复断点。后端助手 API 仍在代码中，不等于 P0 授权上线助手。

最近完成的完整验证：npm verify 106 tests、npm build 成功；Python verify 907 passed / 1 GPU skipped / 1 既有 Starlette/httpx warning。这些是 CPU 工程证据，不能作为 GPU 或业务闭环验收。旧 460/94、101/877 等数字仅属历史。

本机 Node 22.22.3、npm 10.9.8，无 pnpm；workspace Python 3.11.12（系统 Python 不用于本项目）；Docker Desktop 可用，无 NVIDIA GPU。PowerShell 中文乱码通常是 GBK 输出问题，不要改 UTF-8 文件编码。近期 PowerShell 7 出现系统内存/页文件错误时可用 Windows PowerShell -NoProfile；不要以此为由改系统设置。

## 4. 工程边界

- libs/contracts 是跨服务类型唯一来源；服务不互相导入内部模块。保留 monorepo，不为了示意目录整体搬家。
- 前端类型由 tools/gen_openapi_types.py 从实际 API OpenAPI 生成，不手写新跨服务类型。
- 新包同步 uv workspace、mypy_path、pytest testpaths。libs/ui-terminology 是 TS 包，不能被 uv 的 libs/* 泛匹配。
- UI 业务文案走 libs/ui-terminology；算法术语放详情/提示。保留术语与硬编码量值检查，不为绕过测试删除规则。
- 保留旧 JEV/生成契约与测试供 legacy 读取；新 Decision* 契约独立。不要把所有旧测试改成新规则后宣称旧资产可兼容。
- 新领域版本不可变，修改上游创建新版本并标注依赖失效；旧已发布血缘不覆盖。模型包、导出状态、Job、Service 各自状态机。
- 写入使用事务；迁移前一致备份 DB + workspace。禁止 reset 用户改动、初始化覆盖 DB、提交真实运行数据库/上传数据/模型/凭据。
- Windows 删除或移动前核对绝对路径，保持在明确工作区或指定目标内，使用 LiteralPath。测试夹具写 tmp_path，不把 weights/GGUF 假文件丢到根目录。

## 5. 标准命令

```powershell
# 首次安装/依赖变更
npm install
uv sync --all-packages

# API/contracts 变更先生成
uv run python tools/gen_openapi_types.py

# 实现改动后完整检查，不以单测替代
npm run verify
npm run build
npm run verify:py

# 开发：分别运行
npm run dev:api
npm run dev

# 当前 CPU compose
npm run up
npm run logs
npm run down
```

API 默认 9969，Vite 5174。npm run up 是 CPU 实验入口，不能说已运行 GPU Worker 或新 V3 推理服务。GPU extra 隔离于 CPU 主依赖；安装后需验证 wheel/内核与实际加载，不能只做 import 就验收。

## 6. 下一工作与交付证据

按 P3：W1 真实 GPU CLI Spike 首先证明 tokenizer/slot/LoRA/CE+Brier/merge/cal/policy/test；无 GPU 可继续 W2 数据版本、迁移与校准算法的独立 CPU 部分。不要先接上完整训练 UI、展示假结果，再补算法。

每个工作包交付时更新 HANDOFF.md、实施记录、实际验证命令与未完成项。交付必须明确“代码存在”“CPU 逻辑通过”“真实 GPU 通过”“业务效果通过”四个层次，不能混写。历史归档用于追溯，不作为最新状态。
