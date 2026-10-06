# SystemOneStudio

零代码判断模型训练平台：自定义判断任务 → 导入决策样本 → 构造训练集 → 配方 → 候选概率训练 → 校准验证 → 发布服务。

**P3 已批准，正在实施。当前完成工程基线与 V3 CPU 数据准备，尚未完成真实 GPU Spike、完整七步产品及发布闭环。** Demo 和旧生成训练代码不能作为 V3 验收证据。

## 开发与接手

先读 [HANDOFF.md](HANDOFF.md)，然后按下列活动文件继续：

| 文档 | 用途 |
|---|---|
| [AGENTS.md](AGENTS.md) | 当前授权、仓库规范、算法/工程约束 |
| [需求重整 R3](需求重整.md) | 当前产品范围与七步需求 |
| [用户 V3 技术方案](SystemOneStudio_判断模型技术方案_v3.md) | 原始技术依据，原文保留 |
| [技术方案 T3](技术方案.md) | 仓库映射、持久化、执行与发布设计 |
| [开发计划 P3](开发计划.md) | W0–W9 与 A01–A10，已批准 |
| [用户 V3 Demo](SystemOneStudio-demo-v3.html) | 视觉/交互基准，所有成绩和任务均为演示 |
| [P3 实施记录](docs/implementation/2026-10-06-p3.md) | CPU 数据基础、实际证据与剩余工作 |

旧需求、方案与旧Demo在 [归档目录](docs/archive/2026-10-06/README.md)；[历史审计](docs/audit/2026-10-06/项目审计.md) 不代表当前全量状态。

## 本地运行

Node22/npm，Python workspace固定3.11，使用npm和uv。

```powershell
npm ci
uv sync --all-packages

# 独立终端运行
npm run dev:api
npm run dev

# 实现改动后完整检查
npm run verify
npm run build
npm run verify:py

# API/contracts变更后先生成，再检查
uv run python tools/gen_openapi_types.py
```

API默认9969，Vite默认5174。当前 `npm run up` 是CPU单studio Compose入口，没有worker profile。镜像构建尚有明确断点，见HANDOFF第6节。宿主环境设置、持久卷相对路径与新机器注意事项详见HANDOFF。

最近完整检查：前端106 tests、build通过；Python907 passed /1 GPU skipped /1既有warning。本机无NVIDIA GPU。真实训练、合并加载、量化、模型预测与性能尚无V3验收证据。

CPU数据准备：

```powershell
uv run python -m son_data_pipeline.decision_cli --dataset seed.csv --schema schema.json --mapping mapping.json --output prepared-new
```

文件格式、样例、保护机制与检查点详见HANDOFF第7节。结果是DATA_PREPARED/model_trained=false，既有输出目录拒绝覆盖。

## 当前范围

三分类black/white/gray；reason/evidence只作可选审计；候选概率CE+Brier、温度校准与AUTO/REVIEW策略。固定schema、分组划分、冻结测试、版本血缘；先真实GPU CLI再完整UI。先验证Transformers合并模型，GGUF单独验收。首版不含助手、业务模板、自定义Head、RLCD-like或分机代理。

控制面为Web。一体Linux容器发行，GPU主机支持路线为Linux或Windows WSL2/Docker，分别实测；Windows原生训练不在P0。
