# SystemOneStudio

零代码判断模型训练工作台。新方向是候选概率判断、温度校准与自动处理/人工复核策略。产品采用项目展廊和七步沉浸工作台，目标是从决策样本走通真实训练、验证、发布与迁移。

**P3 已获用户批准，正在实施，应用尚未完成闭环。**旧实现基于文本生成训练，不能视为已实现 V3 判断模型。

## 当前活动文档

| 文档 | 用途 | 状态 |
|---|---|---|
| [V3 Demo](SystemOneStudio-demo-v3.html) | 用户编制的视觉与交互基准 | 原文保留；数据为演示 |
| [V3 判断模型技术方案](SystemOneStudio_判断模型技术方案_v3.md) | 用户编制的新技术依据 | 原文保留 |
| [开发计划 P3](开发计划.md) | 范围、差异决策、W0–W9、A01–A10 验收 | 已批准，实施中 |
| [AGENTS.md](AGENTS.md) | 规划门槛、仓库约束和执行规范 | V3 补注优先 |
| [P3 实施记录](docs/implementation/2026-10-06-p3.md) | 工程基线、CPU 数据准备、证据与剩余工作 | 最新状态 |
| [项目审计](项目审计.md) | 早期代码与运行证据 | 历史快照，不代表当前全绿 |

[需求方案.txt](需求方案.txt) 保留未被用户新指示替代的约束；[需求重整 R2](需求重整.md) 与 [技术方案 T2](技术方案.md) 保留追溯。改写前文档及 hash 清单见 [P3 规划前归档](docs/archive/2026-10-06/pre-v3-plan/manifest.json)。

## 当前验证状态

P3 首批实施已修复前端路由/术语类型与 Python 格式问题。当前完整 npm verify 通过（106 个测试）、build 成功；Python verify 通过（907 passed / 1 GPU skipped / 1 既有 warning）。CPU 数据准备 CLI 已实际生成分区与校验清单；V3 产品主链仍未完成。

本机无 NVIDIA GPU。真实训练、合并加载、模型预测、量化与性能尚无当前 V3 验收证据；CPU 算法测试、Demo 动画和固定报告不能替代它们。

## 开发与检查

```powershell
# 首次安装或依赖变更
npm install
uv sync --all-packages

# 开发入口
npm run dev:api
npm run dev

# 契约/API 变更后先生成
uv run python tools/gen_openapi_types.py

# 实现改动后的完整验证
npm run verify
npm run build
npm run verify:py
```

API 默认 9969，Web 开发默认 5174。`npm run up` 是现有 Compose 入口，不能作为已通过 V3 GPU 全链路验收的证明。新发行使用 Linux 容器，兼容路线为 Linux GPU 或满足条件的 Windows WSL2/Docker GPU；Windows 原生训练栈不在 P0。

P3 已获批准，先保护现有改动和恢复工程基线，再做真实 GPU CLI Spike；通过后接入产品主链，最后完成发布、迁移和浏览器体验验收。
