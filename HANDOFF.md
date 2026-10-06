# HANDOFF — SystemOneStudio 开发交接

日期：2026-10-06。交接对象：接手工程师或编码代理。用户已批准 P3 实施，无需重新请求整体授权。

**当前交付的是工程基线和 V3 CPU 数据基础，不是完整判断模型训练产品。** 真 GPU Spike、V3 产品主链、校准/策略、真实发布和完整七步界面均未完成。以下内容用于从现状继续开发，不从头重新规划。

## 1. 先读与文件权威

1. AGENTS.md：现行仓库规范、授权、硬约束、检查命令。
2. 需求重整.md R3：本阶段产品范围、用户旅程和验收边界。
3. SystemOneStudio_判断模型技术方案_v3.md：用户编制的原始算法/产品依据，完整保留。
4. 技术方案.md T3：当前代码到目标架构的映射与落地设计。
5. 开发计划.md P3：已批准的 W0–W9、依赖、A01–A10 验收。
6. SystemOneStudio-demo-v3.html：用户视觉/交互基准；训练、概率、预设项目和服务全是模拟。
7. docs/implementation/2026-10-06-p3.md：前一批实际实施与测量记录。

用户最新指示优先。旧需求和方案在 docs/archive/2026-10-06/，不是当前权威；原始需求已移至 root-cleanup/需求方案.txt。历史审计现为 docs/audit/2026-10-06/项目审计.md。源码注释里的旧需求章节是历史引用，不能据此恢复强制reason、固定旧划分或生成式训练。

实际规范文件叫 AGENTS.md，不另维护一份 AGENT.md。用户 V3 两个原文件不改内容；必要修订落在 R3/T3/P3，记录与原文的关系。

## 2. 产品演进与已批准取舍

用户认为原项目概念好，但仅有 Web 外壳，要求真正跑通、极简操作、良好体验和有完成度的视觉。早期反诈模板、Windows专属控制面和两机协议已取消。

当前首版是 Web 一体应用，内部执行模块分责。先自定义名称/描述/判断问题和三类业务含义；不做业务模板、智能助手或分机代理。GPU宿主可以Linux，也可以Windows，但Windows走WSL2/Docker Linux容器路线，不承诺原生Windows训练。

V3把旧“生成decision/reason JSON”改为“固定候选logits → 概率 → 温度校准 → 策略 → frozen test → 发布”。保留原始LM Head做一次forward受限读出，不先上自定义Head或ModernBERT。

默认 Qwen3.5-2B-Base + LoRA + CE/Brier；0.8B独立验收。P3批准先完成可用Transformers合并模型runtime，GGUF通过转换/logits/parity才开放。RLCD-like与质量增强另阶段做。界面沿用户Demo，但里面的可选模型、技术路线和固定数字不能照搬成生产能力。

## 3. 完成状态：从这里接续

| 工作包 | 当前状态 | 接手重点 |
|---|---|---|
| W0 保护现场/恢复工程基线 | 文件与DB已备份，已知编译/格式问题已修复 | 保留旧资产与当前改动归属，不reset |
| W1 GPU CLI Spike | CPU数据入口已有；真实GPU部分未开始验收 | 首要技术门槛，确认真实模型与候选训练 |
| W2 契约/领域持久化/展廊 | Decision*契约有，API/完整DB/Alembic未接 | 先领域版本与迁移，别复用单JSON模拟 |
| W3 导入/冻结 | CPU函数和CLI有；产品上传映射界面未接 | 复用已测算法并接API/持久资产 |
| W4 构造/配方 | 换序模板重映射有；完整构造/配方未接 | 真平衡/字段换序、追溯、环境能力 |
| W5 训练监管 | 旧Worker和生成训练代码存在 | 新子进程监督与restricted loss不能复用假语义 |
| W6 校准/策略/评测 | 新域与完整算法未实现 | held-out、真实扫描、锁定候选后测试 |
| W7 runtime/模型包 | 旧生成推理/量化存在 | 新概率服务、新进程离线加载与parity |
| W8 迁移/一体发行 | 开发备份存在；产品备份恢复未做 | 一致性、hash、路径重映射、双宿主实测 |
| W9 用户旅程 | 未验收 | 七步无需脚本补库，真实端到端证据 |

当前应用仍是旧Run工作台，不能因为名称/配色相近就称V3已交付。助手入口在新App中不显示，但assistant router/package仍存在。旧项目/场景API部分内存；SQLite场景快照不等于全部领域持久化。

## 4. 代码地图与阅读入口

| 位置 | 看什么 |
|---|---|
| apps/web/src/App.tsx | 当前应用入口，RunWizard仍是旧业务链 |
| apps/web/src/app/routes.ts / routes.test.ts | 动态run路由和刷新/非法路径保护 |
| apps/web/src/features/wizard/RunWizard.tsx | 当前七步壳、上传/旧训练调用；需按V3重建语义 |
| apps/web/src/styles/workspace.css | 上轮深色工作区样式，不是完成的V3视觉 |
| apps/web/src/api/generated | 实际OpenAPI生成文件，不手写领域类型 |
| apps/api/src/son_api/main.py | 启动迁移/configure、旧routers、health/meta、静态Web挂载 |
| apps/api/src/son_api/routers/runs.py | 旧Run动作、上传、队列与CPU能力限制 |
| libs/contracts/src/son_contracts/decision.py | 新V3契约，独立于旧predict.py/JEV |
| services/data-pipeline/src/son_data_pipeline/decision_data.py | canonical_state、render_decision_input、target_slot、split_decision_seeds |
| services/data-pipeline/src/son_data_pipeline/decision_import.py | 显式CSV角色/类型导入，保留实体标识，错误不回显审计内容 |
| services/data-pipeline/src/son_data_pipeline/decision_cli.py | prepare/main，真实CPU数据资产写出 |
| libs/db/src/son_db/database.py | SQLite/WAL、事务和001/002迁移 |
| libs/db/src/son_db/models.py / repositories.py | 旧领域表与操作，不是完整V3领域 |
| services/worker/src/son_worker/__main__.py / queue.py | 旧轮询Worker、任务领取和中断检测 |
| services/worker/src/son_worker/skills.py | 仍用_render_prompt/_render_target生成训练，止于量化 |
| services/worker/src/son_worker/prepare.py | 旧CSV准备与重采样，不直接替代V3数据算法 |
| services/trainer、model-registry、inference、quantizer | 旧适配和GPU代码，要逐条审查，不自动算V3可复用 |
| services/evaluator | 旧分类/性能指标可审查复用，校准/选择性风险未集成 |
| deploy/compose/docker-compose.prod.yml | 只有CPU studio、9969和持久卷 |
| deploy/gpu/Dockerfile.gpu / 运行手册.md | 旧GPU构建路径、依赖/启动问题；未验收的新路线 |

services/calibration是已批准的新增包，目前尚不存在。新增时同步uv workspace、mypy_path、pytest testpaths、镜像workspace manifests。Python导入名与目录名不同，例如son_data_pipeline、son_model_registry、son_jev。

## 5. 已实现的数据基础细节

DecisionSchema为不可变结构，options固定三类；DecisionSample支持可选evidence。DecisionPredictResponse严格概率归一化、choice/confidence与最大候选概率一致；gray允许为高置信类别，action可AUTO或REVIEW。旧PredictResponse仍要求reason是legacy兼容，并非V3回退。

State任意嵌套拒绝目标/审计字段和NaN/Infinity。canonical_state按键排序、UTF-8、稳定null，有限整值float/负零归一；数组顺序和文本不变。不自动猜日期，日期应在输入规范化为ISO字符串。

split_decision_seeds只接受seed；拒绝重复sample_id、相同State跨group、同entity跨group。完整group分配优先于比例，缺类/小样本/共享cal降低独立性都报告。输入拷贝后冻结，verify可检测清单或样本修改。

当前函数边界冻结不是不可绕过的OS访问隔离；trainer必须消费显式training_inputs，禁止自行扫描test。磁盘读取时还要核对artifact清单，不能只信内存hash。P0三项构造需在划分后仅作用train；候选顺序的target_slot已测试六种排列，但还没接训练batch。

CSV字段必须全部显式映射。TEXT默认保留原值，数字选NUMBER，布尔选BOOLEAN，结构化值选JSON；entity不进入State。缺group/entity时要明确independent_rows。函数支持明确label_aliases，CLI目前标签必须规范black/white/gray，不自动猜中文标签。

## 6. 新机器启动与完整检查

仓库远程为 https://github.com/cers-ai/SystemOneStudio.git，当前工作分支main；交接提交应从远程正常clone/pull，不依赖原开发机临时文件。检查实际HEAD和git status，推送状态见本轮最终交付消息。

已知工具：Node22.22.3/npm10.9.8，无pnpm；workspace Python3.11.12（>=3.11,<3.12），不要用系统Python3.14。Docker Desktop可用，无NVIDIA GPU。

```powershell
npm ci
uv sync --all-packages

# 实现变更后的完整检查
npm run verify
npm run build
npm run verify:py

# 修改API/contracts后先生成，再完整检查
uv run python tools/gen_openapi_types.py
```

新增V3契约未被API使用，当前生成文档仍没有其产品端点；63 schemas不是V3接口已实现的证明。gen脚本导入create_app会打开/迁移DB，操作真实开发DB前备份；需要隔离时先设置SON_DB_PATH/SON_WORKSPACE到临时目录。

```powershell
# 两个独立终端
$env:SON_EXECUTION_MODE='cpu'
npm run dev:api

npm run dev
```

API http://127.0.0.1:9969，Vite http://127.0.0.1:5174。Vite代理/api、meta、v1、health、docs。若apps/web/dist已存在，API还会挂载静态Web；改前端后重新build或使用Vite，不把旧dist当最新页面。

```powershell
npm run up
npm run logs
npm run down
```

当前Compose定义为CPU实验，只有studio；镜像构建未在本批验收。已发现apps/api/Dockerfile的Web builder仍COPY根目录不存在的tsconfig.base.json，首次构建前需修复这个明确断点，不能把npm build通过等同Docker镜像可构建。不要使用旧文档的 --profile worker；没有该profile。SON_DATA_DIR相对路径按deploy/compose解析，建议外部持久化用绝对路径。主机默认DB为runtime/systemone.db；Compose内是/data/systemone.db和/data/workspace。

不要默认.env文件会被所有宿主命令加载：uvicorn直接运行需要shell环境变量；Compose有自己的.env查找规则。实际绑定路径用docker compose config核对。

## 7. 可复现CPU数据准备示例

在临时目录或ignored runtime下创建以下文件，仅用于fixture链路检查，不是业务模板或质量证据。

schema.json：

```json
{"schema_id":"fixture_schema","type":"choice","version":1,"question":"依据数值选择类别","options":[{"id":"black","name":"类一","description":"fixture类一"},{"id":"white","name":"类二","description":"fixture类二"},{"id":"gray","name":"类三","description":"fixture类三"}]}
```

mapping.json：

```json
[{"column":"entity","role":"ENTITY_ID"},{"column":"value","role":"STATE_INPUT","value_type":"NUMBER"},{"column":"label","role":"TARGET_CHOICE"},{"column":"evidence","role":"EVIDENCE"}]
```

seed.csv：

```csv
entity,value,label,evidence
001,1,black,audit-only
002,2,white,audit-only
003,3,gray,audit-only
004,4,black,audit-only
005,5,white,audit-only
006,6,gray,audit-only
```

```powershell
uv run python -m son_data_pipeline.decision_cli --dataset seed.csv --schema schema.json --mapping mapping.json --output prepared-new
```

输出必须为DATA_PREPARED、model_trained=false。目录包含schema.json、train/valid/cal/test.jsonl、split_manifest.json、training_inputs.json和artifact_manifest.json。training_inputs没有test路径、带schema哈希；清单记录真实样本/hash/警告。

再次运行同一output应退出码2拒绝覆盖。少于200条显示“链路验证”，缺类也是预期警告。无group显式用--independent-rows；共享模式--mode shared_valid_cal，不能默认自动切换。

之前实测30条fixture得到21/3/3/3，并逐文件校验；见实施记录。它不是训练成功，也不证明选择性风险或业务准确率。

## 8. GPU Spike：拿到节点后的第一项

需要确认GPU型号/显存、驱动、CUDA容器/WSL条件、系统、磁盘、缓存、网络与许可；不要在聊天/仓库存密码私钥。默认2B的具体内存和依赖需实际测量，不从旧Qwen2.5估算沿用。

按以下顺序形成可复现CLI：

1. 锁模型full revision和训练依赖；确认Qwen3.5正确加载类/文本路径/LoRA target modules。
2. 在真实模板边界解析稳定单token；冻结tokenizer/token map，覆盖padding/截断/最后有效slot。
3. 候选softmax；CE smoothing0.05；Brier未平滑one-hot、1/K、weight0.2，分别验证梯度。
4. 真实LoRA训练，只用valid选checkpoint；保留配方、随机种子、显存/耗时/曲线。
5. 合并并新进程重新加载，验证候选读出；不得只测adapter模型。
6. held-out cal拟合T和策略，锁candidate后独立读取test报告。
7. 产出spike_report、config、token_map、cal/policy/eval、merged manifest和复现命令。

没有GPU时W1保持未完成。0.8B/GGUF单独探针，不因为2B或导入依赖成功就开放组合。旧trainer的train extra有torch/TRL/bitsandbytes等宽松约束，不是Qwen3.5验收版本；GPU栈隔离，不能盲目全装进CPU主依赖。

## 9. 可并行准备的独立CPU工作顺序

不需要GPU即可推进，不需要再次整体批准：

1. W2领域持久化：新增schema/data/build/recipe/cal/policy/eval/service/asset关系，保留旧project/run。先迁移基线/备份，接Alembic；不可覆盖001/002历史数据。
2. 契约/API一致：新领域响应用contracts，生成TS类型。API幂等、失效规则和版本绑定有测试；不手写新TS领域接口。
3. W3上传与映射：接现有导入函数，数据质量与分组确认可修复；冻结资产持久化，重启后恢复。
4. W4真实构造：平衡/字段换序/候选换序，来源和样本数准确；不调用教师自证标签。
5. calibration基础：数值稳定概率、NLL/Brier/ECE、logT拟合、风险扫描与review_all；用fixture测数学，不能登记真实calibrated模型。
6. 视觉规范和普通数据页面可准备；完整训练主链UI集成等W1通过，遵守“先证明范式，再接完整UI”。

W5之后子进程监管、恢复、GPU租约、模型登记、runtime和备份按P3继续。每步产出实际证据和交接更新，不只报告模块单测。

## 10. 关键隐患与验收口径

- 旧Worker训练是文本目标，不能把函数改名就算restricted loss；也不能手工登记model/eval来补主链。
- 旧Golden Path E2E含人工夹具，不能据文件名声称业务流程已跑通。
- 旧API的meta echo不是真推理，pending deployment不等于服务运行。
- 校准不能接触test；阈值error分母为自动有效预测，零覆盖error未定义。budget3%是经验输入，不保证线上错误率。
- scalar Temperature不改变argmax，不能以准确率上升作为必然校准成功；raw/cal指标未改善应如实报告。
- 量化改变概率时独立cal/policy，最终测试绑定实际发布runtime；不能使用test选量化配置。
- gray!=REVIEW，AUTO也不等于自动放行；含义由业务schema决定。
- 迁移备份不只复制正在写的.db；WAL/asset需一致快照。恢复不保留PID和RUNNING状态。
- 工作区导出、模型导出和Service运行分开验收；不能把文件存在当格式/加载有效。
- 当前State/字段类型/分组保护为初版基础，仍需产品质量报告、完整字段类型与日期规范化、实际资产重读校验。

## 11. 验证证据与本批整理

前一批完整结果：前端106 tests与build通过；Python907 passed /1 GPU skipped /1既有Starlette/httpx warning。日志在docs/audit/2026-10-06/p3/python-verify.txt。本批文档整理与量化测试夹具隔离的最终重跑日志放在docs/audit/2026-10-06/handoff/；以实际日志为准，失败不能沿用旧数字。

发现test_quantizer.py曾用Path('.')写7字节merged.safetensors与out内placeholder GGUF；本批改用tmp_path，防止测试污染根目录。根目录旧Demo/需求归档、审计移位；模型假文件/输出/工具缓存清理，node_modules/.venv/runtime保留。

本批Git提交包含先前已授权但未提交的代码/数据fixture/文档改动，不应归因为单纯文档变更。真实模型、运行DB、个人上传与.env不提交。尚未实施的域/算法不得在提交说明中声称完成。

## 12. 本机保护快照与异机交接

本机实施前快照：C:\Users\Administrator\AppData\Local\Temp\systemone-p3-approved-9pz_i3wb，含63文件、binary diff及runtime DB backup。该本地DB曾为version2、1project/1run、0datasets/models/jobs；这是历史本机快照，不是新机器默认内容。

本机CLI证据：C:\Users\Administrator\AppData\Local\Temp\systemone-v3-cli-check-94tukqo5。临时目录可能被OS清理，接手者不依赖这些目录；CPU示例和测试可重新生成。

GitHub交接是代码/fixture/文档，不含真实运行状态。若需交用户工作区，用独立渠道传一致DB/资产备份，先核对hash与隐私；产品备份恢复功能尚未实现，不能误认为Git clone已恢复业务数据。

完成下一个工作包后更新本文件中的状态、环境、证据及剩余工作，同时更新需求/设计/计划必要差异。先保留现场、重跑检查，再沿已批准路线继续。

### 根目录清理回执

递归永久删除被自动审批以blocked by policy拒绝，未执行；改用可恢复归档后已完成移出根目录。测试假模型/out及工具缓存保存于本机临时目录 C:\Users\Administrator\AppData\Local\Temp\systemone-root-cleanup-2f8824f3fec844eab81f6acbd8f8d8a3，逐项回执见docs/audit/2026-10-06/handoff/cleanup.json。node_modules、.venv和runtime未删除。
