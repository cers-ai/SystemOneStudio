# SystemOneStudio_改造开发方案.md

> 版本：V1.0  
> 目标：**先把 SystemOneStudio 做成一个真正能跑通、真正可用的轻量系统，再逐步扩展平台能力。**  
> 基础：结合现有 `需求方案.txt`、`技术方案.md`、`开发计划.md` 与当前仓库源码诊断结果整理。  
> 核心调整：**移除 PostgreSQL、Redis；采用 SQLite + 本地文件资产；不做重型分布式架构；优先打通“上传 → 治理 → 合成 → 训练 → 量化 → 评测 → 部署 → 预测”全链路。**

---

# 1. 改造结论

当前 SystemOneStudio **不是空壳项目**。现有代码中，数据治理、数据划分、部分合成、评测指标、训练后端、量化工具封装、推理封装、契约模型、前端页面骨架等都已经有一定实现质量。

但当前最大问题不是“缺几个功能”，而是：

> **大量组件已经实现，但没有被真正串成一条可运行的产品主链。**

当前系统更接近：

```text
若干质量不错的能力组件
+
较完整的前端页面
+
较多单元测试
+
尚未贯通的业务工作流
```

而不是：

```text
业务人员选择场景
→ 上传数据
→ 自动治理
→ 自动合成
→ 选择模型
→ 提交训练
→ GPU执行
→ 量化
→ 评测
→ 部署
→ 在线预测
```

因此，本次改造不继续扩能力、不继续铺平台，而是做一次明确的 **“轻量化 + 全链路穿刺”**。

---

# 2. 本次改造的三条最高原则

## 2.1 原则一：先能用，再平台化

第一阶段只回答一个问题：

> **SystemOneStudio 能不能让一个业务用户从上传数据开始，最终得到一个真实可调用的模型服务？**

在这个问题得到肯定答案之前，不再增加：

- 更多模型；
- 更多训练算法；
- 更复杂的分布式架构；
- 更复杂的画布编排；
- 更多智能助手能力；
- 多租户；
- 模型市场；
- 自动超参搜索；
- Prometheus / Grafana；
- 独立微服务拆分。

---

## 2.2 原则二：数据库和任务系统必须轻

本次明确：

### 移除

- PostgreSQL；
- Redis；
- 当前基于“未来分布式部署”设计出来的外部任务队列依赖。

### 改为

- **SQLite：保存项目、Run、数据资产、任务、模型版本、评测、部署等元数据；**
- **本地文件系统：保存 CSV/JSONL、训练集、合成集、LoRA、模型权重、GGUF、日志等大文件；**
- **SQLite Job Queue + 单机 Worker：执行训练、量化、评测、部署等长任务。**

目标不是追求理论上的高并发，而是：

> **单机可靠、结构清楚、重启可恢复、开发者能看懂、业务链真的能跑通。**

---

## 2.3 原则三：后端状态是真相，前端只展示

当前前端存在自己的 Wizard 状态机，后端又有自己的 Workflow 状态机，两者实际并没有形成同一份运行状态。

改造后必须遵守：

```text
Backend Run State = Source of Truth
Frontend = Backend State Projection
```

前端不能因为用户点了“下一步”，就自行认为该步骤完成。

只有后端真正生成了对应资产，Run 状态才能推进。

---

# 3. 当前代码诊断：必须解决的问题

---

## 3.1 P0：七步向导只是页面导航，没有真正业务串联

当前 `apps/web/src/features/wizard/StepBody.tsx` 中，各步骤的数据没有形成统一上下文。

典型问题包括：

### 问题 A：上传文件只存在于局部组件 State

第二步：

```tsx
case 'seed_uploaded':
  return <UploadStep />;
```

上传后的 `File` 保存在 `UploadStep` 自己内部。

离开页面后，这个文件并没有形成一个真正的 `dataset_id`。

---

### 问题 B：第三步重新创建上传组件

当前代码中：

```tsx
function QualityStep() {
  return <UploadStep />;
}
```

因此第三步并不是使用第二步的数据做质量分析，而是重新创建一套上传页面。

---

### 问题 C：第四步明确传入 `file={null}`

当前存在类似：

```tsx
<SynthStep file={null} />
```

所以第四步并没有真正接收到之前的数据资产。

---

### 问题 D：“下一步”与业务结果无关

当前 Wizard 基本上是：

```text
用户点击下一步
→ wizard.complete(current)
→ 前端状态推进
```

而没有验证：

- Scene 是否真正保存；
- Dataset 是否真正上传；
- Quality 是否真正执行；
- Split 是否真正生成；
- Synth 是否真正生成；
- Model 是否真正选择；
- Training Config 是否真正保存；
- Job 是否真正创建。

### 改造要求

彻底取消“前端自行完成步骤”。

统一改为：

```text
调用后端动作
→ 后端成功创建资产
→ 后端更新 Run 状态
→ 前端重新 GET Run
→ 页面进入下一阶段
```

---

# 4. 新的轻量总体架构

本阶段不要微服务化。

运行形态控制在四个部分：

```text
┌──────────────────────────────────────────────┐
│              SystemOneStudio Web             │
│ React                                        │
│                                              │
│ 七步向导 / 项目 / Run / 训练进度 / 评测       │
└──────────────────────┬───────────────────────┘
                       │ HTTP
                       ▼
┌──────────────────────────────────────────────┐
│                FastAPI Backend               │
│                                              │
│ Project / Run / Dataset / Job                │
│ Model / Evaluation / Deployment              │
│                                              │
│ SQLite Repository                            │
└───────────────┬──────────────────────────────┘
                │
         SQLite + Workspace
                │
                ▼
┌──────────────────────────────────────────────┐
│               Local Job Worker               │
│                                              │
│ Data Prep                                    │
│ Synth                                        │
│ Train                                        │
│ Merge                                        │
│ Quantize                                     │
│ Evaluate                                     │
│ Deploy                                       │
└───────────────┬──────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────┐
│              llama.cpp Server                │
│              /v1/predict                     │
└──────────────────────────────────────────────┘
```

---

# 5. 不再建设“分布式控制面”

现阶段明确取消：

```text
PostgreSQL
Redis
Celery
Prefect
多 Worker
跨节点 Scheduler
分布式锁
服务注册中心
模型对象存储
MinIO
```

第一阶段只支持：

> **一个 SystemOneStudio 实例 + 一个 SQLite + 一个 Worker + 一块 GPU。**

如果未来需要多 GPU、多用户、多节点，再升级。

但现在不为未来假设提前建设复杂度。

---

# 6. SQLite 设计

## 6.1 SQLite 只存元数据，不存大模型文件

SQLite 保存：

- 项目；
- Run；
- Dataset 版本信息；
- Synth 版本信息；
- Training Job；
- Model Version；
- Artifact 索引；
- Evaluation；
- Deployment；
- 简单 Audit。

SQLite 不保存：

- 原始 CSV 大文件；
- 合成数据主体；
- 模型权重；
- LoRA；
- GGUF；
- checkpoint；
- 大型日志。

这些全部放 Workspace 文件系统。

---

## 6.2 数据库位置

默认：

```text
SystemOneStudio/
└─ runtime/
   ├─ systemone.db
   └─ workspace/
```

环境变量：

```text
SON_DB_PATH=./runtime/systemone.db
SON_WORKSPACE=./runtime/workspace
```

---

## 6.3 SQLite 配置

启动时执行：

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
```

原因：

- WAL 允许前端查询和 Worker 写入较稳定地并存；
- 当前只有一个主要 Worker，不存在高并发写冲突；
- 不需要引入额外数据库服务。

---

# 7. 核心数据模型：以 Run 为中心

本次改造最重要的新对象是：

# `Run`

一个 Run 代表：

> **用户从场景选择到最终模型部署的一次完整建模任务。**

---

## 7.1 projects

```text
projects
--------
id
name
description
created_at
updated_at
deleted_at
```

---

## 7.2 runs

```text
runs
----
id
project_id

scene_code
mode

state

dataset_id
split_id
synth_id

base_model_id
training_config_json

training_job_id
model_version_id
evaluation_id
deployment_id

created_at
updated_at
error_message
```

---

## 7.3 datasets

```text
datasets
--------
id
run_id

code
original_filename
source_path

rows
cols
label_column

checksum
status

created_at
```

实际数据保存在：

```text
workspace/{run_id}/datasets/ds_v1/source.csv
```

---

## 7.4 dataset_splits

```text
dataset_splits
--------------
id
dataset_id
code

train_path
valid_path
test_path

train_rows
valid_rows
test_rows

created_at
```

必须继续保持：

```text
test 中 origin=synth 数量恒为 0
```

这是现有代码中值得保留的硬约束。

---

## 7.5 synth_runs

```text
synth_runs
----------
id
run_id
dataset_id

code
method
config_json

output_path
rows

fidelity_score
privacy_json

status
created_at
```

---

## 7.6 jobs

所有长任务统一一张表。

```text
jobs
----
id
run_id

type
status

progress
stage

payload_json
result_json

pid
log_path

created_at
started_at
finished_at

error_message
```

`type`：

```text
SYNTH
TRAIN
QUANTIZE
EVALUATE
DEPLOY
PIPELINE
```

`status`：

```text
QUEUED
RUNNING
SUCCEEDED
FAILED
CANCELLED
INTERRUPTED
```

---

## 7.7 model_versions

```text
model_versions
--------------
id
run_id
code

base_model_id
training_method

adapter_path
merged_model_path

created_at
```

---

## 7.8 artifacts

```text
artifacts
---------
id
model_version_id

type
format
quant

path
checksum
bytes

created_at
```

第一阶段主要：

```text
LORA
MERGED_MODEL
GGUF_Q4_K_M
```

---

## 7.9 evaluations

```text
evaluations
-----------
id
run_id
model_version_id
test_dataset_path

metrics_json
performance_json
error_cases_path

created_at
```

---

## 7.10 deployments

```text
deployments
-----------
id
run_id
model_version_id
artifact_id

runtime
host
port
pid

status
started_at
stopped_at
```

---

# 8. Workspace 文件结构

所有大文件统一按 Run 保存：

```text
runtime/
├─ systemone.db
└─ workspace/
   └─ run_000001/
      ├─ dataset/
      │  ├─ source.csv
      │  ├─ train.csv
      │  ├─ valid.csv
      │  └─ test.csv
      │
      ├─ synth/
      │  └─ syn_v1.csv
      │
      ├─ training/
      │  ├─ config.json
      │  ├─ logs/
      │  ├─ checkpoints/
      │  └─ adapter/
      │
      ├─ model/
      │  ├─ merged/
      │  └─ model-q4_k_m.gguf
      │
      ├─ evaluation/
      │  ├─ predictions.jsonl
      │  ├─ errors.jsonl
      │  └─ report.json
      │
      └─ deployment/
         └─ llama-server.log
```

这套结构比 MinIO / PostgreSQL 混合存储更容易理解、备份和调试。

---

# 9. Run 状态机

前端七步必须完全映射到后端 Run 状态。

建议状态如下：

```text
CREATED

SCENE_READY

DATASET_READY

DATA_QUALITY_READY

TRAIN_SET_READY

MODEL_SELECTED

TRAINING_CONFIGURED

QUEUED

TRAINING

MERGING

QUANTIZING

EVALUATING

MODEL_READY

DEPLOYING

SERVING

FAILED
```

---

# 10. 七步向导重新定义

---

## 第 1 步：选择场景

用户动作：

```text
选择“反诈账户判定”
```

后端：

```http
PATCH /api/runs/{run_id}/scene
```

成功后：

```text
state = SCENE_READY
```

前端不得自己推进状态。

---

## 第 2 步：上传种子数据

只上传一次。

接口：

```http
POST /api/runs/{run_id}/dataset
```

后端：

1. 文件写入 Workspace；
2. 计算 checksum；
3. 识别标签；
4. 创建 `datasets` 记录；
5. 返回 `dataset_id`。

成功：

```text
state = DATASET_READY
```

以后所有操作只传：

```text
dataset_id
```

不再重复上传 CSV。

---

## 第 3 步：数据治理

接口：

```http
POST /api/runs/{run_id}/prepare-data
```

执行：

```text
质量检查
→ 脱敏
→ 数据清洗
→ 7:1.5:1.5 划分
→ 生成 train / valid / test
```

继续复用现有：

- masking；
- quality；
- split；
- gray 标签处理；
- synth 不进入 test 的断言。

成功：

```text
state = DATA_QUALITY_READY
```

---

## 第 4 步：数据合成

当前已有合成能力可以保留，但输出必须资产化。

接口：

```http
POST /api/runs/{run_id}/synth
```

结果不能只返回：

```text
rows
score
privacy
```

必须生成：

```text
syn_v1.csv
+
synth_runs DB record
```

训练输入：

```text
train.csv + syn_v1.csv
```

test 永远：

```text
原始 seed test
```

成功：

```text
state = TRAIN_SET_READY
```

---

## 第 5 步：选择模型

第一阶段只真正支持：

```text
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
```

甚至开发团队可以优先只打通：

```text
Qwen2.5-1.5B-Instruct
```

其它：

```text
Gemma
Llama
自研分类器
```

暂时只保留注册信息，不允许 UI 显示为“已支持”。

接口：

```http
PATCH /api/runs/{run_id}/model
```

成功：

```text
state = MODEL_SELECTED
```

---

## 第 6 步：训练配置

第一阶段只支持：

```text
LoRA SFT
```

暂时不做：

```text
DPO
QLoRA
知识蒸馏
QAT
少样本迁移
```

原因不是这些能力没有价值，而是：

> 当前最重要的是证明 Model Factory 主链成立。

接口：

```http
PATCH /api/runs/{run_id}/training-config
```

成功：

```text
state = TRAINING_CONFIGURED
```

---

## 第 7 步：一键执行

按钮：

```text
开始训练
```

接口：

```http
POST /api/runs/{run_id}/start
```

创建：

```text
PIPELINE Job
```

后台自动执行：

```text
TRAIN
↓
MERGE
↓
QUANTIZE
↓
EVALUATE
↓
MODEL_READY
```

成功后页面显示：

```text
训练完成
模型版本
准确率
召回率
误杀率
F1
GGUF 文件
部署按钮
```

---

# 11. SQLite Job Queue：替代 Redis

不需要 Redis。

Worker 每隔短时间查询：

```sql
SELECT *
FROM jobs
WHERE status='QUEUED'
ORDER BY created_at
LIMIT 1;
```

领取任务时使用事务：

```text
BEGIN IMMEDIATE
→ 检查状态
→ 更新 RUNNING
→ COMMIT
```

MVP 第一阶段：

```text
只允许 1 个 Worker
只同时执行 1 个 GPU Job
```

完全够用。

---

# 12. Worker 设计

新增：

```text
apps/worker/
```

或者：

```text
services/worker/
```

入口：

```bash
python -m son_worker
```

Worker 做三件事：

```text
1. 从 SQLite 读取 QUEUED Job
2. 调现有 Python capability package
3. 持续写回 progress / state / result
```

---

## 12.1 不要重新实现现有能力

Worker 直接复用：

```text
son_data_pipeline
son_synth
son_trainer
son_quantizer
son_evaluator
son_inference
```

这些仍然保持 Python package。

区别是：

> 它们不再被当作独立“微服务”，而是本地能力库。

---

# 13. GPU 镜像改造

当前 GPU Docker 最后启动的仍然是 FastAPI API。

这不合理。

改成：

```dockerfile
CMD ["python", "-m", "son_worker"]
```

API 负责：

```text
控制
状态
查询
提交 Job
```

GPU Worker 负责：

```text
执行
```

第一阶段 API 和 Worker 可以在同一台 GPU 主机。

未来再考虑远程 Worker。

---

# 14. Model Adapter 改造

当前存在：

```text
BaseModelAdapter
```

但真实 Qwen / Gemma / Llama Adapter 尚未接入。

同时 `TorchTrainingBackend` 又直接：

```text
AutoTokenizer
AutoModelForCausalLM
apply_chat_template
```

因此出现两套模型适配逻辑。

必须统一。

---

## 14.1 第一阶段只实现

```text
Qwen25Adapter
```

至少负责：

```text
load_tokenizer
load_model
format_training_prompt
format_inference_prompt
build_lora_config
merge_lora
export_gguf
```

---

## 14.2 Trainer 不再自行判断模型差异

改成：

```text
TrainingBackend
     ↓
BaseModelAdapter
     ↓
Qwen25Adapter
```

后续：

```text
GemmaAdapter
LlamaAdapter
```

只需增加 Adapter。

---

# 15. 训练 API 必须真正存在

当前代码没有完整的平台 Training Job API。

新增：

```http
POST /api/runs/{run_id}/start
GET  /api/jobs/{job_id}
POST /api/jobs/{job_id}/cancel
```

第一阶段不必实现复杂：

```text
pause
resume
priority queue
multi-GPU scheduling
```

失败先支持：

```text
重新执行
```

checkpoint 恢复后续增强。

---

# 16. 训练进度：第一阶段不要搞复杂 WebSocket

为了轻量与稳定，第一阶段前端每 1～2 秒：

```http
GET /api/jobs/{job_id}
```

获取：

```json
{
  "status": "RUNNING",
  "stage": "TRAINING",
  "progress": 43,
  "message": "epoch 1/3",
  "metrics": {
    "loss": 0.82
  }
}
```

等全链路稳定后，再恢复 WebSocket。

避免为了“实时感”增加复杂性。

---

# 17. Evaluation 必须由模型真实产生

当前 Evaluation API 允许调用方直接提交：

```text
y_true
y_pred
scores
```

这只能作为算法库测试能力。

真正产品流程必须改成：

```text
Model Version
+
真实 test.csv
↓
Inference
↓
predictions.jsonl
↓
Evaluator
↓
evaluation report
```

用户不能自行提交一组完美的 `y_pred` 来生成评测报告。

---

# 18. Lineage 必须变成真实血缘

当前 Lineage 主要还是 Schema。

改造后由系统自动生成：

```json
{
  "run": "run_001",
  "scene": "fraud_account",
  "dataset": "ds_v1",
  "split": "split_v1",
  "synth": "syn_v1",
  "base_model": "qwen2.5-1.5b-instruct",
  "training_job": "job_001",
  "model_version": "mv_v1",
  "artifact": "model-q4_k_m.gguf"
}
```

不能由前端任意填写。

---

# 19. 量化第一阶段只支持一种

不要同时生成：

```text
Q4_K_M
Q5_K_M
Q8_0
原生权重
```

第一阶段只做：

```text
Q4_K_M
```

流程：

```text
LoRA
↓
merge
↓
HF merged model
↓
convert_hf_to_gguf
↓
llama-quantize Q4_K_M
↓
checksum
↓
artifact DB record
```

其它量化档位以后再增加。

---

# 20. 部署第一阶段只支持 llama.cpp

不同时支持：

```text
llama.cpp
Ollama
vLLM
文件导出
```

第一阶段真正跑通：

```text
llama.cpp server
```

点击：

```text
部署
```

Worker 真正执行：

```bash
llama-server \
  -m model-q4_k_m.gguf \
  --port 8081
```

然后健康检查：

```http
GET /health
```

成功后：

```text
deployment.status = SERVING
```

而不是只返回一条“请自己运行”的命令。

---

# 21. `/v1/predict` 真正接模型

必须形成：

```text
业务请求
↓
SystemOneStudio API
↓
llama.cpp server
↓
JSON结构化输出
↓
schema validation
↓
返回业务系统
```

第一阶段输出：

```json
{
  "decision": "black",
  "score": 0.92,
  "confidence": 0.88,
  "reason": "..."
}
```

---

# 22. P95 ≤100ms 暂时改为“实测，不承诺”

原需求同时要求：

```text
reason 最大 200 字符
+
P95 ≤100ms
```

这两个目标存在明显冲突。

第一阶段目标：

> **先测出真实数字，不为了达标造假。**

评测报告显示：

```text
P50
P95
P99
TTFT
tokens/s
```

但不把 `100ms` 作为 MVP 阻塞门槛。

第二阶段再决定：

```text
decision fast-path
+
reason async / streaming
```

---

# 23. JEV 第一阶段收缩

在官方规范和基准没有完整落地前：

第一阶段只保留：

```text
JEV Format Compatibility
```

即：

```text
输入输出 Schema 对齐
```

暂时不声明：

```text
L2
L3
风格兼容
完整 JEV 评测兼容
```

所有 UI 上不得展示未经验证的：

```text
JEV L3
```

---

# 24. ScenePicker / ModelShelf / TrainingConfig 必须变成真实操作

当前这些页面很多属于“展示组件”。

改造要求：

### ScenePicker

点击后必须：

```text
PATCH run.scene
```

并存在后端。

### ModelShelf

点击后：

```text
PATCH run.model
```

Run 中有真实：

```text
base_model_id
```

### TrainingConfig

保存后：

```text
PATCH run.training_config
```

而不是只显示推荐值。

---

# 25. 删除前端第二套 Workflow Truth

前端可以保留：

```text
当前页面
展开/收起
临时表单
```

但必须删除前端自主管理：

```text
completed_steps
workflow_state
trained=true
```

后端：

```http
GET /api/runs/{run_id}
```

统一返回：

```json
{
  "state": "TRAINING",
  "current_step": 7,
  "dataset": {...},
  "synth": {...},
  "model": {...},
  "job": {...}
}
```

---

# 26. API 建议重新收敛

核心 API 控制在以下范围。

---

## Project

```text
POST /api/projects
GET  /api/projects
GET  /api/projects/{id}
```

---

## Run

```text
POST /api/projects/{project_id}/runs
GET  /api/runs/{run_id}

PATCH /api/runs/{run_id}/scene
POST  /api/runs/{run_id}/dataset
POST  /api/runs/{run_id}/prepare-data
POST  /api/runs/{run_id}/synth

PATCH /api/runs/{run_id}/model
PATCH /api/runs/{run_id}/training-config

POST /api/runs/{run_id}/start
```

---

## Job

```text
GET  /api/jobs/{job_id}
POST /api/jobs/{job_id}/cancel
```

---

## Model

```text
GET /api/models
GET /api/model-versions/{id}
```

---

## Evaluation

```text
GET /api/model-versions/{id}/evaluation
```

---

## Deployment

```text
POST /api/model-versions/{id}/deploy
GET  /api/deployments/{id}
POST /api/deployments/{id}/stop
```

---

## Runtime

```text
POST /v1/predict
```

够了。

---

# 27. 当前代码：保留、改造、暂缓

---

## 27.1 建议保留

以下代码已有较高价值：

```text
libs/contracts
services/data-pipeline
services/evaluator
services/quantizer
services/inference 中 llama.cpp 封装
services/synth 的 Generator 接口
services/trainer 的基础训练能力
BaseModelAdapter 抽象
前端整体视觉与七步页面框架
```

---

## 27.2 重点重写

```text
apps/api 中 in-memory Store
apps/web Wizard 状态管理
apps/web StepBody 数据传递
orchestrator 状态与任务执行
GPU image entrypoint
Training Job API
Dataset 资产管理
Synth 输出资产化
Evaluation 调用方式
Deployment 执行方式
Model Adapter 实现
```

---

## 27.3 暂缓

```text
Assistant 新功能
Canvas 深度能力
DPO
QLoRA
QAT
知识蒸馏
多模型族适配
Ollama
vLLM
多节点
多 GPU
模型市场
多租户
Prometheus / Grafana
高级通知
```

---

# 28. PostgreSQL / Redis 清理清单

## deploy

从 Docker Compose 删除：

```text
postgres
redis
```

删除：

```text
SON_DATABASE_URL
SON_REDIS_URL
```

新增：

```text
SON_DB_PATH
SON_WORKSPACE
```

---

## Python Dependency

移除无实际需要的：

```text
redis
psycopg
asyncpg
```

如果现有 SQLAlchemy 依赖已经存在，可以继续使用：

```text
SQLAlchemy + SQLite
```

但不要引入复杂数据库框架。

---

# 29. SQLite 实现建议

建议继续使用 SQLAlchemy，但只使用最基础能力：

```text
Engine
Session
ORM Models
Simple Repository
```

不要引入：

```text
复杂 Repository Framework
CQRS
Event Sourcing
Unit of Work Framework
```

迁移第一阶段也不需要 Alembic。

可以采用：

```text
PRAGMA user_version
+
migrations/*.sql
```

例如：

```text
001_init.sql
002_add_deployment.sql
```

启动自动检查版本并顺序执行。

---

# 30. 第一版真正支持的能力边界

为了真正做成，不再“看起来什么都支持”。

第一版明确支持：

### 数据

```text
CSV
结构化表格
black / white / gray
```

### 场景

```text
反诈账户判定
```

### 合成

```text
distribution_fit
```

### 模型

```text
Qwen2.5-1.5B-Instruct
```

验证后增加：

```text
Qwen2.5-3B-Instruct
```

### 训练

```text
LoRA SFT
```

### 量化

```text
GGUF Q4_K_M
```

### 推理

```text
llama.cpp
```

### 评测

```text
Accuracy
Precision
Recall
F1
False Positive Rate
AUC
Format Compliance
P50/P95/P99
```

---

# 31. 第一版暂时不支持的能力必须明确灰掉

前端不要展示“可选”但其实不能用。

例如：

```text
Gemma
Llama
DPO
QAT
QLoRA
知识蒸馏
Ollama
vLLM
JEV L3
```

可以标：

```text
即将支持
```

或者直接隐藏。

不能再出现：

> 注册表里有，所以 UI 看起来像已经支持。

---

# 32. 开发阶段重新划分

不要继续按原 M0-M6 的“组件完成度”排。

改成按纵向闭环排。

---

# P0-1：建立真实 Run 与 SQLite

完成：

```text
SQLite
Project
Run
Dataset
Job
ModelVersion
Artifact
Evaluation
Deployment
```

验收：

```text
重启 API 后项目和 Run 仍存在。
```

---

# P0-2：真正打通第 1～4 步

```text
选择场景
→ 上传一次 CSV
→ dataset_id
→ quality
→ split
→ synth
→ syn_v1
```

验收：

```text
第二步上传一次数据即可。
第三、第四步禁止再次上传。
```

---

# P0-3：真正打通第 5～6 步

```text
选择 Qwen
→ 写入 Run
→ 配置 LoRA
→ 写入 Run
```

验收：

```text
刷新浏览器后选择仍存在。
```

---

# P0-4：建立 SQLite Job Worker

```text
API submit
→ jobs QUEUED
→ Worker claim
→ RUNNING
→ SUCCEEDED / FAILED
```

验收：

```text
API 不执行训练本身。
Worker 停止时任务不会假成功。
```

---

# P0-5：Qwen 真训练

实现：

```text
Qwen25Adapter
+
TorchTrainingBackend
```

真实产生：

```text
LoRA Adapter
```

验收：

```text
不是 mock。
必须存在真实 adapter 文件。
```

---

# P0-6：Merge + Q4_K_M

真实生成：

```text
merged model
+
model.gguf
```

写：

```text
checksum
bytes
artifact record
```

---

# P0-7：真实 Evaluation

```text
test.csv
→ model inference
→ predictions
→ evaluator
→ report
```

验收：

```text
评测结果不能由前端传 y_pred 生成。
```

---

# P0-8：真实 Deployment

```text
点击部署
→ llama-server 真启动
→ health check
→ SERVING
```

---

# P0-9：真实 Predict

用真实 API：

```bash
curl -X POST http://localhost:8000/v1/predict
```

获得真实模型结果。

---

# P0-10：端到端 E2E 测试

这是以后最重要的测试。

增加：

```text
tests/e2e/test_full_pipeline.py
```

测试：

```text
create project
→ create run
→ select scene
→ upload CSV
→ prepare
→ synth
→ select model
→ config
→ submit
→ model version
→ evaluation
→ deployment
→ predict
```

---

# 33. 测试策略必须改变

当前大量单元测试继续保留。

但测试体系变成三层：

```text
Unit Test
+
Integration Test
+
Golden Path E2E
```

E2E 是发布门槛。

---

## 33.1 CPU CI

每次提交运行：

```text
contracts
data pipeline
SQLite
API
Run state
job queue
fake worker integration
frontend
```

---

## 33.2 GPU Acceptance

每个准备发布版本至少跑一次：

```text
真实 Qwen
真实 LoRA
真实 GGUF
真实 llama.cpp
真实 Evaluation
真实 Predict
```

生成一份：

```text
GPU Acceptance Report
```

不再用“代码已实现”代替“能力已验证”。

---

# 34. README 和开发计划必须同步纠偏

删除原来的：

> M0–M5 全部完成。

建议改成：

> 当前已完成主要 UI、契约、数据治理算法及部分模型能力组件；平台正在进行端到端轻量化改造。PostgreSQL / Redis 架构取消，统一采用 SQLite + Local Workspace。当前版本只有在真实完成“上传 → 训练 → 量化 → 评测 → 部署 → 推理”后，才定义为 MVP 完成。

---

# 35. “完成”的新定义

以后任何能力只有满足以下三点才能标：

```text
✅ 已完成
```

必须同时：

### 1. Code

真实实现。

### 2. Integration

真正被主链调用。

### 3. Verification

有真实运行结果。

例如：

```text
TorchTrainingBackend 写完
```

只能叫：

```text
代码实现
```

不能叫：

```text
训练能力完成
```

必须：

```text
UI
→ API
→ Job
→ Worker
→ GPU
→ Adapter
→ Model
```

都跑过才算完成。

---

# 36. 第一版 MVP 的唯一黄金链路

整个开发团队必须围绕这一条链工作：

```text
新建项目
↓
新建 Run
↓
选择反诈账户判定
↓
上传 CSV
↓
生成 ds_v1
↓
质量检查
↓
7:1.5:1.5
↓
合成 syn_v1
↓
选择 Qwen2.5-1.5B
↓
LoRA
↓
Job Queue
↓
Worker
↓
GPU Training
↓
Merge
↓
Q4_K_M
↓
Test Inference
↓
Evaluation
↓
mv_v1
↓
Deploy
↓
llama.cpp
↓
/v1/predict
```

任何一个箭头断掉：

> **MVP 都不能算完成。**

---

# 37. 开发优先级最终排序

## P0 — 现在必须做

1. SQLite 替换当前 in-memory Store；
2. 移除 PostgreSQL / Redis；
3. 引入 Run 核心对象；
4. 修复七步数据断链；
5. Dataset 资产化；
6. Synth 输出资产化；
7. SQLite Job Queue；
8. Local GPU Worker；
9. Qwen Adapter；
10. Training API；
11. Merge + Q4；
12. Evaluation 真调用模型；
13. Deployment 真启动 llama.cpp；
14. `/v1/predict`；
15. Golden Path E2E。

---

## P1 — 全链路跑通后

1. Qwen 3B；
2. QLoRA；
3. checkpoint resume；
4. WebSocket；
5. DPO；
6. Gemma Adapter；
7. Llama Adapter；
8. reason fast-path；
9. JEV 完整评测；
10. 性能优化。

---

## P2 — 后续平台化

1. 多 GPU；
2. Remote Worker；
3. PostgreSQL；
4. Redis；
5. MinIO；
6. 多租户；
7. 模型市场；
8. 完整 Canvas；
9. 自动调参；
10. Assistant 深度编排。

注意：

> PostgreSQL / Redis 不是“永远不需要”，而是 **当前不需要**。

只有真正出现：

```text
多人并发
多实例 API
多 GPU Worker
远程 GPU 集群
高频任务并发
```

之后，再升级基础设施。

---

# 38. 开发团队执行规则

## Rule 1

没有 Golden Path 闭环前，不增加新功能。

## Rule 2

每一次提交都回答：

> 这一改动让黄金链路前进了哪一步？

如果答案是：

> “以后可能有用。”

则默认不进入本轮。

## Rule 3

UI 不得制造假完成状态。

## Rule 4

没有真实文件，就不能生成 Artifact。

## Rule 5

没有真实模型推理，就不能生成 Evaluation。

## Rule 6

没有进程真正启动，就不能显示 Deployment=SERVING。

## Rule 7

所有运行资产必须有：

```text
run_id
path
checksum
status
```

## Rule 8

所有错误必须留下：

```text
job.status=FAILED
error_message
log_path
```

不得吞异常。

---

# 39. 最终验收场景

开发团队最终现场演示必须从一个全新环境开始。

---

## Step 1

启动：

```text
Web
API
Worker
```

不启动：

```text
PostgreSQL
Redis
MinIO
```

---

## Step 2

浏览器新建项目：

```text
反诈 Demo
```

---

## Step 3

选择：

```text
反诈账户判定
```

---

## Step 4

上传：

```text
seed.csv
```

之后全流程不得要求再次上传。

---

## Step 5

完成：

```text
质量检查
split
synth
```

页面能看到：

```text
ds_v1
syn_v1
```

---

## Step 6

选择：

```text
Qwen2.5-1.5B
LoRA
```

---

## Step 7

点击：

```text
开始训练
```

后台：

```text
job
→ worker
→ GPU
```

---

## Step 8

真实产生：

```text
adapter
merged model
GGUF Q4_K_M
```

---

## Step 9

真实 test set 自动评测。

---

## Step 10

点击部署。

真实启动：

```text
llama.cpp server
```

---

## Step 11

页面调用：

```text
/v1/predict
```

真实返回：

```text
decision
score
confidence
reason
latency
```

---

## Step 12

关闭整个系统，再重新启动。

必须仍能看到：

```text
Project
Run
Dataset
ModelVersion
Evaluation
Deployment History
```

这一步用于证明：

> 系统已经不再依赖 in-memory Store。

---

# 40. 最终目标

本轮改造完成后，SystemOneStudio 不追求“能力最多”。

而应该成为：

> **一个结构简单、能真实训练、能真实评测、能真实部署、能真实预测的轻量决策模型训练系统。**

第一阶段的核心产品标准只有一句：

# **全链路是真的。**

在此基础上，再逐步恢复：

```text
多模型
多训练方式
智能助手
画布编排
多 GPU
远程 Worker
企业级数据库
```

否则系统会继续处于：

> “组件很多、测试很多、页面很好看，但用户实际上训不出模型。”

这正是本次改造必须终结的问题。
