# SystemOneStudio 判断模型训练平台技术方案

> 版本：V3.0（Jev-style Decision Model Edition）  
> 日期：2026-10-06  
> 对应原型：`SystemOneStudio-demo-v3.html`  
> 文档性质：开发技术方案 / MVP 实施基线  
> 目标：把 SystemOneStudio 从“小尺寸语言模型微调平台”收敛为“面向业务场景的专用判断模型训练、校准、验证与发布平台”。

---

# 1. 技术结论先行

SystemOneStudio 第一阶段不再建设通用“小模型训练平台”，也不以“LoRA / QLoRA / DPO / 蒸馏”等训练技术作为产品主入口。

本阶段只解决一件事：

> **给定业务 State 和有限、可验证的答案空间，训练一个不生成自由文本、直接输出 Choice 概率分布，并能依据校准置信度进入自动化软件流程的专用判断模型。**

MVP 固定主链：

```text
判断任务定义
    ↓
真实决策样本
    ↓
训练集构造
    ↓
Decision SFT
Candidate CE + Brier
    ↓
Held-out Calibration
Temperature Scaling
    ↓
Threshold Policy
Auto / Review
    ↓
Frozen Test
    ↓
判断模型包 + System One API
```

第一阶段默认训练配方：

```text
Qwen3.5-2B-Base
+ Restricted-Logit Decision Readout
+ LoRA
+ Candidate Cross-Entropy
+ Brier Loss
+ Temperature Scaling
+ Confidence Threshold Policy
```

RLCD-like 强化校准作为 **P1 增强能力**，不是 MVP 主链。

---

# 2. 与上一版技术方案的关键变化

| 上一版思路 | V3 技术方案 |
|---|---|
| 小尺寸 LLM 微调平台 | 专用 Judgment / Decision Model 训练平台 |
| LoRA / QLoRA 是一级“训练方式” | LoRA / QLoRA 降为底层实现参数 |
| 模型生成 `decision/score/confidence/reason` JSON | 模型只产生受限 Choice logits / probabilities |
| reason 强制非空 | reason/evidence 改为数据审计元数据，不是核心模型输出 |
| 训练重点是 loss / accuracy | 同时优化 Accuracy + Calibration + Selective Risk |
| 评测看 Accuracy / F1 / 格式合规 | 加入 Brier / ECE / NLL / Risk-Coverage / Option-order robustness |
| 部署是一份模型权重 | 发布 `Model + Decision Schema + Calibration + Threshold Policy + Evaluation + Lineage` |
| Qwen + 普通生成推理 | 一次前向读取有限候选 logits，不走自回归生成 |
| Jev 兼容作为格式开关 | 学习 Jev 的 typed probabilistic decision 范式，不声称复刻 Jev 私有算法 |

**废止旧约束：** `reason` 不再要求作为模型必选输出。旧文档中“reason 强制非空”的约束由本方案覆盖。

reason / evidence 仍可保留在训练数据中，用途限定为：

1. 样本审计；
2. 专家复核；
3. Hard Negative 构造依据；
4. 独立解释层输入；
5. 未来教师模型数据构造。

但它**不进入判断模型的自由文本生成目标**。

---

# 3. 产品边界

## 3.1 MVP 做什么

第一阶段只支持：

- 自定义判断场景；
- Choice 三分类；
- 固定逻辑 ID：`black / white / gray`；
- 每个项目可分别定义三类的业务含义；
- CSV 种子数据；
- 结构化 / 半结构化 State 序列化；
- 训练集构造与增强；
- Restricted-logit Decision SFT；
- 概率校准；
- 阈值策略；
- Frozen Test；
- 在线单条判断；
- 判断模型包导出；
- 私有 API 服务；
- 单机 GPU 一体部署；
- SQLite + 本地文件存储；
- 全链路版本和证据关联。

## 3.2 MVP 不做什么

第一阶段明确不做：

- 通用聊天模型训练；
- 长文本生成；
- RAG；
- Agent 编排；
- DPO；
- 通用知识蒸馏；
- 任意模型市场；
- 多租户；
- 分布式训练；
- Kubernetes；
- Redis / PostgreSQL / MinIO；
- 自动超参搜索；
- Jev 私有架构复刻；
- Jev 私有 RLCD reward 复刻；
- 自称“Jev compatible L2/L3”。

## 3.3 后续扩展

V1 可扩展：

- `noul`：二元概率 P(true)；
- `score`：有序等级分布和期望分数；
- 动态 Choice 数量；
- Encoder + Option Scorer 架构；
- RLCD-like；
- 教师软标签；
- 多任务共享判断模型；
- CPU / ONNX / Web Runtime。

---

# 4. Jev-style 判断模型的技术定义

SystemOneStudio 内部统一把“判断模型”定义为：

```text
State
 +
Decision Schema
        ↓
Decision Model
        ↓
Candidate Logits
        ↓
Calibration
        ↓
Probability Distribution
        ↓
Threshold Policy
        ↓
AUTO / REVIEW / ESCALATE
```

模型核心职责只有两个：

1. **在被允许的答案空间内选对；**
2. **给出可信的概率。**

模型不负责：

- 写解释；
- 写报告；
- 生成 JSON 文本；
- 决定最终业务动作。

业务动作由 Threshold Policy 决定。

---

# 5. Decision Schema

每个项目必须先固化一个 `Decision Schema`。

MVP 示例：

```json
{
  "schema_id": "dec_schema_001",
  "type": "choice",
  "version": 1,
  "question": "设备质检结果属于哪一类？",
  "options": [
    {
      "id": "black",
      "name": "不通过",
      "description": "存在明确质量问题，不允许自动放行"
    },
    {
      "id": "white",
      "name": "通过",
      "description": "满足放行条件"
    },
    {
      "id": "gray",
      "name": "待复核",
      "description": "证据不足、边界样本或需要人工判断"
    }
  ]
}
```

硬约束：

- option ID 一旦进入训练版本，不允许原地修改；
- option 语义变化必须创建新 schema version；
- 模型输出不允许出现 schema 外标签；
- schema version 必须进入模型 lineage。

---

# 6. MVP 模型构型：Restricted-Logit Decision Model

## 6.1 为什么 MVP 不优先使用自定义 Decision Head

Demo V3 中展示了 `Qwen + Decision Head`。从产品概念上成立，但 MVP 工程实现建议进一步收敛。

如果在 Qwen 后增加自定义 Head：

- GGUF 转换链需要额外适配；
- llama.cpp 标准运行时无法天然识别自定义 Head；
- 导出、量化、推理服务都要维护私有 Runtime；
- MVP 的工程复杂度明显增大。

因此第一阶段推荐：

> **保留 Qwen 原生 LM Head，但完全不做生成，只在一个固定 Decision Slot 上读取有限候选 token 的 logits。**

该方法满足：

- 一次 forward；
- 无自回归 decoding；
- 无自由文本；
- 输出受限；
- 可得到概率；
- 可继续使用标准 LoRA / merge / safetensors；
- 可继续验证 GGUF / llama.cpp 路线。

## 6.2 输入模板

逻辑模板：

```text
<STATE>
{deterministic_serialized_state}
</STATE>

<QUESTION>
{decision_question}
</QUESTION>

<OPTIONS>
A = black: 不通过
B = white: 通过
C = gray: 待复核
</OPTIONS>

<DECISION>
```

模型不生成 `A/B/C`。

系统直接读取 `<DECISION>` 后一个位置的 vocabulary logits：

```text
logit(A)
logit(B)
logit(C)
```

再计算：

```text
p = softmax([logit(A), logit(B), logit(C)])
```

最后映射：

```text
A → black
B → white
C → gray
```

## 6.3 Candidate Token Resolver

不能硬编码认为所有 tokenizer 中 `A/B/C` 一定是稳定单 token。

增加：

```text
CandidateTokenResolver
```

启动模型适配时执行：

1. 检查候选别名 tokenization；
2. 优先寻找稳定单 token 表达；
3. 固化 token ID；
4. 写入 `decision_token_map.json`；
5. 训练、评测、部署必须使用同一映射。

示例：

```json
{
  "A": 32,
  "B": 33,
  "C": 34,
  "mapping": {
    "A": "black",
    "B": "white",
    "C": "gray"
  }
}
```

若候选无法稳定映射为单 token，则该模型 revision 不进入 MVP 可选列表。

---

# 7. 模型适配体系

统一接口：

```python
class DecisionModelAdapter(ABC):
    def validate_revision(self): ...
    def resolve_candidate_tokens(self, schema): ...
    def serialize_state(self, state): ...
    def build_input(self, state, schema, permutation=None): ...
    def load_train_model(self, config): ...
    def candidate_logits(self, model_output, token_map): ...
    def merge_adapter(self): ...
    def export(self): ...
    def load_runtime(self): ...
    def predict_distribution(self, state, schema): ...
```

MVP 只实现：

```text
QwenDecisionAdapter
```

候选底座：

```text
Qwen3.5-2B-Base      默认验收候选
Qwen3.5-0.8B-Base    轻量候选
```

**只有完成以下实测后才能在 UI 中开放：**

- tokenizer / candidate token 校验；
- LoRA 训练通过；
- merge 通过；
- inference parity 通过；
- calibration 可加载；
- quantization 路线验证；
- license / revision 已登记。

实验路线：

```text
ModernBERT-large + Option Scorer
```

该路线不进入 MVP 强制交付。

---

# 8. 数据契约

平台内部统一转为 JSONL Canonical Dataset。

```json
{
  "sample_id": "sample_000001",
  "group_id": "case_001",
  "state": {
    "deviation": 8.1,
    "temperature": 83,
    "surface": "明显缺陷"
  },
  "target": {
    "choice": "black"
  },
  "evidence": "偏差超过阈值且存在明显缺陷",
  "source": {
    "type": "seed",
    "file": "seed.csv",
    "row": 12
  },
  "metadata": {}
}
```

字段角色：

```text
STATE_INPUT
TARGET_CHOICE
EVIDENCE
GROUP_ID
ENTITY_ID
IGNORE
```

必须自动排除：

```text
decision
label
target
reason
evidence
score
confidence
prediction
origin
```

防止答案泄漏。

---

# 9. State 序列化

模型看到的是稳定的 State 表达，不直接依赖原 CSV 行顺序。

默认序列化：

```json
{
  "deviation": 8.1,
  "surface": "明显缺陷",
  "temperature": 83
}
```

要求：

- UTF-8；
- 空值显式表示；
- 数值不随 locale 改写；
- 日期统一 ISO8601；
- 不把 target/evidence 混入 state；
- serializer 有独立 version；
- 同一模型版本必须固定 serializer version。

为降低字段位置偏置，训练时允许做等价 Field-order Augmentation，但验证和线上使用固定 canonical serializer。

---

# 10. 数据划分

## 10.1 原则

```text
train
validation / calibration
test (frozen)
```

test：

- 创建后冻结；
- 不进入任何合成器；
- 不用于超参选择；
- 不用于温度拟合；
- 不用于阈值选择；
- 只有形成候选发布版本时运行。

## 10.2 Group-aware Split

如果存在同一案例的不同变体、翻译、时间切片或扩增版本，必须使用 `group_id` 分组后再 split。

禁止：

```text
原始 Case → train
同一 Case 改写版 → test
```

## 10.3 推荐划分

数据较充分时：

```text
70% train
10% validation
10% calibration
10% test
```

数据较少时，UI 可继续保持 V3 的：

```text
70% train
15% calibration / validation
15% test
```

但报告必须标记：validation 与 calibration 共用会降低校准结论独立性。

样本过少时：

```text
< 200：只允许“链路验证”标签
```

不允许把小样本 Demo 指标包装成生产效果。

---

# 11. 训练集构造引擎

训练集构造不是简单“复制更多行”。

实现三个 P0/P1 策略。

## 11.1 P0：Class Balance

方式：

- WeightedRandomSampler；或
- 可追溯重采样。

目标：

- 缓解类别极不均衡；
- 不声称产生新知识。

## 11.2 P0：Option-order Permutation

训练输入中的：

```text
A/B/C ↔ black/white/gray
```

随机排列，同时同步 target 映射。

例如：

```text
Run 1:
A=black B=white C=gray
Target=A

Run 2:
A=gray B=black C=white
Target=B
```

目标：

- 防止模型学习固定位置；
- 防止永远把 A 当某一类；
- 形成 option-order robustness。

## 11.3 P0：Field-order Permutation

对 JSON / KV 输入做语义等价顺序变化。

要求：

- 只修改表现顺序；
- 不修改业务值；
- lineage 保存 transformation。

## 11.4 P1：Hard Negative

只允许来源可验证的难例进入训练：

```text
专家标注
业务规则验证
历史误判回流
教师模型生成 + 独立 verifier
```

禁止：

```text
让同一个生成器生成样本、生成标签，再用同一个生成器证明正确。
```

---

# 12. 默认训练配方：Domain Decision SFT

训练目标不是 next-token generation，而是 **restricted candidate distribution**。

令候选 logits：

```text
z = [z_black, z_white, z_gray]
```

概率：

```text
p = softmax(z)
```

## 12.1 Candidate Cross-Entropy

```text
L_ce = - Σ y_k log(p_k)
```

支持 label smoothing：

```text
ε = 0.05 (default)
```

## 12.2 Brier Loss

```text
L_brier = 1/K Σ (p_k - y_k)^2
```

## 12.3 总损失

```text
L = L_ce + λ_brier × L_brier
```

默认：

```yaml
ce_weight: 1.0
brier_weight: 0.2
label_smoothing: 0.05
```

这些是起始默认值，不是行业标准常数。必须允许配置并通过真实实验修订。

## 12.4 LoRA

LoRA 是实现技术，不是产品训练范式。

默认建议：

```yaml
rank: 16
alpha: 32
modules: adapter-specific
```

第一阶段先验证：

- Attention projections；
- 必要时 MLP projections；
- LM Head 默认不单独训练，除非适配实验表明需要。

所有具体 target_modules 由 `QwenDecisionAdapter` 固化，前端不直接让业务用户勾层名。

---

# 13. 第二训练配方：Decision Head Build（实验）

这条路线对应 Demo V3 中真正的 `Decision Head`。

结构：

```text
Backbone
   ↓
Pooled Hidden State
   ↓
Decision Head
   ↓
K logits
```

可采用：

```text
LayerNorm
Linear
GELU
Dropout
Linear(K)
```

训练方式：

1. Freeze backbone + train head；
2. 如不足，再开放顶部若干层或 LoRA；
3. CE + Brier；
4. Temperature Scaling。

该路线的运行时不强制走 llama.cpp，优先：

```text
PyTorch
ONNX Runtime
TensorRT（后续）
```

进入正式产品前必须证明：

- 相比 Restricted-logit 路线有显著效果或时延优势；
- 额外 Runtime 成本值得。

---

# 14. 第三训练配方：RLCD-like（P1）

## 14.1 定位

Jev 公开了 RLCD 名称和“优化 calibrated decisions”的目标，但没有公开完整私有 reward、模型架构和 recipe。

因此 SystemOneStudio 只能实现：

> **RLCD-like / calibration-aware RL**

不得在产品文案中声称“复刻 Jev RLCD”。

## 14.2 推荐实现

从已完成 Decision SFT 的 checkpoint 开始。

每个样本得到 candidate distribution：

```text
pθ(a | state)
```

使用 strictly proper scoring reward：

- log score；
- Brier / quadratic score；
- spherical score；
- `score` 类型未来可增加 Ranked Probability Score。

增加：

```text
KL(current || reference_sft)
```

约束分布漂移。

建议实现为独立：

```text
ProperScoreRLTrainer
```

而不是强依赖面向文本生成的 TRL Trainer。

P1 才实现：

```text
Gaussian logit exploration / grouped samples
→ proper-score reward
→ normalized advantage
→ policy gradient
→ KL constraint
```

MVP 数据链、SFT、Calibration 未稳定前不得提前投入 RLCD-like。

---

# 15. Calibration Engine

判断模型“选对”以后还不能直接上线。

Calibration 是独立正式阶段。

## 15.1 Temperature Scaling

模型原始 logits：

```text
z
```

校准后：

```text
p_T = softmax(z / T)
```

在 held-out calibration split 上寻找：

```text
T > 0
```

最小化 NLL。

推荐实现：

```text
torch.optim.LBFGS(log_temperature)
```

建议约束：

```text
0.1 <= T <= 10
```

## 15.2 Calibration 产物

```json
{
  "calibration_id": "cal_001",
  "method": "temperature_scaling",
  "temperature": 1.47,
  "fit_dataset": "ds_cal_003",
  "objective": "nll",
  "created_at": "..."
}
```

## 15.3 Calibration Metrics

必须输出：

```text
Brier Score
ECE
NLL
Reliability Diagram
Confidence Bin Table
```

温度校准不改变 argmax，因此：

```text
Accuracy 理论上保持不变
Confidence / Threshold 行为会改变
```

---

# 16. Threshold Policy Engine

SystemOneStudio 不把：

```text
confidence = 0.9
```

写死成产品规则。

阈值必须从 calibration / validation 数据产生。

## 16.1 默认策略

```text
max_probability >= τ
    → AUTO
else
    → REVIEW
```

## 16.2 推荐阈值搜索

用户输入业务错误预算：

```text
自动区最大允许错误率 <= X%
```

系统扫描：

```text
τ = 0.50 ... 0.99
```

找出满足错误预算时 coverage 最大的阈值。

例如：

```text
目标：auto error <= 3%
结果：τ = 0.91
coverage = 67.8%
review = 32.2%
```

## 16.3 产物

```json
{
  "policy_id": "pol_001",
  "type": "confidence_threshold",
  "threshold": 0.91,
  "selection_dataset": "ds_cal_003",
  "constraint": {
    "metric": "auto_error_rate",
    "max": 0.03
  }
}
```

P1 再支持：

- per-class threshold；
- cost-sensitive threshold；
- reject option；
- OOD escalation。

---

# 17. 判断模型训练全流水线

生产链固定为：

```text
1. Validate Decision Schema
2. Import / Normalize Dataset
3. Leakage Check
4. Group-aware Split
5. Build Train Augmentation
6. Resolve Candidate Tokens
7. Probe GPU / Disk / Model Cache
8. Load Base Revision
9. Decision SFT
10. Select Best Checkpoint
11. Merge LoRA
12. Run Calibration Logits
13. Fit Temperature
14. Scan Threshold Policy
15. Frozen Test
16. Robustness Test
17. Export / Quantize
18. Runtime Parity Test
19. Register Model Version
20. Package / Serve
```

**Frozen Test 只能在第 15 步出现。**

不得为了调 threshold 反复跑 test。

---

# 18. Training Executor

第一阶段不使用 Redis / Celery。

部署三个逻辑进程：

```text
web
worker
runtime
```

## 18.1 Web Process

FastAPI：

- REST API；
- WebSocket / SSE；
- 项目状态；
- 任务提交；
- 报告读取。

不执行长 GPU 任务。

## 18.2 Worker Process

实现：

```text
JobSupervisor
```

职责：

- 轮询 SQLite pending jobs；
- 用 subprocess 启动训练命令；
- PID 关联；
- stdout / stderr 文件化；
- 心跳；
- cancel；
- checkpoint resume；
- 异常退出恢复；
- 更新 SQLite job state。

任务状态：

```text
PENDING
PREPARING
RUNNING
CALIBRATING
EVALUATING
EXPORTING
SUCCEEDED
FAILED
CANCELLING
CANCELLED
INTERRUPTED
```

## 18.3 Runtime Process

模型服务独立进程：

- 加载指定 model version；
- 读取 calibration；
- 读取 threshold policy；
- 提供 `/v1/predict`；
- 健康检查；
- reload / stop。

---

# 19. 技术栈

## 前端

```text
React
Vite
TypeScript
TanStack Query
ECharts
```

不引入复杂状态框架作为 MVP 必选。

## 后端

```text
Python 3.11+
FastAPI
Pydantic v2
SQLAlchemy 2.x
Alembic
SQLite (WAL)
```

## 训练

```text
PyTorch
Transformers
PEFT
Accelerate
Datasets / PyArrow
safetensors
NumPy
scikit-learn
```

MVP 不强制依赖 TRL。

## 量化 / Runtime

主路线：

```text
Transformers Runtime
→ merged safetensors
```

验收后开放：

```text
llama.cpp / llama-cpp-python
GGUF Q4_K_M
```

前提是：

- 当前模型 revision 可转换；
- candidate logits 可读取；
- quantized parity 通过。

自定义 Decision Head 路线：

```text
ONNX Runtime
```

优先于强塞进 GGUF。

---

# 20. SQLite 数据模型

建议核心表：

## `projects`

```text
id
name
description
status
current_scene_version
created_at
updated_at
```

## `decision_schemas`

```text
id
project_id
version
type
schema_json
hash
created_at
```

## `datasets`

```text
id
project_id
kind          seed/train/valid/calibration/test
version
row_count
schema_hash
file_path
sha256
status
created_at
```

## `dataset_samples_index`

只保存轻量索引，不建议把全部大样本文本塞 SQLite：

```text
sample_id
dataset_id
group_id
target_choice
source_type
source_ref
```

真实数据使用：

```text
JSONL / Parquet
```

## `data_build_runs`

```text
id
project_id
input_dataset_id
output_dataset_id
config_json
metrics_json
status
```

## `training_recipes`

```text
id
project_id
version
recipe_type
base_model_id
config_json
hash
```

## `jobs`

```text
id
job_type
project_id
status
pid
progress
stage
config_snapshot
started_at
finished_at
error
```

## `model_versions`

```text
id
project_id
version
base_model_revision
recipe_id
train_dataset_id
adapter_path
merged_path
model_hash
runtime_type
status
```

## `calibrations`

```text
id
model_version_id
method
params_json
dataset_id
metrics_json
```

## `threshold_policies`

```text
id
model_version_id
calibration_id
policy_json
metrics_json
```

## `evaluations`

```text
id
model_version_id
dataset_id
policy_id
metrics_json
report_path
status
```

## `services`

```text
id
model_version_id
policy_id
pid
port
status
health
```

## `artifacts`

```text
id
owner_type
owner_id
artifact_type
path
sha256
size
created_at
```

---

# 21. 文件目录

```text
/workspace/
├── systemone.db
├── projects/
│   └── sc_xxx/
│       ├── schema/
│       │   ├── decision_schema_v1.json
│       │   └── serializer_v1.json
│       ├── data/
│       │   ├── seed/
│       │   ├── train/
│       │   ├── valid/
│       │   ├── calibration/
│       │   └── test/
│       ├── builds/
│       ├── runs/
│       │   └── tr_xxx/
│       │       ├── config.yaml
│       │       ├── logs/
│       │       ├── checkpoints/
│       │       └── metrics.jsonl
│       ├── models/
│       │   └── mv_xxx/
│       │       ├── merged/
│       │       ├── adapter/
│       │       ├── calibration.json
│       │       ├── threshold_policy.json
│       │       ├── decision_schema.json
│       │       ├── token_map.json
│       │       ├── evaluation.json
│       │       ├── lineage.json
│       │       └── checksums.txt
│       └── reports/
├── model-cache/
└── logs/
```

---

# 22. 版本与血缘

统一 ID：

```text
sc_   project / scene
ds_   dataset
syn_  data build
rcp_  recipe
tr_   training run
mv_   model version
cal_  calibration
pol_  threshold policy
ev_   evaluation
svc_  service
```

每个 `mv_*` 必须可以回答：

```text
来自哪个项目？
哪个 Decision Schema？
哪份 seed？
哪份 train / calibration / test？
做过哪些 augmentation？
使用哪个 base model revision？
哪个 recipe？
哪个 git build？
哪个 candidate token map？
哪个 calibration？
哪个 threshold policy？
哪个 frozen test report？
最终文件 hash 是什么？
```

不能回答这些问题的模型，不允许标记为 `RELEASED`。

---

# 23. API 设计

## 项目

```text
GET    /api/projects
POST   /api/projects
GET    /api/projects/{id}
PATCH  /api/projects/{id}
```

## Decision Schema

```text
GET    /api/projects/{id}/decision-schema
POST   /api/projects/{id}/decision-schema/versions
```

## 数据

```text
POST   /api/projects/{id}/datasets/upload
POST   /api/datasets/{id}/prepare
POST   /api/datasets/{id}/build
GET    /api/datasets/{id}/report
```

## Recipe

```text
GET    /api/model-options
GET    /api/recipe-options
POST   /api/projects/{id}/recipes
```

## Training

```text
POST   /api/projects/{id}/training-runs
GET    /api/training-runs/{id}
POST   /api/training-runs/{id}/cancel
POST   /api/training-runs/{id}/resume
GET    /api/training-runs/{id}/events
```

## Calibration / Evaluation

```text
POST   /api/model-versions/{id}/calibrate
POST   /api/model-versions/{id}/threshold-search
POST   /api/model-versions/{id}/evaluate
GET    /api/evaluations/{id}
```

## Runtime

```text
POST   /api/model-versions/{id}/services/start
POST   /api/services/{id}/stop
GET    /api/services/{id}/health
```

## 对外判断 API

```text
POST /v1/predict
```

请求：

```json
{
  "model": "mv_001",
  "state": {
    "deviation": 6.2,
    "temperature": 78,
    "surface": "明显缺陷"
  }
}
```

响应：

```json
{
  "trace_id": "trace_xxx",
  "model": "mv_001",
  "schema_version": 1,
  "choice": "black",
  "probabilities": {
    "black": 0.93,
    "white": 0.02,
    "gray": 0.05
  },
  "confidence": 0.93,
  "action": "AUTO",
  "policy": "pol_001"
}
```

不返回自由文本 `reason`。

后续如需要解释，单独提供：

```text
POST /v1/explain
```

解释层与判断模型解耦。

---

# 24. 训练配置对象

推荐 YAML：

```yaml
recipe_id: rcp_001
recipe_type: restricted_logit_sft

base_model:
  model_id: qwen3.5-2b-base
  revision: "<full_revision>"
  adapter: qwen_decision_v1

schema:
  id: dec_schema_001
  version: 1

candidate_readout:
  type: next_token_restricted_logits
  aliases: [A, B, C]

training:
  epochs: 2
  learning_rate: 5.0e-5
  micro_batch: 4
  gradient_accumulation: 4
  max_length: 1024
  seed: 42

lora:
  enabled: true
  rank: 16
  alpha: 32

objective:
  candidate_ce_weight: 1.0
  label_smoothing: 0.05
  brier_weight: 0.2

calibration:
  method: temperature_scaling
  objective: nll

threshold_policy:
  mode: error_budget
  max_auto_error: 0.03

export:
  merged_safetensors: true
  gguf:
    enabled: true
    quant: Q4_K_M
    require_parity_test: true
```

---

# 25. 训练监控

训练页面不再只画一个总 loss。

实时显示：

```text
Candidate CE
Brier
Total Loss
Validation Accuracy
Validation Brier
Learning Rate
GPU Memory
Step / Epoch
Samples/sec
```

Calibration 阶段：

```text
Raw NLL
Calibrated NLL
Raw ECE
Calibrated ECE
Temperature
```

Threshold 阶段：

```text
Threshold
Coverage
Auto Error Rate
Review Rate
```

---

# 26. 评测体系

## 26.1 判断正确性

必须：

```text
Accuracy
Macro-F1
Per-class Precision
Per-class Recall
Confusion Matrix
```

## 26.2 Calibration

必须：

```text
Brier Score
ECE
NLL
Reliability Diagram
Confidence Bin Table
```

## 26.3 Selective Decision

必须：

```text
Coverage
Review Rate
Auto-zone Error Rate
Risk-Coverage Curve
```

## 26.4 Robustness

至少：

```text
Option-order Flip Rate
Field-order Flip Rate
Missing-field Test
Boundary / Hard-negative Test
High-confidence Wrong Cases
```

P1：

```text
OOD Detection
Prompt / Schema paraphrase robustness
Adversarial boundary
```

## 26.5 性能

```text
P50 latency
P95 latency
Throughput
Peak VRAM / RAM
Model package size
Cold load time
```

第一阶段先**真实测量**，不在技术方案中虚构硬件无关 SLA。

---

# 27. Frozen Test 规则

这是开发必须写进代码的硬约束。

正常训练运行中：

```text
test dataset path 不传给 Trainer
```

只有：

```text
MODEL_CANDIDATE_READY
```

以后才允许 Evaluation Service 读取 test。

每次 test 运行记录：

```text
model_hash
calibration_hash
policy_hash
test_dataset_hash
code_build
run_id
```

如果修改：

- 模型；
- calibration；
- threshold；
- serializer；
- schema；

都视为新 candidate，重新生成 evaluation。

---

# 28. Quantization 与 Parity Gate

量化不能只是“转换成功”。

GGUF 后必须重新跑固定 parity set。

比较：

```text
FP16 / BF16 merged model
vs
Q4_K_M model
```

报告：

```text
Choice Agreement
Probability MAE
Max Probability Drift
Calibration Drift
Latency Change
Memory Change
```

只有通过配置门槛才标记：

```text
DEPLOYABLE
```

建议初始门槛（需实测修订）：

```text
Choice agreement >= 99%
Mean probability drift <= 0.02
```

这是工程初始 Gate，不代表业务质量门槛。

---

# 29. 模型发布包

标准目录：

```text
SystemOneModel-mv_001/
├── model/
│   ├── merged.safetensors
│   └── config.json
├── gguf/
│   └── model-q4_k_m.gguf          # 仅 parity 通过后
├── tokenizer/
├── decision_schema.json
├── decision_token_map.json
├── state_serializer.json
├── calibration.json
├── threshold_policy.json
├── evaluation.json
├── lineage.json
├── checksums.txt
├── api_schema.json
└── README.md
```

发布条件：

```text
权重可加载
+ Decision Schema 可加载
+ Candidate Token Map 可加载
+ Calibration 可加载
+ Policy 可加载
+ Frozen Test 已完成
+ Package Hash 完整
```

缺任何一项都不显示“发布完成”。

---

# 30. 项目展廊与前端状态

首页使用 Demo V3 的项目展廊概念。

每个项目卡显示：

```text
判断任务名称
Decision Schema 类型
当前阶段
数据量
当前模型版本
Calibration 状态
Policy 状态
最新 Frozen Test 状态
更新时间
```

状态示例：

```text
DRAFT
DATA_READY
RECIPE_READY
TRAINING
MODEL_READY
CALIBRATED
VALIDATED
RELEASED
```

注意：

```text
TRAINING_DONE != RELEASED
```

判断模型训练完成后至少还需要：

```text
Calibration
Threshold
Frozen Test
Package Validation
```

---

# 31. 七步 UI 与后台对象对应关系

| UI 步骤 | 后台核心对象 |
|---|---|
| 1 判断任务 | Project + DecisionSchema |
| 2 决策样本 | SeedDataset + DatasetProfile |
| 3 训练集构造 | DataBuildRun + TrainDataset |
| 4 训练配方 | TrainingRecipe + ModelRevision |
| 5 判断模型训练 | TrainingRun + ModelVersion |
| 6 校准验证 | Calibration + ThresholdPolicy + Evaluation |
| 7 发布服务 | ModelPackage + RuntimeService |

不允许前端用一个 JSON 状态字段假装整个链路完成。

---

# 32. 失败、取消与恢复

## 训练失败

必须保留：

```text
最后 checkpoint
stdout/stderr
失败 stage
exit code
GPU 状态摘要
config snapshot
```

## Cancel

前端：

```text
RUNNING → CANCELLING
```

Worker：

```text
SIGTERM
→ grace period
→ SIGKILL if necessary
```

最终：

```text
CANCELLED
```

## 浏览器关闭

不能影响 worker 子进程。

## Server 重启

启动时：

- 检查 jobs 中 RUNNING PID；
- PID 存在则重新 attach；
- PID 不存在则标记 INTERRUPTED；
- 有 checkpoint 时允许 Resume。

---

# 33. 部署架构

第一阶段推荐一体机：

```text
Browser
   │
   ▼
Nginx / App Port
   │
   ▼
FastAPI Web
   │
   ├── SQLite
   ├── Workspace Files
   │
   ├── Worker Process ── GPU Training
   │
   └── Runtime Process ─ GPU/CPU Inference
```

Docker Compose：

```text
systemone-web
systemone-worker
systemone-runtime
```

可以共享：

```text
/workspace
/model-cache
```

Windows GPU 主机：

```text
Windows
→ WSL2
→ Docker Desktop
→ 同一套 Linux 容器
```

Windows 原生 Python 路线不在 MVP 承诺范围。

---

# 34. 镜像与依赖

建议：

```text
Base Image:
NVIDIA CUDA Runtime
+ Python 3.11
+ PyTorch matching CUDA
```

应用镜像预装：

```text
transformers
peft
accelerate
datasets
safetensors
scikit-learn
fastapi
sqlalchemy
```

llama.cpp 建议单独作为：

```text
optional runtime layer
```

避免每次用户部署都现场编译。

---

# 35. 环境探针

训练按钮可用前必须检查：

```text
CUDA available
GPU model
GPU memory
Driver
PyTorch CUDA
Disk free
Workspace writable
Model cache
Base model revision
Tokenizer check
Candidate token check
```

探针结果：

```text
READY
PREPARING
UNSUPPORTED
```

未通过不能只弹“训练失败”。

---

# 36. 安全与数据边界

MVP：

- 默认离线 / 私有部署；
- 上传文件只进入 workspace；
- 敏感字段脱敏规则可选；
- 不默认调用外部模型；
- 教师模型生成属于 P1 插件；
- 每次外部调用未来必须显式标记数据离开本机；
- API 日志不得默认记录完整敏感 State；
- hash、ID、metrics 可记录。

---

# 37. 可观测性

每个训练任务至少记录：

```text
job_id
run_id
project_id
model_revision
recipe_hash
dataset_hash
stage
step
loss_ce
loss_brier
lr
gpu_memory
elapsed
```

每次在线判断：

```text
trace_id
model_version
schema_version
calibration_version
policy_version
latency
choice
confidence
action
```

是否记录完整 state 由项目审计策略决定。

---

# 38. 代码目录

```text
SystemOneStudio/
├── frontend/
│   └── src/
│       ├── gallery/
│       ├── workspace/
│       ├── decision-schema/
│       ├── datasets/
│       ├── data-build/
│       ├── recipes/
│       ├── training/
│       ├── calibration/
│       ├── evaluation/
│       └── release/
│
├── backend/
│   └── systemone/
│       ├── api/
│       ├── db/
│       ├── projects/
│       ├── datasets/
│       ├── schemas/
│       ├── recipes/
│       ├── jobs/
│       ├── artifacts/
│       ├── evaluation/
│       └── services/
│
├── decision_runtime/
│   ├── adapters/
│   │   └── qwen.py
│   ├── serialization/
│   ├── tokens/
│   ├── inference/
│   ├── calibration/
│   └── policy/
│
├── training/
│   ├── datasets/
│   ├── augment/
│   ├── losses/
│   │   ├── candidate_ce.py
│   │   └── brier.py
│   ├── trainers/
│   │   ├── decision_sft.py
│   │   ├── decision_head.py
│   │   └── proper_score_rl.py   # P1
│   ├── calibration/
│   ├── threshold/
│   └── export/
│
├── workspace/
├── docker/
├── tests/
└── docs/
```

---

# 39. 必须做的自动化测试

## Unit

```text
DecisionSchema validation
CandidateTokenResolver
StateSerializer
Candidate logits extraction
CE/Brier loss
Temperature fit
Threshold scan
Lineage hash
```

## Integration

```text
CSV → canonical dataset
Dataset → training
Checkpoint → calibration
Calibration → threshold
Model → frozen test
Merged model → runtime
Package → reload
```

## Inference parity

同一批固定输入：

```text
Training Runtime
Merged Runtime
Quantized Runtime
```

比较 choice / probability。

## Leakage

必须有自动测试证明：

- target 字段未进入 state；
- test 未进入 augment；
- calibration 未进入 gradient training；
- test 未参与 threshold selection。

---

# 40. MVP 验收标准

## A01 项目工作台

- 项目展廊真实保存项目；
- 进入项目为七步工作台；
- 上游修改产生新版本，不覆盖历史资产。

## A02 Decision Schema

- 自定义项目可定义 black/white/gray 业务含义；
- schema version 固化；
- 模型不能输出 schema 外结果。

## A03 数据闭环

- CSV → canonical dataset；
- 字段角色正确；
- 无 label leakage；
- group-aware split；
- test 冻结。

## A04 数据构造

至少真实实现：

- class balance；
- option permutation；
- field-order permutation；
- lineage 可追溯。

## A05 Restricted-logit Training

- 真实 GPU 训练；
- 训练目标只计算 candidate logits；
- CE + Brier 真正参与梯度；
- LoRA 真正合并；
- candidate token mapping 固化。

## A06 Calibration

- held-out 数据 fit Temperature；
- 保存 calibration.json；
- 展示 raw vs calibrated NLL/ECE/Brier；
- 未改善时如实显示，不造“优化成功”。

## A07 Threshold

- 在 calibration / validation 上真实扫描；
- 用户可设 error budget；
- 输出 coverage / error；
- threshold_policy.json 可复现。

## A08 Frozen Test

- test 在此前流程不可见；
- 输出 Accuracy/Macro-F1/Brier/ECE/NLL；
- 输出 Risk-Coverage；
- 输出 Option-order Flip。

## A09 发布

- 模型包可在新进程重新加载；
- `/v1/predict` 返回真实概率；
- action 由 policy 决定；
- service 停止/启动状态真实。

## A10 工作区迁移

- 导出 workspace；
- 在新节点恢复；
- schema/data/recipe/model/calibration/policy/eval lineage 不丢失；
- hash 校验通过。

---

# 41. 开发阶段建议

## Sprint 0：判断模型技术 Spike

只做 CLI，不做完整 UI。

必须打穿：

```text
Qwen3.5-2B Base
→ candidate token resolver
→ restricted logits
→ LoRA
→ CE+Brier
→ temperature
→ threshold
→ frozen test
→ merged model
```

输出：

```text
spike_report.md
training_config.yaml
calibration.json
threshold_policy.json
evaluation.json
```

**先证明训练范式成立，再把 UI 接上。**

## Sprint 1：产品主链

实现：

- Gallery；
- Project；
- Decision Schema；
- CSV；
- dataset split；
- training recipe；
- worker；
- real training；
- calibration；
- evaluation。

## Sprint 2：发布闭环

实现：

- merge；
- quantization；
- parity gate；
- model package；
- runtime API；
- workspace export/restore。

## Sprint 3：质量增强

实现：

- hard negative；
- reliability diagram；
- advanced threshold；
- robustness suite；
- model comparison。

## Sprint 4：RLCD-like Spike

单独评估：

```text
SFT baseline
vs
Proper-score RL
```

只有证明在：

- calibration；
- high-confidence error；
- risk-coverage；

上有稳定收益才进入正式产品。

---

# 42. 第一阶段最重要的技术原则

### 原则 1：不是“生成一个分类答案”

而是：

> **直接训练候选概率分布。**

### 原则 2：Accuracy 不是发布标准

还要看：

> **概率是否可信，错误预算下能自动覆盖多少业务。**

### 原则 3：Calibration 不是报告附属项

而是：

> **模型资产的一部分。**

### 原则 4：Threshold 不是前端写死的 0.9

而是：

> **基于 held-out 数据和业务错误预算得到的版本化 Policy。**

### 原则 5：RLCD-like 不是第一天就上的“高级算法”

而是：

> **在监督判断训练与校准基线成立后，用实验判断是否值得引入。**

### 原则 6：Jev 是产品范式参考，不是可直接复制的开源 recipe

SystemOneStudio 要做的是：

> **建立自己的、可验证、可复现、可工程交付的 Judgment Model Factory。**

---

# 43. 公开路线依据与边界

本方案参考以下公开资料形成，但不把未公开内容当作事实：

1. TypeSafe AI — *Introducing System One Models & Jev*  
   https://typesafe.ai/blog/introducing-system-one-models-and-jev  
   已公开：Jev/System One 的 typed probabilistic decision 定位、新架构、parallel sampler、RLCD 名称与 calibrated-decision 目标。  
   未公开：完整模型结构、完整 reward、完整训练 recipe。

2. Cloudflare — *Introducing Clef: our open-source decision models*  
   https://blog.cloudflare.com/clef-decision-models/  
   公开了冻结 Qwen backbone、routing head、LoRA、label-smoothed CE、Brier、synthetic schema permutation 和二阶段 RLCD 思路。

3. Mapika Decider  
   https://github.com/Mapika/decider  
   公开了一次前向读取 answer-slot logits、CE 监督阶段以及 calibration-aware RL + KL 的实现路线。

4. Bespoke Nimble  
   https://github.com/bespokelabsai/nimble  
   公开了 typed decision scoring、logits → softmax、temperature calibration 的工程方式。

5. Laya / LayaStudio  
   https://github.com/NandhaKishorM/laya  
   https://github.com/biplovgautam/LayaStudio  
   公开了 ModernBERT/mmBERT + option scorer、choice/score/noul、proper scoring / RLCD、held-out calibration 和 Brier/ECE 等评测方法。

因此本方案的 MVP 主路线选择：

```text
Restricted Candidate Logits
+ CE
+ Brier
+ Held-out Temperature
+ Threshold Policy
```

属于对多条公开路线共同工程模式的收敛，而不是宣称复刻任一模型的私有实现。

---

# 44. 最终建议

SystemOneStudio V3 的真正技术核心，不应再是：

```text
“支持多少种模型”
“支持 LoRA 还是 QLoRA”
“能不能点一下训出 GGUF”
```

而应该建立以下五个真正可复用的核心引擎：

```text
1. Decision Schema Engine
2. Decision Dataset Engine
3. Decision Training Engine
4. Calibration & Policy Engine
5. Decision Runtime & Evaluation Engine
```

底座模型只是这五个引擎中的一个可替换部件。

第一版只要把下面这条链真实打穿：

```text
业务样本
→ Restricted-logit Decision SFT
→ Calibration
→ Threshold
→ Frozen Test
→ Typed Probability API
```

SystemOneStudio 就已经不是一个“小模型微调工具”，而是一套真正有独立产品价值的 **判断模型训练与交付系统**。
