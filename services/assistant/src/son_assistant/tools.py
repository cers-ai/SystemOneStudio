"""The tools the assistant can call.

Every tool **reads** platform state. None of them mutate anything, and that is a
design decision rather than an omission: the assistant's job is to lower the
difficulty of using the platform, and a model that can silently start training
runs on someone's cluster is a liability, not a feature. When it believes an
action is needed it says which action and lets the user click it.

Each tool returns small structured data. Returning whole reports would burn the
context window and bury the one number the assistant needed.

Tool names are part of the contract with the model, so they are stable and
snake_case. Descriptions are written for the model, not for humans -- they are the
only thing telling it when to reach for each one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolContext:
    """What the caller knows and the assistant cannot look up for itself."""

    #: Current wizard step, when the request came from the wizard.
    current_step: str | None = None
    #: Names of services currently reachable from the API process.
    available_services: tuple[str, ...] = ()
    gpu_available: bool | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], dict[str, Any]]

    def as_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


def _node_capabilities(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """What this deployment can actually run right now.

    ``gpu_available`` is tri-state on purpose. None means the probe failed, which
    is different from "no GPU"; reporting a failed probe as a confident absence
    makes the assistant tell a GPU node's user they cannot train.
    """
    if ctx.gpu_available is None:
        return {
            "gpu_available": None,
            "available_services": list(ctx.available_services),
            "implication": (
                "无法确定本节点的图形处理器状态（探测失败）。"
                "请让用户确认，或运行 python -m son_cli doctor。"
                "在确认之前，不要断言训练可行或不可行。"
            ),
        }
    return {
        "gpu_available": ctx.gpu_available,
        "available_services": list(ctx.available_services),
        "implication": (
            "训练、压缩导出、推理与响应速度指标均可在本节点执行"
            if ctx.gpu_available
            else "本节点没有图形处理器：数据治理、样本扩增与效果评测可用，"
            "训练与推理需要在具备 GPU 的节点上执行"
        ),
    }


def _workflow_state(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Where the user is in the seven-step flow."""
    return {
        "current_step": ctx.current_step,
        "hint": (
            "若用户不清楚下一步该做什么，先确认当前步骤，再说明该步骤的产出"
            if ctx.current_step
            else "用户尚未进入流程；引导其从场景展廊选模板，或直接进入训练流程"
        ),
    }


def _dataset_quality(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Quality report facts, so advice is grounded in the real numbers."""
    payload = args.get("report")
    if not isinstance(payload, dict):
        return {
            "available": False,
            "guidance": (
                "尚未获得质量报告。请让用户先上传数据并查看质量报告，不要凭空猜测数据质量。"
            ),
        }
    return {
        "available": True,
        "score": payload.get("score"),
        "dimensions": payload.get("dimensions"),
        "label_distribution": payload.get("label_distribution"),
        "anomalies": payload.get("anomalies", []),
        "suggestions": payload.get("suggestions", []),
    }


def _explain_hyperparams(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Why the recommender chose these values."""
    return {
        "values": args.get("values"),
        "explanation_basis": (
            "超参由数据量与模型规格计算得出：学习率随模型规模反向缩放，"
            "批次大小随模型规模下降、随量化训练上升，步数由数据量与时间预算共同决定。"
            "偏好对齐阶段的学习率固定为基础风格适配的一半。"
        ),
        "caveat": ("显存需求与训练时长为经验估算，需在具备 GPU 的节点实测后才可作为承诺。"),
    }


def _diagnose_failure(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Map a raw failure onto the platform's known causes."""
    text = str(args.get("error", "")).lower()

    if "out of memory" in text or "cuda oom" in text:
        return {
            "likely_cause": "显存不足",
            "actions": [
                "改用低显存微调（QLoRA），显存需求约为标准做法的一半",
                "换用 1.5B 规格的底座模型",
                "减小批次大小，同时提高梯度累积步数以保持等效批次不变",
            ],
        }
    if "no kernel image" in text or "cuda error" in text:
        return {
            "likely_cause": "GPU 驱动或 CUDA 版本与容器不匹配",
            "actions": [
                "确认宿主机 nvidia-smi 显示的 CUDA 版本不低于镜像要求",
                "确认 NVIDIA Container Toolkit 已安装：docker run --rm --gpus all "
                "nvidia/cuda 镜像 nvidia-smi",
            ],
        }
    if "not found" in text and ("llama" in text or "quantize" in text):
        return {
            "likely_cause": "推理与压缩工具未安装或不在 PATH 上",
            "actions": ["在具备 GPU 的镜像上运行 deploy/gpu 下的编排文件"],
        }
    if "libcuda" in text or ("no such file" in text and "so" in text):
        return {
            "likely_cause": "容器内未挂载 GPU 驱动",
            "actions": ["确认编排文件中启用了 gpus 配置"],
        }
    return {
        "likely_cause": "无法从错误信息判断",
        "actions": [
            "请用户提供完整的错误输出，不要截断",
            "在该节点上运行 python -m son_cli doctor 并提供输出",
        ],
    }


def _explain_constraint(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Answer 'why is the product like this' from the hard constraints."""
    topic = str(args.get("topic", "")).lower()
    constraints: dict[str, list[str]] = {
        "latency": [
            "响应时间 P95 ≤ 100ms 是硬性目标，并且排在效果指标之前",
            "这决定了推理必须走本地 GGUF 路线，而非托管服务",
            "当前存在一处未解冲突：P95 目标与判定依据 200 字符上限互相矛盾，口径待定案",
        ],
        "label": [
            "判定结果是三分类（黑 / 白 / 灰），不是二分类",
            "灰样本代表证据不足，不应被折叠进黑或白",
            "因此场景创建时拒绝二分类标签体系",
        ],
        "reason": [
            "每条模型输出都必须带判定依据，长度上限 200 字符",
            "这是可解释性与合规的底线，用于审核判定是否合理",
        ],
        "test": [
            "合成数据不得进入测试集",
            "测试集只从原始上传数据中抽取，并在划分后再次断言校验",
        ],
        "quant": [
            "默认压缩档位与评测基线一致，因为不同档位的评测数字不可比",
            "所以只有相同压缩档位下的对比才有意义",
        ],
        "size": [
            "MVP 范围限定在 1.5B–3B 并使用轻量微调",
            "这是针对显存不足的主动取舍，不是遗漏；更大规格已登记但标记为范围外",
        ],
    }
    for key, value in constraints.items():
        if key in topic:
            return {"topic": key, "constraints": value}
    return {
        "topic": topic or "unknown",
        "constraints": [],
        "note": "该问题不在已知约束清单内，请基于实际界面与数据回答，不要编造产品规则",
    }


TOOLS: tuple[Tool, ...] = (
    Tool(
        name="get_node_capabilities",
        description=(
            "查询当前部署节点实际具备的能力：是否有图形处理器、哪些服务可用。"
            "在建议任何训练、部署或性能相关操作之前必须先调用，"
            "因为同一个操作在不同节点上可行性不同。"
        ),
        parameters={"type": "object", "properties": {}},
        handler=_node_capabilities,
    ),
    Tool(
        name="get_workflow_state",
        description="查询用户当前处于七步流程的哪一步。用于判断用户卡在哪一步。",
        parameters={"type": "object", "properties": {}},
        handler=_workflow_state,
    ),
    Tool(
        name="get_dataset_quality",
        description=(
            "查询上传数据的质量报告（评分、各维度、标签分布、异常检测、系统建议）。"
            "在回答数据质量相关问题前调用，避免凭空判断。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "report": {
                    "type": "object",
                    "description": "前端持有的质量报告；没有则不传该参数",
                }
            },
        },
        handler=_dataset_quality,
    ),
    Tool(
        name="explain_hyperparameters",
        description="解释系统为什么推荐了当前这组超参数。回答超参相关问题时调用。",
        parameters={
            "type": "object",
            "properties": {
                "values": {
                    "type": "object",
                    "description": "当前推荐的超参数键值对",
                }
            },
        },
        handler=_explain_hyperparams,
    ),
    Tool(
        name="diagnose_failure",
        description=(
            "根据错误信息定位最可能的原因并给出排查步骤。用户报错时先调用，不要凭猜测回答。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "用户看到的完整错误输出",
                }
            },
        },
        handler=_diagnose_failure,
    ),
    Tool(
        name="explain_product_constraint",
        description=(
            "查询产品的硬性约束及其原因。当用户问'为什么必须这样'时调用，"
            "避免给出与产品设计相悖的建议。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "话题关键词：latency / label / reason / test / quant / size",
                }
            },
        },
        handler=_explain_constraint,
    ),
)

TOOL_BY_NAME: dict[str, Tool] = {tool.name: tool for tool in TOOLS}


def tool_schemas() -> list[dict[str, Any]]:
    return [tool.as_openai_schema() for tool in TOOLS]


def run_tool(name: str, arguments: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Dispatch a tool call.

    An unknown name returns an error rather than raising: the model may invent a
    tool, and a raised exception would abort the whole turn instead of letting
    the model recover.
    """
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return {
            "error": f"未知的工具 {name!r}",
            "available": sorted(TOOL_BY_NAME),
        }
    try:
        return tool.handler(arguments or {}, ctx)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def describe_tools() -> list[dict[str, Any]]:
    """Catalogue for the UI, so 系统管理 can show what the assistant can reach."""
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters} for t in TOOLS
    ]


def tool_names() -> list[str]:
    return [t.name for t in TOOLS]


def dumps(value: Any) -> str:
    """Compact JSON for tool results, so they do not eat the context window."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
