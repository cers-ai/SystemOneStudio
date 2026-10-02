"""Base model registry and adapter abstraction."""

from son_model_registry.adapter import (
    AdapterFactory,
    BaseModelAdapter,
    RegistryModel,
    TrainConfig,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from son_model_registry.qwen_adapter import (
    MODEL_IDS,
    Qwen25Adapter,
    QwenAdapterUnavailable,
    render_features,
    resolve_model_path,
    supported_models,
)
from son_model_registry.seed import REGISTRY_SEED, mvp_models

__all__ = [
    "MODEL_IDS",
    "REGISTRY_SEED",
    "AdapterFactory",
    "BaseModelAdapter",
    "Qwen25Adapter",
    "QwenAdapterUnavailable",
    "RegistryModel",
    "TrainConfig",
    "get_adapter",
    "mvp_models",
    "register_adapter",
    "registered_adapters",
    "render_features",
    "resolve_model_path",
    "supported_models",
]
