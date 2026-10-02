"""Assistant endpoints: configuration in 系统管理, chat everywhere.

The API key is write-only from this module's perspective. There is no GET that
returns it, and a PUT with no key field leaves the existing one alone, so an
operator editing the temperature does not have to re-paste a secret.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from son_assistant import (
    KNOWN_PROVIDERS,
    AssistantPreferences,
    LlmError,
    LlmNotConfigured,
    LlmSettings,
    ToolContext,
    TrainingAssistant,
    default_settings,
    describe_tools,
)

router = APIRouter(prefix="/api/assistant", tags=["assistant"])

_LOCK = threading.Lock()
_SETTINGS: LlmSettings | None = None
_PREFERENCES = AssistantPreferences()


def current_settings() -> LlmSettings:
    """Process-wide settings, seeded from the environment on first use."""
    global _SETTINGS
    with _LOCK:
        if _SETTINGS is None:
            _SETTINGS = default_settings()
        return _SETTINGS


class SettingsView(BaseModel):
    """Read shape. No key, ever."""

    provider: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_s: float
    supports_tools: bool
    enabled: bool
    configured: bool
    has_api_key: bool
    masked_key: str


class ProvidersView(BaseModel):
    providers: dict[str, str]
    needs_api_key: list[str]


class ToolsView(BaseModel):
    tools: list[dict[str, object]]


class PreferencesView(BaseModel):
    ground_with_platform_state: bool
    max_tool_rounds: int
    require_confirmation_for_actions: bool


@router.get("/providers", response_model=ProvidersView, summary="可选服务商")
def providers() -> ProvidersView:
    return ProvidersView(
        providers=KNOWN_PROVIDERS,
        needs_api_key=[p for p in KNOWN_PROVIDERS if p not in {"ollama", "vllm", "local_llamacpp"}],
    )


@router.get("/settings", response_model=SettingsView, summary="读取助手配置")
def get_settings() -> SettingsView:
    return SettingsView(
        **current_settings().public(),
        masked_key=current_settings().masked_key(),
    )


class UpdateSettingsRequest(BaseModel):
    """Write shape.

    ``api_key`` is optional and omitted-means-unchanged, so saving one field does
    not wipe the secret. Send an empty string to clear it deliberately.
    """

    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=64, le=32000)
    timeout_s: float | None = Field(default=None, ge=1.0, le=600.0)
    supports_tools: bool | None = None
    enabled: bool | None = None


@router.put("/settings", response_model=SettingsView, summary="保存助手配置")
def update_settings(request: UpdateSettingsRequest) -> SettingsView:
    global _SETTINGS
    current = current_settings()

    payload: dict[str, object] = {}
    for key in (
        "provider",
        "base_url",
        "model",
        "temperature",
        "max_tokens",
        "timeout_s",
        "supports_tools",
        "enabled",
    ):
        value = getattr(request, key)
        if value is not None:
            payload[key] = value

    # None = leave alone; "" = clear deliberately.
    if request.api_key is not None:
        payload["api_key"] = request.api_key.strip() or None

    # An empty base_url means "resolve it from the provider", so the validator's
    # provider-default logic can run.
    if request.base_url is not None and not request.base_url.strip():
        payload.pop("base_url", None)

    # Only validate a provider that was actually supplied. Checking
    # `payload.get("provider")` unconditionally raised a KeyError on updates
    # that touched other fields, turning a 422 into a 500.
    provider = payload.get("provider")
    if provider is not None and provider not in KNOWN_PROVIDERS and provider != "custom":
        raise HTTPException(status_code=422, detail=f"未知的模型服务商 {provider}")

    try:
        updated = current.model_copy(update=payload)
        # Re-validate: model_copy skips validators, and base_url in particular
        # must be checked.
        updated = LlmSettings.model_validate(updated.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _LOCK:
        _SETTINGS = updated
    return SettingsView(**updated.public(), masked_key=updated.masked_key())


@router.post("/settings/probe", summary="测试连通性")
def probe_settings() -> dict[str, object]:
    """Check the endpoint before claiming the configuration works.

    A save button that reports success for a broken endpoint is worse than no
    save button.
    """
    settings = current_settings()
    if not settings.configured:
        return {"reachable": False, "detail": "配置不完整，尚未启用"}
    result = TrainingAssistant(settings).probe()
    result["model"] = settings.model
    return result


@router.get("/preferences", response_model=PreferencesView, summary="读取助手行为偏好")
def get_preferences() -> PreferencesView:
    return PreferencesView(**_PREFERENCES.model_dump())


class UpdatePreferencesRequest(BaseModel):
    """Partial update: omitting a field leaves it alone.

    The read model is a full object, so accepting it as the write model would
    force a client to round-trip everything it does not intend to change.
    """

    ground_with_platform_state: bool | None = None
    max_tool_rounds: int | None = Field(default=None, ge=1, le=20)
    require_confirmation_for_actions: bool | None = None


@router.put("/preferences", response_model=PreferencesView, summary="保存助手行为偏好")
def update_preferences(request: UpdatePreferencesRequest) -> PreferencesView:
    global _PREFERENCES
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    try:
        _PREFERENCES = AssistantPreferences.model_validate({**_PREFERENCES.model_dump(), **updates})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PreferencesView(**_PREFERENCES.model_dump())


@router.get("/tools", response_model=ToolsView, summary="助手可调用的工具")
def tools() -> ToolsView:
    """Catalogue, so an operator can see exactly what the assistant can reach."""
    return ToolsView(tools=describe_tools())


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


class HistoryTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    current_step: str | None = None
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)
    #: Client-side facts the API cannot look up, e.g. the quality report the
    #: browser already holds. Passed to the model labelled as user context.
    quality_report: dict[str, object] | None = None
    has_data: bool = False
    has_quality: bool = False


class ToolInvocation(BaseModel):
    name: str
    arguments: dict[str, object]
    result: dict[str, object]


class AskResponse(BaseModel):
    text: str
    tool_calls: list[ToolInvocation]
    rounds: int
    elapsed_ms: float
    truncated: bool
    usage: dict[str, int]


def _build_context(request: AskRequest | SuggestRequest) -> ToolContext:
    """Node capabilities come from the platform, not from the client's claim.

    A probe failure stays None rather than collapsing to False: turning "could
    not determine" into "no GPU" makes the assistant tell a GPU node's user they
    cannot train.
    """
    gpu: bool | None
    services: tuple[str, ...] = ("控制面 API",)
    try:
        from son_inference.llamacpp_client import gpu_report

        report = gpu_report()
        gpu = bool(report.get("cuda_available"))
        if "reason" in report and not gpu:
            # torch is absent or CUDA is unavailable: genuinely no GPU.
            gpu = False
        services = ("控制面 API",) + (("训练后端", "推理服务") if gpu else ())
    except Exception:
        gpu = None
        services = ("控制面 API",)

    return ToolContext(
        current_step=request.current_step,
        available_services=services,
        gpu_available=gpu,
    )


@router.post("/ask", response_model=AskResponse, summary="向助手提问")
def ask(request: AskRequest) -> AskResponse:
    """One assistant turn.

    Refusals surface as 409 rather than 200 with an apologetic message, so the
    UI can tell "not configured" from "the model said something unhelpful".
    """
    settings = current_settings()
    if not settings.configured:
        raise HTTPException(
            status_code=409,
            detail=(
                "智能训练助手尚未配置可用的大模型。"
                "请到「系统管理 → 智能训练助手」选择服务商并填写配置；"
                "若使用本地推理服务则无需填写密钥。"
            ),
        )

    assistant = TrainingAssistant(settings, _PREFERENCES)
    ctx = _build_context(request)

    from son_assistant import AssistantMessage

    history = [AssistantMessage(role=t.role, content=t.content) for t in request.history]

    try:
        reply = assistant.ask(
            request.question,
            ctx=ctx,
            history=history,
            extra_context={"quality": request.quality_report} if request.quality_report else None,
        )
    except LlmNotConfigured as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LlmError as exc:
        # Provider errors are 502: the platform is fine, the upstream is not.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        text=reply.text,
        tool_calls=[
            ToolInvocation(name=c["name"], arguments=c["arguments"], result=c["result"])
            for c in reply.trace.tool_calls
        ],
        rounds=reply.trace.rounds,
        elapsed_ms=round(reply.trace.elapsed_ms, 1),
        truncated=reply.trace.truncated,
        usage=reply.trace.usage,
    )


class SuggestRequest(BaseModel):
    current_step: str | None = None
    has_data: bool = False
    has_quality: bool = False


@router.post("/suggest", summary="获取与当前步骤相关的引导语")
def suggest(request: SuggestRequest) -> dict[str, str]:
    """Contextual placeholder text.

    Free: no model call. A suggestion that costs a round trip would be a bad
    default for something rendered on every step change.
    """
    assistant = TrainingAssistant(current_settings(), _PREFERENCES)
    return {
        "text": assistant.suggest(
            _build_context(request),
            has_data=request.has_data,
            has_quality=request.has_quality,
        )
    }
