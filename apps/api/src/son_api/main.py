"""Control-plane FastAPI application.

Scope at M0: serve the OpenAPI contract and the health/version surface, so that
`libs/contracts` types and the generated TypeScript types have a single
generation source. Feature endpoints land per the milestone plan in
开发计划.md (M1 data, M2 training, M3 inference/eval, M5 deploy).
"""

from fastapi import FastAPI
from pydantic import BaseModel

from son_api.routers import datasets, models
from son_contracts import JEV_SPEC_VERSION, JevFlags
from son_contracts.predict import JEV_OUTPUT_SCHEMA, PredictRequest, PredictResponse


class HealthResponse(BaseModel):
    status: str
    service: str
    jev_spec_version: str


class JevSpecResponse(BaseModel):
    """Machine-readable JEV contract served to the frontend.

    Exposing the output schema lets apps/web generate the Playground result
    view and the format-compliance display from one source instead of
    duplicating the field list.
    """

    spec_version: str
    output_schema: dict[str, object]
    default_flags: JevFlags
    note: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="SystemOneStudio Control Plane",
        version="0.1.0",
        description=(
            "Control plane for the zero-code decision model training platform. "
            "The prediction surface is served separately per deployment at /v1/predict."
        ),
    )
    app.include_router(datasets.router)
    app.include_router(models.router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="son-api",
            jev_spec_version=JEV_SPEC_VERSION,
        )

    @app.get("/meta/jev-spec", response_model=JevSpecResponse, tags=["system"])
    def jev_spec() -> JevSpecResponse:
        return JevSpecResponse(
            spec_version=JEV_SPEC_VERSION,
            output_schema=JEV_OUTPUT_SCHEMA,
            default_flags=JevFlags(),
            note=(
                "Paraphrased from 需求方案.txt section 6.1, not the upstream JEV "
                "document. L2/L3 alignment claims are unverified (see 技术方案.md Q3)."
            ),
        )

    @app.post(
        "/meta/echo-prediction",
        response_model=PredictResponse,
        tags=["system"],
        summary="Validate a prediction payload against the JEV output schema",
    )
    def echo_prediction(request: PredictRequest) -> PredictResponse:
        """Contract self-check for the frontend and integration tests.

        Not a real inference path: the actual /v1/predict lives in the
        inference service, mounted per deployment.
        """
        sample = request.sample
        return PredictResponse(
            decision="gray",
            score=0.5,
            confidence=0.5,
            reason=f"contract echo: received {len(sample)} field(s)",
        )

    return app


app = create_app()
