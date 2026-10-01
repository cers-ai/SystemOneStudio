# syntax=docker/dockerfile:1
# Control-plane API image (需求方案.txt apps/api).
#
# Build context must be the repository root: this package is one member of a uv
# workspace with ten siblings, so uv needs every sibling pyproject.toml present.
#
#   docker build -f apps/api/Dockerfile -t son-api:dev .

# ---------------------------------------------------------------------------
# Builder: resolve and install the workspace
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

# Dependency layer first: these files rarely change, so the layer caches.
COPY pyproject.toml uv.lock ./
COPY apps/api/pyproject.toml apps/api/pyproject.toml
COPY libs/contracts/pyproject.toml libs/contracts/pyproject.toml
COPY libs/model-registry/pyproject.toml libs/model-registry/pyproject.toml
COPY services/orchestrator/pyproject.toml services/orchestrator/pyproject.toml
COPY services/data-pipeline/pyproject.toml services/data-pipeline/pyproject.toml
COPY services/synth/pyproject.toml services/synth/pyproject.toml
COPY services/trainer/pyproject.toml services/trainer/pyproject.toml
COPY services/quantizer/pyproject.toml services/quantizer/pyproject.toml
COPY services/evaluator/pyproject.toml services/evaluator/pyproject.toml
COPY services/inference/pyproject.toml services/inference/pyproject.toml
COPY services/jev-adapter/pyproject.toml services/jev-adapter/pyproject.toml

# `--all-packages` is required: plain `uv sync` only installs the *root* project,
# which here is a virtual workspace (package = false) holding just the dev tool
# group. Without it the image builds successfully and contains no application at
# all -- no uvicorn, no fastapi.
#
# `--no-group dev` keeps mypy/pytest/ruff out of a runtime image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-packages --no-group dev

COPY libs/ libs/
COPY services/ services/
COPY apps/api/src/ apps/api/src/

# `--no-editable` matters: an editable install writes a path hook pointing at
# /src in the builder, which does not exist in the runtime image, so `son_api`
# fails to import. A real install puts the packages in site-packages and lets
# this image ship no source tree at all.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-packages --no-group dev --no-editable \
    && /opt/venv/bin/python -c "import son_api.main, uvicorn; print('import ok')"

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Unbuffered output so container logs stream instead of sitting in a buffer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# No compiler toolchain in the final image. psycopg needs libpq at runtime, which
# is why postgresql-client is kept even though the build layer is discarded.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root. Models and datasets land under /data, owned by this user.
RUN groupadd --system --gid 1001 son \
    && useradd --system --uid 1001 --gid son --create-home --home-dir /home/son son \
    && mkdir -p /data \
    && chown -R son:son /data

COPY --from=builder --chown=son:son /opt/venv /opt/venv

# No source tree: --no-editable put every package into site-packages. Keeping the
# source out also means no .pth hook can point at a path that only exists in the
# builder.

WORKDIR /app
USER son

EXPOSE 9969

# /health is served by the control plane itself, so no dependency service has to
# be up for the container to be considered ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9969/health || exit 1

# Single worker on purpose. The project and scene store is in-process memory
# (see apps/api/src/son_api/routers/platform.py), and each uvicorn worker gets
# its own copy -- with --workers 2 a record written through one worker is
# invisible to the other, producing "I created it but it is not there".
# Raise this only once the ORM models land.
CMD ["uvicorn", "son_api.main:app", "--host", "0.0.0.0", "--port", "9969", "--workers", "1"]