# Multi-stage, one file, two targets. The api and worker share every layer up
# to the entrypoint, so the registry stores the dependencies once instead of
# twice and a rebuild of one does not invalidate the other.

# ---------- builder ----------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, as their own layer. Application code changes on every
# commit; the lockfile does not, so this layer stays cached across builds.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra service

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --extra service


# ---------- runtime base ----------
FROM python:3.12-slim AS runtime

# Non-root, with no login shell and no home directory to write into. The
# Kubernetes manifests also mount the root filesystem read-only, so anything
# that needs to write has to say so explicitly.
RUN groupadd --system --gid 1001 judgekit \
 && useradd --system --uid 1001 --gid judgekit --no-create-home --shell /usr/sbin/nologin judgekit

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_JSON=true

WORKDIR /app

COPY --from=builder --chown=judgekit:judgekit /app/.venv /app/.venv
COPY --chown=judgekit:judgekit alembic.ini ./
COPY --chown=judgekit:judgekit migrations/ ./migrations/
COPY --chown=judgekit:judgekit rubrics/ ./rubrics/
COPY --chown=judgekit:judgekit datasets/ ./datasets/

USER judgekit


# ---------- api ----------
FROM runtime AS api

EXPOSE 8000

# Hits the liveness endpoint only. A health check that touched the database
# would restart a healthy container whenever Postgres was briefly slow.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "judgekit.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


# ---------- worker ----------
FROM runtime AS worker

# No port, no health check: a worker serves nothing. Kubernetes decides it is
# alive from the process, and the queue depth metric says whether it is useful.
CMD ["arq", "judgekit.worker.main.WorkerSettings"]
