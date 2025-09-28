# syntax=docker/dockerfile:1.7

############################
# Builder stage
############################
FROM python:3.11-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /src
# Minimal toolchain (keep if you may add C extensions later)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Build from source at this commit
COPY pyproject.toml MANIFEST.in README.md ./
COPY clematis ./clematis
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip build && \
    python -m build --wheel && \
    ls -l dist

############################
# Runtime stage
############################
FROM python:3.11-slim
LABEL org.opencontainers.image.source="https://github.com/${GITHUB_REPOSITORY:-vecipher/Clematis3}"

# Non-root user
RUN useradd -m -u 10001 appuser
WORKDIR /app

# Provide repo-local configs; ensure imports resolve
COPY configs ./configs
ENV PYTHONPATH=/app:$PYTHONPATH

# Offline-by-default runtime
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    CLEMATIS_NETWORK_BAN=1

# Install wheel built in builder stage
COPY --from=builder /src/dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /root/.cache/pip /tmp/*.whl

USER appuser
ENTRYPOINT ["clematis"]
CMD ["--help"]
