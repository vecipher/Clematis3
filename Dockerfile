

# syntax=docker/dockerfile:1.7

############################
# Builder stage
############################
FROM python:3.11-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /src

# Minimal toolchain for building wheels (safe even if pure-Python)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy project files needed to build the wheel
COPY pyproject.toml MANIFEST.in README.md ./
COPY clematis ./clematis
COPY scripts ./scripts

# Build wheel from the current source at this commit
RUN python -m pip install --upgrade pip build \
 && python -m build --wheel \
 && ls -l dist

############################
# Runtime stage
############################
FROM python:3.11-slim

# OCI metadata (lightweight)
LABEL org.opencontainers.image.source="https://github.com/${GITHUB_REPOSITORY:-vecipher/Clematis3}"

# Run as non-root
RUN useradd -m -u 10001 appuser
WORKDIR /app

# Offline-by-default behavior
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    CLEMATIS_NETWORK_BAN=1

# Install the wheel built in the builder stage, then clean
COPY --from=builder /src/dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl \
 && rm -rf /root/.cache/pip /tmp/*.whl

USER appuser

# Default entrypoint is the clematis CLI; remains offline by default
ENTRYPOINT ["clematis"]
CMD ["--help"]