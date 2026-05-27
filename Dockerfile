FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install Python package and dependencies
COPY pyproject.toml .
COPY lift/ lift/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

CMD ["lift"]
