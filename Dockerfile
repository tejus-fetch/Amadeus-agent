FROM python:3.12-slim

# Install system dependencies (libpq for psycopg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

WORKDIR /app

# Copy only dependency files first
COPY pyproject.toml uv.lock ./

# Install deps (cached)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Copy app code
COPY . .

# Activate venv automatically
ENV PATH="/app/.venv/bin:$PATH"

# Run app
CMD ["python", "-m", "app.main"]
