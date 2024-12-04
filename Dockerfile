FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1

# Install required system dependencies
 RUN apt-get update && apt-get install -y \
     gcc \
     libpq-dev \
     && apt-get clean

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ./uv.lock .
COPY ./pyproject.toml .

RUN uv sync --frozen --no-cache

# Copy all necessary python code to run the server
COPY ./*.py ./

# Container Entrypoint
COPY entrypoint.sh .

ENTRYPOINT ["sh", "entrypoint.sh"]
