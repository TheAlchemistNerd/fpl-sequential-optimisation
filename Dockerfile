# ---------------------------------------------------------------
# Multi-stage Dockerfile for the FPL Prediction service
# Stage 1: build dependencies
# ---------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------
# Stage 2: production image
# ---------------------------------------------------------------
FROM python:3.12-slim AS production

WORKDIR /app

# System runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ /app/src/
COPY data/ /app/data/
COPY .env /app/.env

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the FastAPI inference service
CMD ["uvicorn", "src.api.inference_service:app", "--host", "0.0.0.0", "--port", "8000"]
