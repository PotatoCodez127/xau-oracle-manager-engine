# Stage 1: Build-backend compilation isolation
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Generate local package wheels to prevent header bloat in the runtime stage
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r <(pip install build && python -m build --sdist --wheel --outdir /build/dist . && echo "xau-dl-engine") || \
    pip wheel --no-cache-dir --wheel-dir /build/wheels .

# Stage 2: Final immutable runtime deployment image
FROM python:3.10-slim AS runner

WORKDIR /app

# Enforce absolute temporal agreement across trading execution regions
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/*

# Copy workspace directories ensuring exact layout alignment
COPY environment/ ./environment/
COPY features/ ./features/
COPY models/ ./models/
COPY export_tensorboard.py .

ENTRYPOINT ["python"]