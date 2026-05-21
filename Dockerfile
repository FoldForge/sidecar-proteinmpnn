# syntax=docker/dockerfile:1
# GPU image. Base on a CUDA runtime in production; slim python for the gRPC layer
# during early development.
FROM python:3.11-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[model]" || pip install --no-cache-dir grpcio grpcio-tools protobuf pydantic structlog
COPY . .
RUN ./scripts/gen_proto.sh
EXPOSE 50062
ENTRYPOINT ["python", "-m", "foldforge_proteinmpnn.server"]
