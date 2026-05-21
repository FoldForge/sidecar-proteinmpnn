#!/usr/bin/env bash
# Generate Python gRPC stubs from the vendored proto submodule.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=src/foldforge_proteinmpnn/gen
mkdir -p "$OUT"
python -m grpc_tools.protoc \
  -I proto \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  --pyi_out="$OUT" \
  proto/foldforge/common/v1/common.proto \
  proto/foldforge/proteinmpnn/v1/proteinmpnn.proto
# protoc emits package-rooted imports; ensure the gen dir is importable.
find "$OUT" -type d -exec touch {}/__init__.py \;
echo "generated stubs in $OUT"
