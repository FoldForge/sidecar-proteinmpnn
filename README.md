# FoldForge `sidecar-proteinmpnn`

Python gRPC sidecar wrapping **ProteinMPNN** (inverse folding — sequence design for a fixed backbone).

Wraps ProteinMPNN. Loads either the vanilla or soluble weights based on RunRequest.use_soluble_model.

Lighter GPU footprint than the folding models; can batch many sequences per backbone.

## Contract
Implements `ProteinMPNNService` from
[`FoldForge/proto`](https://github.com/FoldForge/proto), vendored here as a git
submodule at `./proto`. Every sidecar exposes the same shape: a server-streaming
`Run` that emits `ProgressEvent` heartbeats and ends with a `RunResult` or
`ErrorDetail`.

## Develop (no GPU needed for the gRPC layer)
```bash
git clone --recurse-submodules git@github.com:FoldForge/sidecar-proteinmpnn.git
cd sidecar-proteinmpnn
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
./scripts/gen_proto.sh        # generate stubs from proto
pytest -q                     # config smoke test
python -m foldforge_proteinmpnn.server   # serves gRPC on :50062
```

Install model deps (GPU box) with `uv pip install -e ".[model]"`.

## Config (env)
| var | default |
|-----|---------|
| `FOLDFORGE_SIDECAR_BIND` | `0.0.0.0:50062` |
| `FOLDFORGE_SIDECAR_WORKERS` | `4` |
| `FOLDFORGE_R2_ENDPOINT` | _(empty)_ |
| `FOLDFORGE_R2_BUCKET` | `foldforge` |
| `FOLDFORGE_GPU_TYPE` | `L40S` |

## Status
MVP skeleton: gRPC server + model wrapper interface. `Run` returns `UNIMPLEMENTED`
until the model integration lands (`src/foldforge_proteinmpnn/model.py`).

## License
Apache-2.0
