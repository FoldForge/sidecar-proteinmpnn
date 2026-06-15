# FoldForge `sidecar-proteinmpnn`

Python gRPC sidecar wrapping **ProteinMPNN** (inverse folding — sequence design
for a fixed backbone).

Unlike the structure-prediction sidecars, ProteinMPNN returns its designs as
**inline FASTA sequences** (`DesignedSequence`), not artifacts — the orchestrator
carries them on to a folding step without an object-store round trip. The real
model loads vanilla or soluble weights based on `use_soluble_model`, shells out to
`protein_mpnn_run.py` under the shared cancellable-subprocess runner, and parses
the ranked designs (best-first by `global_score`).

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
python3.13 -m venv .venv && source .venv/bin/activate   # Python >= 3.10 required
pip install -e ".[dev]"
./scripts/gen_proto.sh        # generate stubs into src/foldforge_proteinmpnn/gen/
pytest -q                     # pure builder + FASTA parser + gRPC integration

# Start the server. NOTE: there is no __main__ / console-script — invoke serve():
python -c "from foldforge_proteinmpnn.server import serve; serve()"   # :50062
```

Install model deps (GPU box) with `pip install -e ".[model]"`.

## Config (env)
| var | default | meaning |
|-----|---------|---------|
| `FOLDFORGE_SIDECAR_BIND` | `0.0.0.0:50062` | gRPC bind address |
| `FOLDFORGE_SIDECAR_WORKERS` | `4` | gRPC thread-pool size |
| `FOLDFORGE_SIDECAR_MOCK` | `1` (on) | mock model (GPU-free); unset for the real CLI |
| `FOLDFORGE_R2_ENDPOINT` | _(empty)_ | object store — used to DOWNLOAD the upstream backbone PDB (this sidecar uploads no artifacts) |
| `FOLDFORGE_R2_BUCKET` | `foldforge` | artifact bucket |
| `FOLDFORGE_GPU_TYPE` | `L40S` | advertised GPU type (lighter than the folding models) |
| `FOLDFORGE_CANCEL_POLL_INTERVAL_S` | `1.0` | cooperative-cancel poll interval (#M2) |
| `FOLDFORGE_CANCEL_GRACE_PERIOD_S` | `10.0` | SIGTERM→SIGKILL grace on cancel (#M2) |

## Status
**Real-model code is complete; GPU inference is the only gated part.**
- **Mock mode** (`FOLDFORGE_SIDECAR_MOCK=1`, default) streams realistic
  `ProgressEvent`s and returns scored inline sequences — the full
  orchestrator → sidecar path (including cross-step sequence hand-off to AF2/Boltz)
  runs with no GPU.
- **Real mode** (`model.py` + `proteinmpnn.py`) downloads the backbone via
  `foldforge_storage`, shells out to `protein_mpnn_run.py` under
  `foldforge_subprocess.run_cancellable` (process-group kill on cancel, #M2), and
  parses the `seqs/*.fa` output (record 0 = native, skipped; 1..N = designs). The
  inference call needs a GPU + an installed ProteinMPNN; `_require()` raises a
  clear error when the entrypoint is absent. See
  [`docs/GPU-DEPLOY.md`](docs/GPU-DEPLOY.md) for the GPU runbook.

GPU-free verification (pure command builder, FASTA parser fixtures, cancel/
group-kill, gRPC client-cancel, mock e2e) all pass via `pytest`.

## License
Apache-2.0
