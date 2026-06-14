# ProteinMPNN sidecar — GPU deployment runbook

This sidecar runs in **mock mode by default** (`FOLDFORGE_SIDECAR_MOCK=1`) so the
whole gateway → orchestrator → sidecar path is exercisable without a GPU. This
document is how to switch it to **real ProteinMPNN inference** on a GPU box.

The code is already wired: `ProteinMPNNModel` downloads the backbone PDB the
request references (produced upstream by RFdiffusion) from the artifact store,
runs `protein_mpnn_run.py` under the shared cancellable runner, and parses the
sampled FASTA into ranked sequences. ProteinMPNN returns **sequences inline** (no
artifact upload) — they're carried on to a folding step (AF2/Boltz). Only the
sampling shell-out needs a GPU; everything around it (cmd build, FASTA parse,
backbone fetch) is GPU-free and unit-tested.

## What ProteinMPNN does

ProteinMPNN does **inverse folding**: given a fixed backbone it samples amino-acid
sequences likely to fold into that shape. It is the middle stage of the flagship
pipeline (RFdiffusion → **ProteinMPNN** → AF2): backbone from RFdiffusion,
sequences here, structure validation from AF2/Boltz. Fast and CPU-runnable, but a
GPU speeds batch sampling.

## Hardware / driver prerequisites

- NVIDIA GPU recommended (ProteinMPNN also runs on CPU, just slower). A modest
  GPU (L40S/A10/T4) is plenty — it's far lighter than AF2/Boltz/RFdiffusion.
- CUDA-enabled PyTorch matching your driver. Verify with `nvcc --version`.
- ProteinMPNN model weights (bundled with the upstream repo under `*_weights/`).

## Install ProteinMPNN

```bash
git clone https://github.com/dauparas/ProteinMPNN.git
cd ProteinMPNN
# the repo ships weights (vanilla_model_weights/, soluble_model_weights/).
# ensure protein_mpnn_run.py is on PATH:
export PATH="$(pwd):$PATH"
which protein_mpnn_run.py
```

The sidecar invokes `protein_mpnn_run.py --pdb_path <backbone> --out_folder <dir>
--num_seq_per_target N --sampling_temp T`, then reads `<dir>/seqs/backbone.fa`.

## Environment

| Variable | Purpose | Example |
|----------|---------|---------|
| `FOLDFORGE_SIDECAR_MOCK` | **Set to `0` to enable the real model.** | `0` |
| `FOLDFORGE_SIDECAR__BIND_ADDR` | gRPC bind address | `0.0.0.0:50062` |
| `FOLDFORGE_R2_ENDPOINT` | object-store endpoint → **required in real mode**: the model DOWNLOADS the upstream backbone PDB from here. Unset = no store, the fetch fails. | `https://<acct>.r2.cloudflarestorage.com` |
| `FOLDFORGE_R2_BUCKET` / `r2_bucket` | bucket holding the backbone artifact | `foldforge` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | R2 keys (read by boto3 from the env; never in the repo) | — |
| `FOLDFORGE_CANCEL_POLL_INTERVAL_S` | cancel-check interval for the subprocess (DEBT #M2) | `1.0` |
| `FOLDFORGE_CANCEL_GRACE_PERIOD_S` | SIGTERM→SIGKILL grace for the subprocess group | `10.0` |

> Unlike the other sidecars, ProteinMPNN **needs the object store in real mode**
> even though it uploads nothing — its input backbone is an artifact reference it
> must fetch.

## Launch (real mode)

```bash
export FOLDFORGE_SIDECAR_MOCK=0
export FOLDFORGE_SIDECAR__BIND_ADDR=0.0.0.0:50062
export FOLDFORGE_R2_ENDPOINT=...      # + AWS_* creds (required: backbone fetch)
export PATH="/path/to/ProteinMPNN:$PATH"
python -c "from foldforge_proteinmpnn.server import serve; serve()"
# logs: sidecar.start ... mode=real
```

`GetCapabilities` will report `model_version=proteinmpnn-real`.

## Verify on the GPU box

1. **Capabilities:** a gRPC `GetCapabilities` returns `proteinmpnn-real`.
2. **Flagship chain:** submit RFdiffusion → ProteinMPNN → AF2 through the gateway.
   The proteinmpnn step should reach SUCCEEDED with `num_sequences` designed
   sequences whose `global_score` VARY (the mock emits fixed ascending scores)
   and are ranked best-first.
3. **Sequence sanity:** each returned FASTA is a distinct amino-acid string of
   the backbone's length; `seq_recovery` is in (0, 1].
4. **Cooperative cancel (DEBT #M2):** start a large `num_sequences` run, then
   cancel the workflow mid-run. Confirm the `protein_mpnn_run.py` process group
   dies (`nvidia-smi` shows the GPU freed within ~`CANCEL_GRACE_PERIOD_S`) — the
   sidecar logs `subprocess cancelled` then `sidecar.run_cancelled`.

## What stays mocked until this is done

Until a GPU box runs the above, `FOLDFORGE_SIDECAR_MOCK=1` keeps the sidecar
serving synthetic sequences. `protein_mpnn_run.py` invocation raises a clear
`RuntimeError` (pointing here) if the entrypoint is absent on PATH, so real mode
fails loudly rather than silently returning fake data.
