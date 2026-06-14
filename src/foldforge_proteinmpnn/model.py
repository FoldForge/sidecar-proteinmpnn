"""Model wrapper for the ProteinMPNN sidecar.

Two implementations live here:

* ``ProteinMPNNModel`` — the real model. Its ``run`` is a TODO until GPU hardware
  and weights are available; everything around it (gRPC server, streaming,
  artifact handling) is already wired so the swap is localized to this method.
* ``MockModel`` — a GPU-free stand-in that emits realistic progress heartbeats
  and returns synthetic artifact references. The server uses it when
  ``FOLDFORGE_SIDECAR_MOCK`` is set (default until the real model is wired), so
  the full orchestrator -> sidecar gRPC path can be exercised end to end without
  a GPU.

Both yield the same shape: a sequence of ``(fraction, message)`` progress tuples
followed by a single ``RunOutput``. The server translates that into the
tool-specific proto ``RunResult``.
"""
from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from foldforge_storage import ArtifactStore


@dataclass
class RunOutput:
    """Normalized result the server maps onto the proto RunResult.

    ``artifacts`` is a list of dicts with keys: uri, content_type, size_bytes,
    sha256. ``sequences`` (ProteinMPNN) is a list of dicts with keys: fasta,
    global_score, seq_recovery, sample_index. ``metrics`` carries tool-specific
    scalars (plddt, scores, ...).
    """

    artifacts: list[dict] = field(default_factory=list)
    sequences: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _artifact(uri: str, content_type: str) -> dict:
    return {"uri": uri, "content_type": content_type, "size_bytes": 0, "sha256": ""}


class ProteinMPNNModel:
    """Real ProteinMPNN wrapper, backed by `protein_mpnn_run.py`.

    Pipeline per run:
      1. Download the backbone PDB referenced by the request from the artifact
         store (the input is a `backbone_pdb` Artifact ref produced upstream by
         RFdiffusion).
      2. Run `protein_mpnn_run.py` for sequence sampling (GPU), under
         `run_cancellable` so a client cancel kills the process group.
      3. Parse the sampled FASTA into ranked sequences (inline — no artifact
         upload; ProteinMPNN's output is sequences, carried on to a folding step).

    ``store`` is injected (defaults to a bucket-scoped ArtifactStore). Inference
    requires a GPU + installed ProteinMPNN; absent that, the helper raises a clear
    error and the sidecar stays in mock mode.
    """

    def __init__(self, store: ArtifactStore | None = None, bucket: str = "foldforge") -> None:
        self._store = store if store is not None else ArtifactStore(bucket=bucket)
        self._loaded = False

    def load(self) -> None:
        # ProteinMPNN loads weights on first invocation; nothing to preload.
        self._loaded = True

    def run(self, params: dict) -> Iterator[tuple[float, str] | RunOutput]:
        # Lazy import so the GPU-free path (and tests) never need ProteinMPNN.
        from . import proteinmpnn

        backbone_uri = params.get("backbone_uri") or ""
        if not backbone_uri:
            raise ValueError("proteinmpnn real model requires a backbone_pdb artifact")
        num_sequences = int(params.get("num_sequences", 8) or 8)
        sampling_temp = float(params.get("sampling_temp", 0.1) or 0.1)
        seed = params.get("seed")
        use_soluble = bool(params.get("use_soluble_model", False))
        omit_aas = params.get("omit_aas") or None

        # Cooperative cancel (DEBT #M2): the server injects a callback that is true
        # once the client cancels / disconnects. ProteinMPNN sampling is usually
        # short, but it's still a GPU subprocess, so we forward the same callback.
        should_cancel = params.get("should_cancel")
        poll_interval_s = float(params.get("cancel_poll_interval_s", 1.0))
        grace_period_s = float(params.get("cancel_grace_period_s", 10.0))

        yield (0.1, "fetching backbone")
        pdb_bytes = self._store.get(backbone_uri)

        with tempfile.TemporaryDirectory(prefix="ff-mpnn-") as tmp:
            out_dir = Path(tmp) / "out"
            yield (0.25, "running protein_mpnn_run.py")
            result = proteinmpnn.run_proteinmpnn(
                pdb_bytes, out_dir, num_sequences,
                sampling_temp=sampling_temp, seed=seed if seed else None,
                use_soluble_model=use_soluble, omit_aas=omit_aas,
                should_cancel=should_cancel,
                poll_interval_s=poll_interval_s,
                grace_period_s=grace_period_s,
            )

            yield (0.95, f"sampled {len(result.sequences)} sequences")
            sequences = [
                {
                    "fasta": s.fasta,
                    "global_score": s.global_score,
                    "seq_recovery": s.seq_recovery,
                    "sample_index": s.sample_index,
                }
                for s in result.sequences
            ]
            yield RunOutput(
                sequences=sequences,
                metrics={"mock": False, "count": len(sequences)},
            )


class MockModel:
    """GPU-free stand-in. Deterministic-ish synthetic run for the ProteinMPNN tool."""

    #: progress stages emitted before completion
    STAGES = ['encoding backbone', 'sampling sequences', 'scoring', 'writing fasta']

    def __init__(self, bucket: str = "foldforge") -> None:
        self._bucket = bucket

    def load(self) -> None:  # nothing to load
        pass

    def run(self, params: dict) -> Iterator[tuple[float, str] | RunOutput]:
        # Cooperative cancel (DEBT #M2): honor the same should_cancel callback the
        # real model forwards, so the cancel path is exercised end-to-end GPU-free.
        from .proteinmpnn import RunCancelled

        should_cancel = params.get("should_cancel")
        step_delay_s = float(params.get("mock_step_delay_s", 0.0))

        n_stages = len(self.STAGES)
        for i, stage in enumerate(self.STAGES, start=1):
            if should_cancel is not None and should_cancel():
                raise RunCancelled("mock run cancelled by client")
            if step_delay_s:
                time.sleep(step_delay_s)
            yield (i / n_stages, stage)

        count = int(params.get("num_sequences", 8) or 8)
        count = max(1, count)
        # Synthesize designed sequences with varied (descending-quality) scores so
        # downstream "pick the best sequence" selection is actually exercised.
        # Each design also gets a few residues mutated by index so the sequences
        # are genuinely distinct (not N copies of one string) — closer to real
        # ProteinMPNN output and avoids the UI looking buggy.
        base = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
        substitutions = "AGSTVLIDEKR"

        def mutate(seq: str, idx: int) -> str:
            chars = list(seq)
            # Mutate `idx` positions (monotonic, no modulo wraparound) so every
            # design index yields a distinct string — idx 0 stays the reference.
            for k in range(idx):
                pos = (idx * 7 + k * 11) % len(chars)
                chars[pos] = substitutions[(idx + k) % len(substitutions)]
            return "".join(chars)

        sequences = [
            {
                "fasta": ">design_%d score=%.3f\n%s"
                % (i, 0.8 + 0.05 * i, mutate(base, i)),
                "global_score": round(0.8 + 0.05 * i, 3),  # lower = better; index 0 best
                "seq_recovery": round(0.62 - 0.02 * i, 3),
                "sample_index": i,
            }
            for i in range(count)
        ]
        yield RunOutput(
            sequences=sequences, metrics={"mock": True, "count": count}
        )
