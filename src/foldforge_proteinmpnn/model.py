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

from collections.abc import Iterator
from dataclasses import dataclass, field


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
    """Real ProteinMPNN wrapper. Loads weights lazily on a GPU box."""

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        # TODO(gpu): import upstream ProteinMPNN inference code and load weights.
        self._loaded = True

    def run(self, params: dict) -> Iterator[tuple[float, str] | RunOutput]:
        # TODO(gpu): replace with real inference. Yield (fraction, message)
        # tuples for progress, then a final RunOutput.
        raise NotImplementedError("ProteinMPNN inference not yet implemented")


class MockModel:
    """GPU-free stand-in. Deterministic-ish synthetic run for the ProteinMPNN tool."""

    #: progress stages emitted before completion
    STAGES = ['encoding backbone', 'sampling sequences', 'scoring', 'writing fasta']

    def __init__(self, bucket: str = "foldforge") -> None:
        self._bucket = bucket

    def load(self) -> None:  # nothing to load
        pass

    def run(self, params: dict) -> Iterator[tuple[float, str] | RunOutput]:
        n_stages = len(self.STAGES)
        for i, stage in enumerate(self.STAGES, start=1):
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
