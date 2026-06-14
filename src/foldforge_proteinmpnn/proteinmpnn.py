"""ProteinMPNN CLI integration for the real model.

ProteinMPNN does inverse folding: given a backbone PDB it samples amino-acid
sequences. Unlike the structure-prediction sidecars it returns **sequences
inline** (no artifact upload) — the orchestrator carries them on to a folding
step (af2/boltz) without an object-store round trip.

Split so the GPU-free parts are unit-testable on any box and only the actual
sampling needs a GPU:

* `build_proteinmpnn_cmd(...)` — PURE: assembles the `protein_mpnn_run.py` argv
  (input PDB dir, output dir, num_seq_per_target, sampling_temp, seed, soluble
  weights, fixed positions / omit AAs).
* `parse_proteinmpnn_fasta(text)` — PURE: parses a ProteinMPNN `seqs/*.fa` file
  into ranked `DesignedSequence`s. The first record is the native sequence
  (skipped); each sampled record's header carries `score=`/`global_score=` and
  `seq_recovery=`. Verified against a fixture matching the real header schema.
* `run_proteinmpnn(...)` — shells out under the shared `run_cancellable` so a
  client cancel kills the GPU subprocess GROUP. Requires a GPU + installed
  ProteinMPNN; raises a clear error when the entrypoint is absent (degrades
  loudly, never silently).

ProteinMPNN output contract (confirmed against dauparas/ProteinMPNN):
  out_dir/seqs/{pdbstem}.fa  — FASTA where record 0 is the input/native
  (header: ">name, score=..., global_score=..., ...") and records 1..N are the
  sampled designs (header: ">T=0.1, sample=1, score=..., global_score=...,
  seq_recovery=...").
"""
from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from foldforge_subprocess import RunCancelled, run_cancellable

log = structlog.get_logger()

__all__ = [
    "RunCancelled",
    "run_cancellable",
    "DesignedSequence",
    "build_proteinmpnn_cmd",
    "parse_proteinmpnn_fasta",
    "run_proteinmpnn",
]

# header float fields, e.g. "score=1.234", "global_score=1.5", "seq_recovery=0.62"
_SCORE_RE = re.compile(r"\bscore=([0-9.]+)")
_GLOBAL_RE = re.compile(r"\bglobal_score=([0-9.]+)")
_RECOVERY_RE = re.compile(r"\bseq_recovery=([0-9.]+)")
_SAMPLE_RE = re.compile(r"\bsample=(\d+)")


@dataclass
class DesignedSequence:
    """One sampled sequence from a ProteinMPNN run (mirrors proto Sequence)."""

    fasta: str          # a one-record FASTA (header + sequence)
    global_score: float  # mean negative log-likelihood, lower is better
    seq_recovery: float
    sample_index: int


@dataclass
class ProteinMPNNResult:
    sequences: list[DesignedSequence] = field(default_factory=list)


def build_proteinmpnn_cmd(
    entrypoint: str,
    pdb_path: str,
    out_dir: str,
    num_sequences: int,
    sampling_temp: float = 0.1,
    seed: int | None = None,
    use_soluble_model: bool = False,
    omit_aas: list[str] | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Assemble the `protein_mpnn_run.py` argv (pure, GPU-free).

    Uses ProteinMPNN's standard flags. `pdb_path` is a single backbone PDB
    (`--pdb_path`); outputs land under `{out_dir}/seqs/`. `sampling_temp` is
    passed as a space-joined list (ProteinMPNN accepts multiple temps). Soluble
    weights, an explicit seed, and omitted AAs are appended only when set.
    """
    cmd = [
        entrypoint,
        "--pdb_path", pdb_path,
        "--out_folder", out_dir,
        "--num_seq_per_target", str(max(1, num_sequences)),
        "--sampling_temp", str(sampling_temp),
        "--batch_size", "1",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if use_soluble_model:
        cmd += ["--use_soluble_model"]
    if omit_aas:
        cmd += ["--omit_AAs", "".join(omit_aas)]
    if extra:
        cmd += extra
    return cmd


def parse_proteinmpnn_fasta(text: str) -> ProteinMPNNResult:
    """Parse a ProteinMPNN `seqs/*.fa` into ranked sampled sequences (pure).

    Record 0 is the native/input sequence (skipped). Each sampled record's header
    carries `global_score=` (preferred) / `score=` and `seq_recovery=`; we sort
    best-first by global_score (lower = better) and renumber sample_index by rank.
    """
    records: list[tuple[str, str]] = []  # (header, sequence)
    header: str | None = None
    seq_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        elif line.strip():
            seq_lines.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq_lines)))

    # Skip record 0 (native). Parse the sampled designs.
    designs: list[DesignedSequence] = []
    for header, seq in records[1:]:
        gm = _GLOBAL_RE.search(header) or _SCORE_RE.search(header)
        rm = _RECOVERY_RE.search(header)
        sm = _SAMPLE_RE.search(header)
        global_score = float(gm.group(1)) if gm else 0.0
        seq_recovery = float(rm.group(1)) if rm else 0.0
        sample_index = int(sm.group(1)) if sm else len(designs)
        designs.append(
            DesignedSequence(
                fasta=f">sample={sample_index} global_score={global_score} "
                f"seq_recovery={seq_recovery}\n{seq}",
                global_score=global_score,
                seq_recovery=seq_recovery,
                sample_index=sample_index,
            )
        )

    # Rank best-first (lower global_score is better) and renumber by rank.
    designs.sort(key=lambda d: d.global_score)
    for rank, d in enumerate(designs):
        d.sample_index = rank
    return ProteinMPNNResult(sequences=designs)


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise RuntimeError(
            f"{binary} not found on PATH. Install ProteinMPNN (dauparas/ProteinMPNN) "
            f"on a GPU box; see sidecar-proteinmpnn/docs/GPU-DEPLOY.md. The sidecar "
            f"runs in mock mode (FOLDFORGE_SIDECAR_MOCK=1) until then."
        )
    return path


def run_proteinmpnn(
    pdb_bytes: bytes,
    out_dir: Path,
    num_sequences: int,
    sampling_temp: float = 0.1,
    seed: int | None = None,
    use_soluble_model: bool = False,
    omit_aas: list[str] | None = None,
    entrypoint: str = "protein_mpnn_run.py",
    should_cancel: Callable[[], bool] | None = None,
    poll_interval_s: float = 1.0,
    grace_period_s: float = 10.0,
) -> ProteinMPNNResult:
    """Run inference with `protein_mpnn_run.py` and parse the sampled sequences.

    Writes `pdb_bytes` to a backbone file, shells out under `run_cancellable`
    (client cancel kills the process GROUP and raises RunCancelled), then parses
    `{out_dir}/seqs/{stem}.fa`. Requires a GPU; raises a clear error if the
    entrypoint is missing.
    """
    binary = _require(entrypoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = out_dir / "backbone.pdb"
    pdb_path.write_bytes(pdb_bytes)
    cmd = build_proteinmpnn_cmd(
        binary, str(pdb_path), str(out_dir), num_sequences,
        sampling_temp=sampling_temp, seed=seed,
        use_soluble_model=use_soluble_model, omit_aas=omit_aas,
    )
    log.info("proteinmpnn.run", cmd=" ".join(cmd))
    run_cancellable(
        cmd,
        log_dir=out_dir,
        should_cancel=should_cancel,
        poll_interval_s=poll_interval_s,
        grace_period_s=grace_period_s,
    )
    fa = out_dir / "seqs" / "backbone.fa"
    if not fa.exists():
        raise RuntimeError(f"proteinmpnn produced no sequences at {fa}")
    return parse_proteinmpnn_fasta(fa.read_text())
