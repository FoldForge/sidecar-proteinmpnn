"""GPU-free tests for the ProteinMPNN integration (cmd, FASTA parse, cancel).

The real `protein_mpnn_run.py` sampling needs a GPU and is NOT exercised here.
What IS verified without a GPU:
  * build_proteinmpnn_cmd renders the standard argv (flags, soluble, omit AAs).
  * parse_proteinmpnn_fasta parses a real-schema seqs/*.fa: skips the native
    record, ranks designs best-first by global_score, renumbers sample_index.
  * run_proteinmpnn raises a clear error when the entrypoint is absent.
  * MockModel honors should_cancel. The shared run_cancellable machinery is
    tested in foldforge-pylib.
"""
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from foldforge_proteinmpnn.proteinmpnn import (  # noqa: E402
    RunCancelled,
    build_proteinmpnn_cmd,
    parse_proteinmpnn_fasta,
    run_proteinmpnn,
)
from foldforge_proteinmpnn.model import MockModel, RunOutput  # noqa: E402


# ---- cmd builder (pure) --------------------------------------------------

def test_build_cmd_minimal():
    cmd = build_proteinmpnn_cmd("protein_mpnn_run.py", "/in/bb.pdb", "/out", 8)
    assert cmd[0] == "protein_mpnn_run.py"
    assert "--pdb_path" in cmd and "/in/bb.pdb" in cmd
    assert "--out_folder" in cmd and "/out" in cmd
    assert "--num_seq_per_target" in cmd and "8" in cmd
    assert "--sampling_temp" in cmd


def test_build_cmd_options():
    cmd = build_proteinmpnn_cmd(
        "protein_mpnn_run.py", "/in/bb.pdb", "/out", 4,
        sampling_temp=0.2, seed=42, use_soluble_model=True, omit_aas=["C", "M"],
    )
    assert "--seed" in cmd and "42" in cmd
    assert "--use_soluble_model" in cmd
    assert "--omit_AAs" in cmd
    i = cmd.index("--omit_AAs")
    assert cmd[i + 1] == "CM"


# ---- FASTA parser (pure, fixture matching real schema) -------------------

REAL_FA = """\
>backbone, score=1.5000, global_score=1.5000, fixed_chains=[], designed_chains=['A']
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
>T=0.1, sample=1, score=1.1000, global_score=1.1000, seq_recovery=0.6200
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVA
>T=0.1, sample=2, score=0.9000, global_score=0.9000, seq_recovery=0.7000
MKTGYIAKQRQISFVKSHFSRQLEERLGLIEVA
"""


def test_parse_skips_native_and_ranks_best_first():
    result = parse_proteinmpnn_fasta(REAL_FA)
    # 2 sampled designs (native record 0 skipped).
    assert len(result.sequences) == 2
    # Best-first by global_score (0.9 < 1.1).
    assert result.sequences[0].global_score == 0.9
    assert result.sequences[1].global_score == 1.1
    # sample_index renumbered by rank.
    assert [s.sample_index for s in result.sequences] == [0, 1]
    assert result.sequences[0].seq_recovery == 0.7
    # FASTA payload carries the actual sequence.
    assert "MKTGYIAKQRQ" in result.sequences[0].fasta


def test_parse_empty_when_only_native():
    result = parse_proteinmpnn_fasta(">native score=1.0\nMKTAY\n")
    assert result.sequences == []


def test_run_proteinmpnn_raises_without_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError) as ei:
        run_proteinmpnn(b"HEADER\nEND\n", tmp_path / "out", num_sequences=1)
    assert "protein_mpnn_run.py" in str(ei.value)
    assert "GPU" in str(ei.value)


# ---- cooperative cancel (mock, GPU-free) ---------------------------------

def test_mock_model_raises_when_cancelled():
    model = MockModel()
    gen = model.run({"num_sequences": 4, "should_cancel": lambda: True})
    with pytest.raises(RunCancelled):
        for _ in gen:
            pass


def test_mock_model_completes_without_cancel():
    model = MockModel()
    items = list(model.run({"num_sequences": 3, "should_cancel": lambda: False}))
    assert any(isinstance(i, RunOutput) for i in items)
