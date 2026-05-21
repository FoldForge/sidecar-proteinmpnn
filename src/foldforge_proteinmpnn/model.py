"""Thin wrapper around the ProteinMPNN model.

Wraps ProteinMPNN. Loads either the vanilla or soluble weights based on RunRequest.use_soluble_model.

Keeping the model behind this interface lets the gRPC server (server.py) be
unit-tested with a fake, and lets the heavy deps (torch, model weights) stay out
of the import path until actually needed on a GPU box.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class RunOutput:
    """Normalized result the server translates into the proto RunResult."""
    artifacts: list[dict]
    metrics: dict


class ProteinMPNNModel:
    """Loads weights lazily; `run` yields progress fractions then a final output."""

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        # TODO: import upstream ProteinMPNN inference code and load weights.
        self._loaded = True

    def run(self, params: dict) -> Iterator[tuple[float, str] | RunOutput]:
        """Yield (fraction, message) progress tuples, then a final RunOutput.

        TODO: replace with real inference. The server consumes this generator and
        emits ProgressEvent heartbeats for each tuple.
        """
        raise NotImplementedError("ProteinMPNN inference not yet implemented")
