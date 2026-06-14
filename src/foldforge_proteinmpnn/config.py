"""Runtime configuration from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    bind_addr: str = os.environ.get("FOLDFORGE_SIDECAR_BIND", "0.0.0.0:50062")
    max_workers: int = int(os.environ.get("FOLDFORGE_SIDECAR_WORKERS", "4"))
    # R2 object storage for artifact exchange (PDB/CIF/MSA blobs).
    r2_endpoint: str = os.environ.get("FOLDFORGE_R2_ENDPOINT", "")
    r2_bucket: str = os.environ.get("FOLDFORGE_R2_BUCKET", "foldforge")
    default_gpu_type: str = os.environ.get("FOLDFORGE_GPU_TYPE", "L40S")
    # Cooperative cancel (DEBT #M2): how often (seconds) the proteinmpnn subprocess
    # poll loop checks for client cancellation. Too high: cancellation is detected
    # late, so the GPU keeps burning longer after the client gave up. Too low:
    # needless wakeups. ProteinMPNN sampling is short, so 1s is generous.
    cancel_poll_interval_s: float = float(
        os.environ.get("FOLDFORGE_CANCEL_POLL_INTERVAL_S", "1.0")
    )
    # Grace period (seconds) between SIGTERM and SIGKILL when cancelling the
    # proteinmpnn subprocess group. Too high: a process ignoring SIGTERM keeps the
    # GPU that much longer. Too low: no time to release the CUDA context cleanly.
    cancel_grace_period_s: float = float(
        os.environ.get("FOLDFORGE_CANCEL_GRACE_PERIOD_S", "10.0")
    )


settings = Settings()
