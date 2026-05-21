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


settings = Settings()
