"""Config smoke test — runs without GPU, model weights, or generated stubs."""
from foldforge_proteinmpnn.config import settings


def test_defaults():
    assert settings.bind_addr.endswith(":50062")
    assert settings.max_workers >= 1
    assert settings.r2_bucket
