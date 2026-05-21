"""gRPC server for the ProteinMPNN sidecar.

Implements ProteinMPNNService from the FoldForge proto contract. The streaming `Run` RPC
drives the model generator, emitting common.v1.ProgressEvent heartbeats for each
progress tuple and terminating with a tool-specific RunResult (or an ErrorDetail
on failure).

Runner selection:
* ``FOLDFORGE_SIDECAR_MOCK`` set (default) -> MockModel: GPU-free synthetic run
  so the orchestrator <-> sidecar gRPC path works end to end without hardware.
* unset -> the real ProteinMPNNModel (raises until GPU inference is wired).

``build_server`` constructs (but does not start) a gRPC server, so tests can bind
an ephemeral port; ``serve`` builds, starts and blocks.
"""
from __future__ import annotations

import os
import sys
from concurrent import futures
from pathlib import Path

import grpc
import structlog

# Generated stubs live under src/foldforge_proteinmpnn/gen after scripts/gen_proto.sh.
_GEN = Path(__file__).parent / "gen"
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

from .config import settings  # noqa: E402
from .model import MockModel, RunOutput, ProteinMPNNModel  # noqa: E402

log = structlog.get_logger()


def _load_stubs():
    """Import generated modules, with a clear error if codegen hasn't run."""
    try:
        from foldforge.proteinmpnn.v1 import proteinmpnn_pb2, proteinmpnn_pb2_grpc  # type: ignore
        from foldforge.common.v1 import common_pb2  # type: ignore
        return proteinmpnn_pb2, proteinmpnn_pb2_grpc, common_pb2
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Generated gRPC stubs missing. Run ./scripts/gen_proto.sh first."
        ) from e


def use_mock() -> bool:
    """Mock is the default until the real model is wired; opt out explicitly."""
    return os.environ.get("FOLDFORGE_SIDECAR_MOCK", "1") not in ("0", "false", "False")


def _make_servicer(pb2, pb2_grpc, common_pb2, model):
    def _progress(fraction: float, message: str):
        ev = common_pb2.ProgressEvent(
            state=common_pb2.JOB_STATE_RUNNING, fraction=float(fraction), message=message
        )
        ev.at.GetCurrentTime()
        return pb2.RunUpdate(progress=ev)

    def _build_result(pb2, common_pb2, request, out: RunOutput):
        n = int(out.metrics.get("count", 1) or 1)
        seqs = [
            pb2.Sequence(
                fasta=">mock_seq_%d\nMOCKAMINOACIDSEQUENCEPLACEHOLDER" % i,
                global_score=0.0, seq_recovery=0.0, sample_index=i,
            )
            for i in range(n)
        ]
        return pb2.RunUpdate(result=pb2.RunResult(sequences=seqs))

    class Servicer(getattr(pb2_grpc, "ProteinMPNNServiceServicer")):
        def Run(self, request, context):  # noqa: N802 (grpc naming)
            params = _params_from_request(request)
            try:
                final = None
                for item in model.run(params):
                    if isinstance(item, RunOutput):
                        final = item
                        break
                    fraction, message = item
                    yield _progress(fraction, message)
                if final is None:
                    final = RunOutput()
                yield _build_result(pb2, common_pb2, request, final)
            except NotImplementedError as e:
                yield pb2.RunUpdate(
                    error=common_pb2.ErrorDetail(code="UNIMPLEMENTED", message=str(e), retryable=False)
                )
            except Exception as e:  # pragma: no cover - defensive
                log.error("sidecar.run_error", error=str(e))
                yield pb2.RunUpdate(
                    error=common_pb2.ErrorDetail(code="INTERNAL", message=str(e), retryable=True)
                )

        def GetCapabilities(self, request, context):  # noqa: N802
            mode = "mock" if use_mock() else "real"
            return pb2.Capabilities(model_version="proteinmpnn-" + mode)

    return Servicer()


def build_server(bind_addr: str | None = None):
    """Construct a gRPC server (not started). Returns (server, bound_port)."""
    pb2, pb2_grpc, common_pb2 = _load_stubs()
    model = MockModel(bucket=settings.r2_bucket) if use_mock() else ProteinMPNNModel()
    model.load()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=settings.max_workers))
    getattr(pb2_grpc, "add_ProteinMPNNServiceServicer_to_server")(
        _make_servicer(pb2, pb2_grpc, common_pb2, model), server
    )
    port = server.add_insecure_port(bind_addr or settings.bind_addr)
    return server, port


def serve() -> None:
    server, port = build_server()
    mode = "mock" if use_mock() else "real"
    log.info("sidecar.start", service="ProteinMPNNService", port=port, mode=mode)
    server.start()
    server.wait_for_termination()


def _params_from_request(request) -> dict:
    """Flatten a typed RunRequest into a plain dict the mock model reads.

    The real model reads the typed request directly; the mock only needs scalar
    fields, so repeated and message-typed fields are skipped.
    """
    out: dict = {}
    for field in request.DESCRIPTOR.fields:
        # `is_repeated` is the modern API (protobuf >=5); fall back to `label`
        # on older runtimes. `cpp_type` for message detection is not deprecated.
        is_repeated = getattr(field, "is_repeated", None)
        if is_repeated is None:
            is_repeated = field.label == field.LABEL_REPEATED
        is_message = field.cpp_type == field.CPPTYPE_MESSAGE
        if is_repeated or is_message:
            continue
        try:
            out[field.name] = getattr(request, field.name)
        except Exception:
            continue
    return out


if __name__ == "__main__":
    serve()
