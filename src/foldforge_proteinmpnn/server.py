"""gRPC server for the ProteinMPNN sidecar.

Implements ProteinMPNNService from the FoldForge proto contract. The streaming `Run`
RPC emits common.v1.ProgressEvent heartbeats and terminates with a RunResult or
common.v1.ErrorDetail, mirroring every other sidecar.
"""
from __future__ import annotations

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

log = structlog.get_logger()


def _load_stubs():
    """Import generated modules, with a clear error if codegen hasn't run."""
    try:
        from foldforge.proteinmpnn.v1 import proteinmpnn_pb2_grpc  # type: ignore
        from foldforge.proteinmpnn.v1 import proteinmpnn_pb2  # type: ignore
        from foldforge.common.v1 import common_pb2  # type: ignore
        return proteinmpnn_pb2, proteinmpnn_pb2_grpc, common_pb2
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Generated gRPC stubs missing. Run ./scripts/gen_proto.sh first."
        ) from e


def serve() -> None:
    pb2, pb2_grpc, common_pb2 = _load_stubs()

    service_cls = getattr(pb2_grpc, "ProteinMPNNServiceServicer")

    class Servicer(service_cls):
        def Run(self, request, context):  # noqa: N802 (grpc naming)
            # TODO: drive ProteinMPNNModel.run() and yield RunUpdate messages.
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("ProteinMPNN Run not yet implemented")
            return iter(())

        def GetCapabilities(self, request, context):  # noqa: N802
            return pb2.Capabilities(model_version="proteinmpnn-dev")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=settings.max_workers))
    add_fn = getattr(pb2_grpc, "add_ProteinMPNNServiceServicer_to_server")
    add_fn(Servicer(), server)
    server.add_insecure_port(settings.bind_addr)
    log.info("sidecar.start", service="ProteinMPNNService", bind=settings.bind_addr)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
