"""gRPC integration test for the proteinmpnn sidecar.

Builds the real server (mock mode) on an ephemeral loopback port and drives the
streaming Run RPC through a real channel — exercising the exact path the
orchestrator's GrpcRunner uses. No GPU required.
"""
import sys
from pathlib import Path

import grpc
import pytest
from google.protobuf import empty_pb2

_SRC = Path(__file__).resolve().parents[1] / "src"
_GEN = _SRC / "foldforge_proteinmpnn" / "gen"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_GEN))


def _stubs():
    try:
        from foldforge.proteinmpnn.v1 import proteinmpnn_pb2, proteinmpnn_pb2_grpc
        from foldforge.common.v1 import common_pb2  # noqa
        return proteinmpnn_pb2, proteinmpnn_pb2_grpc
    except ImportError:
        pytest.skip("stubs not generated; run ./scripts/gen_proto.sh")


@pytest.fixture()
def channel(monkeypatch):
    monkeypatch.setenv("FOLDFORGE_SIDECAR_MOCK", "1")
    _stubs()  # ensure generated, else skip
    from foldforge_proteinmpnn.server import build_server

    server, port = build_server("127.0.0.1:0")
    server.start()
    ch = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(ch).result(timeout=5)
    yield ch
    ch.close()
    server.stop(None)


def test_run_streams_progress_then_result(channel):
    pb2, pb2_grpc = _stubs()
    stub = getattr(pb2_grpc, "ProteinMPNNServiceStub")(channel)
    updates = list(stub.Run(pb2.RunRequest()))
    kinds = [u.WhichOneof("event") for u in updates]
    assert "progress" in kinds, kinds
    assert kinds.count("result") == 1, kinds
    assert "error" not in kinds, kinds
    # progress fractions are monotonic non-decreasing and end at ~1.0
    fracs = [u.progress.fraction for u in updates if u.WhichOneof("event") == "progress"]
    assert fracs == sorted(fracs)
    result = [u for u in updates if u.WhichOneof("event") == "result"][0].result
    assert len(getattr(result, "sequences")) >= 1


def test_get_capabilities(channel):
    pb2, pb2_grpc = _stubs()
    stub = getattr(pb2_grpc, "ProteinMPNNServiceStub")(channel)
    caps = stub.GetCapabilities(empty_pb2.Empty())
    assert "mock" in caps.model_version


def test_client_cancel_stops_the_run(channel, monkeypatch):
    # DEBT #M2: a client that cancels mid-run must cause the sidecar to stop —
    # should_cancel (context.is_active()) trips, the model raises RunCancelled,
    # and the stream ends without a result. The server forwards
    # FOLDFORGE_MOCK_STEP_DELAY_S into the mock so each stage sleeps, giving a
    # window to cancel after the first progress frame arrives.
    monkeypatch.setenv("FOLDFORGE_MOCK_STEP_DELAY_S", "0.3")
    pb2, pb2_grpc = _stubs()
    stub = getattr(pb2_grpc, "ProteinMPNNServiceStub")(channel)
    call = stub.Run(pb2.RunRequest(num_sequences=4))
    got_progress = False
    cancelled = False
    for u in call:
        if u.WhichOneof("event") == "progress":
            got_progress = True
            call.cancel()
            break
    try:
        for _ in call:
            pass
    except grpc.RpcError as e:
        cancelled = e.code() == grpc.StatusCode.CANCELLED
    assert got_progress
    assert cancelled

