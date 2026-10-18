"""Tests for model serialization and ONNX parity."""

import pytest

from legal_rag.indexing.export import export_embedding_to_onnx, verify_onnx_parity


@pytest.fixture(scope="module")
def onnx_model_path(tmp_path_factory):
    fn = tmp_path_factory.mktemp("models") / "embedder_test.onnx"
    export_embedding_to_onnx(output_path=fn)
    return fn


def test_onnx_export_and_parity(onnx_model_path):
    assert onnx_model_path.exists()
    assert onnx_model_path.stat().st_size > 100 * 1024  # > 100 KB

    is_close, max_diff, benchmarks = verify_onnx_parity(
        onnx_model_path,
        atol=1e-3,
    )
    assert is_close is True
    assert max_diff < 1e-3
    assert benchmarks["onnx_latency_ms"] > 0
    assert benchmarks["pytorch_latency_ms"] > 0
