"""Tests for Module 3 serving patterns, batch scoring, streaming consumers, and release guards."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from legal_rag.indexing.chunker import LegalChunk
from legal_rag.serving.accelerated_runtime import mean_pooling
from legal_rag.serving.batch import BatchScorer
from legal_rag.serving.stream_consumer import LegalStreamConsumer
from serving.canary_watcher import CanaryWatcher
from serving.service import LegalRAGService


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    sample_chunk = LegalChunk(
        chunk_id="chunk_1",
        article_number=44,
        citation="Egyptian Civil Code, Article 44",
        book="General Provisions",
        chapter="Natural Persons",
        section="",
        topic="Legal Capacity",
        is_repealed=False,
        text="سن الرشد إحدى وعشرون سنة ميلادية كاملة.",
        text_en="",
        source_page=12,
        paragraph_index=0,
    )
    store.search.return_value = [(sample_chunk, 0.95)]
    store.documents_indexed = 1149
    return store


def test_batch_scorer_scoring_and_parquet_export(tmp_path, mock_vector_store):
    """Verify batch scorer processes queries and exports partitioned parquet."""
    scorer = BatchScorer(vector_store=mock_vector_store, batch_size=4)
    queries = ["سؤال قانوني 1", "سؤال قانوني 2", "سؤال قانوني 3"]

    results = scorer.score_queries(queries, top_k=2)
    assert len(results) == 3
    assert results[0]["top_sources"] == [44]
    assert results[0]["relevance_scores"] == [0.95]

    benchmark_metrics = scorer.run_benchmark(queries, output_dir=tmp_path)
    assert benchmark_metrics["total_queries_scored"] == 3
    assert benchmark_metrics["throughput_qps"] > 0
    assert Path(benchmark_metrics["output_file"]).exists()
    assert benchmark_metrics["cost_saving_factor"] > 0


def test_stream_consumer_success(mock_vector_store):
    """Test successful message handling and XACK acknowledgement."""
    mock_redis = MagicMock()
    consumer = LegalStreamConsumer.__new__(LegalStreamConsumer)
    consumer.client = mock_redis
    consumer.stream_key = "test_stream"
    consumer.group_name = "test_grp"
    consumer.consumer_name = "test_c1"
    consumer.output_stream = "test_out"
    consumer.dlq_stream = "test_dlq"
    consumer.vector_store = mock_vector_store

    event_data = {
        "query_id": "test-123",
        "question": "ما هي شروط الأهلية؟",
        "top_k": "1",
        "timestamp_ms": "1000.0",
    }
    result = consumer.process_message("1-0", event_data)

    assert result["status"] == "success"
    assert result["msg_id"] == "1-0"
    assert mock_redis.xack.called
    assert mock_redis.xadd.called


def test_stream_consumer_dlq_on_poison_message(mock_vector_store):
    """Test poison message triggers Dead Letter Queue and XACK."""
    mock_redis = MagicMock()
    consumer = LegalStreamConsumer.__new__(LegalStreamConsumer)
    consumer.client = mock_redis
    consumer.stream_key = "test_stream"
    consumer.group_name = "test_grp"
    consumer.consumer_name = "test_c1"
    consumer.output_stream = "test_out"
    consumer.dlq_stream = "test_dlq"
    consumer.vector_store = mock_vector_store

    poison_event = {"query_id": "bad-999", "poison": "true"}
    result = consumer.process_message("2-0", poison_event)

    assert result["status"] == "dlq_routed"
    assert mock_redis.xadd.called
    assert mock_redis.xack.called


def test_canary_watcher_evaluate_cycle():
    """Verify watcher detects simulated breach and calls trigger_rollback."""
    watcher = CanaryWatcher(p95_sla_ms=1000.0, max_error_rate=0.05)
    with patch.object(watcher, "trigger_rollback") as mock_rollback:
        # Cycle 1: breach detected (1 violation)
        triggered1 = watcher.evaluate_cycle(cycle_num=1, simulate_breach=True)
        assert not triggered1
        assert watcher.consecutive_violations == 1
        assert not mock_rollback.called

        # Cycle 2: consecutive breach triggers rollback
        triggered2 = watcher.evaluate_cycle(cycle_num=2, simulate_breach=True)
        assert triggered2
        assert mock_rollback.called


@pytest.mark.asyncio
async def test_bentoml_service_methods(mock_vector_store):
    """Test BentoML service endpoints with mocked dependencies."""
    with (
        patch("serving.service.LegalVectorStore", return_value=mock_vector_store),
        patch("serving.service.LegalRAGPipeline") as mock_pipe_cls,
    ):
        mock_pipeline = MagicMock()

        async def mock_answer(*args, **kwargs):
            return {
                "answer": "إجابة قانونية معتمدة",
                "sources": ["Egyptian Civil Code, Article 44"],
                "total_latency_ms": 45.2,
            }

        mock_pipeline.answer_question = mock_answer
        mock_pipe_cls.return_value = mock_pipeline

        service = LegalRAGService()
        health_res = service.health()
        assert health_res["status"] == "healthy"
        assert health_res["documents_indexed"] == 1149

        ask_res = await service.ask("ما هي شروط الأهلية؟")
        assert "answer" in ask_res
        assert ask_res["model_version"] == "v0.3.0"

        batch_res = service.ask_batch(["سؤال 1", "سؤال 2"])
        assert len(batch_res) == 2
        assert batch_res[0]["article_numbers"] == [44]


def test_mean_pooling_calculation():
    """Verify mathematical mean pooling implementation."""
    model_output = (np.array([[[1.0, 2.0], [3.0, 4.0]]]),)
    attention_mask = np.array([[1, 1]])
    pooled = mean_pooling(model_output, attention_mask)
    expected = np.array([[2.0, 3.0]])
    assert np.allclose(pooled, expected)


def test_stream_producer_publish():
    """Verify stream producer emits query events with microsecond timestamps."""
    from legal_rag.serving.stream_producer import LegalStreamProducer

    with patch("redis.Redis") as mock_redis_cls:
        mock_client = MagicMock()
        mock_client.xadd.return_value = "100-0"
        mock_redis_cls.return_value = mock_client

        producer = LegalStreamProducer(host="localhost", port=6379)
        msg_id = producer.publish_query("ما هو سن الرشد؟")
        assert msg_id == "100-0"
        assert mock_client.xadd.called


def test_stream_consumer_setup_group_busygroup():
    """Verify stream consumer gracefully handles existing BUSYGROUP error."""
    import redis

    with patch("redis.Redis") as mock_redis_cls:
        mock_client = MagicMock()
        mock_client.xgroup_create.side_effect = redis.exceptions.ResponseError(
            "BUSYGROUP Consumer Group name already exists"
        )
        mock_redis_cls.return_value = mock_client

        consumer = LegalStreamConsumer(host="localhost", port=6379, vector_store=MagicMock())
        assert consumer.group_name == "legal_rag_workers"


def test_stream_consumer_run_poll_loop(mock_vector_store):
    """Verify stream consumer loop reads and processes messages."""
    consumer = LegalStreamConsumer.__new__(LegalStreamConsumer)
    mock_redis = MagicMock()
    mock_redis.xreadgroup.return_value = [
        (
            "legal_rag_requests",
            [("1-0", {"query_id": "q1", "question": "سؤال", "top_k": "1", "timestamp_ms": "1000"})],
        )
    ]
    consumer.client = mock_redis
    consumer.stream_key = "legal_rag_requests"
    consumer.group_name = "legal_rag_workers"
    consumer.consumer_name = "w1"
    consumer.output_stream = "out"
    consumer.dlq_stream = "dlq"
    consumer.vector_store = mock_vector_store

    results = consumer.run_poll_loop(max_messages=1, count_per_read=1)
    assert len(results) == 1
    assert results[0]["status"] == "success"


def test_batch_main_cli(tmp_path):
    """Test batch module CLI invocation."""
    from legal_rag.serving.batch import main as batch_main

    mock_metrics = {
        "total_queries_scored": 2,
        "elapsed_seconds": 0.1,
        "throughput_qps": 20.0,
        "peak_memory_mb": 0.5,
        "cost_per_million_batch_usd": 0.05,
        "cost_per_million_webservice_usd": 1.35,
        "cost_saving_factor": 27.0,
        "output_file": str(tmp_path / "preds.parquet"),
    }
    with (
        patch("sys.argv", ["batch.py", "--num-samples", "2", "--output-dir", str(tmp_path)]),
        patch("legal_rag.serving.batch.BatchScorer.run_benchmark", return_value=mock_metrics),
    ):
        batch_main()


@pytest.mark.asyncio
async def test_accelerated_runtime_benchmarks_mocked(tmp_path):
    """Test accelerated runtime benchmark suite functions."""
    from legal_rag.serving.accelerated_runtime import benchmark_embeddings, benchmark_llm_generation

    with (
        patch("legal_rag.serving.accelerated_runtime.AutoModel") as mock_pt,
        patch("legal_rag.serving.accelerated_runtime.AutoTokenizer") as mock_tok,
        patch("onnxruntime.InferenceSession") as mock_ort,
    ):
        dummy_pt_inst = MagicMock()
        dummy_pt_inst.return_value.last_hidden_state.numpy.return_value = np.zeros((1, 5, 384))
        mock_pt.from_pretrained.return_value = dummy_pt_inst

        dummy_tok_inst = MagicMock()
        dummy_tok_inst.return_value = {
            "input_ids": np.zeros((1, 5), dtype=np.int64),
            "attention_mask": np.ones((1, 5), dtype=np.int64),
        }
        mock_tok.from_pretrained.return_value = dummy_tok_inst

        dummy_ort_inst = MagicMock()
        dummy_ort_inst.run.return_value = [np.zeros((1, 5, 384))]
        inp1 = MagicMock()
        inp1.name = "input_ids"
        inp2 = MagicMock()
        inp2.name = "attention_mask"
        dummy_ort_inst.get_inputs.return_value = [inp1, inp2]
        mock_ort.return_value = dummy_ort_inst

        res = benchmark_embeddings(iterations=2)
        assert len(res["results"]) == 3

    with patch("legal_rag.rag.generator.LegalGenerator") as mock_gen_cls:
        mock_gen = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield "كلمة1 "
            yield "كلمة2 "

        mock_gen.generate_stream = mock_stream
        mock_gen_cls.return_value = mock_gen

        llm_res = await benchmark_llm_generation(concurrency_levels=[1, 2])
        assert len(llm_res) == 2
        assert llm_res[0]["concurrency"] == 1
