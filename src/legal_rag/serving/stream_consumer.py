"""Redis Streams consumer group worker with XACK, dead-letter queue, and end-to-end latency tracking."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from typing import Any

import redis

from legal_rag.indexing.vector_store import LegalVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("legal_rag.serving.stream_consumer")


class LegalStreamConsumer:
    """Stream consumer group worker enforcing at-least-once delivery, XACK, and DLQ."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        stream_key: str = "legal_rag_requests",
        group_name: str = "legal_rag_workers",
        consumer_name: str = "worker-1",
        output_stream: str = "legal_rag_responses",
        dlq_stream: str = "legal_rag_dlq",
        vector_store: LegalVectorStore | None = None,
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.stream_key = stream_key
        self.group_name = group_name
        self.consumer_name = consumer_name
        self.output_stream = output_stream
        self.dlq_stream = dlq_stream

        self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)
        self.vector_store = vector_store or LegalVectorStore()
        self._setup_group()

    def _setup_group(self):
        """Ensure stream and consumer group exist."""
        try:
            self.client.xgroup_create(self.stream_key, self.group_name, id="0", mkstream=True)
            logger.info(
                f"Consumer group '{self.group_name}' created on stream '{self.stream_key}'."
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{self.group_name}' already exists.")
            else:
                raise

    def process_message(self, msg_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Process a single stream message with at-least-once delivery and DLQ routing on failure."""
        start_worker_time = time.time()
        try:
            # Check for poison message simulation
            if data.get("poison") == "true":
                raise ValueError("Simulated poison message triggering Dead Letter Queue.")

            query_id = data.get("query_id", "unknown")
            question = data.get("question", "")
            top_k = int(data.get("top_k", 3))
            produced_timestamp_ms = float(data.get("timestamp_ms", start_worker_time * 1000))

            if not question:
                raise ValueError("Missing 'question' in event data payload.")

            # Perform retrieval
            search_results = self.vector_store.search(question, top_k=top_k)
            sources = [chunk.citation for chunk, _ in search_results]
            article_numbers = [chunk.article_number for chunk, _ in search_results]

            current_time_ms = time.time() * 1000
            e2e_latency_ms = round(current_time_ms - produced_timestamp_ms, 2)
            worker_duration_ms = round((time.time() - start_worker_time) * 1000, 2)

            response_payload = {
                "query_id": query_id,
                "question": question,
                "sources": json.dumps(sources),
                "article_numbers": json.dumps(article_numbers),
                "e2e_latency_ms": str(e2e_latency_ms),
                "worker_duration_ms": str(worker_duration_ms),
                "processed_by": self.consumer_name,
            }

            # 1. Publish to output stream
            self.client.xadd(self.output_stream, response_payload)

            # 2. Acknowledge message (XACK)
            self.client.xack(self.stream_key, self.group_name, msg_id)

            logger.info(
                f"Successfully processed {msg_id} (query_id={query_id}) - "
                f"e2e_latency={e2e_latency_ms}ms, worker_duration={worker_duration_ms}ms. XACK sent."
            )
            return {"status": "success", "msg_id": msg_id, "e2e_latency_ms": e2e_latency_ms}

        except Exception as exc:  # noqa: BLE001
            # Poison message handling: forward to DLQ and acknowledge original stream to avoid blocking
            logger.error(
                f"Error processing {msg_id}: {exc}. Routing to Dead Letter Queue '{self.dlq_stream}'."
            )
            dlq_payload = {
                "failed_msg_id": msg_id,
                "original_payload": json.dumps(data),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": str(time.time() * 1000),
            }
            self.client.xadd(self.dlq_stream, dlq_payload)
            self.client.xack(self.stream_key, self.group_name, msg_id)
            return {"status": "dlq_routed", "msg_id": msg_id, "error": str(exc)}

    def run_poll_loop(
        self, max_messages: int = 10, count_per_read: int = 5
    ) -> list[dict[str, Any]]:
        """Read and process messages up to max_messages."""
        processed_records = []
        messages_handled = 0

        while messages_handled < max_messages:
            # Read from group
            entries = self.client.xreadgroup(
                groupname=self.group_name,
                consumername=self.consumer_name,
                streams={self.stream_key: ">"},
                count=count_per_read,
                block=1000,
            )

            if not entries:
                break

            for stream_name, messages in entries:
                for msg_id, data in messages:
                    res = self.process_message(msg_id, data)
                    processed_records.append(res)
                    messages_handled += 1
                    if messages_handled >= max_messages:
                        break

        return processed_records


def main():
    parser = argparse.ArgumentParser(description="Redis Stream Consumer Group Worker.")
    parser.add_argument(
        "--test-mode", action="store_true", help="Run simulated in-process self-test"
    )
    parser.add_argument("--max-messages", type=int, default=10, help="Max messages to process")
    args = parser.parse_args()

    if args.test_mode:
        logger.info("Executing self-test mode for stream consumer logic...")
        # Self-test logic asserting XACK and DLQ logic
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        consumer = LegalStreamConsumer.__new__(LegalStreamConsumer)
        consumer.client = mock_client
        consumer.stream_key = "test_requests"
        consumer.group_name = "test_group"
        consumer.consumer_name = "test_worker"
        consumer.output_stream = "test_responses"
        consumer.dlq_stream = "test_dlq"
        consumer.vector_store = LegalVectorStore()

        # Test valid message
        res1 = consumer.process_message(
            "1001-0",
            {
                "query_id": "q1",
                "question": "ما هي شروط الأهلية؟",
                "top_k": "2",
                "timestamp_ms": str(time.time() * 1000),
            },
        )
        assert res1["status"] == "success", f"Expected success, got {res1}"
        assert mock_client.xack.called, "XACK was not called on successful message!"
        assert mock_client.xadd.called, "Output stream xadd was not called!"

        # Test poison message routing to DLQ
        mock_client.reset_mock()
        res2 = consumer.process_message(
            "1002-0",
            {"query_id": "q2", "poison": "true"},
        )
        assert res2["status"] == "dlq_routed", f"Expected dlq_routed, got {res2}"
        assert mock_client.xack.called, "XACK was not called to acknowledge DLQ-routed message!"

        print(
            json.dumps(
                {"test_result": "PASS", "valid_message": res1, "dlq_message": res2}, indent=2
            )
        )
        return

    consumer = LegalStreamConsumer()
    consumer.run_poll_loop(max_messages=args.max_messages)


if __name__ == "__main__":
    main()
