"""Unit tests for configuration and structured logging."""

import logging

from legal_rag.config import Settings
from legal_rag.logging_conf import get_correlation_id, set_correlation_id, setup_logging


def test_settings_defaults():
    s = Settings()
    assert s.API_PORT == 8000
    assert s.API_VERSION == "0.1.0"
    assert "Egyptian-Civil-Code.pdf" in str(s.RAW_DATA_PATH)
    assert s.CHUNKING_STRATEGY == "article"


def test_correlation_id_context():
    test_id = "test-uuid-12345"
    set_correlation_id(test_id)
    assert get_correlation_id() == test_id


def test_setup_logging_runs():
    setup_logging("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    # restore INFO
    setup_logging("INFO")
