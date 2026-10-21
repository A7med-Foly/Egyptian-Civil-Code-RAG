"""Evidently AI Drift Monitoring Suite and Automated Retraining Integration.

Runs statistical data and embedding drift tests over inference query logs,
persists drift metrics into PostgreSQL, and executes closed-loop retraining
triggers via Apache Airflow when distribution drift is detected.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("legal_rag.monitoring.evidently")

# Attempt imports for Evidently (supports both legacy 0.4+ and 0.7+ modular structures)
try:
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        EVIDENTLY_AVAILABLE = True
    except ImportError:
        Report = None  # type: ignore
        DataDriftPreset = None  # type: ignore
        EVIDENTLY_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import Json

    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore
    Json = None  # type: ignore
    PSYCOPG2_AVAILABLE = False


DEFAULT_DB_URL = os.getenv(
    "DRIFT_DATABASE_URL",
    "postgresql://mlflow:mlflowpassword@localhost:5433/mlflow",
)


def generate_reference_dataset(n_samples: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generate canonical baseline feature distribution of Egyptian Civil Code queries."""
    rng = np.random.default_rng(seed)

    query_length = rng.normal(loc=65.0, scale=12.0, size=n_samples).clip(min=15, max=150)
    word_count = (query_length / 6.5).round().astype(int).clip(min=3)
    has_legal_terms = rng.choice([1, 0], size=n_samples, p=[0.95, 0.05])
    retrieval_score = rng.normal(loc=0.82, scale=0.06, size=n_samples).clip(min=0.4, max=0.99)
    latency_ms = rng.normal(loc=350.0, scale=60.0, size=n_samples).clip(min=100, max=1200)
    num_citations = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.25, 0.50, 0.20, 0.05])

    return pd.DataFrame(
        {
            "query_length": query_length,
            "word_count": word_count,
            "has_legal_terms": has_legal_terms,
            "retrieval_score": retrieval_score,
            "latency_ms": latency_ms,
            "num_citations": num_citations,
        }
    )


def generate_current_dataset(
    n_samples: int = 100,
    drift: bool = False,
    drift_type: str = "data",
    seed: int = 101,
) -> pd.DataFrame:
    """Generate production query features for verification (either control or shifted)."""
    rng = np.random.default_rng(seed)

    if not drift:
        # Control distribution matching reference
        query_length = rng.normal(loc=65.5, scale=11.5, size=n_samples).clip(min=15, max=150)
        word_count = (query_length / 6.5).round().astype(int).clip(min=3)
        has_legal_terms = rng.choice([1, 0], size=n_samples, p=[0.94, 0.06])
        retrieval_score = rng.normal(loc=0.81, scale=0.06, size=n_samples).clip(min=0.4, max=0.99)
        latency_ms = rng.normal(loc=355.0, scale=58.0, size=n_samples).clip(min=100, max=1200)
        num_citations = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.26, 0.49, 0.20, 0.05])
    else:
        # Shifted distribution
        if drift_type == "data":
            # Significant vocabulary and length inflation (e.g. conversational prompt drift)
            query_length = rng.normal(loc=160.0, scale=35.0, size=n_samples).clip(min=80, max=300)
            word_count = (query_length / 5.0).round().astype(int).clip(min=15)
            has_legal_terms = rng.choice([1, 0], size=n_samples, p=[0.40, 0.60])
            retrieval_score = rng.normal(loc=0.55, scale=0.12, size=n_samples).clip(
                min=0.1, max=0.85
            )
            latency_ms = rng.normal(loc=750.0, scale=150.0, size=n_samples).clip(min=300, max=2500)
            num_citations = rng.choice([0, 1, 2], size=n_samples, p=[0.50, 0.35, 0.15])
        else:
            # Concept/score degradation drift
            query_length = rng.normal(loc=70.0, scale=15.0, size=n_samples).clip(min=20, max=160)
            word_count = (query_length / 6.5).round().astype(int).clip(min=3)
            has_legal_terms = rng.choice([1, 0], size=n_samples, p=[0.70, 0.30])
            retrieval_score = rng.normal(loc=0.42, scale=0.08, size=n_samples).clip(
                min=0.1, max=0.65
            )
            latency_ms = rng.normal(loc=520.0, scale=110.0, size=n_samples).clip(min=150, max=1800)
            num_citations = rng.choice([0, 1], size=n_samples, p=[0.70, 0.30])

    return pd.DataFrame(
        {
            "query_length": query_length,
            "word_count": word_count,
            "has_legal_terms": has_legal_terms,
            "retrieval_score": retrieval_score,
            "latency_ms": latency_ms,
            "num_citations": num_citations,
        }
    )


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run Evidently data drift report over reference and current feature distributions."""
    if not EVIDENTLY_AVAILABLE:
        raise RuntimeError("Evidently is not available in the active environment.")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report_dict = report.as_dict()

    # Parse DatasetDriftMetric and DataDriftTable
    dataset_metric = report_dict["metrics"][0]["result"]
    table_metric = report_dict["metrics"][1]["result"] if len(report_dict["metrics"]) > 1 else {}

    drift_by_columns: dict[str, Any] = {}
    raw_col_drift = table_metric.get("drift_by_columns", {})
    for col_name, stats in raw_col_drift.items():
        drift_by_columns[col_name] = {
            "drift_detected": stats.get("drift_detected", False),
            "drift_score": stats.get("drift_score", 0.0),
            "stat_test": stats.get("stat_test", "ks" if "ks" in str(stats) else "unknown"),
        }

    return {
        "dataset_drift": bool(dataset_metric["dataset_drift"]),
        "share_drifted_features": float(dataset_metric["share_of_drifted_columns"]),
        "number_drifted_features": int(dataset_metric["number_of_drifted_columns"]),
        "total_features": int(dataset_metric["number_of_columns"]),
        "drift_by_column": drift_by_columns,
        "report_summary": {
            "timestamp": datetime.now(UTC).isoformat(),
            "reference_rows": len(reference_df),
            "current_rows": len(current_df),
        },
    }


def init_drift_db(database_url: str | None = None) -> None:
    """Create the PostgreSQL drift_metrics table if it does not already exist."""
    if not PSYCOPG2_AVAILABLE:
        logger.warning("psycopg2 is not installed; skipping PostgreSQL table initialization.")
        return

    db_url = database_url or DEFAULT_DB_URL
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS drift_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            test_suite_name VARCHAR(100) NOT NULL,
            dataset_drift BOOLEAN NOT NULL,
            share_drifted_features DOUBLE PRECISION NOT NULL,
            number_drifted_features INTEGER NOT NULL,
            total_features INTEGER NOT NULL,
            drift_by_column JSONB NOT NULL,
            report_summary JSONB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_drift_metrics_ts ON drift_metrics(timestamp DESC);
        """
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("drift_metrics table initialized successfully in PostgreSQL.")
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not initialize drift_metrics table in PostgreSQL (%s).", e)


def save_drift_report(
    drift_result: dict[str, Any],
    database_url: str | None = None,
    suite_name: str = "daily_rag_query_drift",
) -> int | None:
    """Insert drift report into PostgreSQL drift_metrics table."""
    if not PSYCOPG2_AVAILABLE:
        return None

    db_url = database_url or DEFAULT_DB_URL
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        insert_sql = """
        INSERT INTO drift_metrics (
            test_suite_name,
            dataset_drift,
            share_drifted_features,
            number_drifted_features,
            total_features,
            drift_by_column,
            report_summary
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        cur.execute(
            insert_sql,
            (
                suite_name,
                drift_result["dataset_drift"],
                drift_result["share_drifted_features"],
                drift_result["number_drifted_features"],
                drift_result["total_features"],
                Json(drift_result["drift_by_column"]),
                Json(drift_result["report_summary"]),
            ),
        )
        inserted_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Persisted drift report #%s to PostgreSQL successfully.", inserted_id)
        return inserted_id
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to persist drift metrics to PostgreSQL (%s).", e)
        return None


def trigger_retraining_on_drift(
    drift_result: dict[str, Any],
    dag_id: str = "legal_rag_continuous_training",
) -> bool:
    """Trigger continuous retraining in Airflow if distribution drift exceeds threshold."""
    if not drift_result.get("dataset_drift", False):
        logger.info(
            "No distribution drift detected (share=%.2f). Skipping retraining trigger.",
            drift_result.get("share_drifted_features", 0.0),
        )
        return False

    logger.warning(
        "DISTRIBUTION DRIFT DETECTED (drifted %d/%d features). Triggering Airflow DAG %s...",
        drift_result["number_drifted_features"],
        drift_result["total_features"],
        dag_id,
    )

    # Trigger via Docker CLI first
    cmd = [
        "docker",
        "exec",
        "legal-rag-airflow-scheduler",
        "airflow",
        "dags",
        "trigger",
        dag_id,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Successfully triggered Airflow DAG %s: %s", dag_id, res.stdout.strip())
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Docker CLI trigger failed (%s). Retraining trigger completed with notification.", e
        )
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Evidently Drift Monitoring Test ===")
    init_drift_db()

    ref = generate_reference_dataset(150)
    print(f"Generated reference baseline ({len(ref)} samples)")

    # 1. Control Run (should NOT drift)
    ctrl = generate_current_dataset(100, drift=False)
    ctrl_res = run_drift_analysis(ref, ctrl)
    print(
        f"Control Run: dataset_drift={ctrl_res['dataset_drift']} (drifted {ctrl_res['number_drifted_features']}/{ctrl_res['total_features']})"
    )
    assert not ctrl_res["dataset_drift"], "Control distribution falsely flagged as drifted!"

    # 2. Shifted Run (MUST drift)
    shifted = generate_current_dataset(100, drift=True, drift_type="data")
    shift_res = run_drift_analysis(ref, shifted)
    print(
        f"Shifted Run: dataset_drift={shift_res['dataset_drift']} (drifted {shift_res['number_drifted_features']}/{shift_res['total_features']})"
    )
    assert shift_res["dataset_drift"], "Shifted distribution failed to trigger drift!"

    # 3. Save to PostgreSQL
    record_id = save_drift_report(shift_res)
    print(f"Saved drift record to PostgreSQL with ID: {record_id}")
