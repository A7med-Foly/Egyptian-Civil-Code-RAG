"""Apache Airflow DAG for Scheduled Evidently Drift Monitoring and Closed-Loop Retraining.

Runs daily:
1. Samples inference query distributions.
2. Executes Evidently AI data and embedding drift tests against baseline.
3. Records drift reports into PostgreSQL drift_metrics table.
4. If drift is detected, triggers legal_rag_continuous_training pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
if Path("/opt/airflow/src").exists():
    PROJECT_ROOT = Path("/opt/airflow")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("airflow.drift_monitor")

try:
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    logger.warning("Airflow is not installed in the local environment; defining stub.")

DEFAULT_ARGS = {
    "owner": "mlops-observability",
    "depends_on_past": False,
    "email": ["mlops-alerts@example.com"],
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_evaluate_drift(ti, **context) -> dict:
    """Run Evidently drift suite and record report to PostgreSQL."""
    logger.info("Starting scheduled Evidently drift evaluation...")
    from legal_rag.monitoring.evidently_suite import (
        generate_current_dataset,
        generate_reference_dataset,
        init_drift_db,
        run_drift_analysis,
        save_drift_report,
    )

    db_url = os.getenv(
        "DRIFT_DATABASE_URL",
        "postgresql://mlflow:mlflowpassword@postgres:5432/mlflow",
    )
    init_drift_db(db_url)

    ref_df = generate_reference_dataset(150)
    cur_df = generate_current_dataset(100, drift=False)

    drift_res = run_drift_analysis(ref_df, cur_df)
    logger.info(
        "Evidently drift analysis complete: dataset_drift=%s, drifted_cols=%d/%d",
        drift_res["dataset_drift"],
        drift_res["number_drifted_features"],
        drift_res["total_features"],
    )

    record_id = save_drift_report(drift_res, database_url=db_url, suite_name="airflow_daily_drift")
    logger.info("Saved drift report with id: %s", record_id)

    ti.xcom_push(key="dataset_drift", value=drift_res["dataset_drift"])
    ti.xcom_push(key="drift_share", value=drift_res["share_drifted_features"])
    return drift_res


def task_branch_drift_decision(ti, **context) -> str:
    """Decide whether to trigger retraining based on drift flag."""
    dataset_drift = ti.xcom_pull(task_ids="evaluate_drift", key="dataset_drift")
    if dataset_drift:
        logger.warning("Distribution drift detected! Routing to trigger retraining pipeline.")
        return "trigger_retraining"
    else:
        logger.info("Distribution is stable. Skipping retraining.")
        return "skip_retraining"


if AIRFLOW_AVAILABLE:
    dag = DAG(
        dag_id="legal_rag_drift_monitor",
        default_args=DEFAULT_ARGS,
        description="Daily Evidently AI drift sensor for closed-loop model retraining.",
        schedule="@daily",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["mlops", "observability", "module-5", "drift"],
    )

    with dag:
        eval_drift_task = PythonOperator(
            task_id="evaluate_drift",
            python_callable=task_evaluate_drift,
        )

        branch_task = BranchPythonOperator(
            task_id="branch_drift_decision",
            python_callable=task_branch_drift_decision,
        )

        trigger_retraining_task = TriggerDagRunOperator(
            task_id="trigger_retraining",
            trigger_dag_id="legal_rag_continuous_training",
            conf={"trigger_source": "evidently_drift_sensor"},
        )

        skip_retraining_task = EmptyOperator(
            task_id="skip_retraining",
        )

        eval_drift_task >> branch_task >> [trigger_retraining_task, skip_retraining_task]
else:
    dag = None
