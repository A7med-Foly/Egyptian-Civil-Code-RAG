"""Unit tests for MLflow Model Registry and promotion lifecycle."""

import mlflow
import pytest

from legal_rag.registry import LegalModelRegistry


@pytest.fixture
def temp_mlflow_env(tmp_path):
    """Set up isolated MLflow SQLite tracking environment."""
    db_path = tmp_path / "test_mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def test_registry_init(temp_mlflow_env):
    """Verify registry client initializes with custom tracking URI."""
    registry = LegalModelRegistry(
        model_name="TestModel",
        tracking_uri=temp_mlflow_env,
    )
    assert registry.model_name == "TestModel"
    assert registry.tracking_uri == temp_mlflow_env


def test_registry_promotion_lifecycle(temp_mlflow_env, tmp_path):
    """Test full model registration, initial promotion, and comparison logic."""
    registry = LegalModelRegistry(
        model_name="TestLegalRAG",
        tracking_uri=temp_mlflow_env,
    )

    # 1. Create Run 1
    with mlflow.start_run(run_name="Run-1") as run1:
        run1_id = run1.info.run_id
        mlflow.log_metric("faithfulness_score", 0.78)
        art_file = tmp_path / "eval1.json"
        art_file.write_text('{"score": 0.78}', encoding="utf-8")
        mlflow.log_artifact(str(art_file), artifact_path="evaluation")

    # 2. Promote Run 1 to Production (initial promotion)
    promoted1 = registry.promote_if_better(
        candidate_run_id=run1_id,
        metric="faithfulness_score",
    )
    assert promoted1 is True
    prod_model = registry.get_production_model()
    assert prod_model is not None
    assert prod_model["run_id"] == run1_id

    # 3. Create Run 2 with lower score (should not promote, stages instead)
    with mlflow.start_run(run_name="Run-2") as run2:
        run2_id = run2.info.run_id
        mlflow.log_metric("faithfulness_score", 0.72)
        art_file2 = tmp_path / "eval2.json"
        art_file2.write_text('{"score": 0.72}', encoding="utf-8")
        mlflow.log_artifact(str(art_file2), artifact_path="evaluation")

    promoted2 = registry.promote_if_better(
        candidate_run_id=run2_id,
        metric="faithfulness_score",
    )
    assert promoted2 is False
    prod_after = registry.get_production_model()
    assert prod_after["run_id"] == run1_id

    # 4. Create Run 3 with higher score (should beat Run 1 and promote to Production)
    with mlflow.start_run(run_name="Run-3") as run3:
        run3_id = run3.info.run_id
        mlflow.log_metric("faithfulness_score", 0.89)
        art_file3 = tmp_path / "eval3.json"
        art_file3.write_text('{"score": 0.89}', encoding="utf-8")
        mlflow.log_artifact(str(art_file3), artifact_path="evaluation")

    promoted3 = registry.promote_if_better(
        candidate_run_id=run3_id,
        metric="faithfulness_score",
    )
    assert promoted3 is True
    prod_final = registry.get_production_model()
    assert prod_final["run_id"] == run3_id
