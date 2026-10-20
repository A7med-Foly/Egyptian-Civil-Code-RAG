"""MLflow Model Registry integration and promotion lifecycle management."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

try:
    from legal_rag.config import settings

    DEFAULT_TRACKING_URI = settings.MLFLOW_TRACKING_URI
except Exception:  # noqa: BLE001
    DEFAULT_TRACKING_URI = "http://localhost:5000"

logger = logging.getLogger("legal_rag.registry")


class LegalModelRegistry:
    """Manages model versioning, registry, and stage promotion in MLflow."""

    def __init__(
        self,
        model_name: str = "EgyptianLegalRAG",
        tracking_uri: str | None = None,
    ):
        self.model_name = model_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_model_from_run(
        self,
        run_id: str,
        artifact_path: str = "evaluation",
        tags: dict[str, Any] | None = None,
    ) -> str:
        """Register a model artifact from an MLflow run into the Model Registry."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info(f"Registering model from {model_uri} as '{self.model_name}'...")

        # Ensure registered model exists
        try:
            self.client.create_registered_model(self.model_name)
        except Exception:  # noqa: BLE001, S110
            pass  # Already exists

        model_version = self.client.create_model_version(
            name=self.model_name,
            source=model_uri,
            run_id=run_id,
            tags=tags or {},
        )
        version_num = str(model_version.version)
        logger.info(f"Created model version {version_num} for '{self.model_name}'.")
        return version_num

    def transition_stage(
        self,
        version: str,
        stage: str,
        archive_existing_versions: bool = True,
    ) -> None:
        """Transition a model version to 'Staging', 'Production', or 'Archived'."""
        logger.info(f"Transitioning version {version} of '{self.model_name}' to {stage}...")
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

    def get_production_model(self) -> dict[str, Any] | None:
        """Fetch the current Production model version and its metrics."""
        try:
            latest_versions = self.client.get_latest_versions(
                self.model_name, stages=["Production"]
            )
            if not latest_versions:
                return None
            prod_version = latest_versions[0]
            run = self.client.get_run(prod_version.run_id)
            return {
                "version": prod_version.version,
                "run_id": prod_version.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not retrieve production model: {e}")
            return None

    def promote_if_better(
        self,
        candidate_run_id: str,
        metric: str = "faithfulness_score",
        higher_is_better: bool = True,
        min_improvement: float = 0.0,
    ) -> bool:
        """Promote candidate run to Production if its metric exceeds current Production."""
        candidate_run = self.client.get_run(candidate_run_id)
        candidate_metric = candidate_run.data.metrics.get(metric)

        if candidate_metric is None:
            raise ValueError(f"Metric '{metric}' not found in candidate run {candidate_run_id}")

        prod = self.get_production_model()
        if prod is None:
            logger.info(
                f"No existing Production model found. Promoting candidate {candidate_run_id} directly."
            )
            version = self.register_model_from_run(candidate_run_id)
            self.transition_stage(version, "Production")
            return True

        prod_metric = prod["metrics"].get(metric, 0.0)
        logger.info(
            f"Comparing candidate ({metric}={candidate_metric:.4f}) vs "
            f"Production v{prod['version']} ({metric}={prod_metric:.4f})"
        )

        is_better = (
            (candidate_metric >= prod_metric + min_improvement)
            if higher_is_better
            else (candidate_metric <= prod_metric - min_improvement)
        )

        if is_better:
            logger.info("Candidate run is superior. Registering and promoting to Production.")
            version = self.register_model_from_run(candidate_run_id)
            self.transition_stage(version, "Production")
            return True
        else:
            logger.info(
                f"Candidate did not beat Production. Registering to Staging (retaining Production v{prod['version']})."
            )
            version = self.register_model_from_run(candidate_run_id)
            self.transition_stage(version, "Staging")
            return False


def register_and_promote_model(
    run_id: str,
    model_name: str = "EgyptianLegalRAG",
    target_stage: str = "Production",
    tracking_uri: str | None = None,
) -> str:
    """Convenience helper to register a run and transition to target stage."""
    registry = LegalModelRegistry(model_name=model_name, tracking_uri=tracking_uri)
    version = registry.register_model_from_run(run_id=run_id)
    registry.transition_stage(version=version, stage=target_stage)
    return version
