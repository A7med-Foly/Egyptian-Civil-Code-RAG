"""Statistical drift detection algorithms for tabular and embedding distributions.

Implements all six core methods required by MLOps Module 5:
1. Kolmogorov-Smirnov (KS) test (continuous distributions)
2. Chi-Square goodness-of-fit test (categorical distributions)
3. Wasserstein Distance / Earth Mover's Distance (magnitude of shift)
4. Kullback-Leibler (KL) & Jensen-Shannon (JS) divergence
5. Maximum Mean Discrepancy (MMD) & Binary Domain Classifier (high-dim embeddings)
6. Streaming Drift Detection: ADWIN & DDM (concept drift & error streams)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger("legal_rag.monitoring.drift_stats")


@dataclass
class DriftResult:
    """Standardized result returned by drift detection algorithms."""

    method: str
    drift_detected: bool
    score: float
    threshold: float
    p_value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# 1. Kolmogorov-Smirnov (KS) Test for Continuous Features
# =====================================================================
def ks_test(
    reference: np.ndarray | list[float],
    current: np.ndarray | list[float],
    alpha: float = 0.05,
) -> DriftResult:
    """Perform two-sample Kolmogorov-Smirnov test to detect distribution shift.

    Null hypothesis (H0): reference and current samples come from the same distribution.
    Fires if p-value < alpha.
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)

    stat, p_val = stats.ks_2samp(ref, curr)
    drift = bool(p_val < alpha)

    return DriftResult(
        method="KS-Test",
        drift_detected=drift,
        score=float(stat),
        threshold=alpha,
        p_value=float(p_val),
        details={
            "statistic": float(stat),
            "sample_size_ref": len(ref),
            "sample_size_curr": len(curr),
        },
    )


# =====================================================================
# 2. Chi-Square Goodness-of-Fit Test for Categorical Features
# =====================================================================
def chi2_test(
    reference_counts: dict[str, int],
    current_counts: dict[str, int],
    alpha: float = 0.05,
) -> DriftResult:
    """Perform Chi-square test on categorical distributions.

    Compares observed current frequencies against expected reference frequencies.
    """
    categories = sorted(set(reference_counts.keys()) | set(current_counts.keys()))
    total_ref = sum(reference_counts.values())
    total_curr = sum(current_counts.values())

    if total_ref == 0 or total_curr == 0:
        raise ValueError("Frequency counts must be non-empty.")

    # Expected frequencies based on reference distribution proportions
    observed = np.array([current_counts.get(c, 0) for c in categories], dtype=float)
    expected_prob = np.array(
        [reference_counts.get(c, 0) / total_ref for c in categories], dtype=float
    )

    # Avoid zero expected counts by smoothing
    eps = 1e-5
    expected_prob = (expected_prob + eps) / (expected_prob + eps).sum()
    expected = expected_prob * total_curr

    chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)
    drift = bool(p_val < alpha)

    return DriftResult(
        method="Chi-Square",
        drift_detected=drift,
        score=float(chi2_stat),
        threshold=alpha,
        p_value=float(p_val),
        details={
            "categories": categories,
            "observed": observed.tolist(),
            "expected": expected.tolist(),
        },
    )


# =====================================================================
# 3. Wasserstein Distance (Earth Mover's Distance)
# =====================================================================
def wasserstein_test(
    reference: np.ndarray | list[float],
    current: np.ndarray | list[float],
    threshold: float = 0.15,
) -> DriftResult:
    """Compute Wasserstein-1 distance to measure physical shift magnitude.

    Unlike p-value tests that fire on large sample sizes, Wasserstein scales directly
    with the actual geometric distance between distributions.
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)

    # Normalize by pooled standard deviation for scale invariance if scale > 0
    pooled_std = np.std(np.concatenate([ref, curr]))
    dist = stats.wasserstein_distance(ref, curr)
    norm_dist = dist / pooled_std if pooled_std > 1e-8 else dist

    drift = bool(norm_dist >= threshold)

    return DriftResult(
        method="Wasserstein",
        drift_detected=drift,
        score=float(norm_dist),
        threshold=threshold,
        p_value=None,
        details={"raw_distance": float(dist), "normalized_distance": float(norm_dist)},
    )


# =====================================================================
# 4. Kullback-Leibler (KL) & Jensen-Shannon (JS) Divergence
# =====================================================================
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Compute Kullback-Leibler divergence D_KL(P || Q).

    Note: KL divergence is asymmetric: D_KL(P || Q) != D_KL(Q || P).
    """
    p_norm = np.asarray(p, dtype=float) + eps
    p_norm /= p_norm.sum()

    q_norm = np.asarray(q, dtype=float) + eps
    q_norm /= q_norm.sum()

    return float(np.sum(p_norm * np.log(p_norm / q_norm)))


def js_divergence_test(
    p: np.ndarray,
    q: np.ndarray,
    threshold: float = 0.10,
    eps: float = 1e-9,
) -> DriftResult:
    """Compute Jensen-Shannon divergence JS(P, Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M).

    JS divergence is symmetric, smooth, and bounded in [0, 1] (when using log base 2).
    """
    p_arr = np.asarray(p, dtype=float) + eps
    p_arr /= p_arr.sum()

    q_arr = np.asarray(q, dtype=float) + eps
    q_arr /= q_arr.sum()

    m = 0.5 * (p_arr + q_arr)
    kl_pm = np.sum(p_arr * np.log2(p_arr / m))
    kl_qm = np.sum(q_arr * np.log2(q_arr / m))

    js_score = float(0.5 * kl_pm + 0.5 * kl_qm)
    drift = bool(js_score >= threshold)

    return DriftResult(
        method="JS-Divergence",
        drift_detected=drift,
        score=js_score,
        threshold=threshold,
        p_value=None,
        details={"is_symmetric": True, "bounded_range": "[0, 1]"},
    )


# =====================================================================
# 5. MMD & Domain Classifier for High-Dimensional Embeddings
# =====================================================================
def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Compute Maximum Mean Discrepancy (MMD) with an RBF kernel.

    MMD tests whether two sets of multidimensional embeddings are drawn
    from the same distribution without succumbing to multiple-testing collapse.
    """
    m, n = len(X), len(Y)
    if gamma is None:
        # Median heuristic for gamma
        combined = np.vstack([X[:100], Y[:100]])
        dists = np.linalg.norm(combined[:, None] - combined[None, :], axis=-1)
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        gamma = 1.0 / (2 * (med**2) + 1e-8)

    def rbf(A, B):
        diff = A[:, None, :] - B[None, :, :]
        return np.exp(-gamma * np.sum(diff**2, axis=-1))

    K_xx = rbf(X, X)
    K_yy = rbf(Y, Y)
    K_xy = rbf(X, Y)

    mmd_sq = (np.sum(K_xx) / (m * m)) - (2 * np.sum(K_xy) / (m * n)) + (np.sum(K_yy) / (n * n))
    return float(np.sqrt(max(0.0, mmd_sq)))


def domain_classifier_test(
    reference_embeddings: np.ndarray,
    current_embeddings: np.ndarray,
    auc_threshold: float = 0.65,
    random_state: int = 42,
) -> DriftResult:
    """Train a binary domain classifier to distinguish Reference (0) vs Current (1).

    - If AUC ≈ 0.5 (near random guessing), reference and current are indistinguishable (No Drift).
    - If AUC rises significantly (> auc_threshold), distribution has drifted.
    """
    X_ref = np.asarray(reference_embeddings)
    X_curr = np.asarray(current_embeddings)

    y_ref = np.zeros(len(X_ref))
    y_curr = np.ones(len(X_curr))

    X = np.vstack([X_ref, X_curr])
    y = np.concatenate([y_ref, y_curr])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    clf = LogisticRegression(max_iter=300, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_pred_proba))

    mmd_val = compute_mmd(X_ref, X_curr)
    drift = bool(auc >= auc_threshold)

    # Top feature coefficients indicating which dimensions drifted most
    top_feature_indices = np.argsort(np.abs(clf.coef_[0]))[-5:][::-1].tolist()

    return DriftResult(
        method="MMD-Domain-Classifier",
        drift_detected=drift,
        score=auc,
        threshold=auc_threshold,
        p_value=None,
        details={
            "auc": auc,
            "mmd": mmd_val,
            "top_drift_dimensions": top_feature_indices,
            "interpretation": (
                "Drift detected" if drift else "Control distribution matches reference (AUC ≈ 0.5)"
            ),
        },
    )


# =====================================================================
# 6. Streaming Drift Detection: ADWIN & DDM
# =====================================================================
class DDM:
    """Drift Detection Method (Gama et al.) for sequential binary error streams.

    Maintains current error rate p_i and standard deviation s_i = sqrt(p_i * (1 - p_i) / i).
    Triggers:
    - Warning: p_i + s_i >= p_min + 2.0 * s_min
    - Drift:   p_i + s_i >= p_min + 3.0 * s_min
    """

    def __init__(
        self, min_instances: int = 30, warning_level: float = 2.0, drift_level: float = 3.0
    ):
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()

    def reset(self) -> None:
        """Reset internal tracking statistics."""
        self.n = 0
        self.error_count = 0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.in_warning = False
        self.in_drift = False

    def update(self, error: int | bool) -> dict[str, Any]:
        """Update detector with latest sample (1 for error, 0 for success)."""
        self.n += 1
        if bool(error):
            self.error_count += 1

        p = self.error_count / self.n
        s = np.sqrt(p * (1.0 - p) / self.n) if self.n > 1 else 0.0

        if self.n >= self.min_instances:
            if p + s < self.p_min + self.s_min:
                self.p_min = p
                self.s_min = s

            effective_s_min = max(self.s_min, 0.01)
            if (p + s >= self.p_min + self.drift_level * effective_s_min) and (p > self.p_min):
                self.in_drift = True
                self.in_warning = False
            elif (p + s >= self.p_min + self.warning_level * effective_s_min) and (p > self.p_min):
                self.in_warning = True
                self.in_drift = False
            else:
                self.in_warning = False
                self.in_drift = False

        return {
            "n": self.n,
            "error_rate": float(p),
            "in_warning": self.in_warning,
            "in_drift": self.in_drift,
        }


class ADWIN:
    """Adaptive Windowing algorithm for streaming concept drift detection.

    Grows window W while data is stationary; shrinks window when sub-window means differ
    statistically using the Hoeffding bound.
    """

    def __init__(self, delta: float = 0.002, max_window: int = 1000):
        self.delta = delta
        self.max_window = max_window
        self.window: list[float] = []
        self.drift_detected = False

    def reset(self) -> None:
        self.window.clear()
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Add value to window and test for significant split."""
        self.window.append(float(value))
        if len(self.window) > self.max_window:
            self.window.pop(0)

        self.drift_detected = False
        n = len(self.window)

        if n >= 20:
            # Check potential cut points
            w = np.array(self.window)
            for split in range(10, n - 10, 5):
                w0 = w[:split]
                w1 = w[split:]
                n0, n1 = len(w0), len(w1)
                m0, m1 = np.mean(w0), np.mean(w1)

                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                epsilon = np.sqrt((1.0 / (2 * m)) * np.log(4.0 * n / self.delta))

                if abs(m0 - m1) > epsilon:
                    self.drift_detected = True
                    # Drop older segment upon drift
                    self.window = self.window[split:]
                    break

        return self.drift_detected
