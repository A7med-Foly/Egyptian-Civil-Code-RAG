"""Automated Canary SLA Health Guard and Watcher.

Continuously monitors canary candidate health, p95 latency, and error rate.
Triggers automatic rollback via serving/rollback.sh if SLAs are violated
for two consecutive evaluation intervals.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("serving.canary_watcher")


class CanaryWatcher:
    """Monitors canary candidate SLAs and invokes automated rollback upon failure."""

    def __init__(
        self,
        canary_url: str = "http://localhost:8000",
        p95_sla_ms: float = 1200.0,
        max_error_rate: float = 0.05,
        poll_interval_sec: float = 2.0,
        rollback_script: str = "./serving/rollback.sh",
    ):
        self.canary_url = canary_url.rstrip("/")
        self.p95_sla_ms = p95_sla_ms
        self.max_error_rate = max_error_rate
        self.poll_interval_sec = poll_interval_sec
        self.rollback_script = rollback_script
        self.consecutive_violations = 0
        self.max_allowed_consecutive_violations = 2

    def sample_canary_health(
        self, sample_size: int = 10, simulate_breach: bool = False
    ) -> tuple[float, float]:
        """Collect sample queries to compute p95 latency and error rate."""
        if simulate_breach:
            # Simulate a slow candidate or error-prone model
            latencies = [1650.0 + (i * 20) for i in range(sample_size)]
            error_rate = 0.15
            p95 = float(np.percentile(latencies, 95))
            return p95, error_rate

        latencies = []
        errors = 0
        probe_query = {"question": "ما هي شروط أهلية التعاقد؟", "top_k": 3}

        for _ in range(sample_size):
            try:
                t0 = time.perf_counter()
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(f"{self.canary_url}/ask", json=probe_query)
                    dur_ms = (time.perf_counter() - t0) * 1000
                    if resp.status_code == 200:
                        latencies.append(dur_ms)
                    else:
                        errors += 1
            except Exception:  # noqa: BLE001
                errors += 1

        p95 = float(np.percentile(latencies, 95)) if latencies else 9999.0
        error_rate = errors / sample_size
        return p95, error_rate

    def evaluate_cycle(self, cycle_num: int, simulate_breach: bool = False) -> bool:
        """Run one evaluation cycle. Returns True if rollback was triggered."""
        p95, error_rate = self.sample_canary_health(sample_size=10, simulate_breach=simulate_breach)
        sla_breached = (p95 > self.p95_sla_ms) or (error_rate > self.max_error_rate)

        logger.info(
            f"Cycle {cycle_num}: Canary p95={p95:.1f}ms (SLA: <{self.p95_sla_ms}ms), "
            f"Error Rate={error_rate * 100:.1f}% (SLA: <{self.max_error_rate * 100:.1f}%)"
        )

        if sla_breached:
            self.consecutive_violations += 1
            logger.warning(
                f"SLA VIOLATION #{self.consecutive_violations} detected! "
                f"(Threshold is {self.max_allowed_consecutive_violations} consecutive violations)"
            )
            if self.consecutive_violations >= self.max_allowed_consecutive_violations:
                logger.error(
                    "CRITICAL: Canary exceeded SLA tolerance for 2 consecutive cycles. "
                    "TRIGGERING AUTOMATIC ROLLBACK!"
                )
                self.trigger_rollback()
                return True
        else:
            self.consecutive_violations = 0
            logger.info("Canary health check PASSED within SLA thresholds.")

        return False

    def trigger_rollback(self):
        """Invoke rollback script and log execution."""
        logger.info(f"Executing rollback script: {self.rollback_script}")
        res = subprocess.run([self.rollback_script], check=False, capture_output=True, text=True)
        logger.info("Rollback script output:\n" + res.stdout)
        if res.returncode == 0:
            logger.info("AUTOMATED ROLLBACK EXECUTED SUCCESSFULLY.")
        else:
            logger.error(f"Rollback script failed with code {res.returncode}: {res.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Canary SLA Watcher.")
    parser.add_argument(
        "--simulate-breach", action="store_true", help="Simulate slow model violating SLA"
    )
    parser.add_argument("--cycles", type=int, default=3, help="Number of monitoring cycles to run")
    args = parser.parse_args()

    watcher = CanaryWatcher()
    for c in range(1, args.cycles + 1):
        triggered = watcher.evaluate_cycle(cycle_num=c, simulate_breach=args.simulate_breach)
        if triggered:
            print("Automatic guard stopped watcher following successful rollback.")
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
