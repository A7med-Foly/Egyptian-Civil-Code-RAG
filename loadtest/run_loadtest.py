"""Locust headless benchmark runner with automated metrics parsing and HTML report export."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("loadtest.runner")


def run_locust_test(
    host: str = "http://localhost:8000",
    users: int = 50,
    spawn_rate: int = 5,
    run_time: str = "30s",
    report_name: str = "locust_fastapi",
) -> dict:
    """Run headless locust test and parse CSV metrics."""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_prefix = reports_dir / report_name
    html_report = reports_dir / f"{report_name}.html"

    cmd = [
        "uv",
        "run",
        "locust",
        "-f",
        "loadtest/locustfile.py",
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "--run-time",
        run_time,
        "--host",
        host,
        "--html",
        str(html_report),
        "--csv",
        str(csv_prefix),
    ]

    logger.info(f"Running load test against {host} with {users} concurrent users for {run_time}...")
    start_time = time.time()
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    duration = time.time() - start_time

    logger.info(f"Locust run completed in {duration:.1f}s. Exit code: {res.returncode}")

    # Parse generated CSV stats
    stats_file = reports_dir / f"{report_name}_stats.csv"
    metrics = {
        "host": host,
        "concurrent_users": users,
        "run_time": run_time,
        "html_report": str(html_report),
        "requests_total": 0,
        "failures_total": 0,
        "rps": 0.0,
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "p99_ms": 0.0,
    }

    if stats_file.exists():
        with open(stats_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Name") == "Aggregated":
                    metrics["requests_total"] = int(row.get("Request Count", 0))
                    metrics["failures_total"] = int(row.get("Failure Count", 0))
                    metrics["rps"] = float(row.get("Requests/s", 0.0))
                    metrics["p50_ms"] = float(row.get("50%", 0.0))
                    metrics["p95_ms"] = float(row.get("95%", 0.0))
                    metrics["p99_ms"] = float(row.get("99%", 0.0))

    logger.info(
        f"Summary: RPS={metrics['rps']:.2f}, p50={metrics['p50_ms']:.1f}ms, "
        f"p95={metrics['p95_ms']:.1f}ms, p99={metrics['p99_ms']:.1f}ms, "
        f"Failures={metrics['failures_total']}/{metrics['requests_total']}"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Locust load test.")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--spawn-rate", type=int, default=5)
    parser.add_argument("--run-time", default="30s")
    parser.add_argument("--report-name", default="locust_fastapi")
    args = parser.parse_args()

    metrics = run_locust_test(
        host=args.host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        run_time=args.run_time,
        report_name=args.report_name,
    )
    print(json.dumps(metrics, indent=2))
