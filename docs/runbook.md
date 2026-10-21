# Production On-Call Runbook: Arabic Legal RAG System

**Target System:** Arabic Legal Document Q&A RAG (`Egyptian Civil Code, 1948`)  
**On-Call Tier:** 24/7 Production MLOps Engineering  
**Escalation Window:** 15-minute response SLA for Critical Symptom Alerts  
**Last Verified:** October 2026 (`v0.5.0`)

---

## 1. Quickstart: 3am Immediate Triage

When your pager fires at 3am, execute this 4-step diagnostic protocol in order:

```bash
# 1. Check all Docker containers and exporter health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 2. Check Prometheus scrape targets (verify all UP)
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | grep -E '("job"|"health")'

# 3. Check firing alerts in Alertmanager
curl -s http://localhost:9093/api/v2/alerts | python3 -m json.tool

# 4. Probe API health endpoint directly
curl -i http://localhost:8000/health
```

---

## 2. Symptom vs. Cause Diagnostic Table

Always treat **Symptoms** (client-impacting SLA breaches) with immediate traffic mitigations before diagnosing underlying **Causes**.

| Alert Name | Severity | Type | Symptom (User Impact) | Primary Cause Candidates | Immediate Action (< 5 min) |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **`LegalRAGServiceDown`** | `critical` | Symptom | Blackbox probe returns non-200 or times out. Users receive 502/503. | Container crashed, OOM killed, port conflict, or network deadlock. | `docker restart legal-rag-v1 legal-rag-v2 legal-rag-nginx` |
| **`LegalRAGHighP95Latency`** | `critical` | Symptom | $p95$ request latency exceeds $2.0\text{s}$ over 5m window. Slow UX. | LLM provider throttling, unbatched CPU inference, vector search saturation. | Switch LLM provider or execute Canary Rollback (`./serving/rollback.sh`). |
| **`LegalRAGHighErrorRate`** | `critical` | Symptom | HTTP $5xx$ error rate exceeds $1.0\%$ over 5m window. | Bad model deployment, schema parsing bug, or upstream API timeout. | Execute instant rollback to stable release: `./serving/rollback.sh`. |
| **`LegalRAGDriftDetected`** | `warning` | Cause | Evidently AI flags $> 33\%$ drifted features ($P(X)$ or embedding shift). | Lawyers asking questions on new decrees, dialect shift, or non-statutory topics. | Verify drift in Grafana, trigger closed-loop retraining in Airflow. |
| **`LegalRAGContainerResourceExhaustion`** | `warning` | Cause | CPU $> 85\%$ or Memory $> 90\%$ sustained for 5 minutes. | High concurrent traffic burst, ONNX thread contention, memory leak. | Scale worker processes or restart degraded container. |
| **`LegalRAGPredictionVolumeDrop`** | `warning` | Cause | Ingestion traffic drops $> 50\%$ compared to same hour last week. | Ingress routing failure, upstream gateway outage, DNS resolution drop. | Check Nginx access logs and upstream client connectivity. |

---

## 3. Actionable Playbooks

### Playbook A: Instant Canary Rollback (< 1 second)
Use when a newly promoted candidate release (`v2`) exhibits elevated errors or latency spikes.

```bash
# Execute sub-second automated rollback to stable v1 (100% stable, 0% candidate)
./serving/rollback.sh

# Verify Nginx upstream routing is restored
docker exec legal-rag-nginx nginx -T | grep -A 5 "upstream legal_rag_backend"
```

---

### Playbook B: Closed-Loop Airflow Retraining Trigger
Use when `LegalRAGDriftDetected` fires or Evidently AI records distribution shifts in PostgreSQL.

```bash
# 1. Inspect recent drift records in PostgreSQL
uv run python -c "
import psycopg2
conn = psycopg2.connect('postgresql://mlflow:mlflowpassword@localhost:5433/mlflow')
cur = conn.cursor()
cur.execute('SELECT timestamp, share_drifted_features, number_drifted_features, drift_by_column FROM drift_metrics ORDER BY timestamp DESC LIMIT 3;')
for row in cur.fetchall():
    print(row)
conn.close()
"

# 2. Trigger the continuous training Airflow DAG
docker exec legal-rag-airflow-scheduler airflow dags trigger legal_rag_continuous_training -c '{"trigger_source": "oncall_manual_drift_remediation"}'

# 3. Monitor DAG execution state
docker exec legal-rag-airflow-scheduler airflow dags list-runs -d legal_rag_continuous_training -s
```

---

### Playbook C: Rate Limiting & Denial-of-Service Defense
Use when users report false `429 Too Many Requests` or malicious clients flood the API.

```bash
# 1. Check rate limiter Prometheus counter
curl -s http://localhost:9090/api/v1/query?query=sum(rate(legal_rag_rate_limit_hits_total[5m]))by(client_id,status)

# 2. Flush rate limit token buckets in Redis (resets client limits instantly)
docker exec legal-rag-redis redis-cli FLUSHDB
```

---

### Playbook D: Prometheus & Alertmanager Config Reload
Use when updating alert rules or recording rules without restarting containers.

```bash
# Reload Prometheus rules dynamically without container downtime
curl -X POST http://localhost:9090/-/reload

# Verify Alertmanager configuration syntax and reload
curl -X POST http://localhost:9093/-/reload
```

---

## 4. Top 10 Monitoring Anti-Patterns

1. **Alerting on Causes instead of Symptoms:** Never page the on-call engineer at 3am because CPU reached 80% if user latency is 20ms and errors are zero. Page on user-facing symptoms; ticket causes.
2. **Cardinality Bombing:** Never put high-cardinality values (user IP, query text, correlation UUID) in Prometheus metric labels. Use bounded categorical values (`model_version`, `status`, `stage`).
3. **The Multiprocess Scraping Trap:** Running multi-worker Uvicorn/Gunicorn without `PROMETHEUS_MULTIPROC_DIR` causes metrics to jump backward and drop increments. Always enforce `MultiProcessCollector`.
4. **Static Hardcoded Thresholds on Non-Stationary Signals:** Hardcoding static error count alerts (> 100 errors) fails during low traffic (3am) and causes storm alerts during high traffic (noon). Use rates and window ratios (`rate(...) > 0.01`).
5. **Missing Alert Inhibition:** If the entire container is down, Alertmanager must inhibit latency and error alerts for that container. Only page once on `LegalRAGServiceDown`.
6. **Hallucinated Retraining Loops:** Triggering automated model retraining without an offline evaluation gate (`faithfulness >= 0.75`) risks promoting corrupted models. Always enforce evaluation branching.
7. **Unbounded Sliding Windows:** Computing quantile histograms over `[1h]` instead of `[5m]` masks recent catastrophic spikes. Use multi-window burn rates.
8. **Logging PII to Central Telemetry:** Never log unmasked Egyptian National IDs or telephone numbers to Grafana, Langfuse, or ELK. Always redact in memory first.
9. **No Fallback for External Telemetry:** If Langfuse or Redis goes down, the core `/ask` API must continue answering queries with graceful local fallbacks.
10. **Silent Broken Dashboards:** Dashboards created via UI click-ops vanish upon container restart. Always provision dashboards as code in Git (`provisioning/dashboards`).
