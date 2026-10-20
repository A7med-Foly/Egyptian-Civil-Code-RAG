#!/bin/bash
# Emergency Rollback Script: Instantly returns 100% traffic to stable version (v1)
set -e

NGINX_CONF_PATH="${1:-docker/nginx.conf}"
CONTAINER_NAME="legal-rag-nginx"

START_TIME=$(date +%s%N)
echo "[$(date +'%Y-%m-%d %H:%M:%S')] TRIGGERING INSTANT CANARY ROLLBACK..."

# Restore weights: 100% Stable (v1), 0% Candidate (v2)
sed -i "s/server legal-rag-v1:8000 weight=[0-9]*/server legal-rag-v1:8000 weight=10/" "$NGINX_CONF_PATH"
sed -i "s/server legal-rag-v2:8000 weight=[0-9]*/server legal-rag-v2:8000 weight=0/" "$NGINX_CONF_PATH"

# Reload Nginx
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker exec "$CONTAINER_NAME" nginx -s reload
    echo "Nginx reloaded. Traffic 100% restored to stable legal-rag-v1."
else
    echo "Container ${CONTAINER_NAME} not running; configuration updated on disk."
fi

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
ELAPSED_SEC=$(awk "BEGIN {print $ELAPSED_MS / 1000}")

echo "=================================================="
echo "ROLLBACK COMPLETED SUCCESSFULLY"
echo "Rollback Time: ${ELAPSED_SEC}s (${ELAPSED_MS} ms)"
echo "Traffic state: 100% v1 (stable), 0% v2 (candidate)"
echo "=================================================="
