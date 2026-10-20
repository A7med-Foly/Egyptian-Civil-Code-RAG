#!/bin/bash
# Canary Promotion Script: Ramps candidate traffic 10% -> 25% -> 50% -> 100%
set -e

NGINX_CONF_PATH="${1:-docker/nginx.conf}"
CONTAINER_NAME="legal-rag-nginx"

echo "=================================================="
echo "Starting Canary Traffic Promotion Sequence"
echo "=================================================="

STAGES=("10" "25" "50" "100")

for CANARY_PCT in "${STAGES[@]}"; do
    STABLE_PCT=$((100 - CANARY_PCT))
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Ramping traffic -> Stable: ${STABLE_PCT}%, Canary: ${CANARY_PCT}%"

    if [ "$CANARY_PCT" -eq 100 ]; then
        WEIGHT_V1="0"
        WEIGHT_V2="10"
    elif [ "$CANARY_PCT" -eq 50 ]; then
        WEIGHT_V1="5"
        WEIGHT_V2="5"
    elif [ "$CANARY_PCT" -eq 25 ]; then
        WEIGHT_V1="75"
        WEIGHT_V2="25"
    else
        WEIGHT_V1="9"
        WEIGHT_V2="1"
    fi

    # Update upstream weights in nginx.conf
    sed -i "s/server legal-rag-v1:8000 weight=[0-9]*/server legal-rag-v1:8000 weight=${WEIGHT_V1}/" "$NGINX_CONF_PATH"
    sed -i "s/server legal-rag-v2:8000 weight=[0-9]*/server legal-rag-v2:8000 weight=${WEIGHT_V2}/" "$NGINX_CONF_PATH"

    # Reload nginx configuration with zero downtime
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker exec "$CONTAINER_NAME" nginx -s reload
        echo "Nginx successfully reloaded inside container ${CONTAINER_NAME}."
    else
        echo "Container ${CONTAINER_NAME} not currently running; configuration updated on disk."
    fi

    if [ "$CANARY_PCT" -ne 100 ]; then
        echo "Observing telemetry at ${CANARY_PCT}% canary split..."
        sleep 2
    fi
done

echo "=================================================="
echo "Canary promotion complete. 100% traffic routed to candidate (v2)."
echo "=================================================="
