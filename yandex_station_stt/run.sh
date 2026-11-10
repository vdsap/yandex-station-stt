#!/bin/sh
set -e

CONFIG_PATH=/data/options.json
DEBUG_FLAG=""

if [ -f "$CONFIG_PATH" ]; then
    DEBUG=$(grep -o '"debug":\s*true' "$CONFIG_PATH" || true)
    if [ -n "$DEBUG" ]; then
        DEBUG_FLAG="--debug"
    fi
fi

echo "[INFO] Starting Yandex Station STT (Wyoming server on port 10300)..."

# Announce service to Home Assistant via Supervisor Discovery API
if [ -n "$SUPERVISOR_TOKEN" ]; then
    echo "[INFO] Registering Wyoming service in Home Assistant discovery..."
    DISCOVERY_HOST="${HOSTNAME:-localhost}"
    curl -s -X POST \
        -H "Authorization: Bearer ${SUPERVISOR_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"service\": \"wyoming\", \"config\": {\"host\": \"${DISCOVERY_HOST}\", \"port\": 10300}}" \
        http://supervisor/discovery || echo "[WARN] Failed to send discovery payload"
fi

exec python3 -m yandex_stt.server --uri tcp://0.0.0.0:10300 --models-dir /app/models $DEBUG_FLAG
