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
exec python3 -m yandex_stt.server --uri tcp://0.0.0.0:10300 --models-dir /app/models $DEBUG_FLAG
