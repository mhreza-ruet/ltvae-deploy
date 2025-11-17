#!/usr/bin/env bash
set -e

# Allow runtime overrides via env
export MODEL_PATH=${MODEL_PATH:-app/checkpoints/best_model.pth}
export VOCAB_JSON=${VOCAB_JSON:-app/token_to_idx.json}
export PROP_HEAD_PATH=${PROP_HEAD_PATH:-app/checkpoints/property_head_best.pt}
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}

echo "[startup] MODEL_PATH=$MODEL_PATH"
echo "[startup] VOCAB_JSON=$VOCAB_JSON"
echo "[startup] PROP_HEAD_PATH=$PROP_HEAD_PATH"

# Warm the model once (optional; safe if it fails)
python - <<'PY' || true
import requests, time, os
url = "http://127.0.0.1:8000/health"
for _ in range(2):
    try:
        requests.get(url, timeout=0.2)
        break
    except Exception:
        time.sleep(0.2)
PY

# Launch FastAPI
exec uvicorn app.main:app --host 0.0.0.0 --port 8000