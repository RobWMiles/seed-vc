#!/usr/bin/env bash
# Boot the Seed-VC RunPod worker. handler.py keeps Seed-VC's models
# loaded between jobs to avoid the ~10s checkpoint reload per request.
set -euo pipefail
cd /app
exec python -u handler.py
