#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/labinnovaciont/Desktop/LAIA"
PYTHON="${ROOT}/deployment/laia_model/.venv/bin/python"

cd "$ROOT"
exec "$PYTHON" -m app.calibration "$@"
