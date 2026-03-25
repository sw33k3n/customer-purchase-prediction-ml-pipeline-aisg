#!/usr/bin/env bash
set -euo pipefail

echo "=== AIAP23 Pipeline Runner ==="
echo "Working directory: $(pwd)"

# Pick a Python executable
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "ERROR: Python not found. Install Python 3 or activate your virtual environment."
  exit 1
fi

echo "Python: $($PYTHON_BIN --version)"

mkdir -p outputs/models outputs/metrics

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
# Disable Python warnings for cleaner runtime output.
export PYTHONWARNINGS="ignore"

DEFAULT_DB_URL="https://techassessment.blob.core.windows.net/aiap23-assessment-data/online_shopping.db"
DB_PATH="${1:-${ONLINE_SHOPPING_DB_PATH:-${DEFAULT_DB_URL}}}"
TABLE_NAME="${2:-${ONLINE_SHOPPING_TABLE_NAME:-online_shopping}}"

TEMP_DB_PATH=""
if [[ "${DB_PATH}" =~ ^https?:// ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to download database URL."
    exit 1
  fi
  TEMP_DB_PATH="$(mktemp /tmp/online_shopping.XXXXXX.db)"
  echo "Downloading DB to temporary path: ${TEMP_DB_PATH}"
  curl -fL "${DB_PATH}" -o "${TEMP_DB_PATH}"
  DB_PATH="${TEMP_DB_PATH}"
fi

cleanup() {
  if [ -n "${TEMP_DB_PATH}" ] && [ -f "${TEMP_DB_PATH}" ]; then
    rm -f "${TEMP_DB_PATH}"
  fi
}
trap cleanup EXIT

if [ -f "src/main.py" ]; then
  ENTRYPOINT="src/main.py"
elif [ -f "src/pipeline.py" ]; then
  ENTRYPOINT="src/pipeline.py"
else
  echo "ERROR: Could not find an entrypoint. Expected src/main.py or src/pipeline.py."
  exit 1
fi

echo "Running entrypoint: ${ENTRYPOINT}"
$PYTHON_BIN "${ENTRYPOINT}" --db-path "${DB_PATH}" --table-name "${TABLE_NAME}"

echo "=== Done ==="
