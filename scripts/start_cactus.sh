#!/usr/bin/env bash
# Start the Cactus Compute LLM server in the foreground.
#
# Cactus is the alternative LLM runtime; install via
#   uv sync --extra rpi-cactus
# It exposes an OpenAI-compatible HTTP server.
#
# Usage:
#   source models/.env
#   ./scripts/start_cactus.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "$REPO_ROOT/.venv/bin" ]]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

: "${CACTUS_MODEL_PATH:?CACTUS_MODEL_PATH must be set (source models/.env)}"
: "${CACTUS_PORT:=8080}"
: "${CACTUS_HOST:=127.0.0.1}"

if ! command -v cactus-server >/dev/null 2>&1; then
  echo "cactus-server not found on PATH." >&2
  echo "Install it via: uv sync --extra rpi-cactus (and re-run if .venv is missing)." >&2
  exit 127
fi

exec cactus-server \
  --model "$CACTUS_MODEL_PATH" \
  --host "$CACTUS_HOST" \
  --port "$CACTUS_PORT" \
  --openai_compat
