#!/usr/bin/env bash
# Start the LiteRT-LM OpenAI-compatible server in the foreground.
#
# Uses `src/servers/litert_lm_mtp_server.py`, a thin FastAPI wrapper
# around the LiteRT-LM Python SDK that enables Multi-Token Prediction
# (MTP). The bundled `litert-lm serve` CLI does not expose the
# --enable-speculative-decoding flag, so this wrapper is necessary to
# turn MTP on in server mode.
#
# LiteRT-LM is the Google AI Edge LLM runtime; install it via
#   uv sync --extra rpi-litertlm
# After `s2s-rpi-setup --fetch --llm-backend litert-lm` the model is
# already imported into LiteRT-LM's local registry. This script only
# starts the server; it does NOT import.
#
# Usage:
#   source models/.env
#   ./scripts/start_litert_lm.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "$REPO_ROOT/.venv/bin" ]]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

: "${LITERT_LM_PORT:=9379}"
: "${LITERT_LM_HOST:=127.0.0.1}"
: "${LITERT_LM_DISABLE_MTP:=}"
: "${LITERT_LM_MAX_NUM_TOKENS:=}"

if ! command -v litert-lm-mtp-server >/dev/null 2>&1; then
  echo "litert-lm-mtp-server not found on PATH." >&2
  echo "Re-install: uv sync --extra rpi-litertlm" >&2
  exit 127
fi

# Resolve the model path. Prefer LITERT_LM_MODEL_PATH (set by
# s2s-rpi-setup); fall back to the alias-resolved location.
if [[ -n "${LITERT_LM_MODEL_PATH:-}" ]]; then
  MODEL_PATH="$LITERT_LM_MODEL_PATH"
else
  # Default LiteRT-LM registry layout: ~/.litert-lm/models/<alias>/model.litertlm
  _alias_dir="${HOME}/.litert-lm/models/${LITERT_LM_MODEL_ALIAS:-gemma4-e2b}"
  MODEL_PATH="${_alias_dir}/model.litertlm"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[warn] LiteRT-LM model file not found at $MODEL_PATH" >&2
  echo "       Import it via: litert-lm import <HF repo> <file>" >&2
fi

echo "[info] serving model: $MODEL_PATH"
echo "[info] MTP: $(if [[ -n "$LITERT_LM_DISABLE_MTP" ]]; then echo disabled; else echo enabled; fi)"

ARGS=(--host "$LITERT_LM_HOST" --port "$LITERT_LM_PORT" --model "$MODEL_PATH")
if [[ -n "$LITERT_LM_DISABLE_MTP" ]]; then
  ARGS+=(--disable-mtp)
fi
if [[ -n "$LITERT_LM_MAX_NUM_TOKENS" ]]; then
  ARGS+=(--max-num-tokens "$LITERT_LM_MAX_NUM_TOKENS")
fi

exec litert-lm-mtp-server "${ARGS[@]}"
