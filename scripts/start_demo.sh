#!/usr/bin/env bash
# start_demo.sh — start the upstream demo Web-UI on https://<pi-ip>:7860/ with WSS.
#
# The demo proxies a microphone stream over the OpenAI Realtime WebSocket to the
# speech-to-speech pipeline. Both endpoints must be TLS-enabled so the browser
# grants getUserMedia() (browsers reject HTTP LAN origins).
#
# Usage:
#   ./scripts/start_demo.sh                  # auto-detect certs + LAN IP
#   PI_IP=10.0.0.5 ./scripts/start_demo.sh  # override the IP if needed
#
# Requires:
#   - certs at models/tls/<ip>+localhost.pem and -key.pem (created by `mkcert`)
#   - the pipeline running on https://<pi-ip>:8765/ (started via rpi_start.sh
#     once certs exist)
#
# The script can be invoked from any directory; it resolves paths absolutely.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="$REPO_ROOT/.venv/bin"
DEMO_DIR="$REPO_ROOT/demo"
LOG_DIR="$REPO_ROOT/models/log"
LOG_FILE="$LOG_DIR/demo-ui.log"

# Prepend venv bin so uvicorn resolves to the project's interpreter.
if [[ -d "$VENV_BIN" ]]; then
  export PATH="$VENV_BIN:$PATH"
else
  echo "[fail] no .venv at $VENV_BIN — run \`uv sync --extra rpi\` first." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

# Auto-detect the Pi's LAN IP. Override with PI_IP=...
PI_IP="${PI_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
if [[ -z "$PI_IP" ]]; then
  echo "[fail] could not determine PI_IP. Set PI_IP=<addr> and retry." >&2
  exit 1
fi

# Auto-discover TLS certs. Prefer the stable realtime-cert.pem if present,
# otherwise pick any *+localhost.pem with a matching -key.pem sibling.
TLS_CERT=""
TLS_KEY=""
if [[ -f "$REPO_ROOT/models/tls/realtime-cert.pem" && -f "$REPO_ROOT/models/tls/realtime-key.pem" ]]; then
  TLS_CERT="$REPO_ROOT/models/tls/realtime-cert.pem"
  TLS_KEY="$REPO_ROOT/models/tls/realtime-key.pem"
else
  for cert in "$REPO_ROOT"/models/tls/*+localhost.pem; do
    [[ -f "$cert" ]] || continue
    key="${cert%.pem}-key.pem"
    if [[ -f "$key" ]]; then
      TLS_CERT="$cert"
      TLS_KEY="$key"
      break
    fi
  done
fi
if [[ -z "$TLS_CERT" ]]; then
  echo "[fail] no TLS cert found under models/tls/. Run \`mkcert $PI_IP localhost\` first." >&2
  exit 1
fi

# Default to WSS so the browser can connect. Override with SPEECH_TO_SPEECH_URL=
# for non-TLS setups (e.g. SSH tunnel demo).
: "${SPEECH_TO_SPEECH_URL:=wss://$PI_IP:8765/v1/realtime}"
export SPEECH_TO_SPEECH_URL

# Expose the mkcert root CA so browser-PCs can fetch it via /rootCA.pem and
# trust the local TLS certs (otherwise the WebSocket dial silently fails with
# close code 1015 — browsers do NOT show a "proceed anyway" dialog for WS).
MKCERT_CAROOT="$("$VENV_BIN/mkcert" -CAROOT 2>/dev/null)/rootCA.pem"
export MKCERT_CAROOT
if [[ -f "$MKCERT_CAROOT" ]]; then
  echo "[info] mkcert CA available at: https://$PI_IP:7860/rootCA.pem"
else
  echo "[warn] mkcert rootCA.pem not found at $MKCERT_CAROOT — re-run: $VENV_BIN/mkcert -install"
fi

echo "[info] starting demo UI on https://$PI_IP:7860/  (log: $LOG_FILE)"
echo "[info] pipeline target: $SPEECH_TO_SPEECH_URL"
echo "[info] cert: $TLS_CERT"

# Stop any old demo instance.
pkill -f 'uvicorn.*demo' 2>/dev/null || true
sleep 1

exec uvicorn \
  --app-dir "$DEMO_DIR" \
  --ssl-keyfile "$TLS_KEY" \
  --ssl-certfile "$TLS_CERT" \
  server:app \
  --host 0.0.0.0 \
  --port 7860 \
  >>"$LOG_FILE" 2>&1
