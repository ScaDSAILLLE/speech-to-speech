#!/usr/bin/env bash
# verify_rpi_fork.sh — End-to-End smoke test for the RPi fork.
#
# Run from the repo root:
#   bash scripts/verify_rpi_fork.sh
#
# Reports pass/fail at each step and exits 1 on the first hard failure.
# Run once and read the output. Does NOT restart anything (assumes the
# model servers are already running via ./scripts/rpi_start.sh).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TLS_CERT="$REPO_ROOT/models/tls/realtime-cert.pem"
MOONSHINE_PORT="${MOONSHINE_STT_PORT:-9001}"
SUPERTONIC_PORT="${SUPERTONIC_TTS_PORT:-9002}"
LITERT_LM_PORT="${LITERT_LM_PORT:-9379}"
PIPELINE_PORT="${S2S_WS_PORT:-8765}"
DEMO_PORT=7860
PIPELINE_TLS=1

red() { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
section() { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }

ok=0
fail=0

check_http() {
  local url="$1" expect_status="${2:-200}"
  local code
  code=$(curl -sk -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null || echo "000")
  if [[ "$code" == "$expect_status" ]]; then
    green "  ✓ $url → $code"
    ok=$((ok+1))
    return 0
  else
    red "  ✗ $url → $code (expected $expect_status)"
    fail=$((fail+1))
    return 1
  fi
}

section "Listening ports"
ss -lntp 2>&1 | grep -E ':9001|:9002|:9379|:8765|:7860' | head -10

section "HTTP probes (HTTP, no TLS — internal-only model servers)"
check_http "http://127.0.0.1:${MOONSHINE_PORT}/health"
check_http "http://127.0.0.1:${SUPERTONIC_PORT}/v1/audio/voices"
check_http "http://127.0.0.1:${LITERT_LM_PORT}/v1/models"

section "HTTPS probes (with mkcert cert — pipeline + demo)"
if [[ -f "$TLS_CERT" ]]; then
  check_http "https://192.168.178.101:${PIPELINE_PORT}/v1/pool"
  check_http "https://192.168.178.101:${DEMO_PORT}/api/config"
else
  yellow "  ! TLS cert not found at $TLS_CERT — skipping HTTPS checks"
fi

section "Demo /api/config payload"
if [[ -f "$TLS_CERT" ]]; then
  CFG=$(curl -sk --cacert "$TLS_CERT" --max-time 5 "https://192.168.178.101:${DEMO_PORT}/api/config")
  echo "$CFG" | head -1
  if echo "$CFG" | grep -q "wss://192.168.178.101:${PIPELINE_PORT}/v1/realtime"; then
    green "  ✓ demo s2sUrl points at WSS pipeline"
    ok=$((ok+1))
  else
    red "  ✗ demo s2sUrl wrong: $CFG"
    fail=$((fail+1))
  fi
fi

section "LiteRT-LM chat (functional)"
LLM_RESP=$(curl -s --max-time 60 -X POST "http://127.0.0.1:${LITERT_LM_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-e2b","messages":[{"role":"user","content":"Reply with just OK."}],"max_tokens":10,"stream":false}' 2>/dev/null | head -1)
echo "  $LLM_RESP" | head -c 200; echo
if echo "$LLM_RESP" | grep -q '"content"'; then
  green "  ✓ LLM responds with content"
  ok=$((ok+1))
else
  red "  ✗ LLM did not respond with content"
  fail=$((fail+1))
fi

section "Supertonic TTS (functional)"
TTS_RESP=$(curl -s --max-time 30 -X POST "http://127.0.0.1:${SUPERTONIC_PORT}/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"input":"Test.","voice":"M1","response_format":"wav","sample_rate":16000}' \
  -o /tmp/test.wav -w '%{http_code} %{size_download}' 2>/dev/null)
echo "  HTTP $TTS_RESP"
if file /tmp/test.wav 2>/dev/null | grep -q 'WAVE'; then
  green "  ✓ valid WAV file produced"
  ok=$((ok+1))
else
  red "  ✗ no valid WAV file"
  fail=$((fail+1))
fi

section "Moonshine STT (functional)"
STT_RESP=$(curl -s --max-time 30 -F file=@/tmp/test.wav -F model=UsefulSensors/moonshine-base "http://127.0.0.1:${MOONSHINE_PORT}/v1/audio/transcriptions" 2>&1)
echo "  $STT_RESP"
if echo "$STT_RESP" | grep -q '"text"'; then
  green "  ✓ Moonshine STT responds"
  ok=$((ok+1))
else
  red "  ✗ Moonshine STT did not respond (got: $STT_RESP)"
  fail=$((fail+1))
fi

section "WebSocket round-trip"
if [[ -f "$TLS_CERT" ]]; then
  if uv run python3 "$REPO_ROOT/scripts/_ws_round_trip.py" 2>&1; then
    green "  ✓ WS round-trip succeeded"
    ok=$((ok+1))
  else
    red "  ✗ WS round-trip failed"
    fail=$((fail+1))
  fi
fi

echo
section "Summary"
green "Passed: $ok"
[[ $fail -gt 0 ]] && red "Failed: $fail" || green "Failed: 0"
exit $fail
