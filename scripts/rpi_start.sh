#!/usr/bin/env bash
# rpi_start.sh — orchestrate the three model servers and the speech-to-speech pipeline.
#
# Usage:
#   source models/.env             # exports HF_HOME, ports, etc.
#   ./scripts/rpi_start.sh         # foreground: servers in background, pipeline foreground
#   ./scripts/rpi_start.sh --bg    # everything in background; PIDs in models/run/
#   ./scripts/rpi_start.sh --stop  # stop everything we previously started
#   ./scripts/rpi_start.sh --status
#
# Layout created under models/:
#   run/<name>.pid                 PID file per process
#   log/<name>.log                 full stdout/stderr per process (always written)
#   log/<name>.live.log            shorthand for `tail -f` from the terminal
#
# Requires:
#   - models/.env sourced (or env vars already in the environment)
#   - moonshine-stt-server / supertonic-tts-server / litert-lm / speech-to-speech on PATH
#
# All servers start in parallel; the script waits for each TCP port to bind
# before declaring success.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
RUN_DIR="$MODELS_DIR/run"
LOG_DIR="$MODELS_DIR/log"
ENV_FILE="$MODELS_DIR/.env"
VENV_BIN="$REPO_ROOT/.venv/bin"

mkdir -p "$RUN_DIR" "$LOG_DIR"

# Make uv-installed entry points visible (speech-to-speech, moonshine-stt-server,
# supertonic-tts-server, litert-lm, s2s-rpi-setup). Without this the launcher's
# `command -v` checks fail and `exec speech-to-speech` errors with "not found".
if [[ -d "$VENV_BIN" ]]; then
  export PATH="$VENV_BIN:$PATH"
else
  echo "[warn] no .venv at $REPO_ROOT/.venv — did you run \`uv sync --extra rpi\` yet?"
  echo "       Falling back to global PATH; entry points may not be found."
fi

# Source models/.env if present (do not overwrite pre-set values).
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Defaults if not in .env
: "${MOONSHINE_STT_PORT:=9001}"
: "${SUPERTONIC_TTS_PORT:=9002}"
: "${LITERT_LM_PORT:=9379}"        # LiteRT-LM's documented default
: "${S2S_WS_HOST:=0.0.0.0}"
: "${S2S_WS_PORT:=8765}"

# Auto-discover TLS certs for the WebSocket server. When mkcert produced
# models/tls/realtime-{cert,key}.pem (or any *+localhost.pem with a matching
# -key.pem sibling) the pipeline speaks WSS, which is the only way browsers
# grant getUserMedia() on LAN IPs (HTTPS-or-localhost rule).
TLS_CERT=""
TLS_KEY=""
if [[ -d "$MODELS_DIR/tls" ]]; then
  # Prefer the stable realtime-cert.pem if present.
  if [[ -f "$MODELS_DIR/tls/realtime-cert.pem" && -f "$MODELS_DIR/tls/realtime-key.pem" ]]; then
    TLS_CERT="$MODELS_DIR/tls/realtime-cert.pem"
    TLS_KEY="$MODELS_DIR/tls/realtime-key.pem"
  else
    # Otherwise pick the first *.pem with a matching -key.pem sibling.
    for cert in "$MODELS_DIR"/tls/*+localhost.pem; do
      [[ -f "$cert" ]] || continue
      key="${cert%.pem}-key.pem"
      if [[ -f "$key" ]]; then
        TLS_CERT="$cert"
        TLS_KEY="$key"
        break
      fi
    done
  fi
fi

PIDS_TO_CLEAN=()

cleanup() {
  local code=$?
  if (( ${#PIDS_TO_CLEAN[@]} > 0 )); then
    echo
    echo "[shutdown] stopping background processes"
    for pid in "${PIDS_TO_CLEAN[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done
    # Also clean up any PID files we created but didn't track (e.g. on signal).
    for pidfile in "$RUN_DIR"/*.pid; do
      [[ -f "$pidfile" ]] || continue
      pid="$(cat "$pidfile" 2>/dev/null || true)"
      if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
      rm -f "$pidfile"
    done
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local host="$1" port="$2" name="$3" timeout="${4:-60}"
  local elapsed=0
  while (( elapsed < timeout )); do
    if (echo > "/dev/tcp/$host/$port") >/dev/null 2>&1; then
      echo "[ok] $name listening on $host:$port"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "[fail] $name did not bind $host:$port within ${timeout}s" >&2
  return 1
}

# Start a background process and stream its stdout/stderr to:
#   1. The current terminal (so the user sees live progress), with a per-server
#      `[name]` prefix prepended to each line.
#   2. $LOG_DIR/$name.log (full unprefixed log for `tail -f`).
#
# Background so the launcher can keep starting the other services in parallel.
start_bg() {
  local name="$1"; shift
  local pidfile="$RUN_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[skip] $name already running (pid $(cat "$pidfile"))"
    return 0
  fi

  echo "[start] $name (logs: $logfile, live in this terminal)"
  # Stream every line to:
  #   1. the terminal on stderr, prefixed with [name] for clarity, and
  #   2. $logfile unprefixed, for later `tail -f`.
  # We redirect the subshell's stdout to the log file (raw lines) but keep
  # stderr untouched, so the inner printf's `>&2` reaches the terminal.
  (
    "$@" 2>&1 | while IFS= read -r line || [[ -n "$line" ]]; do
      printf '[%s] %s\n' "$name" "$line" >&2
      printf '%s\n' "$line"
    done
  ) >"$logfile" &
  local pid=$!
  echo "$pid" >"$pidfile"
  PIDS_TO_CLEAN+=("$pid")
  sleep 0.2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[fail] $name exited immediately; see $logfile" >&2
    rm -f "$pidfile"
    return 1
  fi
}

# Stop a previously-started background service and remove its PID file.
stop_one() {
  local name="$1"
  local pidfile="$RUN_DIR/$name.pid"
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[stop] $name (pid $pid)"
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pidfile"
}

stop_all() {
  local name
  for pidfile in "$RUN_DIR"/*.pid; do
    [[ -f "$pidfile" ]] || continue
    name="$(basename "$pidfile" .pid)"
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[stop] $name (pid $pid)"
      kill "$pid" 2>/dev/null || true
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$pidfile"
  done
  echo "[ok] all stopped"
}

print_status() {
  local name pidfile pid state
  for pidfile in "$RUN_DIR"/*.pid; do
    [[ -f "$pidfile" ]] || continue
    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile" 2>/dev/null || echo "-")"
    if [[ "$pid" != "-" ]] && kill -0 "$pid" 2>/dev/null; then
      state="running"
    else
      state="stopped"
    fi
    printf "  %-22s pid=%-6s state=%s\n" "$name" "$pid" "$state"
  done
}

# Pipeline CLI flags shared by foreground/background runners.
PIPELINE_ARGS=(
  --mode realtime
  --ws_host "$S2S_WS_HOST"
  --ws_port "$S2S_WS_PORT"
  --stt faster-whisper
  --faster_whisper_stt_model_name "base.en"
  --faster_whisper_stt_device cpu
  --faster_whisper_stt_compute_type int8
  --faster_whisper_stt_gen_language en
  --llm_backend chat-completions
  --model_name "${LITERT_LM_MODEL_ALIAS:-gemma4-e2b}"
  --responses_api_base_url "http://127.0.0.1:${LITERT_LM_PORT}/v1"
  --responses_api_api_key ""
  --tts supertonic-http
  --supertonic_http_tts_base_url "http://127.0.0.1:${SUPERTONIC_TTS_PORT}/v1"
  # VAD tuning for the RPi fork. The upstream defaults (64 ms min silence,
  # 0.6 threshold) were designed for clean studio mics; on a Pi with consumer
  # speakers the model's TTS echo kept triggering new turns. 900 ms silence +
  # 0.7 threshold + 320 ms speech pad absorb both the echo and natural phrase
  # pauses (typical intra-sentence pauses run 600-800 ms; the 1.5 s reopen
  # window keeps the same turn alive when the speaker resumes mid-thought).
  # NB: --enable_realtime_transcription is intentionally OFF: it makes the
  # VAD yield progressive audio chunks to STT while the user is still
  # speaking, but Moonshine can't stream partial transcripts (the seq2seq
  # model needs the full utterance). Each progressive chunk produces a
  # separate STT call with very little audio (200-800 ms), gets truncated by
  # the per-second token cap, and the resulting 1-2-word "transcript" wins
  # on the wire. With real-time off, the VAD waits for the soft-end silence
  # (900 ms) and STT receives the full turn audio at once. Flip on with
  # --enable_vad_realtime when switching to a streaming-capable STT backend.
  --thresh 0.7
  --min_silence_ms 900
  --min_speech_ms 320
  --speech_pad_ms 500
  --speculative_reopen_ms 1500
  --unanswered_reopen_ms 9000
  --enable_live_transcription
)
# Add TLS flags only when certs were found. Without these the server speaks
# ws:// (insecure) and browsers refuse getUserMedia() on LAN IPs.
if [[ -n "$TLS_CERT" && -n "$TLS_KEY" ]]; then
  PIPELINE_ARGS+=(
    --ws_ssl_certfile "$TLS_CERT"
    --ws_ssl_keyfile "$TLS_KEY"
  )
  echo "[info] TLS enabled: cert=$TLS_CERT"
else
  echo "[info] TLS disabled (no certs in models/tls/); browser mic will be blocked on non-localhost."
fi

run_pipeline_foreground() {
  local pipeline_log="$LOG_DIR/speech-to-speech.log"
  echo "[run] speech-to-speech (logs: $pipeline_log, live in this terminal)"
  exec speech-to-speech "${PIPELINE_ARGS[@]}"
}

run_pipeline_background() {
  start_bg speech-to-speech speech-to-speech "${PIPELINE_ARGS[@]}"
}

case "${1:-start}" in
  --stop|stop)
    stop_all
    ;;
  --status|status)
    print_status
    ;;
  --bg|start)
    # Helper: abort cleanly if a server fails to come up. We don't use plain
    # `set -e` because we want to skip optional servers with a warning rather
    # than abort the whole stack.
    bail_on_fail() {
      local rc=$?
      if (( rc != 0 )); then
        echo "[fail] aborting startup; cleaning up already-started services" >&2
        stop_one moonshine-stt-server
        stop_one supertonic-tts-server
        stop_one litert-lm
        stop_one cactus-server
        stop_one speech-to-speech
        exit "$rc"
      fi
    }

    # All four services start in parallel; wait_for_port is called for each.
    # The launcher exits non-zero if any required port fails to bind.

    # 1. STT server. Only needed for backends that talk to an external HTTP
    # STT (moonshine-http). In-process backends (faster-whisper, parakeet-tdt,
    # whisper, etc.) load the model directly in the pipeline process and need
    # no separate server.
    _stt_backend=""
    for ((i = 0; i < ${#PIPELINE_ARGS[@]}; i++)); do
      if [[ "${PIPELINE_ARGS[$i]}" == "--stt" && $((i + 1)) -lt ${#PIPELINE_ARGS[@]} ]]; then
        _stt_backend="${PIPELINE_ARGS[$((i + 1))]}"
        break
      fi
    done

    if [[ "$_stt_backend" == "moonshine-http" ]]; then
      if command -v moonshine-stt-server >/dev/null 2>&1; then
        start_bg moonshine-stt-server moonshine-stt-server \
          --host 127.0.0.1 --port "$MOONSHINE_STT_PORT" \
          --model "${MOONSHINE_DEFAULT_MODEL:-UsefulSensors/moonshine-base}" \
          --dtype "${MOONSHINE_DTYPE:-float32}" \
          && wait_for_port 127.0.0.1 "$MOONSHINE_STT_PORT" moonshine-stt-server 180 \
          || bail_on_fail
      else
        echo "[warn] moonshine-stt-server not on PATH; STT pipeline will fail."
      fi
    else
      echo "[info] STT backend=$_stt_backend — in-process, no separate STT server needed."
      # Free the port if a leftover moonshine-stt-server is still listening.
      _stale_pid=$(ss -lntp 2>/dev/null | grep ":${MOONSHINE_STT_PORT} " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
      if [[ -n "$_stale_pid" ]]; then
        echo "[info] killing leftover moonshine-stt-server on :${MOONSHINE_STT_PORT} (pid=$_stale_pid)"
        kill -TERM "$_stale_pid" 2>/dev/null || true
      fi
    fi

    # 2. TTS server
    if command -v supertonic-tts-server >/dev/null 2>&1; then
      start_bg supertonic-tts-server supertonic-tts-server \
        --host 127.0.0.1 --port "$SUPERTONIC_TTS_PORT" \
        && wait_for_port 127.0.0.1 "$SUPERTONIC_TTS_PORT" supertonic-tts-server 180 \
        || bail_on_fail
    else
      echo "[warn] supertonic-tts-server not on PATH; TTS pipeline will fail."
    fi

    # 3. LLM server (LiteRT-LM or Cactus). Both speak OpenAI-compat and bind the
    # same port (9379 for LiteRT-LM, by convention). Only one should run at a
    # time — set S2S_LLM_BACKEND in models/.env to choose.
    if [[ "${S2S_LLM_BACKEND:-litert-lm}" == "litert-lm" ]] && command -v litert-lm >/dev/null 2>&1; then
      local_litert_args=(serve --host 127.0.0.1 --port "$LITERT_LM_PORT")
      if [[ "${LITERT_LM_VERBOSE:-1}" == "1" ]]; then
        local_litert_args+=(--verbose)
      fi
      start_bg litert-lm litert-lm "${local_litert_args[@]}" \
        && wait_for_port 127.0.0.1 "$LITERT_LM_PORT" litert-lm 300 \
        || bail_on_fail
    elif [[ "${S2S_LLM_BACKEND:-litert-lm}" == "cactus" ]] && command -v cactus-server >/dev/null 2>&1; then
      start_bg cactus-server cactus-server \
        --model "${CACTUS_MODEL_PATH:-/dev/null}" --host 127.0.0.1 --port "$LITERT_LM_PORT" \
        && wait_for_port 127.0.0.1 "$LITERT_LM_PORT" cactus-server 300 \
        || bail_on_fail
    else
      echo "[warn] no LLM runtime configured (S2S_LLM_BACKEND='${S2S_LLM_BACKEND:-unset}'); pipeline will not respond to user input."
    fi

    echo
    echo "[info] model servers up. WebSocket endpoint: ws://$S2S_WS_HOST:$S2S_WS_PORT/v1/realtime"
    echo

    if [[ "${1:-start}" == "--bg" ]]; then
      run_pipeline_background
      echo "[ok] all processes started in background; use --status to inspect."
      echo "[info] tail logs with: tail -f $LOG_DIR/*.log"
      # Detach: disable the trap that would clean up our own children.
      trap - EXIT INT TERM
      exit 0
    else
      run_pipeline_foreground
    fi
    ;;
  *)
    echo "Usage: $0 {start|--bg|--stop|--status}" >&2
    exit 64
    ;;
esac
