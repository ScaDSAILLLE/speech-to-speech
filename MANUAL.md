# MANUAL — speech-to-speech RPi fork

This document is the comprehensive reference for the Raspberry Pi fork. It complements `README.md` (quickstart) and `AGENTS.md` (repo conventions) by going deep on architecture, deployment, and operations.

> The fork is **foreign**: nothing here is ever pushed back to `huggingface/speech-to-speech`. See the [Hosting](#hosting) section at the bottom for how to move the result into our own git.

---

## 1. Why a fork

The upstream project ships a low-latency, modular voice pipeline (`VAD → STT → LLM → TTS`) with an OpenAI Realtime WebSocket front-end. It works well on a Mac or a CUDA box. On a Raspberry Pi we hit three practical issues:

1. **Heavy defaults.** `qwen3-tts` (GGML/CUDA) and `parakeet-tdt` (nano-parakeet) are not Pi-friendly out of the box, and the macOS / CUDA markers in `pyproject.toml` don't apply.
2. **Models live in `~/.cache`.** We need weights under `<repo>/models/` so the repo is self-contained and the same checkout works on every Pi.
3. **One Python process holds every model.** On a Pi with limited RAM (4-8 GB), a single process that loads VAD + STT + LLM + TTS is tight. Splitting each model into its own process gives us independent restarts and clearer failure boundaries.

The fork keeps the upstream **orchestration** (`s2s_pipeline.py`, the OpenAI Realtime server, the queueing pipeline) and adds three thin OpenAI-HTTP-server shims plus in-process OpenAI-HTTP-client handlers for STT and TTS. The LLM slot was already an OpenAI-HTTP client; we just point it at LiteRT-LM or Cactus instead of a hosted provider.

## 2. Architecture in detail

### 2.1 Process topology

```
┌────────────────────┐
│   browser / client │  ws://<host>:8765/v1/realtime
└─────────┬──────────┘
          │
┌─────────▼──────────────────────────────────────────────────────────┐
│  speech-to-speech  (process #1, Realtime WebSocket + pipeline)       │
│   ├─ VAD handler                                                        │
│   ├─ TranscriptionNotifier                                              │
│   ├─ MoonshineHttpSTTHandler  ──HTTP──►  moonshine-stt-server (:9001)   │
│   ├─ ChatCompletionsApiModelHandler ──HTTP──►  litert-lm serve (:9379)  │
│   └─ SupertonicHttpTTSHandler ──HTTP──►  supertonic-tts-server (:9002)  │
└────────────────────────────────────────────────────────────────────────┘
```

Each model-server is its own Python process and binds to `127.0.0.1` only. The pipeline process binds `0.0.0.0:8765` for the browser UI.

### 2.2 Why HTTP instead of subprocess

Three reasons:

1. **Independent restarts.** If LiteRT-LM OOMs, you restart only that process; the pipeline keeps its session state.
2. **Process isolation.** A crash in the TTS server can't take down the WebSocket pipeline.
3. **Swappable LLM runtime.** Both LiteRT-LM and Cactus Compute expose OpenAI-compat HTTP servers natively, so the in-process LLM client (`responses-api` / `chat-completions` backends) handles both unchanged. The HTTP shims for STT/TTS keep the same shape.

### 2.3 What the OpenAI-HTTP shims add

| File | Endpoint | Models |
|---|---|---|
| `src/servers/moonshine_stt_server.py` | `POST /v1/audio/transcriptions` | `moonshine/base`, `moonshine/streaming-medium` |
| `src/servers/supertonic_tts_server.py` | `POST /v1/audio/speech` | `supertonic-3` (default voice `M1`) |

Each server:

- Loads its model once at startup (`setup()` in the handler sense, but `lifespan`-based on FastAPI).
- Exposes `/health` and `/v1/models` for diagnostics.
- Listens on the port from env / CLI flag.
- Writes logs to `models/log/<name>.log` when run via `rpi_start.sh`.

### 2.4 What the in-process handlers do

`MoonshineHttpSTTHandler` (`src/speech_to_speech/STT/moonshine_http_handler.py`):

- Opens an `openai.OpenAI` client against `MOONSHINE_STT_BASE_URL`.
- On `process(vad_audio)`: encodes the audio as in-memory WAV and calls `client.audio.transcriptions.create(...)`. Yields one `Transcription` event.
- Honours speculative-turn gating inherited from `BaseSTTHandler`.

`SupertonicHttpTTSHandler` (`src/speech_to_speech/TTS/supertonic_http_handler.py`):

- Opens an `openai.OpenAI` client against `SUPERTONIC_TTS_BASE_URL`.
- On `process(text_input)`: calls `client.audio.speech.create(...)` with `response_format="pcm"`. Splits the returned int16 mono PCM into `BLOCKSIZE`-sample chunks (default 512 ≈ 32 ms @ 16 kHz) and yields each as `np.ndarray`.
- Honours voice overrides from session updates.
- Yields `AUDIO_RESPONSE_DONE` on `EndOfResponse` so downstream flushes.

### 2.5 Where the changes land

```
src/speech_to_speech/s2s_pipeline.py
  - imports + Literal entries for the new backends
  - new args dataclass slots
  - new branches in get_stt_handler / get_tts_handler
  - argument wiring through build_pipeline / _build_pipeline_handlers / _build_realtime_pipeline_unit

src/speech_to_speech/arguments_classes/
  - moonshine_http_stt_arguments.py
  - supertonic_http_tts_arguments.py
  - module_arguments.py: extended Literals

src/speech_to_speech/STT/moonshine_http_handler.py         (new)
src/speech_to_speech/TTS/supertonic_http_handler.py        (new)

src/servers/moonshine_stt_server.py                         (new, entry: moonshine-stt-server)
                                                  # uses transformers.MoonshineForConditionalGeneration
src/servers/supertonic_tts_server.py                        (new, entry: supertonic-tts-server)
                                                  # uses the official supertonic package

src/s2s_rpi/setup.py                                        (new, entry: s2s-rpi-setup)
                                                  # handles litert-lm import for Gemma 4 E2B/E4B
scripts/rpi_start.sh  / rpi_stop.sh  / rpi_status.sh        (new)
scripts/start_litert_lm.sh  / start_cactus.sh               (new)
deploy/supervisord/speech-to-speech.conf                    (new)
deploy/systemd/speech-to-speech-{moonshine,supertonic,litert-lm,pipeline}.service
deploy/README.md                                            (new)
```

Upstream files left untouched: `api/openai_realtime/*`, the entire `pipeline/` orchestration, the queue plumbing, the `realtime_server` lifecycle, `demo/`, `tests/`.

> **Important:** the pipeline connects to LiteRT-LM through `--llm_backend chat-completions`, not `responses-api`. LiteRT-LM's documented OpenAI surface is `/v1/chat/completions` only; it does not implement `/v1/responses`. The two OpenAI-compat slots share the same connection flags, so the existing `--responses_api_*` arguments are reused — they just target the chat-completions endpoint instead.

## 3. Installation

### 3.1 One-time prep (Raspberry Pi OS Bookworm 64-bit)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libportaudio2 libsndfile1 ffmpeg git
pip install --break-system-packages uv
git clone https://github.com/<your-org>/speech-to-speech-rpi.git
cd speech-to-speech-rpi
```

### 3.2 Sync dependencies

```bash
uv sync --extra rpi            # default: moonshine + litert-lm + supertonic
# or:
uv sync --extra rpi-cactus     # alternative LLM runtime
# add-ons:
uv sync --extra faster-whisper # opt-in STT fallback
```

### 3.3 Bootstrap the workspace

```bash
s2s-rpi-setup --sync --fetch
source models/.env
```

`s2s-rpi-setup` does three things:

1. Creates `models/{huggingface,moonshine,supertonic,litert-lm,cactus,run,log}` and writes `.gitkeep`s into each (so `git` keeps the directory structure).
2. Writes `models/.env` exporting every required env var.
3. With `--fetch`, downloads the recommended weights via `hf` or `huggingface-cli`.

CLI flags:

```
--llm-backend {litert-lm|cactus}      which LLM runtime to configure (default litert-lm)
--litert-lm-model {e2b|e4b}           which Gemma 4 variant to import (default e2b)
--moonshine-model NAME                HF repo id of the Moonshine checkpoint (default UsefulSensors/moonshine-base)
--supertonic-voice VOICE              default Supertonic voice id (default M1)
--moonshine-stt-port PORT             default 9001
--supertonic-tts-port PORT            default 9002
--litert-lm-port PORT                 default 9379 (LiteRT-LM documented default)
--fetch                               download weights via huggingface-cli AND run `litert-lm import`
--sync                                run `uv sync --extra <extras>` first
--extra NAME                          repeat to add multiple extras
--force-env                           overwrite models/.env if it already exists
```

LiteRT-LM models available out of the box:

| `--litert-lm-model` | Hugging Face repo | Alias after import | Size |
|---|---|---|---|
| `e2b` (default) | `litert-community/gemma-4-E2B-it-litert-lm` | `gemma4-e2b` | ~1.5 GB |
| `e4b` | `litert-community/gemma-4-E4B-it-litert-lm` | `gemma4-e4b` | ~2.5 GB |

The env file is plain shell, so any consumer can do `set -a; source models/.env; set +a`.

### 3.4 Verify a clean install

```bash
python -c "import speech_to_speech, moonshine_onnx, supertonic, litert_lm" 2>&1 | head -20
speech-to-speech --help | head -30
moonshine-stt-server --help
supertonic-tts-server --help
s2s-rpi-setup --help
```

If any of these fail, the most common cause is a missing optional dep — re-run `uv sync --extra rpi`.

## 4. Running

### 4.1 The launcher (foreground)

```bash
source models/.env
./scripts/rpi_start.sh
```

The script starts the three model servers in the background (PID files in `models/run/`, logs in `models/log/`), waits for the ports to bind, then `exec`s the realtime pipeline in the foreground. Ctrl-C stops everything.

### 4.2 The launcher (background)

```bash
./scripts/rpi_start.sh --bg
./scripts/rpi_start.sh --status
./scripts/rpi_start.sh --stop
```

`--bg` returns immediately after spawning the pipeline in the background. Use `--status` for a one-line health readout and `--stop` for a clean shutdown.

### 4.3 Manual run (each process in its own terminal)

Useful when iterating on one component.

```bash
# Terminal 1
moonshine-stt-server --host 127.0.0.1 --port 9001

# Terminal 2
supertonic-tts-server --host 127.0.0.1 --port 9002

# Terminal 3 (after `s2s-rpi-setup --fetch --llm-backend litert-lm` has
# imported the model into LiteRT-LM's local registry):
litert-lm serve --host 127.0.0.1 --port 9379 --verbose

# Terminal 4 (after the three are up)
speech-to-speech \
  --mode realtime \
  --ws_host 0.0.0.0 \
  --ws_port 8765 \
  --stt moonshine-http \
  --moonshine_http_stt_base_url http://127.0.0.1:9001/v1 \
  --llm_backend chat-completions \
  --responses_api_base_url http://127.0.0.1:9379/v1 \
  --responses_api_api_key "" \
  --tts supertonic-http \
  --supertonic_http_tts_base_url http://127.0.0.1:9002/v1
```

### 4.4 Talk to it

```bash
# Python client (upstream script):
python scripts/listen_and_play_realtime.py --host 127.0.0.1 --port 8765

# Browser UI (upstream demo):
export SPEECH_TO_SPEECH_URL=ws://<pi-ip>:8765/v1/realtime
cd demo
uv pip install -r requirements.txt
uv run uvicorn --app-dir . server:app --host 0.0.0.0 --port 7860
# open http://<pi-ip>:7860/
```

The browser requires `localhost` or HTTPS for microphone access. From a different machine, use SSH port-forwarding:

```bash
ssh -L 7860:127.0.0.1:7860 -L 8765:127.0.0.1:8765 pi@<pi-ip>
# then open http://localhost:7860/
```

### 4.5 Direct HTTP calls (no pipeline)

Each server is independently testable:

```bash
# Moonshine STT
curl -F file=@test.wav -F model=UsefulSensors/moonshine-base http://127.0.0.1:9001/v1/audio/transcriptions
# {"text":"...","language":null,"model":"UsefulSensors/moonshine-base"}

# Moonshine STT streaming
curl -F file=@test.wav -F model=UsefulSensors/moonshine-streaming-medium http://127.0.0.1:9001/v1/audio/transcriptions

# Supertonic TTS (WAV). Voices are M1..M5 (male) and F1..F5 (female).
curl -X POST http://127.0.0.1:9002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello world","voice":"M1","response_format":"wav","lang":"en"}' \
  --output /tmp/hello.wav

# Supertonic TTS (raw PCM at 16 kHz, German voice)
curl -X POST http://127.0.0.1:9002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hallo welt","voice":"F1","response_format":"pcm","sample_rate":16000,"lang":"de"}' \
  --output /tmp/hallo.pcm

# LiteRT-LM exposes /v1/chat/completions only (not /v1/responses).
curl http://127.0.0.1:9379/v1/models

curl -X POST http://127.0.0.1:9379/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-e2b","messages":[{"role":"user","content":"Hi"}]}'
```

### 4.6 TLS for browser microphone access (mkcert)

The browser refuses `getUserMedia()` on plain HTTP for any host that is not
`localhost` or `127.0.0.1`. `http://192.168.178.101:7860/` therefore shows the
microphone prompt but the conversation never starts. The fix is HTTPS, and
since this is a fully-local setup we use mkcert for a self-signed CA that the
browser PCs can trust.

#### On the Pi

```bash
# Install mkcert into the project venv (binary download, no PyPI package).
curl -sL -o .venv/bin/mkcert \
  https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-arm64
chmod +x .venv/bin/mkcert

# Install the local CA into the Pi's system trust store (needed so curl on
# the Pi trusts its own certificates).
.venv/bin/mkcert -install

# Generate a cert that covers both the LAN IP and `localhost` — the demo UI
# may be reached by either depending on the network setup.
mkdir -p models/tls
.venv/bin/mkcert \
  -cert-file models/tls/realtime-cert.pem \
  -key-file  models/tls/realtime-key.pem \
  192.168.178.101 localhost
```

`models/tls/` is git-ignored — the private key must not be committed.

#### Restart the pipeline with TLS

`scripts/rpi_start.sh` auto-detects `models/tls/realtime-cert.pem` and
`models/tls/realtime-key.pem` and passes `--ws_ssl_certfile` /
`--ws_ssl_keyfile` to `speech-to-speech`. Log output should show:

```
[info] TLS enabled: cert=/home/.../models/tls/realtime-cert.pem
```

#### Start the demo UI with TLS

```bash
SPEECH_TO_SPEECH_URL="wss://192.168.178.101:8765/v1/realtime" \
  scripts/start_demo.sh
```

The script auto-detects certs and binds the demo UI on `https://0.0.0.0:7860`.

#### On the browser PC: trust the mkcert CA

**Quickest path** — open the install helper from the running demo (no scp needed):

1. Open `https://192.168.178.101:7860/`, click through the certificate warning.
2. Visit `https://192.168.178.101:7860/install-ca.html` — a step-by-step
   page with a big **Download rootCA.pem** button and per-browser
   instructions (Firefox, Chrome/Edge on Linux/macOS/Windows).
3. Import `rootCA.pem` into the browser (the helper page has the exact steps).
4. **Fully restart the browser** (quit all windows, reopen) — Firefox in
   particular does not pick up new CAs until a hard restart.

The same CA-download link also appears in two more places so you can grab
the file whenever you need it:

- The Settings dialog → below the speech-to-speech server URL field
- The red error banner that appears when a WebSocket connection fails with
  close code `1015` (TLS handshake error)

If the demo isn't reachable yet, copy the CA manually:

```bash
# On the Pi:
scp "$(.venv/bin/mkcert -CAROOT)/rootCA.pem" <browser-user>@<browser-ip>:/tmp/
```

Import `rootCA.pem` into the browser's trust store:

- **Chrome (Linux)**: `certutil -d sql:$HOME/.pki/nssdb -A -t "C,," -n mkcert -i /tmp/rootCA.pem`
- **Chrome (macOS)**: open `rootCA.pem` → "Keychain Access" → "System" → "Get Info" → Trust → "Always Trust"
- **Chrome (Windows)**: double-click → "Install Certificate" → "Local Machine" → "Trusted Root Certification Authorities"
- **Firefox (any OS)**: Settings → Privacy & Security → Certificates → "View Certificates" → Import → tick "Trust this CA to identify websites"

Restart the browser after importing the CA.

#### Browser test

On the browser PC:

```
https://192.168.178.101:7860/
```

`https://192.168.178.101:7860/` should now load without a "Not secure"
warning. Click the orb → browser asks for microphone permission → grant →
speak → hear the answer.

#### Fallback: SSH tunnel (no TLS required)

If the browser-PC import of the mkcert CA is troublesome (corporate
firewall blocks cert download, browser refuses to import, etc.):

```bash
# On the browser-PC:
ssh -L 7860:127.0.0.1:7860 -L 8765:127.0.0.1:8765 pi@192.168.178.101

# Then stop the demo and restart with the loopback URL so the WebSocket
# also resolves to localhost:
pkill -f 'uvicorn.*demo'
SPEECH_TO_SPEECH_URL="ws://localhost:8765/v1/realtime" \
  .venv/bin/uvicorn --app-dir demo server:app --host 127.0.0.1 --port 7860

# In the browser (same machine):
http://localhost:7860/
```

The browser sees `http://localhost:7860/` — localhost IS a secure context, so
getUserMedia() works without TLS.

## 5. Deployment options

The three options below are all supported and use the same process set. Choose based on your operational preferences — the launcher is the most portable.

| Option | Auto-restart | Log rotation | systemd integration | Setup cost |
|---|---|---|---|---|
| `scripts/rpi_start.sh` | ❌ (Ctrl-C stops all) | ❌ | ❌ | zero |
| `deploy/supervisord/speech-to-speech.conf` | ✅ | ✅ | ❌ | small |
| `deploy/systemd/*.service` | ✅ | ✅ (journal) | ✅ | small |

Detailed recipes live in [`deploy/README.md`](./deploy/README.md).

## 6. Environment variables

The setup script writes these into `models/.env`. All are exported via `set -a; source models/.env; set +a`.

| Variable | Set by | Purpose |
|---|---|---|
| `HF_HOME` | setup | Hugging Face cache root. Defaults to `<repo>/models/huggingface`. |
| `HUGGINGFACE_HUB_CACHE` | setup | `hf hub` cache. Subdirectory of `HF_HOME`. |
| `TRANSFORMERS_CACHE` | setup | `transformers` cache. Subdirectory of `HF_HOME`. |
| `HF_HUB_DISABLE_PROGRESS_BARS` | setup | Quiets download progress in the logs. |
| `MOONSHINE_MODEL_DIR` | setup | Where the moonshine-stt-server looks for weights. |
| `MOONSHINE_DEFAULT_MODEL` | setup | Default HF repo id for the STT server. |
| `MOONSHINE_STT_PORT` / `MOONSHINE_STT_BASE_URL` | setup | Port + base URL for the STT server. |
| `SUPERTONIC_MODEL_DIR` | setup | Where the supertonic-tts-server looks for weights. |
| `SUPERTONIC_DEFAULT_VOICE` | setup | Default voice id (`M1..M5` / `F1..F5`). |
| `SUPERTONIC_TTS_PORT` / `SUPERTONIC_TTS_BASE_URL` | setup | Port + base URL for the TTS server. |
| `LITERT_LM_MODEL_ALIAS` | setup (when `--llm-backend litert-lm`) | Alias of the imported LiteRT-LM model (e.g. `gemma4-e4b`). |
| `LITERT_LM_PORT` / `LITERT_LM_BASE_URL` | setup | Port + base URL for the LLM server (default port 9379). |
| `CACTUS_MODEL_PATH` | setup (when `--llm-backend cactus`) | Path to the Cactus Compute model directory. |
| `S2S_LLM_BACKEND` | setup | Which LLM runtime the launcher should start. |
| `S2S_WS_HOST` / `S2S_WS_PORT` | setup | Bind address for the realtime pipeline. |

`rpi_start.sh` is the only consumer that needs them in one shell. Once the servers are up, the env vars are no longer required by the pipeline itself — only by the launchers.

## 7. Troubleshooting

### `command not found` for `speech-to-speech`, `moonshine-stt-server`, `supertonic-tts-server`, `s2s-rpi-setup`

These are installed into `<repo>/.venv/bin/` by `uv sync`, but `.venv/bin` is not on your shell's `PATH` by default. Three equivalent fixes:

```bash
# Option A — invoke via uv (recommended for one-off commands)
uv run s2s-rpi-setup --help
uv run speech-to-speech --help

# Option B — activate the venv for the current shell
source .venv/bin/activate
s2s-rpi-setup --help

# Option C — prepend the venv to PATH
export PATH="$PWD/.venv/bin:$PATH"
s2s-rpi-setup --help
```

The bundled launcher (`scripts/rpi_start.sh`) auto-prepends `.venv/bin` to PATH at startup, so as long as the venv exists you do **not** need to activate manually before running it.

### "address already in use" on :9001 / :9002 / :9379 / :8765

```bash
./scripts/rpi_start.sh --stop
ss -lntp | grep -E '9001|9002|9379|8765'   # find what is still bound
```

### Moonshine server crashes on startup

Check the log: `tail -n 50 models/log/moonshine-stt.log`. Common causes:

- `MOONSHINE_MODEL_DIR` points to a non-existent path. Verify with `ls -la "$MOONSHINE_MODEL_DIR"`.
- The model download was incomplete. Re-run `hf download UsefulSensors/moonshine-base --local-dir "$MOONSHINE_MODEL_DIR"`.
- The first run downloads ~60 MB (base) or ~250 MB (streaming-medium) from `huggingface.co`. The launcher waits up to 180 s for the port to bind; if your network is slower, increase `wait_for_port` in `scripts/rpi_start.sh` (search for `wait_for_port 127.0.0.1 "$MOONSHINE_STT_PORT"`).

### Pipeline says "did not bind 127.0.0.1:9001 within 180s"

First-run downloads take longer than warm starts. Either:

- Wait for the model download to complete (it is cached in `HF_HOME`), then re-run the launcher.
- Pre-fetch with `hf download UsefulSensors/moonshine-base --local-dir "$MOONSHINE_MODEL_DIR"`.
- Bump the timeout (see above).

### Supertonic returns empty audio

- Confirm the model directory has the expected files: `ls "$SUPERTONIC_MODEL_DIR"` should show voice embeddings and ONNX files.
- Try with `response_format=wav` first; the WAV path skips the inline header and is easier to inspect with `file out.wav`.

### LLM responses time out

- `LITERT_LM_BASE_URL` should be the OpenAI-compat base, e.g. `http://127.0.0.1:9379/v1`. The `chat-completions` client appends `/chat/completions`. LiteRT-LM does **not** expose `/v1/responses`, so make sure `--llm_backend chat-completions` (not `responses-api`) is set.
- `curl http://127.0.0.1:9379/v1/models` should list `gemma4-e2b` (or `gemma4-e4b`) after `s2s-rpi-setup --fetch`. If the model isn't there, run `litert-lm list` to check the registry.
- Check `models/log/litert-lm.log` for the model load progress. Gemma 4 E2B on a Pi can take 30-60 s on first cold start; E4B takes 60-90 s.

### `command not found: litert-lm`

The CLI binary installed by `[rpi-litertlm]` is named `litert-lm`, not `litert-lm-server`. If the launcher prints this warning, `uv sync --extra rpi-litertlm` did not run or the venv is not on `PATH`. The bundled launcher auto-prepends `.venv/bin` to `PATH`, so this should resolve automatically once the venv exists.

### Pipeline says "All session slots are in use"

Only relevant in realtime mode with multiple parallel clients. Either close the extra tab or raise `--num_pipelines`.

### Demo UI gets "ws disconnected mid-send"

- The browser requires `localhost` or HTTPS for microphone access. Use `http://localhost:7860/`, or set up a TLS reverse proxy.
- Confirm the browser machine can reach the Pi: `nc -vz <pi-ip> 8765`.

## 8. Performance expectations

Indicative numbers on a Raspberry Pi 5 (8 GB, active cooler, Bookworm 64-bit, 2400 MHz):

| Stage | Latency (cold) | Latency (warm) | Notes |
|---|---|---|---|
| Moonshine-base STT | 200-400 ms | 80-150 ms | 16 kHz mono, ~5 s utterance |
| LiteRT-LM Gemma 4 E2B (CPU, 4 threads) | 30-60 s (load) + ~5-8 tok/s | ~5-8 tok/s | First generation is slow; subsequent turns warm the prompt cache |
| LiteRT-LM Gemma 4 E4B (CPU, 4 threads) | 60-90 s (load) + ~3-6 tok/s | ~3-6 tok/s | Higher quality, slower than E2B |
| Supertonic 3 TTS | 1-2 s (load) + ~150-300 ms per sentence | ~150-300 ms per sentence | |

Total conversational latency, end-to-end, after warm-up: **~1.5-2.5 s** for a short answer. For interactive dialogue, set `--responses_api_disable_thinking true` (default).

If latency is the bottleneck, lower `--num_pipelines 1` (already the default), drop `live_transcription_update_interval` to 0.3, and switch `moonshine-http` to `moonshine-base` (not `streaming-medium`).

## 9. Development workflow

```bash
uv sync --extra rpi --extra faster-whisper
pytest tests/ -x
ruff check src/
```

New code style rules: see `pyproject.toml`'s `[tool.ruff]` section (line length 120). The upstream conventions are kept: handler subclasses of `BaseHandler[TIn, TOut]` (or `BaseSTTHandler`), dataclass-based arguments under `arguments_classes/`, OpenAI-compat HTTP clients with the `openai` SDK.

## 10. What's intentionally out of scope

See `AGENTS.md` for the full list. Highlights:

- Republishing to PyPI or running `.github/workflows/publish.yml`.
- Editing anything under `archive/`.
- Pointing the LLM at OpenAI / HF Inference Providers / OpenRouter as a "temporary" fallback.
- Committing model weights, `models/`, `dist/`, `build/`, or `uv.lock`.

## 11. Hosting

This fork lives in a foreign repository. We move it to our own git before tagging.

```bash
# 1. Create the new repo on our host (e.g. gitlab.example.org/speech/speech-to-speech-rpi)
#    Make sure the default branch is `main` and the visibility is private/internal as needed.

# 2. From the working copy, re-target the remotes.
cd /path/to/speech-to-speech
git remote rename origin upstream              # the foreign fork, keep for syncs
git remote add origin git@<host>:<org>/speech-to-speech-rpi.git

# 3. Initial commit.
git add -A
git commit -m "Initial RPi fork: moonshine + litert-lm + supertonic"
git push -u origin main

# 4. Tag a release once stable.
git tag -a v0.1.0 -m "First working RPi port"
git push origin v0.1.0

# 5. To pick up upstream changes later (read-only sync):
git fetch upstream
git merge upstream/main --no-ff -m "Merge upstream main"
```

Branch protection: enable on `main` in the new repo. CI is intentionally minimal — see "Testing" below; we don't run the upstream pytest suite as a gating check because it imports CUDA-specific code paths we no longer ship.

## 12. Testing

```bash
pytest tests/ -x -q                      # upstream tests; some will skip without MLX
ruff check src/                           # lint
s2s-rpi-setup --help                      # smoke: CLI is wired
moonshine-stt-server --help
supertonic-tts-server --help
```

End-to-end smoke (after a real install with all weights):

```bash
./scripts/rpi_start.sh &
sleep 20
curl -fsS http://127.0.0.1:9001/health
curl -fsS http://127.0.0.1:9002/health
curl -fsS http://127.0.0.1:9379/v1/models
python scripts/listen_and_play_realtime.py --host 127.0.0.1 --port 8765
```

When we have a CI runner that can install `litert-lm` and `moonshine`, this becomes a real CI workflow.

## 13. Migration notes

When lifting this into our own git:

- Replace `<your-org>` in `pyproject.toml` `[project.urls]`, README, and this manual.
- Replace the git URL in the README quickstart.
- Drop `archive/` from the initial commit if you want a smaller history (we keep it for parity with upstream).
- Decide on a license footer; the inherited Apache-2.0 stays.
- Decide whether the new repo is public or private; the upstream fork is Apache-2.0, so a public mirror is legally fine but we default to internal.

---

Last touched: see `git log --oneline -1` on the file `MANUAL.md`.
