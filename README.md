# speech-to-speech — Raspberry Pi fork

> [!NOTE]
> **Fork notice.** This repository is a fork of [`huggingface/speech-to-speech`](https://github.com/huggingface/speech-to-speech), maintained at **ScaDS.AI Living Lab** (Universität Leipzig) and re-wired for fully-local inference on a Raspberry Pi 5. All upstream code is © The HuggingFace Inc. team, licensed under the Apache License 2.0 — see [`LICENSE`](./LICENSE). The list of changes lives in the **What changed vs upstream** section below and in [`CHANGELOG.md`](./CHANGELOG.md).

Fully-local, voice-agent pipeline for the Raspberry Pi. VAD → STT → LLM → TTS, with every model served from the same machine and every stage talking OpenAI-compatible protocols.

> **Heads-up.** This fork is trimmed and re-wired for a CPU-first Pi. The upstream codebase remains the source of truth for the realtime WebSocket server, the OpenAI Realtime protocol implementation, and the demo UI; we add a swappable in-process HTTP-client layer for STT and TTS so each model can run in its own process.

---

## Why this fork exists

End-to-end voice assistant that runs entirely on a Raspberry Pi 5.
No cloud, no hosted inference, no telemetry. OpenAI Realtime WebSocket
on top, swappable backends per stage. Target: interactive voice
conversation where the LLM is the only component where you can hear the
Pi work — STT, VAD, TTS are all sub-second on a Cortex-A76.

Proven stack (July 2026, RPi 5 / Bookworm 64-bit / Cortex-A76):

| Stage | Backend | Latency on Pi CPU |
|-------|---------|-------------------|
| VAD | Silero VAD v5 | < 50 ms |
| STT | `faster-whisper base.en` (CTranslate2 int8, in-process) | ~14 s for 5 s audio |
| LLM | LiteRT-LM Gemma 4 E2B (XNNPack) + per-session KV-cache pool | ~13 s cold, **~3–4 s warm** |
| TTS | Supertonic 3 (ONNX) | ~1 s |

Total round-trip ≈ 20 s (down from ≈30 s before the LLM
Conversation-Pool fix). **STT is now the dominant slow stage** —
see [`CHANGELOG.md`](./CHANGELOG.md) → "Backlog" for the list of
open work (parakeet-TDT for faster STT + partials, 1B-class
LLM as a fallback, real echo cancellation, live-transcription
support).

---

## What ships here

| Stage | Default RPi backend | OpenAI-compat surface | Notes |
|---|---|---|---|
| VAD | Silero VAD v5 | — | Built-in, CPU-only. |
| STT | [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) `base.en` (`faster-whisper`, in-process) | `POST /v1/audio/transcriptions` (own process if `--stt moonshine-http`) | CTranslate2 int8. Moonshine available as opt-in fallback via `--stt moonshine-http` (server in `servers/moonshine_stt_server.py`, runs on `:9001`). |
| LLM | [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) Gemma 4 E2B (`chat-completions`) | `POST /v1/chat/completions` | No in-process code change; we point the existing client at `litert-lm serve` on `:9379`. Model is imported via `litert-lm import` during setup. |
| TTS | [Supertonic 3](https://github.com/supertone-inc/supertonic) (`supertonic-http`) | `POST /v1/audio/speech` | Server in `servers/supertonic_tts_server.py`, runs on `:9002`. Uses the official `supertonic` Python package. |
| Realtime | OpenAI Realtime WebSocket | `ws://host:8765/v1/realtime` | Reused from upstream; unchanged. |

Cactus Compute runtime is an alternative to LiteRT-LM (`uv sync --extra rpi-cactus`).

## Quickstart (Raspberry Pi 5, Bookworm 64-bit)

```bash
git clone <this-repo>
cd <this-repo>

uv sync --extra rpi              # pulls faster-whisper, litert-lm, supertonic (moonshine opt-in)
uv run s2s-rpi-setup --fetch     # creates models/, downloads weights, writes models/.env
                                 # (uses the venv-managed entry point)

source models/.env
./scripts/rpi_start.sh           # foreground; Ctrl-C stops everything
# or:
./scripts/rpi_start.sh --bg      # daemonised; PIDs in models/run/, logs in models/log/

# From a second terminal (or another machine on the LAN):
curl -sk https://<pi-ip>:7860/api/config    # demo UI config (should 200)
curl -sk https://<pi-ip>:8765/v1/pool       # pipeline health
./scripts/verify_rpi_fork.sh               # full smoke test
```

The browser-side demo UI is at `https://<pi-ip>:7860/`. The mkcert root
CA lives at `models/tls/rootCA.pem` — import it into the browser's
trust store once, then a hard reload of the page is enough.

> `uv run <cmd>` invokes the entry points installed under `.venv/bin/`. The bundled launcher auto-prepends `.venv/bin` to `PATH`, so once `uv sync` has run, `./scripts/rpi_start.sh` works from any shell without further activation.

`models/.env` (created by `s2s-rpi-setup`) exports the required environment variables:

```
HF_HOME=<repo>/models/huggingface
HUGGINGFACE_HUB_CACHE=<repo>/models/huggingface/hub
TRANSFORMERS_CACHE=<repo>/models/huggingface
MOONSHINE_MODEL_DIR=<repo>/models/moonshine
MOONSHINE_DEFAULT_MODEL=UsefulSensors/moonshine-base        # only used if --stt moonshine-http
SUPERTONIC_MODEL_DIR=<repo>/models/supertonic
SUPERTONIC_DEFAULT_VOICE=M1
LITERT_LM_MODEL_ALIAS=gemma4-e2b          # alias inside litert-lm's local registry
MOONSHINE_STT_BASE_URL=http://127.0.0.1:9001/v1
SUPERTONIC_TTS_BASE_URL=http://127.0.0.1:9002/v1
LITERT_LM_BASE_URL=http://127.0.0.1:9379/v1   # litert-lm's documented default port
```

Nothing lands in `~/.cache/huggingface`; the repo stays self-contained.

## Architecture

```
+------------------+    ws://host:8765/v1/realtime    +-----------------------+
|   Browser / UI   | <--------------------------------> | speech-to-speech      |
|   (demo/)        |                                    | realtime server       |
+------------------+                                    +--------+-----+--------+
                                                               |     |     |
                              +--------------------------------+     |     +-----------------------------+
                              | in-process OpenAI-HTTP-Client       |            in-process OpenAI-HTTP-Client
                              v                                      v                                          v
                     +-------------------+                +-----------------------+                      +--------------------+
                     | Moonshine STT     |                | LiteRT-LM             |                      | Supertonic 3 TTS   |
                     | HTTP server       |                | Gemma 4 server        |                      | HTTP server        |
                     | :9001             |                | :9379                 |                      | :9002              |
                     | /v1/audio/        |                | /v1/chat/completions  |                      | /v1/audio/speech   |
                     |   transcriptions  |                | (imported via         |                      |                    |
                     | (only if          |                | `litert-lm import`)   |                      |                    |
                     |  --stt moonshine-http)              |                        |                      |                    |
                     +-------------------+                +-----------------------+                      +--------------------+
```

Default: STT (`faster-whisper`) runs in-process in the pipeline. The Moonshine STT subprocess on `:9001` only starts when `--stt moonshine-http` is requested. TTS subprocess on `:9002` and LLM subprocess on `:9379` always run. All subprocesses bound to `127.0.0.1` by default.

## What changed vs. upstream

- `pyproject.toml` slimmed: CUDA-only and macOS-only wheel markers are gone. New extras: `[rpi]`, `[rpi-moonshine]`, `[rpi-litertlm]`, `[rpi-cactus]`, `[rpi-supertonic]`, plus a fallback `[faster-whisper]`.
- New in-process handlers: `MoonshineHttpSTTHandler`, `SupertonicHttpTTSHandler`. Each is a thin OpenAI-client wrapper around its corresponding model-server process.
- New model-server processes in `src/servers/` (FastAPI): `moonshine-stt-server`, `supertonic-tts-server`. Both installable as console scripts.
- New launcher trio: `scripts/rpi_start.sh` (foreground / `--bg`), `scripts/rpi_stop.sh`, `scripts/rpi_status.sh` with PID/log/run-dir conventions under `models/run/` and `models/log/`.
- `models/.env` writer `s2s-rpi-setup` for reproducible installs.
- Half-duplex mic in the browser: `track.enabled = false` while TTS is playing, re-enabled on `response.done`. Fixes the TTS-echo → runaway-VAD loop.
- VAD tuning for Pi speakers + half-duplex mic: silence threshold 64 ms → 900 ms; threshold 0.6 → 0.7; speech_pad 500 ms; extended reopen windows.
- TLS via mkcert so the browser's `getUserMedia()` works on the LAN.
- Decoupled `--enable_realtime_transcription` from `--enable_live_transcription` (was forcing VAD progressive emission with non-streaming STT — caused the "Hello." for "hello, how are you" symptom).
- `models/`, `dist/`, `build/`, `uv.lock`, `*.profraw`, `*.profdata` are gitignored. The PyPI publish workflow is intentionally not ported — see `AGENTS.md` for why.

The realtime WebSocket server, the OpenAI Realtime protocol implementation, and the demo UI are **unchanged** from upstream (with one local patch: `websocket_router.py` sends `session.updated` after a successful `session.update`, which the upstream code forgot).

## Verification

```bash
# 1. Setup commands available
uv run s2s-rpi-setup --help

# 2. LiteRT-LM alone (always running)
curl -X POST http://127.0.0.1:9379/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-e2b","messages":[{"role":"user","content":"Hi"}]}'

# 3. Supertonic TTS alone
curl -X POST http://127.0.0.1:9002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"hello","voice":"M1","response_format":"wav"}' --output /tmp/out.wav

# 4. Full pipeline (after launcher is up)
./scripts/verify_rpi_fork.sh

# 5. Manual WS round-trip with a known WAV (diagnostics)
./scripts/_ws_pipeline_probe.py /path/to/test.wav
```

## Documentation

- [`MANUAL.md`](./MANUAL.md) — full architecture, deployment options (launcher), troubleshooting, manual run commands.
- [`AGENTS.md`](./AGENTS.md) — repo instructions for AI agents working on this fork (onboarding: what this is, where things live, hard-won lessons, backlog pointer).
- [`CHANGELOG.md`](./CHANGELOG.md) — project history by theme (project framing, STT switch, bug fixes, TLS, packaging, cleanup), plus the active backlog.
- [`demo/README.md`](./demo/README.md) — browser UI setup (upstream docs).

## License

Apache-2.0 (inherited from upstream). See [`LICENSE`](./LICENSE).
