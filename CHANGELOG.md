# Changelog

All notable changes to the RPi fork are recorded here. The format is
loose — sections appear as topics arise. Newest entries on top.

The "Backlog" section at the bottom tracks open work.

---

## [unreleased] — PoC milestone, Moonshine → faster-whisper switch, LiteRT-LM MTP wired in

### Theme: LLM round-trip latency — per-session Conversation pool + extra_body flatten fix

The big win this milestone: the LLM is no longer the bottleneck on
warm turns. Two bugs were stacked on top of each other.

- **Added** per-session `ConversationPool` in
  `src/servers/litert_lm_mtp_server.py`. Each WebSocket session gets
  its own persistent `litert_lm.Conversation`; the KV-cache survives
  across turns so each new user message only decodes the new tokens,
  not the full history. Idle sessions evict after
  `--idle-eviction-seconds` (default 30 min). One-shot calls without
  a session ID (e.g. the pipeline's startup warmup probe) still work
  and just don't share KV-cache with anyone.
- **Added** real cancellation: when a new request arrives for an
  existing session that is still generating, the previous
  `Conversation.cancel_process()` is called. This is the C++-side
  interrupt (LiteRT-LM logs `SessionAdvanced::CancelProcess`),
  not a `httpx`-stream abort on the client side. Cancelled
  conversations are dropped and replaced, because LiteRT-LM's
  `cancel_process()` during the prefill phase leaves the
  conversation in a state where the next `send_message` raises
  `Session is not prefilled yet`.
- **Fixed** `ChatCompletionsLanguageModel` (`chat_completions_language_model.py`)
  merges `session_id` (sourced from
  `runtime_config.session_id`, which `RealtimeService.register()`
  now writes into) into the OpenAI request's `extra_body`.
- **Fixed** `src/servers/litert_lm_mtp_server.py` reads `session_id`
  from either `req["extra_body"]` *or* top-level `req`. The OpenAI
  Python SDK 1.x flattens `extra_body` into the top-level JSON body
  on the wire (no wrapper key), so without this the server saw
  everything but the session id.
- **Added** `RuntimeConfig.session_id: str | None` field;
  `RealtimeService.register()` writes
  `state.runtime_config.session_id = state.session_id` once per
  WS connection so the value is visible to every handler that
  reads `runtime_config`.
- **Measured** on Pi 5 with Gemma 4 E2B (10 user turns in one WS
  session, "rocky / robert"-style name-following):

  | | Before pool | After pool |
  |---|---|---|
  | Turn 1 (cold) | ~13 s | ~13 s (conv creation + first prefill) |
  | Turn 2-10 (warm) | ~13 s each | **2.9–4.3 s each** |
  | Total over 10 turns | ~130 s | ~45 s |
  | Pool stats (live) | n/a | `total_created=2, total_reused=10` |

  MTP is still wired in but contributes no measurable speedup on
  Pi-5-CPU (see entry below); the dominant win here is KV-cache
  reuse.

### Bug: stale `.pyc` cache shipped an old `service.register()`

- **Fixed** the symptom of "history still broken after restart":
  the project's `__pycache__/*.pyc` files from a 17:56 build still
  contained the pre-`session_id` version of
  `RealtimeService.register()`. A fresh `import` recompiled from
  source and worked, but long-running pipeline/server processes
  kept loading the cached module. The fix is to delete
  `__pycache__/` before restarting; documented in AGENTS.md
  "Hard-won lessons" so the next agent doesn't repeat the
  30-minute loop.

### Theme: LLM inference — MTP wiring

- **Added** `src/servers/litert_lm_mtp_server.py`: a FastAPI wrapper
  around the LiteRT-LM Python SDK that enables Multi-Token
  Prediction (`Engine.enable_speculative_decoding`) and exposes an
  OpenAI-compatible `/v1/chat/completions` endpoint. Necessary because
  the bundled `litert-lm serve` CLI does not expose the
  `--enable-speculative-decoding` flag, so MTP cannot be enabled in
  the standard server. The wrapper also keeps the model in-process
  across requests, eliminating the ~24 s cold-cache compile penalty
  that `litert-lm serve` incurs on every fresh request.
- **Changed** `scripts/start_litert_lm.sh`: now starts the new
  MTP-aware server. `LITERT_LM_MODEL_PATH` (added to `models/.env`)
  controls which `.litertlm` file is loaded; falls back to the default
  registry layout under `~/.litert-lm/models/`. Set
  `LITERT_LM_DISABLE_MTP=1` in the env to disable MTP without code
  changes.
- **Added** `litert-lm-mtp-server` console-script entry point in
  `pyproject.toml`; re-run `uv pip install -e .` after pulling.
- **Changed** `src/s2s_rpi/setup.py`: writes `LITERT_LM_MODEL_PATH`
  alongside `LITERT_LM_MODEL_ALIAS` so subsequent runs are
  reproducible.
- **Measured** on Pi 5 with Gemma 4 E2B, "List 3 fun facts about Mars"
  prompt (~50-token reply): MTP off ~8.5 s warm, MTP on ~8.4 s warm.
  **No measurable speedup on Pi-5-CPU** (Google's "up to 3×" claim is
  for mobile GPUs). MTP is wired in for future GPU migration
  (Reachy Mini, NPU add-ons) where it pays off; on Pi 5 today the
  effective gain is the eliminated cold-start (24 s → ~4 s for the
  first request).

## [unreleased] — PoC milestone, Moonshine → faster-whisper switch

### Theme: project framing

- New repository section in `README.md` ("Why this fork exists")
  explaining the motivation and listing the proven Pi stack with
  measured latencies (VAD / STT / LLM / TTS).
- New `CHANGELOG.md` (this file) so the next maintainer can follow
  what changed and why without git archaeology.
- New `AGENTS.md` structure aimed at getting a coding agent
  productive in 60 s: what this is, the proven stack, quickstart,
  where things live, backends per stage, hard-won lessons, backlog
  pointer, out-of-scope rules.

### Theme: STT backend

- **Switched default STT from `moonshine-http` to `faster-whisper`.**
  Moonshine was found to hallucinate on real-world audio — emitting
  "Hello." for a user saying "hello, how are you" even after the
  full audio reached the model. The seq2seq architecture has documented
  repetition issues on short segments; `faster-whisper base.en` with
  int8 quantization gives clean transcripts at acceptable latency
  (~14 s for 5 s audio on Cortex-A76).
- `scripts/rpi_start.sh` now picks `faster-whisper` automatically;
  the launcher only starts the `moonshine-stt-server` subprocess
  when `--stt moonshine-http` is requested (in-process STT otherwise).
- Moonshine remains available as opt-in; `moonshine-streaming-medium`
  weights stay downloaded for that path.
- Stopped the always-on `moonshine-stt-server` process; port 9001 is
  free unless explicitly enabled.

### Bug: audio truncation on every turn ("Hello." for "hello, how are you")

- **Root cause**: `enable_live_transcription=True` was implicitly
  forcing `enable_realtime_transcription=True` in
  `s2s_pipeline.py:556` and `:743`. That made the VAD emit
  progressive 200–800 ms audio chunks to STT while the user was still
  speaking. Moonshine (non-streaming) could only return 1–2-word
  hallucinations from those slices, which then raced to the client as
  the "user transcript" via `conversation.item.input_audio_transcription.completed`.
- **Fix**: decoupled the two flags. Added `--enable_vad_realtime`
  (default `false`) so the VAD only emits progressive chunks when
  explicitly requested. STT now waits for the soft-end silence
  (900 ms) and receives the full turn audio at once.
- Hardened the `MoonshineHttpSTTHandler` to drop any `mode="progressive"`
  chunks as a defensive net for future STT configurations.
- Verified with `scripts/_ws_pipeline_probe.py`: a 4.96 s WAV now
  reaches Moonshine with the full 4.96 s of audio (was 0.82 s before
  the fix); faster-whisper then returns the correct transcript
  end-to-end.

### Bug: mic echo loop during TTS playback

- **Root cause**: With consumer speakers, the Pi's own TTS audio was
  re-captured by the mic, triggering new VAD turns ("Pipeline 0:
  speech during response: cancelled, queue flushed") and infinite
  STT → LLM-cancel → STT cycles.
- **Fix**: half-duplex in the browser. When the WS client detects the
  AI speaking (`response.output_audio.delta` events), it sets
  `track.enabled = false` on the mic MediaStream; on `response.done`
  it re-enables. The manual mute button keeps user priority (user mute
  wins over the AI-driven mute).
- VAD silence threshold raised from 64 ms (upstream default) to
  900 ms; threshold from 0.6 to 0.7; speech_pad from 0 to 500 ms;
  reopen windows extended (`speculative_reopen_ms` 1000 → 1500,
  `unanswered_reopen_ms` 7000 → 9000) so natural mid-sentence pauses
  (typical 600–800 ms) don't cut the turn.

### Bug: server missing `session.updated` ack

- **Symptom**: After `session.update`, the OpenAI Realtime client
  (the upstream demo) didn't receive the `session.updated` event back
  and timed out on `session.update → next event` waits.
- **Fix**: `websocket_router.py` now sends `session.updated` after a
  successful `session.update`. Upstream code forgot this; the patch
  is one local change to `api/openai_realtime/websocket_router.py`.

### Feature: TLS for the browser-side mic API

- Browsers refuse `getUserMedia()` on plain HTTP LAN origins. Added a
  mkcert-based self-signed CA pipeline: certs generated at
  `models/tls/realtime-cert.pem` covering the Pi's LAN IP + localhost;
  the launcher passes `--ws_ssl_certfile / --ws_ssl_keyfile` to the
  pipeline and to the demo UI.
- The demo serves `/rootCA.pem` directly so the user can download
  and trust the CA in Firefox/Chrome without external tooling.

### Feature: debuggability

- New `scripts/_ws_pipeline_probe.py` exercises the full WS round-trip
  with a known WAV (default: 5 s Watson reference, real human speech).
  It reports audio length actually received by STT, transcript events,
  and server-side timing. Caught the audio-truncation bug above by
  showing the STT call received only 0.82 s of audio instead of 5 s.
- `scripts/verify_rpi_fork.sh` runs the probe plus per-server health
  checks and prints a pass/fail summary.

### Theme: packaging and ops

- New launcher trio: `scripts/rpi_start.sh`, `scripts/rpi_stop.sh`,
  `scripts/rpi_status.sh` with PID/log/run-dir conventions under
  `models/run/` and `models/log/`.
- New `s2s-rpi-setup` CLI (`src/s2s_rpi/setup.py`) does model fetch
  + env-file generation. Content-compare on `models/.env` so re-running
  doesn't churn the file when nothing changed.
- New `pyproject.toml` extras: `[rpi]` umbrella, plus per-backend
  extras (`rpi-moonshine`, `rpi-litertlm`, `rpi-cactus`,
  `rpi-supertonic`, `faster-whisper`).
- `src/servers/moonshine_stt_server.py`: OpenAI-compatible wrapper
  around `transformers.MoonshineForConditionalGeneration` /
  `MoonshineStreamingForConditionalGeneration`. Token cap raised from
  6.5 to 13 tokens/s of audio (model card's recommendation caps too
  tightly for conversational turns).
- `src/servers/supertonic_tts_server.py`: OpenAI-compatible wrapper
  around the `supertonic` Python package.
- New WS clients: `STT/moonshine_http_handler.py`,
  `TTS/supertonic_http_handler.py`, plus matching `*_arguments.py`
  dataclasses registered in `module_arguments.py`.

### Theme: cleanup

- Dropped unused `deploy/` (systemd units + supervisord config +
  README) — replaced by `scripts/rpi_start.sh`. The new launcher is
  simpler and integrates with the per-process PID/log conventions
  used everywhere else.
- Dropped orphan `demo/install-ca.html` — superseded by the demo's
  in-app error banner download link (`/rootCA.pem`).
- Dropped `default.profraw` (stale LLVM profiling artifact).
- `.gitignore` now excludes `*.profraw` / `*.profdata` to keep the
  repo clean from cargo-pgo / llvm-profdata outputs going forward.

### Theme: docs

- New `MANUAL.md` — installation, TLS setup, model storage convention,
  pipeline tuning, troubleshooting, smoke check.
- Rewrote `AGENTS.md` and `README.md` for the RPi context (the
  upstream instructions don't apply).

---

## Backlog

### LLM is the round-trip bottleneck (largely resolved)

As of the Conversation-Pool fix (see top of [unreleased]),
warm-turn LLM latency dropped from ~13 s to ~3–4 s on Pi 5
Gemma 4 E2B. The LLM is no longer the dominant term in the
~30 s round-trip; STT (~14 s for 5 s audio with faster-whisper
base.en int8) is now the slow stage. Possible next levers:

- **Smaller model** for headroom: `gemma3-1b-it` (1B) via
  LiteRT-LM halves the remaining 3-4 s and also reduces power.
- **STT replacement**: `nvidia/parakeet-tdt-0.6b-v3` via
  `onnx-asr` is faster + gives partial transcription (see
  "STT parity" below).
- **llama.cpp** as a second LLM backend (see "Reference: LLM
  engines considered"): broader model coverage, mature CPU
  path, OpenAI-compatible `llama-server`. Not implemented yet
  because LiteRT-LM + Conversation-Pool is fast enough on
  Gemma 4 E2B. Add it when a non-Gemma model is needed or
  when tooling around Gemma-3 1B / Llama-3.2 1B variants
  becomes useful. Cost: one new `scripts/start_llama_cpp.sh`
  + GGUF download hook in `s2s-rpi-setup` (a few hours).
- **Cactus Compute runtime** (`uv sync --extra rpi-cactus`) —
  designed for edge ARM, often faster than LiteRT-LM on
  Cortex-A76.

### Reference: LLM engines considered

Quick comparison of the engines evaluated for this fork
(currently only LiteRT-LM is wired in):

| Engine | 1–2B model on Pi 5 | Tool calling | OpenAI-Compat | Worth adding? |
|---|---|---|---|---|
| **LiteRT-LM** (current) | Gemma 4 E2B: ~7.6 t/s decode (HF model card, 4 threads, XNNPACK) | Function-calling API + dedicated `FunctionGemma` | `litert-lm serve` since v0.13 | yes, already done |
| **llama.cpp** | no official Pi-5 numbers; community estimates 12–18 t/s for 1B Q4_K_M | GBNF grammar-constrained, `llama-server` | yes (`llama-server`) | yes — backlog candidate |
| **mistral.rs** | no Pi-5 numbers; similar to llama.cpp | strict-schema mode, server-side agentic loop | yes + Anthropic-Messages-Compat | maybe (best tool-calling story, less Pi 5 focus) |
| **Cactus Compute** | no ARM numbers; extrapolated 5–8 t/s for CQ4 Gemma 4 E2B from Mac M-series | own 26 M "Needle" distilled model | yes (`cactus serve`) | research only |
| **vLLM** | not viable on Pi 5 (no BF16, ≥3 GB PyTorch overhead) | yes | yes | no (wrong architecture for edge) |
| **LiteLLM** | n/a (proxy only) | n/a | is OpenAI-Compat | no (violates local-only) |

Full analysis with sources lives in the project history (search for
"Stufe 1/2/3" — the conversation that produced this section). The
short version: LiteRT-LM is the right choice for now; llama.cpp is
the fallback when a non-Gemma model is needed.

### Bug: `response.create` race

Pipeline log shows the LLM being POSTed 0.5 s after `session.update`,
before the VAD has had time to soft-end and the STT has produced a
transcript. Symptom in the probe: `audio=0.00 s`, LLM runs on empty
input, eventually times out. Has not been reproduced in the browser
session yet — keep an eye on it.

### STT parity: parakeet-tdt

`nvidia/parakeet-tdt-0.6b-v3` via `onnx-asr` (pure-Python, no PyTorch
needed on the STT side) is on deck as a follow-up. Slightly better
WER than faster-whisper base.en, plus true partial-transcription
support — would also enable a real-time transcript overlay in the
demo UI.

### Echo cancellation

Half-duplex works but feels clunky (mic truly off while AI speaks).
Real acoustic echo cancellation on the Pi's audio capture would let
us keep the mic open and use speakers without feedback loops.
Software options: `speexdsp`, `webrtc-audio-processing`.

### Live transcription

`--enable_live_transcription` is set but no current STT backend
streams partials. Parakeet-TDT or a streaming Whisper backend would
unblock the live caption overlay in the demo.
