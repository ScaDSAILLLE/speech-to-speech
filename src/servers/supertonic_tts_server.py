"""Supertonic 3 TTS HTTP server.

Exposes the OpenAI-compatible `POST /v1/audio/speech` endpoint, backed by
the `supertonic` Python package (v1.3.x). Supertonic loads its model and
voice styles from the Hugging Face Hub on first use; with `HF_HOME` pointed
at `<repo>/models/huggingface`, every download lands in the repo and not in
`~/.cache`.

Request body (JSON, OpenAI spec):

    {
      "model": "supertonic-3",        # optional, default "supertonic-3"
      "input": "Hello world",          # required, text to synthesise
      "voice": "M1",                   # required, voice id (M1..M5 / F1..F5)
      "response_format": "pcm",        # "pcm" (raw int16 mono) or "wav"
      "sample_rate": 16000,            # optional, default 16000 (resampled from native 44100)
      "speed": 1.05,                   # optional, default 1.05
      "lang": "en"                     # optional, one of supertonic.AVAILABLE_LANGUAGES
    }

Response: audio bytes. When `stream=true` is sent, the response is sent as
chunked transfer-encoding with the same sample format.

Environment variables consumed if flags are omitted:

  SUPERTONIC_MODEL_DIR     HF cache (HF_HOME is honoured as well)
  SUPERTONIC_DEFAULT_MODEL  default model id (default: supertonic-3)
  SUPERTONIC_DEFAULT_VOICE  default voice id (default: M1)
  SUPERTONIC_THREADS        intra_op_num_threads for ONNX runtime
  SUPERTONIC_TTS_PORT       TCP port
"""

from __future__ import annotations

import argparse
import io
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("supertonic-tts-server")


# Voices known to ship with `Supertone/supertonic-3` (verified against the HF
# repo). `get_voice_style` lazily downloads missing styles to `voice_styles/`
# under the model cache.
KNOWN_VOICES: tuple[str, ...] = (
    "M1", "M2", "M3", "M4", "M5",
    "F1", "F2", "F3", "F4", "F5",
)


class SpeechRequest(BaseModel):
    model: str = Field(default="supertonic-3")
    input: str = Field(..., min_length=1)
    voice: str | None = None
    response_format: str = Field(default="pcm")
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    speed: float = Field(default=1.05, gt=0.0, le=4.0)
    lang: str | None = None
    stream: bool = Field(default=False)


class TTSBackend:
    """Lazy wrapper around the `supertonic` package."""

    def __init__(
        self,
        model_name: str,
        model_dir: Path | None,
        threads: int | None,
    ) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.threads = threads
        self._tts: Any = None
        self._voice_cache: dict[str, Any] = {}

    def warmup(self) -> None:
        if self.model_dir is not None:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.model_dir))
            logger.info("HF_HOME -> %s", self.model_dir)

        import supertonic  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {"model": self.model_name, "auto_download": True}
        if self.model_dir is not None:
            # Keep the model cache under the repo instead of ~/.cache/supertonic3.
            # HF_HOME only redirects HuggingFace Hub downloads, not the
            # supertonic-internal cache path.
            kwargs["model_dir"] = str(self.model_dir)
        if self.threads is not None and self.threads > 0:
            kwargs["intra_op_num_threads"] = self.threads
        logger.info("Loading Supertonic model %s (threads=%s)", self.model_name, self.threads)
        self._tts = supertonic.TTS(**kwargs)
        logger.info("Supertonic ready, native sample_rate=%d Hz", self._tts.sample_rate)

    def list_voices(self) -> list[str]:
        return list(KNOWN_VOICES)

    def _get_voice(self, voice: str) -> Any:
        assert self._tts is not None, "backend not warmed up"
        if voice not in self._voice_cache:
            self._voice_cache[voice] = self._tts.get_voice_style(voice)
        return self._voice_cache[voice]

    def synthesize(
        self,
        text: str,
        voice: str,
        target_rate: int,
        speed: float,
        lang: str | None,
    ) -> np.ndarray:
        assert self._tts is not None, "backend not warmed up"
        vs = self._get_voice(voice)
        wav, _durations = self._tts.synthesize(
            text=text,
            voice_style=vs,
            speed=speed,
            total_steps=8,
            lang=lang,
            verbose=False,
        )
        # wav has shape (1, samples), float32 in [-1, 1]. Squeeze to 1-D.
        samples = np.asarray(wav).reshape(-1).astype(np.float32, copy=False)

        native_rate = int(self._tts.sample_rate)
        if target_rate != native_rate:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(native_rate, target_rate)
            samples = resample_poly(samples, target_rate // g, native_rate // g).astype(np.float32, copy=False)
        return samples

    def stream_chunks(
        self,
        text: str,
        voice: str,
        target_rate: int,
        speed: float,
        lang: str | None,
        blocksize: int,
    ) -> "AsyncIterator[bytes]":
        samples = self.synthesize(text, voice, target_rate, speed, lang)
        pcm = _float_to_int16_bytes(samples)
        # 16-bit mono PCM: blocksize samples == blocksize*2 bytes.
        byte_blocksize = blocksize * 2
        for offset in range(0, len(pcm), byte_blocksize):
            yield pcm[offset : offset + byte_blocksize]


def _float_to_int16_bytes(samples: np.ndarray) -> bytes:
    arr = np.clip(samples, -1.0, 1.0)
    pcm = (arr * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(_float_to_int16_bytes(samples))
    return buf.getvalue()


def create_app(backend: TTSBackend, default_voice: str, blocksize: int = 512) -> FastAPI:
    app = FastAPI(title="Supertonic 3 TTS (OpenAI-compat)", version="0.2.0")

    @app.on_event("startup")
    def _startup() -> None:
        backend.warmup()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "default_voice": default_voice}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": "supertonic", "object": "model", "owned_by": "Supertone"},
                {"id": "supertonic-2", "object": "model", "owned_by": "Supertone"},
                {"id": "supertonic-3", "object": "model", "owned_by": "Supertone"},
            ],
        }

    @app.get("/v1/audio/voices")
    async def list_voices() -> dict[str, Any]:
        return {"voices": backend.list_voices()}

    @app.post("/v1/audio/speech")
    async def synthesize(req: SpeechRequest) -> Response:
        voice = req.voice or default_voice
        try:
            samples = backend.synthesize(
                text=req.input,
                voice=voice,
                target_rate=req.sample_rate,
                speed=req.speed,
                lang=req.lang,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("supertonic synthesis failed")
            raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc

        if req.response_format == "wav":
            wav_bytes = _samples_to_wav_bytes(samples, req.sample_rate)
            return Response(content=wav_bytes, media_type="audio/wav")

        if req.response_format == "pcm":
            return Response(
                content=_float_to_int16_bytes(samples),
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": str(req.sample_rate),
                    "X-Sample-Format": "int16",
                },
            )

        raise HTTPException(
            status_code=400,
            detail=f"unsupported response_format {req.response_format!r}; use 'pcm' or 'wav'",
        )

    @app.post("/v1/audio/speech/stream")
    async def synthesize_stream(req: SpeechRequest) -> StreamingResponse:
        voice = req.voice or default_voice
        return StreamingResponse(
            backend.stream_chunks(
                req.input, voice, req.sample_rate, req.speed, req.lang, blocksize
            ),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(req.sample_rate),
                "X-Sample-Format": "int16",
            },
        )

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="supertonic-tts-server")
    p.add_argument("--host", default=os.environ.get("SUPERTONIC_TTS_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("SUPERTONIC_TTS_PORT", "9002")))
    p.add_argument(
        "--model",
        default=os.environ.get("SUPERTONIC_DEFAULT_MODEL", "supertonic-3"),
        help="Supertonic model id. One of 'supertonic', 'supertonic-2', 'supertonic-3'.",
    )
    p.add_argument(
        "--default-voice",
        default=os.environ.get("SUPERTONIC_DEFAULT_VOICE", "M1"),
        help="Voice id used when a request omits `voice`. One of M1..M5 / F1..F5.",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ["SUPERTONIC_MODEL_DIR"]) if os.environ.get("SUPERTONIC_MODEL_DIR") else None,
        help="Directory for HF cache. Defaults to $SUPERTONIC_MODEL_DIR.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=int(os.environ["SUPERTONIC_THREADS"]) if os.environ.get("SUPERTONIC_THREADS") else None,
        help="intra_op_num_threads for ONNX Runtime. Default: auto.",
    )
    p.add_argument("--blocksize", type=int, default=512, help="PCM block size for streaming responses.")
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    backend = TTSBackend(
        model_name=args.model,
        model_dir=args.model_dir,
        threads=args.threads,
    )
    app = create_app(backend, args.default_voice, blocksize=args.blocksize)

    import uvicorn

    logger.info(
        "Supertonic TTS server listening on http://%s:%d (model=%s voice=%s)",
        args.host,
        args.port,
        args.model,
        args.default_voice,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
