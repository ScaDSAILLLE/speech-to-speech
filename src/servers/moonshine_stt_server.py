"""Moonshine STT HTTP server.

Exposes the OpenAI-compatible `POST /v1/audio/transcriptions` endpoint,
backed by the Hugging Face `UsefulSensors/moonshine-{base,streaming-medium}`
checkpoint loaded once at startup via the official Transformers integration.

By default the server listens on 127.0.0.1:9001 and uses the `moonshine-base`
checkpoint. Override via flags or environment variables:

  --host 0.0.0.0         bind address
  --port 9001            TCP port
  --model UsefulSensors/moonshine-base
                        HF repo id of the Moonshine checkpoint
  --device cpu           torch device
  --dtype float32        torch dtype (float32, float16)

Environment variables (consumed only if flags are omitted):

  MOONSHINE_MODEL_DIR     local HF cache (HF_HOME is honoured as well)
  MOONSHINE_DEFAULT_MODEL default checkpoint id

Launch via the entry point installed by pyproject.toml:

  moonshine-stt-server --port 9001 --model UsefulSensors/moonshine-base

Notes on the model:

  * Moonshine is a sequence-to-sequence ASR model, so it is loaded with
    `transformers.MoonshineForConditionalGeneration` + `transformers.AutoProcessor`.
  * The processor's `feature_extractor.sampling_rate` is the model's native
    sample rate (16 kHz for both base and streaming-medium).
  * To avoid hallucination loops we cap `max_new_tokens` at
    `6.5 * num_samples / sample_rate`, matching the upstream model card.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger("moonshine-stt-server")


def _decode_audio_to_array(raw: bytes, target_rate: int) -> "Any":
    """Decode an uploaded audio file to a 1-D numpy array at target_rate.

    Prefers `librosa` (robust to webm/ogg/mp3/wav). Falls back to `soundfile`
    + scipy resample if librosa is unavailable.
    """
    try:
        import io as _io

        import librosa  # type: ignore[import-not-found]

        audio, _sr = librosa.load(_io.BytesIO(raw), sr=target_rate, mono=True)
        return audio.astype("float32", copy=False)
    except ImportError:
        import tempfile

        import soundfile as sf
        from scipy.signal import resample_poly

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()
            audio, sr = sf.read(tmp.name, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if int(sr) != target_rate:
            from math import gcd

            g = gcd(int(sr), target_rate)
            audio = resample_poly(audio, target_rate // g, int(sr) // g).astype("float32", copy=False)
        return audio.astype("float32", copy=False)


class MoonshineBackend:
    """Lazy wrapper around the Transformers Moonshine model.

    Picks the right model class based on the model id:
    - `moonshine-{tiny,base}` → `MoonshineForConditionalGeneration`
    - `moonshine-streaming-{tiny,small,medium}` → `MoonshineStreamingForConditionalGeneration`

    Keeps the FastAPI factory and the route handlers free of torch/transformers
    imports so the module is importable even when the heavier deps are missing.
    """

    def __init__(self, model_id: str, device: str, dtype: str) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._model: Any = None
        self._processor: Any = None
        self._is_streaming: bool = "streaming" in model_id

    def warmup(self) -> None:
        import torch
        from transformers import AutoProcessor

        logger.info("Loading Moonshine model %s on %s (dtype=%s)", self.model_id, self.device, self.dtype)
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        if self._is_streaming:
            from transformers import MoonshineStreamingForConditionalGeneration

            model_cls = MoonshineStreamingForConditionalGeneration
        else:
            from transformers import MoonshineForConditionalGeneration

            model_cls = MoonshineForConditionalGeneration
        self._model = model_cls.from_pretrained(
            self.model_id, torch_dtype=torch_dtype
        ).to(self.device)

    @property
    def sample_rate(self) -> int:
        if self._processor is None:
            return 16_000
        return int(self._processor.feature_extractor.sampling_rate)

    def transcribe(self, audio: "Any", language: str | None) -> str:
        import torch
        import time

        assert self._model is not None and self._processor is not None, "backend not warmed up"
        sample_rate = self.sample_rate
        t0 = time.time()
        logger.info("transcribe(): %.2fs of audio", len(audio) / sample_rate)
        inputs = self._processor(audio, return_tensors="pt", sampling_rate=sample_rate)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Cap generated length to avoid the seq2seq hallucination loops the
        # model card warns about. The card recommends 6.5 tokens/s of audio;
        # that turns out to be too tight for short conversational turns
        # ("Can we" / "If you" / "I will" were getting truncated to the
        # model's first 1-2 words). We double it to ~13 tokens/s — still
        # bounded, but enough headroom for natural sentences.
        seq_lens = inputs["attention_mask"].sum(dim=-1)
        token_limit_factor = 13.0 / sample_rate
        max_new_tokens = max(1, int((seq_lens * token_limit_factor).max().item()))

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self._processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        logger.info("transcribe() took %.1fs, produced %d tokens: %r", time.time() - t0, len(generated_ids[0]), text[:80])

        # The processor is language-agnostic; we honour the request's `language`
        # only as a passthrough echo in the response.
        _ = language
        return text


def create_app(backend: MoonshineBackend) -> FastAPI:
    app = FastAPI(title="Moonshine STT (OpenAI-compat)", version="0.1.0")

    @app.on_event("startup")
    def _startup() -> None:
        backend.warmup()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "sample_rate": backend.sample_rate, "model": backend.model_id}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": "moonshine/base", "object": "model", "owned_by": "UsefulSensors"},
                {"id": "moonshine/streaming-medium", "object": "model", "owned_by": "UsefulSensors"},
                {"id": "UsefulSensors/moonshine-base", "object": "model", "owned_by": "UsefulSensors"},
                {"id": "UsefulSensors/moonshine-streaming-medium", "object": "model", "owned_by": "UsefulSensors"},
            ],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form("UsefulSensors/moonshine-base"),
        language: str | None = Form(None),
        response_format: str = Form("json"),
    ) -> JSONResponse:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty audio upload")

        try:
            audio = _decode_audio_to_array(raw, backend.sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.exception("audio decode failed")
            raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc

        try:
            text = backend.transcribe(audio, language)
        except Exception as exc:  # noqa: BLE001
            logger.exception("moonshine inference failed")
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

        if not text:
            logger.debug("moonshine: empty transcript")
            return JSONResponse({"text": "", "language": language, "model": model})

        return JSONResponse({"text": text, "language": language, "model": model})

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="moonshine-stt-server")
    p.add_argument("--host", default=os.environ.get("MOONSHINE_STT_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("MOONSHINE_STT_PORT", "9001")))
    p.add_argument(
        "--model",
        default=os.environ.get("MOONSHINE_DEFAULT_MODEL", "UsefulSensors/moonshine-base"),
        help="HF repo id of the Moonshine checkpoint. Use 'UsefulSensors/moonshine-streaming-medium' for streaming.",
    )
    p.add_argument("--device", default=os.environ.get("MOONSHINE_DEVICE", "cpu"))
    p.add_argument(
        "--dtype",
        default=os.environ.get("MOONSHINE_DTYPE", "float32"),
        choices=("float32", "float16", "bfloat16"),
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ["MOONSHINE_MODEL_DIR"]) if os.environ.get("MOONSHINE_MODEL_DIR") else None,
        help="If set, pre-populate HF_HOME with this directory so weights land under it.",
    )
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.model_dir is not None:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(args.model_dir))
        logger.info("HF_HOME -> %s", args.model_dir)

    backend = MoonshineBackend(model_id=args.model, device=args.device, dtype=args.dtype)
    app = create_app(backend)

    import uvicorn

    logger.info(
        "Moonshine STT server listening on http://%s:%d (model=%s)",
        args.host,
        args.port,
        args.model,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
