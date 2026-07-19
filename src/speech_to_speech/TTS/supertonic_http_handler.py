"""SupertonicHttpTTSHandler — TTS handler that talks to a supertonic-tts-server.

Behaviour:
  - On `setup()`, opens an `openai.OpenAI` client against
    `--supertonic_http_tts_base_url`.
  - On `process(text_input)`, calls `client.audio.speech.create(...)` with
    `response_format="pcm"` (default) and yields fixed-size int16 numpy arrays
    to the output queue, sized by `--supertonic_http_tts_blocksize`.
  - `EndOfResponse` items yield `AUDIO_RESPONSE_DONE` so downstream knows
    the current turn's audio is complete.
  - Cancellation is handled by the base pipeline; in-flight HTTP calls time
    out via the configured timeout.

The voice id can be overridden per-request if the OpenAI Realtime client
sends session.audio.output.voice; this handler honours that override.
"""

from __future__ import annotations

import logging
from threading import Event
from typing import Any, Iterator, Optional

import numpy as np
from openai import OpenAI

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)


class SupertonicHttpTTSHandler(BaseHandler[TTSIn, TTSOut]):
    def setup(
        self,
        should_listen: Event,
        base_url: str = "http://127.0.0.1:9002/v1",
        api_key: str = "not-needed",
        model_name: str = "supertonic-3",
        voice: str = "M1",
        sample_rate: int = 16_000,
        blocksize: int = 512,
        response_format: str = "pcm",
        timeout_s: float = 60.0,
        voice_override: Optional[str] = None,
        cancel_scope: Optional[CancelScope] = None,
        speculative_turns: Optional[SpeculativeTurnTracker] = None,
        **_kwargs: Any,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.model_name = model_name
        self.default_voice = voice
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.response_format = response_format
        self.voice_override = voice_override
        logger.info(
            "SupertonicHttpTTSHandler ready: base_url=%s voice=%s sample_rate=%d blocksize=%d",
            base_url,
            voice,
            sample_rate,
            blocksize,
        )

    @property
    def min_time_to_debug(self) -> float:
        return 0.1

    def _resolve_voice(self, tts_input: TTSIn) -> str:
        if self.voice_override:
            return self.voice_override
        session_voice = getattr(tts_input, "voice", None) if tts_input is not None else None
        if session_voice:
            return session_voice
        return self.default_voice

    def _stream_pcm(self, text: str, voice: str) -> Iterator[np.ndarray]:
        if self.response_format == "wav":
            resp = self.client.audio.speech.create(
                model=self.model_name,
                input=text,
                voice=voice,
                response_format="wav",
            )
            raw = resp.read()
            pcm = _wav_bytes_to_pcm(raw, expected_rate=self.sample_rate)
        else:
            resp = self.client.audio.speech.create(
                model=self.model_name,
                input=text,
                voice=voice,
                response_format="pcm",
            )
            pcm = resp.read()

        arr = np.frombuffer(pcm, dtype=np.int16)
        for start in range(0, len(arr), self.blocksize):
            chunk = arr[start : start + self.blocksize]
            if len(chunk) < self.blocksize:
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield chunk

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale Supertonic TTS input")
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        gen = self.cancel_scope.generation if self.cancel_scope else None
        text = tts_input.text or ""
        voice = self._resolve_voice(tts_input)
        if not text.strip():
            return

        try:
            for chunk in self._stream_pcm(text, voice):
                if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                    logger.info("Supertonic TTS cancelled (interruption)")
                    return
                yield chunk
        except Exception as exc:  # noqa: BLE001
            logger.exception("supertonic-http TTS failed: %s", exc)

    def cleanup(self) -> None:
        client = getattr(self, "client", None)
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


def _wav_bytes_to_pcm(raw: bytes, expected_rate: int) -> bytes:
    """Strip the WAV header and return raw int16 mono PCM, optionally resampling."""
    import io
    import wave

    with wave.open(io.BytesIO(raw), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        import numpy as np

        arr = np.frombuffer(frames, dtype=np.int16 if sampwidth == 2 else f"i{sampwidth * 8}").astype(np.int16)
    else:
        arr = np.frombuffer(frames, dtype=np.int16).copy()

    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if rate != expected_rate:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(rate), int(expected_rate))
        arr = resample_poly(arr, expected_rate // g, int(rate) // g).astype(np.int16)

    return arr.tobytes()
