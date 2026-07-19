"""MoonshineHttpSTTHandler — STT handler that talks to a moonshine-stt-server.

The server is launched separately (see `moonshine-stt-server` console script
and `scripts/rpi_start.sh`). This handler is a thin OpenAI-compatible STT
client; all model loading happens in the server process.

Behaviour:
  - On `setup()`, opens an `openai.OpenAI` client against `--moonshine_http_stt_base_url`.
  - On `process(vad_audio)`, sends the audio buffer as multipart upload to
    `/v1/audio/transcriptions` and yields a single `Transcription` event.
  - Cancellation is handled by the base pipeline (`cancel_scope`); in-flight
    HTTP requests time out via the configured timeout.

If you want streaming partial transcripts, set
`--moonshine_http_stt_model_name moonshine/streaming-medium` and ensure the
server is started with that same id.
"""

from __future__ import annotations

import io
import logging
import os
import wave
from typing import Any, Iterator, Optional

from openai import OpenAI

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)


def _to_wav_bytes(audio: Any, sample_rate: int) -> bytes:
    """Encode an int16/float numpy array as a 16-bit mono WAV.

    The moonshine server accepts WAV via librosa/soundfile, so we hand it a
    proper container rather than a raw buffer.
    """
    import numpy as np

    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    arr = np.asarray(audio).squeeze()
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype(np.int16)
    elif arr.dtype != np.int16:
        pcm = arr.astype(np.int16)
    else:
        pcm = arr

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class MoonshineHttpSTTHandler(BaseSTTHandler):
    """STT handler backed by a moonshine-stt-server HTTP endpoint."""

    def setup(
        self,
        base_url: str = "http://127.0.0.1:9001/v1",
        api_key: str = "not-needed",
        model_name: str = "moonshine/base",
        language: Optional[str] = None,
        timeout_s: float = 30.0,
        sample_rate: int = 16_000,
        **_kwargs: Any,
    ) -> None:
        os.environ.setdefault("OPENAI_LOG", "warning")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate
        logger.info(
            "MoonshineHttpSTTHandler ready: base_url=%s model=%s sample_rate=%d",
            base_url,
            model_name,
            sample_rate,
        )

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if not hasattr(vad_audio, "audio"):
            logger.debug("moonshine-http: skipping non-audio item %r", vad_audio)
            return

        # Skip progressive VAD emissions. Moonshine is a seq2seq model that
        # needs the full utterance to produce a stable transcript; running it
        # on 200-800 ms slices while the user is still speaking wastes an
        # HTTP round-trip per slice and yields 1-2-word hallucinations that
        # race the real transcript on the wire. The VAD still emits the
        # `final` VADAudio on soft-end silence, which we do transcribe.
        if getattr(vad_audio, "mode", "final") == "progressive":
            return

        wav_bytes = _to_wav_bytes(vad_audio.audio, self.sample_rate)
        try:
            resp = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=("audio.wav", wav_bytes, "audio/wav"),
                language=self.language,
                response_format="json",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("moonshine-http transcription failed: %s", exc)
            return

        text = getattr(resp, "text", None) or ""
        text = text.strip()
        if not text:
            logger.debug("moonshine-http: empty transcript")
            return

        yield Transcription(
            text=text,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    def cleanup(self) -> None:
        client = getattr(self, "client", None)
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
