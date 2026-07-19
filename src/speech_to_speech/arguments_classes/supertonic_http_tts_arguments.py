from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SupertonicHttpTTSHandlerArguments:
    supertonic_http_tts_base_url: str = field(
        default="http://127.0.0.1:9002/v1",
        metadata={"help": "Base URL of the supertonic-tts-server. Default is http://127.0.0.1:9002/v1."},
    )
    supertonic_http_tts_api_key: str = field(
        default="not-needed",
        metadata={"help": "API key sent to the Supertonic server. The local server ignores it."},
    )
    supertonic_http_tts_model_name: str = field(
        default="supertonic-3",
        metadata={"help": "Model id forwarded as the `model` field of the speech request."},
    )
    supertonic_http_tts_voice: str = field(
        default="M1",
        metadata={"help": "Default Supertonic voice id used when the request does not specify one. M1..M5 / F1..F5 ship with Supertone/supertonic-3."},
    )
    supertonic_http_tts_sample_rate: int = field(
        default=16000,
        metadata={"help": "Audio sample rate of the response, must match the pipeline's playback rate. Default is 16000."},
    )
    supertonic_http_tts_blocksize: int = field(
        default=512,
        metadata={"help": "Number of int16 samples per yielded audio chunk. Default is 512 (~32ms @ 16kHz)."},
    )
    supertonic_http_tts_response_format: str = field(
        default="pcm",
        metadata={"help": "Either 'pcm' (raw int16 mono) or 'wav' (PCM wrapped in a WAV header). Default is 'pcm'."},
    )
    supertonic_http_tts_timeout_s: float = field(
        default=60.0,
        metadata={"help": "HTTP request timeout in seconds. Default is 60."},
    )
    supertonic_http_tts_voice_override: Optional[str] = field(
        default=None,
        metadata={"help": "If set, this voice id is used for every request, ignoring session-level voice updates."},
    )
