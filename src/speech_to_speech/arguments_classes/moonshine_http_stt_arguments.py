from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MoonshineHttpSTTHandlerArguments:
    moonshine_http_stt_base_url: str = field(
        default="http://127.0.0.1:9001/v1",
        metadata={"help": "Base URL of the moonshine-stt-server. Default is http://127.0.0.1:9001/v1."},
    )
    moonshine_http_stt_api_key: str = field(
        default="not-needed",
        metadata={"help": "API key sent to the moonshine server. The local server ignores it."},
    )
    moonshine_http_stt_model_name: str = field(
        default="moonshine/base",
        metadata={
            "help": (
                "Moonshine checkpoint id, forwarded as the `model` form field. "
                "Use 'moonshine/base' for the default or 'moonshine/streaming-medium' for streaming."
            )
        },
    )
    moonshine_http_stt_language: Optional[str] = field(
        default=None,
        metadata={"help": "Optional ISO language code forwarded as the `language` form field."},
    )
    moonshine_http_stt_timeout_s: float = field(
        default=30.0,
        metadata={"help": "HTTP request timeout in seconds. Default is 30."},
    )
    moonshine_http_stt_sample_rate: int = field(
        default=16000,
        metadata={"help": "Sample rate the pipeline expects from VAD. Default is 16000."},
    )
