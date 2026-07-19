from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WebSocketStreamerArguments:
    ws_host: str = field(
        default="0.0.0.0",
        metadata={
            "help": "The host IP address for the WebSocket server. Default is '0.0.0.0' which binds to all "
            "available interfaces on the host machine."
        },
    )
    ws_port: int = field(
        default=8765,
        metadata={"help": "The port number on which the WebSocket server listens. Default is 8765."},
    )
    ws_ssl_keyfile: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the TLS private key for the WebSocket server. When set together "
                "with --ws_ssl_certfile, the server speaks WSS (WebSocket Secure) on the same "
                "port. Required for browser microphone access over LAN: https://<pi-ip>:<port>/ "
                "is a secure context, http:// is not."
            )
        },
    )
    ws_ssl_certfile: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the TLS certificate for the WebSocket server. Used together with "
                "--ws_ssl_keyfile to enable WSS."
            )
        },
    )
