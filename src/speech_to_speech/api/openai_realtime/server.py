import logging
import threading
from threading import Event

import uvicorn

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.websocket_router import create_app

logger = logging.getLogger(__name__)


class RealtimeServer:
    """
    Pipeline handler for the OpenAI Realtime API mode.

    Owns a pool of isolated PipelineUnits and a single uvicorn server.
    The websocket route claims the next free unit on each accept and releases
    it on disconnect; once all units are in use, further connections are rejected.
    """

    def __init__(
        self,
        stop_event: Event,
        pool: list[PipelineUnit],
        host: str = "0.0.0.0",
        port: int = 8765,
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
    ) -> None:
        if not pool:
            raise ValueError("RealtimeServer requires at least one PipelineUnit in the pool")
        if (ssl_keyfile is None) != (ssl_certfile is None):
            raise ValueError(
                "RealtimeServer: --ws_ssl_keyfile and --ws_ssl_certfile must be set together."
            )
        self.stop_event = stop_event
        self.pool = pool
        self.host = host
        self.port = port
        self.ssl_keyfile = ssl_keyfile
        self.ssl_certfile = ssl_certfile

    def run(self) -> None:
        """Start the FastAPI/uvicorn server (called from a ThreadManager thread)."""
        app = create_app(pool=self.pool, stop_event=self.stop_event)

        scheme = "wss" if self.ssl_keyfile else "ws"
        logger.info(
            f"OpenAI Realtime API starting on {scheme}://{self.host}:{self.port}/v1/realtime (pool size {len(self.pool)})"
        )

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            ssl_keyfile=self.ssl_keyfile,
            ssl_certfile=self.ssl_certfile,
            log_level="info",
        )
        server = uvicorn.Server(config)

        server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]

        def _watch_stop() -> None:
            self.stop_event.wait()
            server.should_exit = True

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()

        server.run()
