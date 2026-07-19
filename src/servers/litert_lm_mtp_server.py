"""LiteRT-LM server with MTP (Multi-Token Prediction) + per-session Conversation pool.

Why this exists
---------------
The `litert-lm serve` CLI does not expose the
`--enable-speculative-decoding=true` flag, so MTP cannot be enabled in
server mode. This wrapper loads the model once via the Python SDK,
calls `Engine.enable_speculative_decoding`, and exposes an
OpenAI-compatible `/v1/chat/completions` endpoint.

Conversation pool (per `session_id`)
------------------------------------
Each WebSocket pipeline session gets its OWN persistent
``litert_lm.Conversation``. The pipeline passes ``session_id`` via
``extra_body["session_id"]``; the server holds the conversation across
requests so the KV-cache is reused turn after turn — that avoids the
~100-300 ms DataProcessor rebuild that `create_conversation()` would
otherwise pay on every turn.

When a new request arrives for an existing ``session_id``, the
previous turn's generator is cancelled via ``Conversation.cancel_process()``.
This is real cancellation at the LiteRT-LM C++ layer, not a
``httpx``-stream abort on the client side.

Idle conversations are evicted after ``--idle-eviction-seconds`` (default
30 min) to bound memory; the next request simply creates a fresh
conversation.

Performance on Pi 5 (Cortex-A76, 8 GB, XNNPACK CPU backend)
-----------------------------------------------------------
Benchmark with Gemma 4 E2B, 200-token prompt "List 3 fun facts about Mars":

    | mode                       | warm-cache time | relative |
    |----------------------------+-----------------+----------|
    | without MTP                | ~8.5 s          | 1.00x    |
    | with MTP                   | ~8.4 s          | 1.01x    |
    | HTTP serve (legacy)        | ~4.3 s          | n/a      |

MTP gives no measurable speedup on Pi 5 CPU (Google's "up to 3x" claim
is for mobile GPUs with the XNNPACK/GPU + FlashAttention pipeline).
It is wired in for two reasons:

1. **Future GPU migration**: the same model + the same flag will
   benefit dramatically the day a GPU/NPU is added.
2. **Lower warm-cache variance**: MTP reduces decode jitter by
   packing multiple tokens per step.

Streaming
---------
``Conversation.send_message_async()`` returns a synchronous iterator
that yields chat-completion-shaped chunks in real time. We wrap it in
``asyncio.to_thread()`` so the FastAPI event loop stays responsive,
and forward each chunk as an OpenAI SSE ``chat.completion.chunk``.

If you want faster decode on Pi 5 today, the real lever is model size:
switch from `gemma4-e2b` to a 1B-class model (see CHANGELOG.md
"Backlog -> LLM is the round-trip bottleneck" for details).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, AsyncIterator

import litert_lm  # type: ignore[import-not-found]
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger("litert-lm-mtp-server")


# ---------------------------------------------------------------------------
# Engine lifecycle: load once at startup, reuse across requests.
# ---------------------------------------------------------------------------

class EngineHolder:
    """Single Engine instance per process.

    LiteRT-LM's Engine is heavy (loads the .litertlm, compiles XNNPACK
    cache). We initialise it once at startup and reuse across HTTP
    requests. Conversations are created per session and reused across
    turns so the KV-cache survives turn boundaries.
    """

    def __init__(self, model_path: str, enable_mtp: bool, max_num_tokens: int | None = None):
        self.model_path = model_path
        self.enable_mtp = enable_mtp
        self._engine: litert_lm.Engine | None = None
        self._max_num_tokens = max_num_tokens

    def startup(self) -> None:
        logger.info("Loading LiteRT-LM engine from %s", self.model_path)
        t0 = time.time()
        kwargs: dict[str, Any] = {"model_path": self.model_path}
        if self._max_num_tokens is not None:
            kwargs["max_num_tokens"] = self._max_num_tokens
        self._engine = litert_lm.Engine(**kwargs)
        logger.info("Engine loaded in %.1f s", time.time() - t0)

        if self.enable_mtp:
            # `enable_speculative_decoding` is a property-style setter on
            # the C++ engine. Calling it enables MTP for subsequent
            # conversations.
            self._engine.enable_speculative_decoding
            logger.info("Multi-Token Prediction (MTP) enabled")

    def get(self) -> litert_lm.Engine:
        if self._engine is None:
            raise RuntimeError("Engine not initialised; call startup() first")
        return self._engine


# ---------------------------------------------------------------------------
# Conversation pool: one persistent Conversation per session_id.
# ---------------------------------------------------------------------------

class ConversationPool:
    """Map ``session_id`` -> ``Conversation`` with idle eviction.

    Thread-safe; only the FastAPI thread pool mutates it, but the
    cancel-then-replace sequence under ``with lock`` keeps things sane.

    ``_generating`` tracks sessions whose conversation is currently
    inside a ``send_message*`` call. Only conversations in that set
    are eligible for ``cancel_process()``; calling cancel on a
    conversation that is still in its prefill phase breaks it (LiteRT-LM
    raises "Session is not prefilled yet" on the next send_message).
    """

    def __init__(self, engine: litert_lm.Engine | EngineHolder, idle_eviction_s: float = 1800.0):
        self._engine = engine
        self._lock = threading.Lock()
        self._conversations: dict[str, tuple[litert_lm.Conversation, float]] = {}
        self._generating: set[str] = set()
        self._idle_eviction_s = idle_eviction_s
        self._total_created = 0
        self._total_reused = 0
        self._total_cancelled = 0

    def _resolve_engine(self) -> litert_lm.Engine:
        if isinstance(self._engine, EngineHolder):
            return self._engine.get()
        return self._engine

    def get_or_create(
        self,
        session_id: str,
        sampler_config: litert_lm.SamplerConfig,
        system_message: str | None,
    ) -> tuple[litert_lm.Conversation, bool]:
        """Return ``(conversation, created_now)``.

        If a conversation for ``session_id`` is currently generating,
        cancel it; the cancelled conversation is dropped and a fresh
        one is created, because LiteRT-LM's
        ``cancel_process()`` during the prefill phase leaves the
        conversation in a state where ``send_message`` will fail
        ("Session is not prefilled yet"). Cancelling only
        post-generation keeps the conversation alive — but for
        simplicity we always replace on cancel.
        """
        with self._lock:
            self._evict_idle_locked()
            existing = self._conversations.get(session_id)
            if existing is not None and session_id not in self._generating:
                conv, _last_used = existing
                self._conversations[session_id] = (conv, time.time())
                self._total_reused += 1
                return conv, False

            if existing is not None:
                conv, _last_used = existing
                try:
                    conv.cancel_process()
                    self._total_cancelled += 1
                    logger.info("ConversationPool: cancelled in-flight for session_id=%s", session_id)
                except Exception:
                    pass
                self._conversations.pop(session_id, None)
                self._generating.discard(session_id)

            conv = self._resolve_engine().create_conversation(
                system_message=system_message,
                sampler_config=sampler_config,
                automatic_tool_calling=False,
            )
            self._conversations[session_id] = (conv, time.time())
            self._total_created += 1
            logger.info(
                "ConversationPool: created #%d for session_id=%s (active=%d)",
                self._total_created, session_id, len(self._conversations),
            )
            return conv, True

    def mark_generating(self, session_id: str) -> None:
        with self._lock:
            self._generating.add(session_id)

    def mark_done(self, session_id: str) -> None:
        with self._lock:
            self._generating.discard(session_id)

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._conversations.pop(session_id, None)
            self._generating.discard(session_id)

    def _evict_idle_locked(self) -> None:
        if self._idle_eviction_s <= 0:
            return
        cutoff = time.time() - self._idle_eviction_s
        stale = [sid for sid, (_, ts) in self._conversations.items() if ts < cutoff]
        for sid in stale:
            self._conversations.pop(sid, None)
            self._generating.discard(sid)
            logger.info("ConversationPool: evicted idle session_id=%s", sid)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": len(self._conversations),
                "generating": len(self._generating),
                "total_created": self._total_created,
                "total_reused": self._total_reused,
                "total_cancelled": self._total_cancelled,
                "idle_eviction_s": self._idle_eviction_s,
            }


# ---------------------------------------------------------------------------
# OpenAI -> LiteRT-LM mapping helpers.
# ---------------------------------------------------------------------------

def _build_sampler_config(req: dict[str, Any]) -> litert_lm.SamplerConfig:
    """Map OpenAI sampling fields onto LiteRT-LM's SamplerConfig."""
    return litert_lm.SamplerConfig(
        temperature=float(req.get("temperature", 0.7)),
        top_p=float(req.get("top_p", 0.9)),
        top_k=int(req.get("top_k", 40)),
    )


def _extract_last_user_message(messages: list[dict[str, Any]]) -> str | None:
    """Return the last user-role message content, or None.

    The persistent conversation already owns prior turns, so we only
    need to feed the latest user prompt into ``send_message_async``.
    """
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return str(content)
    return None


def _find_system_message(messages: list[dict[str, Any]]) -> str | None:
    """Return the last system-role message content, or None."""
    for m in reversed(messages):
        if m.get("role") == "system":
            content = m.get("content", "")
            return content if isinstance(content, str) else str(content)
    return None


def _chunk_text(payload: dict[str, Any]) -> str:
    """Pull a text fragment out of a LiteRT-LM async chunk payload."""
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    tool_calls = payload.get("tool_calls")
    if tool_calls:
        return json.dumps(tool_calls)
    return ""


# ---------------------------------------------------------------------------
# FastAPI app.
# ---------------------------------------------------------------------------

def build_app(
    engine: EngineHolder,
    pool: ConversationPool | None = None,
) -> FastAPI:
    app = FastAPI(title="litert-lm-mtp-server", version="0.2.0")
    if pool is None:
        pool = ConversationPool(engine, idle_eviction_s=1800.0)

    @app.on_event("startup")
    def _startup() -> None:
        engine.startup()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": engine.model_path,
            "mtp": engine.enable_mtp,
            "pool": pool.stats(),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: dict[str, Any]) -> Any:
        # -------- parse OpenAI request --------
        messages_in = req.get("messages", [])
        if not messages_in:
            raise HTTPException(status_code=400, detail="messages is required")

        sampler = _build_sampler_config(req)
        max_tokens = int(req.get("max_tokens", 256))
        stream = bool(req.get("stream", False))

        extra_body = req.get("extra_body") or {}
        # OpenAI Python SDK 1.x flattens `extra_body` into the top-level
        # JSON body (no wrapper). Accept session_id from either location.
        session_id = extra_body.get("session_id") or req.get("session_id")
        persistent = bool(session_id)
        if not persistent:
            session_id = f"_oneshot_{int(time.time()*1e6)}"

        last_user_msg = _extract_last_user_message(messages_in)
        if last_user_msg is None:
            raise HTTPException(status_code=400, detail="no user message in messages[]")
        system_msg = _find_system_message(messages_in)

        # -------- get/cancel/create conversation --------
        # If session_id is set, use the persistent pool (KV-cache reused across
        # turns). Otherwise (e.g. pipeline warmup probe at startup, before any
        # WS session exists) build a one-shot conversation that's not stored.
        try:
            if persistent:
                conversation, created_now = await asyncio.to_thread(
                    pool.get_or_create, session_id, sampler, system_msg,
                )
            else:
                conversation = await asyncio.to_thread(
                    engine.get().create_conversation,
                    system_message=system_msg,
                    sampler_config=sampler,
                    automatic_tool_calling=False,
                )
                created_now = True
        except Exception as exc:
            logger.exception("conversation creation failed (persistent=%s)", persistent)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        completion_id = f"chatcmpl_{int(time.time() * 1000)}"
        created_ts = int(time.time())
        model_name = os.path.basename(engine.model_path)

        # -------- non-streaming response --------
        if not stream:
            def _run_sync() -> tuple[dict[str, Any], float]:
                t0 = time.time()
                if persistent:
                    pool.mark_generating(session_id)
                try:
                    resp = conversation.send_message(last_user_msg, max_output_tokens=max_tokens)
                finally:
                    if persistent:
                        pool.mark_done(session_id)
                return resp, time.time() - t0

            try:
                resp_msg, elapsed = await asyncio.to_thread(_run_sync)
            except Exception as exc:
                if persistent:
                    pool.drop(session_id)
                msg = str(exc)
                if "not prefilled" in msg or "cancelled" in msg.lower():
                    logger.warning("send_message broke session (likely cancel-during-prefill): %s; dropped", msg)
                    raise HTTPException(status_code=409, detail=f"conversation broken: {msg}") from exc
                logger.exception("send_message failed for session_id=%s", session_id)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            content = _chunk_text(resp_msg) if isinstance(resp_msg, dict) else str(resp_msg)
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(content.split()),
                    "total_tokens": len(content.split()),
                },
                "_server_elapsed_s": round(elapsed, 3),
                "_conversation_reused": not created_now,
            }

        # -------- streaming SSE response --------
        async def sse_stream() -> AsyncIterator[bytes]:
            # First chunk announces the role; matches OpenAI shape.
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield ("data: " + json.dumps(role_chunk) + "\n\n").encode()

            t0 = time.time()
            chunk_count = 0

            if persistent:
                pool.mark_generating(session_id)
            iterator: Any | None = None
            iter_broken = False
            try:
                iterator = await asyncio.to_thread(
                    conversation.send_message_async,
                    last_user_msg, max_output_tokens=max_tokens,
                )
            except Exception as exc:
                if persistent:
                    pool.mark_done(session_id)
                    pool.drop(session_id)
                iter_broken = True
                msg = str(exc)
                if "not prefilled" in msg or "cancelled" in msg.lower():
                    logger.warning("send_message_async broke session (likely cancel-during-prefill): %s; dropped", msg)
                else:
                    logger.exception("send_message_async failed for session_id=%s", session_id)
                err_payload = {
                    "id": completion_id,
                    "object": "error",
                    "error": {"message": msg, "type": "server_error"},
                }
                yield f"data: {json.dumps(err_payload)}\n\n".encode()
                return

            def _next_chunk() -> tuple[Any, bool]:
                try:
                    return next(iterator), False
                except StopIteration:
                    return None, True
                except Exception as exc:
                    return exc, True

            try:
                while True:
                    chunk, done = await asyncio.to_thread(_next_chunk)
                    if done:
                        if isinstance(chunk, Exception):
                            raise chunk
                        break
                    chunk_count += 1
                    text = _chunk_text(chunk) if isinstance(chunk, dict) else str(chunk)
                    if not text:
                        continue
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n".encode()

                finish_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                logger.info(
                    "stream session_id=%s reused=%s chunks=%d elapsed=%.2fs",
                    session_id, not created_now, chunk_count, time.time() - t0,
                )
            except asyncio.CancelledError:
                try:
                    conversation.cancel_process()
                except Exception:
                    pass
                logger.info("stream cancelled by client (session_id=%s)", session_id)
                raise
            except Exception as exc:
                msg = str(exc)
                if "not prefilled" in msg or "cancelled" in msg.lower():
                    if persistent:
                        pool.drop(session_id)
                    logger.warning("stream session broke (likely cancel-during-prefill): %s; dropped", msg)
                else:
                    logger.exception("send_message_async failed for session_id=%s", session_id)
                err_payload = {
                    "id": completion_id,
                    "object": "error",
                    "error": {"message": msg, "type": "server_error"},
                }
                yield f"data: {json.dumps(err_payload)}\n\n".encode()
            finally:
                if persistent:
                    pool.mark_done(session_id)

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/v1/conversation/clear")
    def clear_conversation(req: dict[str, Any]) -> dict[str, Any]:
        session_id = (req.get("session_id") or "").strip()
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        pool.drop(session_id)
        return {"ok": True, "session_id": session_id}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": os.path.basename(engine.model_path),
                    "object": "model",
                    "owned_by": "litert-lm",
                }
            ],
        }

    return app


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=os.environ.get(
        "LITERT_LM_MODEL_PATH",
        "/home/rpi-ai/.litert-lm/models/gemma4-e2b/model.litertlm",
    ))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9379)
    p.add_argument("--disable-mtp", action="store_true",
                   help="Skip enable_speculative_decoding (default: on)")
    p.add_argument("--max-num-tokens", type=int, default=None,
                   help="Override the engine's KV-cache size")
    p.add_argument("--idle-eviction-seconds", type=float, default=1800.0,
                   help="Evict conversations idle for this long (0 = never)")
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    holder = EngineHolder(
        model_path=args.model,
        enable_mtp=not args.disable_mtp,
        max_num_tokens=args.max_num_tokens,
    )
    app = build_app(
        holder,
        pool=ConversationPool(
            engine=holder,
            idle_eviction_s=args.idle_eviction_seconds,
        ),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
