#!/usr/bin/env python3
"""End-to-end WebSocket round-trip test for the RPi fork.

Connects to wss://192.168.178.101:8765/v1/realtime with the demo's exact
session.update payload, sends one input_audio_buffer.append frame (silence
at 24 kHz PCM16), creates a response, and reads frames until response.done.

Exit 0 on success, non-zero on failure. Verbose output for debugging.
"""
import base64
import json
import os
import socket
import ssl
import struct
import sys
import time

HOST = "192.168.178.101"
PORT = 8765
CERT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "tls", "realtime-cert.pem",
)


def _frame(opcode: int, payload: bytes) -> bytes:
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    header = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header += bytes([0x80 | n])
    elif n < 65536:
        header += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        header += bytes([0x80 | 127]) + struct.pack(">Q", n)
    return header + mask + masked


def _read_frame(sock: socket.socket, timeout: float = 30.0):
    sock.settimeout(timeout)
    buf = b""
    while len(buf) < 2:
        chunk = sock.recv(4096)
        if not chunk:
            return None, None
        buf += chunk
    opcode = buf[0] & 0x0F
    length = buf[1] & 0x7F
    idx = 2
    if length == 126:
        while len(buf) < idx + 2:
            buf += sock.recv(4096)
        length = struct.unpack(">H", buf[idx:idx + 2])[0]
        idx += 2
    elif length == 127:
        while len(buf) < idx + 8:
            buf += sock.recv(4096)
        length = struct.unpack(">Q", buf[idx:idx + 8])[0]
        idx += 8
    while len(buf) < idx + length:
        buf += sock.recv(4096)
    return opcode, buf[idx:idx + length]


def _connect():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    sock = socket.create_connection((HOST, PORT), timeout=10)
    sock = ctx.wrap_socket(sock, server_hostname=HOST)
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET /v1/realtime HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    sock.sendall(req.encode())
    sock.settimeout(10)
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = sock.recv(4096)
        if not chunk:
            break
        resp += chunk
    if b"101 Switching Protocols" not in resp:
        raise RuntimeError(f"WS upgrade failed:\n{resp.decode(errors='replace')}")
    return sock


def main() -> int:
    print(f"[1/6] connecting to wss://{HOST}:{PORT}/v1/realtime")
    sock = _connect()
    print("      ✓ upgraded")

    # 1. session.created
    op, p = _read_frame(sock, timeout=10)
    if not p or json.loads(p).get("type") != "session.created":
        raise RuntimeError(f"expected session.created, got opcode={op} payload={p!r}")
    print("[2/6] received session.created")

    # 2. session.update (the demo's exact payload, voice=M1 to match Supertonic)
    sock.sendall(
        _frame(
            0x1,
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": "Reply in one short sentence.",
                        "audio": {"output": {"voice": "M1"}},
                    },
                }
            ).encode(),
        )
    )
    print("[3/6] sent session.update (voice=M1)")

    # 3. session.updated confirmation (with our upstream patch, the server
    # sends this back; without the patch it stays silent and we just continue)
    op, p = _read_frame(sock, timeout=5)
    if p:
        t = json.loads(p).get("type")
        if t == "session.updated":
            print("[4/6] ✓ server confirmed session.updated")
        else:
            print(f"[4/6] (no session.updated; got {t} — proceeding)")
    else:
        print("[4/6] (no session.updated — proceeding)")

    # 4. input_audio_buffer.append with a 1-second 440 Hz tone at 24 kHz mono PCM16.
    # Silence alone gets interpreted as "user stopped speaking" immediately; a
    # synthetic tone gives the LLM something to actually respond to.
    import math

    sample_rate = 24000
    duration_s = 1.0
    samples = []
    for i in range(int(sample_rate * duration_s)):
        v = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        samples.append(v)
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    sock.sendall(
        _frame(
            0x1,
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                }
            ).encode(),
        )
    )
    print(f"[5/6] sent {duration_s:.1f}s of 440 Hz tone ({len(pcm)} bytes PCM16)")

    # 5. response.create — ask the model to respond (modalities include
    # audio so the pipeline can stream Supertonic TTS back).
    sock.sendall(
        _frame(
            0x1,
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"modalities": ["text", "audio"]},
                }
            ).encode(),
        )
    )
    print("[6/6] waiting for response (max 120s)...")

    transcript = None
    audio_chunks = 0
    audio_bytes = 0
    event_counts: dict = {}
    deadline = time.time() + 120
    while time.time() < deadline:
        op, p = _read_frame(sock, timeout=60)
        if not op:
            print("      ! connection closed by server")
            break
        # Binary frames carry PCM16 audio deltas (24 kHz, mono).
        if op == 0x2:
            audio_chunks += 1
            audio_bytes += len(p)
            if audio_chunks <= 3 or audio_chunks % 10 == 0:
                print(f"      audio chunk #{audio_chunks}: {len(p)} bytes")
            continue
        # WS control frames: 0x8 close, 0x9 ping, 0xA pong.
        if op in (0x8, 0x9, 0xA):
            print(f"      ws ctrl frame op={op} len={len(p)}")
            if op == 0x9:
                # Send pong back so the server doesn't drop us.
                sock.sendall(_frame(0xA, p))
            if op == 0x8:
                break
            continue
        # Text frames are JSON control / text deltas.
        try:
            msg = json.loads(p)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"      ! non-JSON text frame (op={op}, {len(p)} bytes): {p[:60]!r}")
            print(f"        decode error: {e}")
            continue
        t = msg.get("type", "<no-type>")
        event_counts[t] = event_counts.get(t, 0) + 1
        if t == "response.text.delta" or t == "response.output_audio_transcript.delta":
            transcript = (transcript or "") + msg.get("delta", "")
            print(f"      delta: {msg.get('delta', '')[:80]!r}")
        elif t == "response.output_audio.delta":
            audio_chunks += 1
            audio_bytes += len(msg.get("delta", ""))
        elif t == "response.done":
            print(
                f"      ✓ response.done (audio: {audio_chunks} chunks, "
                f"{audio_bytes} bytes base64, events: {sorted(event_counts.items())})"
            )
            break
        elif t and "error" in t.lower():
            print(f"      ✗ server error: {msg.get('error') or msg}")
            sock.close()
            return 1

    sock.close()
    if transcript:
        print(f"\n  transcript: {transcript!r}")
    print(f"  audio chunks received: {audio_chunks}")
    if not transcript and audio_chunks == 0:
        print("  ✗ no response received from the pipeline")
        return 1
    print("  ✓ WS round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
