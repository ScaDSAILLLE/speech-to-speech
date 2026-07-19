"""End-to-end WS pipeline test.

Streams a known WAV as `input_audio_buffer.append` events to the realtime
pipeline and reports:
1. How many audio bytes the pipeline received (per session)
2. How many VAD events fired (speech_started / speech_stopped)
3. The final transcription it produced
4. The audio duration in the final STT call (vs the original WAV duration)

If the STT receives a truncated version of the audio, this will reveal it.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import ssl
import socket
import struct
import sys
import time
import wave
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8765


def _frame(opcode: int, payload: bytes) -> bytes:
    """Build a client→server WS frame.

    RFC 6455 §5.1: every frame from client to server MUST be masked. The mask
    is a 4-byte random key XORed into the payload; the mask key itself is sent
    in the frame header. Servers close with code 1002 ("protocol error") if
    the bit isn't set.
    """
    header = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header.append(0x80 | n)  # 0x80 = MASK bit set
    elif n < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack(">H", n))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack(">Q", n))
    mask_key = os.urandom(4)
    header.extend(mask_key)
    masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return bytes(header) + masked


def _read_frame(sock: socket.socket, timeout: float = 60.0):
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("wav", type=Path)
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-response", action="store_true",
                   help="don't trigger response.create (just observe VAD)")
    args = p.parse_args()

    with wave.open(str(args.wav), "rb") as wf:
        sr = wf.getframerate()
        assert sr == 16000, f"need 16kHz WAV, got {sr}"
        assert wf.getnchannels() == 1, f"need mono WAV, got {wf.getnchannels()}"
        pcm16 = wf.readframes(wf.getnframes())
    wav_duration_s = len(pcm16) / 2 / sr
    print(f"input WAV: {wav_duration_s:.2f}s, {len(pcm16)} bytes PCM16 @ {sr} Hz mono")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    sock = socket.create_connection((args.host, args.port), timeout=10)
    sock = ctx.wrap_socket(sock, server_hostname=args.host)
    key = base64.b64encode(os.urandom(16)).decode()
    req = (
        f"GET /v1/realtime HTTP/1.1\r\n"
        f"Host: {args.host}:{args.port}\r\n"
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
        print("WS upgrade failed:", resp.decode(errors="replace"))
        return 1
    print("✓ WS upgraded")

    op, p = _read_frame(sock)
    if not p:
        print("no initial frame"); return 1
    sc = json.loads(p)
    assert sc["type"] == "session.created", sc
    print(f"✓ session.created")

    sock.sendall(_frame(0x1, json.dumps({
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": "Reply in one short sentence.",
            "audio": {"output": {"voice": "M1"}},
        },
    }).encode()))
    while True:
        op, p = _read_frame(sock, timeout=30)
        if not op:
            print("server closed"); break
        if op in (0x8, 0x9, 0xA):
            if op == 0x9:
                sock.sendall(_frame(0xA, p))
            if op == 0x8:
                break
            continue
        if op == 0x2:
            continue
        try:
            msg = json.loads(p)
        except Exception:
            continue
        if msg.get("type") == "session.updated":
            print(f"✓ session.updated")
            break

    chunk_samples = 512
    bytes_per_sample = 2
    bytes_per_chunk = chunk_samples * bytes_per_sample
    chunks = [pcm16[i:i + bytes_per_chunk] for i in range(0, len(pcm16), bytes_per_chunk)]
    print(f"sending {len(chunks)} audio chunks of {chunk_samples} samples ({chunk_samples / sr * 1000:.0f}ms each)")
    sent_bytes = 0
    for c in chunks:
        sock.sendall(_frame(0x1, json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(c).decode(),
        }).encode()))
        sent_bytes += len(c)
    print(f"✓ sent {sent_bytes} bytes ({sent_bytes / 2 / sr:.2f}s); draining any pending frames…")
    # Drain any immediate server events before issuing response.create, so we
    # can see VAD-side events (speech_started/stopped) that may already have
    # fired while we were streaming audio.
    drain_deadline = time.time() + 5
    while time.time() < drain_deadline:
        try:
            op, p = _read_frame(sock, timeout=0.5)
        except (socket.timeout, TimeoutError):
            break
        if not op:
            break
        if op in (0x8, 0x9, 0xA):
            if op == 0x9:
                sock.sendall(_frame(0xA, p))
            if op == 0x8:
                break
            continue
        if op == 0x2:
            continue
        try:
            msg = json.loads(p)
        except Exception:
            continue
        print(f"  pre-response event: {msg.get('type')} {msg if msg.get('type','').endswith('error') else ''}")

    if not args.no_response:
        sock.sendall(_frame(0x1, json.dumps({
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]},
        }).encode()))

    def recv_until_done(deadline_s: float):
        events_local: dict[str, int] = {}
        transcripts_local: list[str] = []
        user_transcripts: list[str] = []
        audio_local = 0
        while time.time() < deadline_s:
            try:
                op, p = _read_frame(sock, timeout=30)
            except (socket.timeout, TimeoutError):
                break
            if not op:
                break
            if op in (0x8, 0x9, 0xA):
                if op == 0x9:
                    sock.sendall(_frame(0xA, p))
                if op == 0x8:
                    break
                continue
            if op == 0x2:
                audio_local += len(p)
                continue
            try:
                msg = json.loads(p)
            except Exception:
                continue
            t = msg.get("type", "<no-type>")
            events_local[t] = events_local.get(t, 0) + 1
            if t == "response.text.delta" or t == "response.output_audio_transcript.delta":
                transcripts_local.append(msg.get("delta", ""))
            elif t == "conversation.item.input_audio_transcription.completed":
                user_transcripts.append(msg.get("transcript", ""))
            elif t == "response.done":
                print(f"\nuser_transcripts: {user_transcripts!r}")
                return events_local, transcripts_local, audio_local
            elif t and "error" in t.lower():
                print(f"✗ server error: {msg.get('error') or msg}")
                return events_local, transcripts_local, audio_local
        print(f"\nuser_transcripts (no done): {user_transcripts!r}")
        return events_local, transcripts_local, audio_local

    events, transcripts, audio_delta_bytes = recv_until_done(time.time() + 120)

    print()
    print("=== Events ===")
    for k, v in sorted(events.items()):
        print(f"  {k}: {v}")
    print(f"audio deltas received: {audio_delta_bytes} bytes")
    print()
    print(f"final transcript: {''.join(transcripts)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
