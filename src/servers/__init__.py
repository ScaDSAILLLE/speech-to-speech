"""Standalone model-server entry points for the speech-to-speech RPi fork.

Each module here implements one OpenAI-compatible HTTP service:
  - `moonshine_stt_server`: POST /v1/audio/transcriptions
  - `supertonic_tts_server`: POST /v1/audio/speech
"""
