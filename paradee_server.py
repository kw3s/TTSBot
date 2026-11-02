#!/usr/bin/env python3
"""Paradee TTS micro-service: wraps the local Paradee model in HTTP.

Paradee (https://github.com/sahilmahendrakar/paradee) is a tiny on-device
ONNX model - it cannot run inside Vercel serverless functions, so it lives
here instead, on any machine that can host Python + onnxruntime (a laptop,
a VPS, the Instabox trial-box, ...). The Telegram bot reaches it through
the thin client in paradee.py, exactly like it reaches the IndexTTS and
AuK HuggingFace Spaces.

Run (inside the venv where paradee is installed):
    PARADEE_API_KEY=<secret> python3 paradee_server.py
    # serves on 0.0.0.0:$PARADEE_PORT (default 8765)

API:
    GET  /        -> {"ok": true, "service": "paradee", "voices": [...]}
    POST /tts     -> audio/wav bytes
        JSON body: {"text": "...", "voice": "af_heart", "speed": 1.0}
        Header X-API-Key must match $PARADEE_API_KEY when that is set.

Stdlib only (plus paradee/numpy) - no extra web framework to install.
"""

import io
import json
import os
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# The sibling paradee.py in this repo is the bot's *client* module. When
# this server is started from the repo directory, Python puts that
# directory first on sys.path and `import paradee` would load the client
# instead of the installed paradee package. Drop the script's directory
# from sys.path so the real package is always found.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path
            if os.path.abspath(p or ".") != _SCRIPT_DIR]

API_KEY = os.environ.get("PARADEE_API_KEY", "")
PORT = int(os.environ.get("PARADEE_PORT", "8765"))
VOICES = ["af_heart", "am_puck", "bf_isabella", "bm_fable"]

_engine = None
_lock = threading.Lock()


def engine():
    """Load the model once, lazily (first request pays the load cost)."""
    global _engine
    if _engine is None:
        from paradee import Paradee
        _engine = Paradee(version="1.2", quantized=True)
    return _engine


def synth_wav(text, voice="af_heart", speed=1.0):
    from paradee import SAMPLE_RATE
    with _lock:  # the G2P pipeline is not thread-safe; serialise requests
        audio = engine()(text, speed=speed, voice=voice)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._json(200, {"ok": True, "service": "paradee",
                             "voices": VOICES})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/tts":
            self._json(404, {"ok": False, "error": "not found"})
            return
        if API_KEY and self.headers.get("X-API-Key") != API_KEY:
            self._json(401, {"ok": False, "error": "bad api key"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            self._json(400, {"ok": False, "error": "invalid JSON body"})
            return
        text = (payload.get("text") or "").strip()
        voice = payload.get("voice") or "af_heart"
        try:
            speed = float(payload.get("speed") or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        if not text:
            self._json(400, {"ok": False, "error": "text is empty"})
            return
        if voice not in VOICES:
            self._json(400, {"ok": False,
                             "error": f"unknown voice {voice!r}; "
                                      f"use one of {VOICES}"})
            return
        try:
            wav = synth_wav(text, voice=voice, speed=speed)
        except Exception as exc:  # keep the service alive on bad input
            self._json(500, {"ok": False, "error": f"synthesis failed: {exc}"})
            return
        self._send(200, wav, "audio/wav")


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"paradee server on 0.0.0.0:{PORT} "
          f"(api key {'set' if API_KEY else 'NOT set - open endpoint'})",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
