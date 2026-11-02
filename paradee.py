#!/usr/bin/env python3
"""Client for a Paradee TTS server (see paradee_server.py). No deps.

    POST $PARADEE_URL/tts  {"text": ..., "voice": ..., "speed": ...}
        -> audio/wav bytes

Environment:
    PARADEE_URL      base url of the server (default http://127.0.0.1:8765)
    PARADEE_API_KEY  shared secret, sent as X-API-Key (optional)
    PARADEE_VOICE    default voice (default af_heart)

CLI:
    python3 paradee.py "text to speak" [-o out.wav] [--voice bf_isabella]
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

SERVER = os.environ.get("PARADEE_URL", "http://127.0.0.1:8765").rstrip("/")
API_KEY = os.environ.get("PARADEE_API_KEY", "")
DEFAULT_VOICE = os.environ.get("PARADEE_VOICE", "af_heart")
VOICES = ("af_heart", "am_puck", "bf_isabella", "bm_fable")


def generate(text, voice=None, speed=1.0, timeout=120):
    """Synthesize one chunk of text; returns WAV bytes."""
    payload = json.dumps({
        "text": text,
        "voice": voice or DEFAULT_VOICE,
        "speed": speed,
    }).encode()
    headers = {"Content-Type": "application/json",
               "User-Agent": "fish-tts-bot/1.0"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{SERVER}/tts", data=payload,
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise RuntimeError(f"paradee server HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"paradee server unreachable at {SERVER}: {last}")


def main():
    parser = argparse.ArgumentParser(
        description="Paradee TTS via a paradee_server.py instance")
    parser.add_argument("text", help="text to synthesize")
    parser.add_argument("-o", "--output", default="paradee_out.wav")
    parser.add_argument("--voice", default=None, choices=VOICES,
                        help=f"default voice (env PARADEE_VOICE or af_heart)")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    audio = generate(args.text, voice=args.voice, speed=args.speed)
    with open(args.output, "wb") as fh:
        fh.write(audio)
    print(f"wrote {len(audio)} bytes to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
