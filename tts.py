#!/usr/bin/env python3
"""fish-tts: local text-to-speech CLI powered by Fish Audio.

Usage:
    tts.py "say this"            speak literal text
    echo "text" | tts.py         speak piped text
    tts.py                       speak current clipboard contents
    tts.py -c                    force-read clipboard
    tts.py "text" --save out.mp3 --no-play
    tts.py --model s2.1-pro --voice <reference_id> "text"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid

API_URL = "https://api.fish.audio/v1/tts"
VALID_FORMATS = ("mp3", "wav", "opus", "pcm")
VALID_MODELS = ("s2-pro", "s2.1-pro", "s2.1-pro-free", "s1")

CLIPBOARD_BACKENDS = [
    ("wl-paste", ["wl-paste", "--no-newline"], "wayland"),
    ("xclip", ["xclip", "-selection", "clipboard", "-o"], "x11"),
    ("xsel", ["xsel", "--clipboard", "--output"], "x11"),
    ("pbpaste", ["pbpaste"], "macos"),
    ("termux-clipboard-get", ["termux-clipboard-get"], "termux"),
    (
        "powershell",
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-Clipboard -Raw",
        ],
        "windows",
    ),
]

PLAY_BACKENDS = {
    "mpv": {
        "cmd": ["mpv", "--really-quiet", "--no-video", "{file}"],
        "formats": VALID_FORMATS,
    },
    "ffplay": {
        "cmd": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "{file}"],
        "formats": VALID_FORMATS,
    },
    "play": {
        "cmd": ["play", "-q", "{file}"],
        "formats": VALID_FORMATS,
    },
    "afplay": {
        "cmd": ["afplay", "{file}"],
        "formats": VALID_FORMATS,
    },
    "paplay": {
        "cmd": ["paplay", "{file}"],
        "formats": ("wav",),
    },
    "aplay": {
        "cmd": ["aplay", "-q", "{file}"],
        "formats": ("wav",),
    },
}


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def load_env_file(path):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def get_api_key(cli_value):
    if cli_value:
        return cli_value
    load_env_file(os.path.join(script_dir(), ".env"))
    load_env_file(os.path.expanduser("~/.config/fish-tts/.env"))
    return os.environ.get("FISH_API_KEY")


def which(name):
    found = shutil.which(name)
    if name == "powershell.exe" and not found and sys.platform.startswith("win"):
        return name
    return found


def read_clipboard():
    tried = []
    for name, cmd, _platform in CLIPBOARD_BACKENDS:
        exe = which(cmd[0])
        if not exe:
            tried.append(f"{name} (not installed)")
            continue
        cmd = list(cmd)
        cmd[0] = exe
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=True
            )
        except (subprocess.SubprocessError, OSError) as exc:
            tried.append(f"{name} (failed: {exc})")
            continue
        text = proc.stdout.rstrip("\r\n")
        if text.strip():
            return text
        tried.append(f"{name} (empty)")
    print("error: could not read clipboard. Tried:", file=sys.stderr)
    for entry in tried:
        print(f"  - {entry}", file=sys.stderr)
    print("hint: install one of: wl-paste (Wayland), xclip/xsel (X11), "
          "pbpaste (macOS), or 'pip install pyperclip'", file=sys.stderr)
    sys.exit(2)


def gather_text(args):
    if args.text:
        return " ".join(args.text)
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    return read_clipboard()


def pick_player(audio_format):
    for name, backend in PLAY_BACKENDS.items():
        if audio_format not in backend["formats"]:
            continue
        if which(name):
            return name, [which(name)] + backend["cmd"][1:]
    return None, None


def synthesize(text, api_key, model, voice, audio_format, speed, lang):
    payload = {"text": text, "format": audio_format}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "model": model,
    }
    if voice:
        payload["reference_id"] = voice
    if lang:
        payload["language"] = lang
    if speed is not None:
        payload["prosody"] = {"speed": float(speed)}
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace").strip()
        except Exception:
            pass
        if exc.code == 401:
            print("error: 401 unauthorized - API key rejected.", file=sys.stderr)
            print("check FISH_API_KEY in .env or your account at "
                  "https://fish.audio/app/api-keys", file=sys.stderr)
        elif exc.code == 402:
            print("error: 402 payment required - insufficient API credit.", file=sys.stderr)
            print("Fish Audio API credit is separate from platform credit. "
                  "Top up at https://fish.audio/app/developers", file=sys.stderr)
        else:
            print(f"error: HTTP {exc.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"error: cannot reach {API_URL}: {exc.reason}", file=sys.stderr)
        sys.exit(1)


def create_voice_model(api_key, wav_bytes, title):
    """Clone a voice on fish.audio from wav bytes.

    Returns the raw model dict ({'_id', 'state', ...}). Raises RuntimeError
    instead of sys.exit so callers can decide how to surface failures.
    """
    boundary = "fishtts" + uuid.uuid4().hex
    parts = []
    for key, value in (
        ("type", "tts"),
        ("title", title),
        ("train_mode", "fast"),
        ("visibility", "private"),
    ):
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        )
    parts.append(
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="voices"; '
        'filename="reference.wav"\r\nContent-Type: audio/wav\r\n\r\n'
    )
    body = "".join(parts).encode("utf-8") + wav_bytes + \
        f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.fish.audio/model",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"model create failed: HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach api.fish.audio: {exc.reason}")


def get_voice_model(api_key, model_id):
    req = urllib.request.Request(
        f"https://api.fish.audio/model/{model_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"model lookup failed: {exc}")


def play_audio(path, player_cmd):
    cmd = [part.replace("{file}", path) for part in player_cmd]
    try:
        subprocess.run(cmd, check=False)
    except OSError as exc:
        print(f"error: playback failed: {exc}", file=sys.stderr)
        sys.exit(1)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="fish-tts",
        description="Local TTS via Fish Audio. Reads clipboard when no text given.",
    )
    parser.add_argument("text", nargs="*", help="text to speak (default: clipboard)")
    parser.add_argument("-c", "--clipboard", action="store_true",
                        help="force reading text from the clipboard")
    parser.add_argument("--model", default="s2.1-pro-free", choices=VALID_MODELS,
                        help="TTS model (default: s2.1-pro-free)")
    parser.add_argument("--voice", default=None,
                        help="reference_id of a cloned/community voice")
    parser.add_argument("--format", default="mp3", choices=VALID_FORMATS,
                        dest="audio_format", help="audio format (default: mp3)")
    parser.add_argument("--speed", type=float, default=None,
                        help="speech speed multiplier, e.g. 1.5")
    parser.add_argument("--lang", default=None,
                        help="hint language, e.g. en, zh, ja")
    parser.add_argument("--save", metavar="FILE",
                        help="also write audio to FILE ('-' uses a temp name printed to stdout)")
    parser.add_argument("--no-play", action="store_true",
                        help="do not play audio locally")
    parser.add_argument("--api-key", default=None,
                        help="override API key (else $FISH_API_KEY / .env)")
    parser.add_argument("--list-backends", action="store_true",
                        help="show detected clipboard/audio backends and exit")
    args = parser.parse_args(argv)
    if args.clipboard:
        args.text = []
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.list_backends:
        print("clipboard:")
        for name, cmd, _p in CLIPBOARD_BACKENDS:
            status = "ok" if which(cmd[0]) else "missing"
            print(f"  [{status}] {name}")
        print("playback:")
        for name, backend in PLAY_BACKENDS.items():
            status = "ok" if which(name) else "missing"
            print(f"  [{status}] {name} ({', '.join(backend['formats'])})")
        return 0

    api_key = get_api_key(args.api_key)
    if not api_key:
        print("error: no API key. Put FISH_API_KEY=<key> in "
              f"{os.path.join(script_dir(), '.env')}", file=sys.stderr)
        return 2

    text = gather_text(args)
    if not text.strip():
        print("error: no text provided (args/stdin/clipboard all empty)", file=sys.stderr)
        return 2

    audio = synthesize(
        text=text,
        api_key=api_key,
        model=args.model,
        voice=args.voice,
        audio_format=args.audio_format,
        speed=args.speed,
        lang=args.lang,
    )
    print(f"synthesized {len(audio)} bytes ({args.audio_format})", file=sys.stderr)

    out_path = None
    cleanup = False
    if args.save:
        if args.save == "-":
            fd, out_path = tempfile.mkstemp(prefix="fishtts_", suffix=f".{args.audio_format}")
            os.close(fd)
            print(out_path)
        else:
            out_path = os.path.abspath(os.path.expanduser(args.save))

    if not args.no_play:
        _player, player_cmd = pick_player(args.audio_format)
        if not player_cmd:
            fallback = os.path.abspath(f"fishtts_output.{args.audio_format}")
            with open(fallback, "wb") as fh:
                fh.write(audio)
            print(f"warning: no audio player found (install mpv or ffmpeg). "
                  f"Audio saved to {fallback}", file=sys.stderr)
            out_path = fallback
        else:
            if out_path is None:
                fd, out_path = tempfile.mkstemp(
                    prefix="fishtts_", suffix=f".{args.audio_format}"
                )
                os.close(fd)
                cleanup = True
            with open(out_path, "wb") as fh:
                fh.write(audio)
            play_audio(out_path, player_cmd)

    if args.save and out_path and not os.path.exists(out_path):
        with open(out_path, "wb") as fh:
            fh.write(audio)

    if cleanup and out_path and os.path.exists(out_path):
        os.remove(out_path)
    return 0

    if cleanup and out_path and os.path.exists(out_path):
        os.remove(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
