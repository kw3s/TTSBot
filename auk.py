#!/usr/bin/env python3
"""Client for the official AuK HuggingFace Space (Tencent-Hunyuan AuK).

AuK is Tencent's 1.5B speech generation + editing foundation model. The official
Space exposes a public Gradio queue API, so this client needs nothing but the
standard library (works on Termux, Vercel, anywhere).

Gradio queue protocol used here:
    POST /gradio_api/upload                            -> server-side file path
    POST /gradio_api/call/run_generate_with_pe         -> {"event_id": ...}
    GET  /gradio_api/call/run_generate_with_pe/<id>    (SSE) -> result file

Published endpoint parameters (the API lists 8, but the function takes 10 —
ref_text and gen_text are hidden from /info and must still be sent, else the
positional data array misaligns and the Space answers "error: null"):
    use_pe        bool    Prompt Enhancer. When False, gen_seconds must be > 0.
    variant       str     "AuK (Base)" or "AuK-Flash ⚡"
    audio         file    reference audio (upload first). None => Instruct TTS.
    instruction   str     natural-language instruction; carries the text to speak
    gen_seconds   float   target duration; 0 is only valid with use_pe=True
    ref_text      str     optional transcript of the reference audio (None is fine)
    gen_text      str     optional target text override (None is fine)
    nfe           int     base-model sampling steps (Flash ignores it)
    cfg           float   base-model CFG strength (Flash ignores it)
    seed          int     RNG seed

Instruction templates (from the official cookbook):
    zero-shot:  Say the following with the same voice: "{text}"
    instruct:   Based on the following description: "{voice}", generate speech content "{text}".

CLI:
    python3 auk.py "text to speak" --ref voice.wav -o out.wav      # zero-shot clone
    python3 auk.py "text to speak" --instruct "warm male narrator" --seconds 5
    python3 auk.py "text" --variant flash                          # fast 4-step
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

SPACE = os.environ.get("AUK_SPACE", "https://tencent-auk.hf.space").rstrip("/")
ENDPOINT = "run_generate_with_pe"
HF_TOKEN = os.environ.get("HF_TOKEN") or None

VARIANT_BASE = "AuK (Base)"
VARIANT_FLASH = "AuK-Flash ⚡"
VARIANT_ALIASES = {
    "base": VARIANT_BASE,
    "auk": VARIANT_BASE,
    "flash": VARIANT_FLASH,
    "auk-flash": VARIANT_FLASH,
}

# Speech rate used to size gen_seconds when the caller doesn't supply one.
# ~15 characters/second of English is a decent average for this model.
CHARS_PER_SECOND = 15.0
MIN_SECONDS = 1.5
# ZeroGPU has a per-request budget; keep any single call comfortably inside it.
MAX_SECONDS = 25.0


def _headers(extra=None):
    hdrs = {"User-Agent": "ttsbot-auk/1.0"}
    if HF_TOKEN:
        hdrs["Authorization"] = f"Bearer {HF_TOKEN}"
    hdrs.update(extra or {})
    return hdrs


def _post(url, body=None, headers=None, timeout=180, retries=4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=_headers(headers))
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            # 4xx other than 429 won't fix themselves on retry.
            if exc.code != 429 and 400 <= exc.code < 500:
                raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from None
            last = RuntimeError(f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {retries} tries: {last}")


def resolve_variant(name):
    if not name:
        return VARIANT_FLASH
    if name in (VARIANT_BASE, VARIANT_FLASH):
        return name
    key = str(name).strip().lower()
    if key in VARIANT_ALIASES:
        return VARIANT_ALIASES[key]
    raise ValueError(f"unknown variant {name!r}; use base/flash")


def estimate_seconds(text, speed=1.0):
    """Rough target duration for `text`; `speed` > 1 means slower/longer."""
    secs = max(MIN_SECONDS, len(text) / CHARS_PER_SECOND) * float(speed or 1.0)
    return round(min(MAX_SECONDS, secs), 2)


def upload_ref(wav_bytes, filename="ref.wav"):
    """Upload a reference clip; returns the server-side path Gradio gave us."""
    boundary = "auk" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    out = _post(
        f"{SPACE}/gradio_api/upload",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=300,
    )
    files = json.loads(out)
    if not files:
        raise RuntimeError("upload returned no file path")
    return files[0]


def _get(url, timeout=300, retries=4):
    """GET bytes (result files are served over GET; POST returns 405)."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and 400 <= exc.code < 500:
                raise RuntimeError(f"HTTP {exc.code} from {url}") from None
            last = exc
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def _extract_audio(value, timeout=300):
    """Pull audio bytes out of whatever shape the Space returned."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.startswith("data:audio") and "," in value:
            return base64.b64decode(value.split(",", 1)[1])
        if value.startswith("http"):
            return _get(value, timeout=timeout)
        if value.startswith("/"):
            return _get(f"{SPACE}/gradio_api/file={value}", timeout=timeout)
        return None
    if isinstance(value, dict):
        for key in ("url", "path"):
            got = _extract_audio(value.get(key), timeout=timeout)
            if got:
                return got
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            got = _extract_audio(item, timeout=timeout)
            if got:
                return got
        return None
    return None


def generate(
    instruction,
    ref_path=None,
    variant=VARIANT_FLASH,
    use_pe=False,
    gen_seconds=None,
    ref_text=None,
    gen_text=None,
    nfe=32,
    cfg=2.0,
    seed=42,
    timeout=600,
    on_status=None,
):
    """Synthesize audio. `instruction` carries the text (see module docstring).

    `ref_path` is a local path (uploaded here) or an already-uploaded server path.
    Returns WAV bytes.
    """
    if not instruction or not instruction.strip():
        raise ValueError("instruction must not be empty")

    variant = resolve_variant(variant)

    audio_field = None
    if ref_path:
        ref_str = str(ref_path)
        if os.path.exists(ref_str):
            # Local file: upload it and use the server-side path we get back.
            with open(ref_str, "rb") as fh:
                server_path = upload_ref(fh.read(), filename=os.path.basename(ref_str))
        else:
            # Not on this disk, so assume it is already an uploaded server path.
            server_path = ref_str
        # Gradio 6 only accepts the file reference with a nested meta._type;
        # a bare path (or a top-level _type) makes the Space answer error: null.
        audio_field = {"path": server_path, "meta": {"_type": "gradio.FileData"}}

    if use_pe:
        seconds = float(gen_seconds or 0)
    else:
        seconds = float(gen_seconds or estimate_seconds(instruction))
        if seconds <= 0:
            raise ValueError("gen_seconds must be > 0 when use_pe is False")

    # NOTE: 10 positional args. /gradio_api/info only advertises 8; ref_text and
    # gen_text must still be present or the remaining args shift and the Space
    # returns {"error": null} without ever running the model.
    data = [bool(use_pe), variant, audio_field, instruction,
            seconds, ref_text, gen_text, int(nfe), float(cfg), int(seed)]

    payload = json.dumps({"data": data}).encode()
    out = _post(
        f"{SPACE}/gradio_api/call/{ENDPOINT}",
        payload,
        {"Content-Type": "application/json"},
        timeout=180,
    )
    event_id = json.loads(out).get("event_id")
    if not event_id:
        raise RuntimeError(f"no event_id in response: {out[:200]}")

    url = f"{SPACE}/gradio_api/call/{ENDPOINT}/{event_id}"
    req = urllib.request.Request(url, headers=_headers({"Accept": "text/event-stream"}))
    event = None
    with urllib.request.urlopen(req, timeout=timeout) as stream:
        for raw in stream:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload_line = line.split(":", 1)[1].strip()
                if event == "complete":
                    result = json.loads(payload_line)
                    audio = _extract_audio(result)
                    if not audio:
                        raise RuntimeError(f"complete event had no audio: {str(result)[:300]}")
                    return audio
                if event == "error":
                    raise RuntimeError(f"space error: {payload_line[:300]}")
                if event in ("process_starts", "generating", "estimation") and on_status:
                    on_status(event)
    raise RuntimeError(f"stream ended without a result (last event: {event})")


def main():
    parser = argparse.ArgumentParser(description="AuK speech generation via the official HF Space")
    parser.add_argument("text", help="text to speak")
    parser.add_argument("-o", "--output", default="auk_out.wav")
    parser.add_argument("--ref", default=None, help="local reference .wav (zero-shot voice clone)")
    parser.add_argument("--instruct", default=None,
                        help="voice description; enables Instruct TTS (no reference needed)")
    parser.add_argument("--variant", default="flash", help="base|flash (default: flash)")
    parser.add_argument("--seconds", type=float, default=None,
                        help="target duration in seconds (default: estimated)")
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pe", action="store_true", help="enable the Prompt Enhancer")
    args = parser.parse_args()

    if args.instruct:
        instruction = (f'Based on the following description: "{args.instruct}", '
                       f'generate speech content "{args.text}".')
    else:
        instruction = f'Say the following with the same voice: "{args.text}"'

    print(f"variant={resolve_variant(args.variant)} pe={args.pe} "
          f"ref={args.ref or 'none'} seconds={args.seconds or 'auto'}", file=sys.stderr)
    audio = generate(
        instruction,
        ref_path=args.ref,
        variant=args.variant,
        use_pe=args.pe,
        gen_seconds=args.seconds,
        nfe=args.nfe,
        cfg=args.cfg,
        seed=args.seed,
        on_status=lambda e: print(f"  status: {e}", file=sys.stderr),
    )
    with open(args.output, "wb") as fh:
        fh.write(audio)
    print(f"wrote {len(audio)} bytes to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
