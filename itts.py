#!/usr/bin/env python3
"""Client for the IndexTTS-2.5 HuggingFace Space demo API (no deps).

Uses the public Gradio queue protocol:
    POST /gradio_api/upload          -> server-side file path
    POST /gradio_api/call/gen_single -> {"event_id": ...}
    GET  /gradio_api/call/gen_single/<id> (SSE) -> result file url

CLI:
    python3 itts.py "text to speak" [-o out.wav] [--ref voice.wav]
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

SPACE = os.environ.get("INDEX_TTS_SPACE",
                       "https://indexteam-indextts-2-5-demo.hf.space")

DEFAULTS = dict(
    emo_mode="Same as the voice reference",
    emo_weight=0.65,
    max_text_tokens_per_segment=120,
    duration_factor=1.0,
    do_sample=True,
    top_p=0.8,
    top_k=30,
    temperature=0.8,
    length_penalty=0.0,
    num_beams=3,
    repetition_penalty=10.0,
    max_mel_tokens=1500,
)


def _post(url, body=None, headers=None, timeout=120):
    last = None
    for attempt in range(4):
        try:
            hdrs = {"User-Agent": "fish-tts-bot/1.0"}
            hdrs.update(headers or {})
            req = urllib.request.Request(url, data=body, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed: {last}")


def _get_stream(url, timeout=600):
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    return urllib.request.urlopen(req, timeout=timeout)


def upload_ref(wav_bytes, filename="prompt.wav"):
    boundary = "itts" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()
    out = _post(
        f"{SPACE}/gradio_api/upload",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=180,
    )
    files = json.loads(out)
    if not files:
        raise RuntimeError("upload returned no file path")
    return files[0]


def generate(text, ref_server_path, lang="EN"):
    data = [
        DEFAULTS["emo_mode"],           # emo_control_method
        ref_server_path,                # prompt (voice reference)
        text,                           # text
        lang,                           # lang_choice
        None,                           # emo_ref_path
        DEFAULTS["emo_weight"],
        *[0.0] * 8,                     # vec1..vec8 (unused in mode 0)
        "",                             # emo_text
        False,                          # emo_random
        DEFAULTS["max_text_tokens_per_segment"],
        DEFAULTS["duration_factor"],
        DEFAULTS["do_sample"],
        DEFAULTS["top_p"],
        DEFAULTS["top_k"],
        DEFAULTS["temperature"],
        DEFAULTS["length_penalty"],
        DEFAULTS["num_beams"],
        DEFAULTS["repetition_penalty"],
        DEFAULTS["max_mel_tokens"],
    ]
    payload = json.dumps({"data": data}).encode()
    out = _post(
        f"{SPACE}/gradio_api/call/gen_single",
        payload,
        {"Content-Type": "application/json"},
        timeout=120,
    )
    event_id = json.loads(out).get("event_id")
    if not event_id:
        raise RuntimeError(f"no event_id in response: {out[:200]}")

    status = ""
    with _get_stream(f"{SPACE}/gradio_api/call/gen_single/{event_id}") as stream:
        event = None
        for raw in stream:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event == "complete":
                result = json.loads(line.split(":", 1)[1].strip())
                entry = result[0]
                url = entry.get("url") if isinstance(entry, dict) else None
                path = entry.get("path") if isinstance(entry, dict) else str(entry)
                if not url:
                    url = f"{SPACE}/gradio_api/file={path}"
                return _post(url, b"", {}, timeout=300)
            elif line.startswith("data:") and event == "error":
                raise RuntimeError(f"space error: {line[:300]}")
            elif line.startswith("data:") and event == "process_starts":
                print("generation started...", file=sys.stderr)
    raise RuntimeError(f"stream ended without result (last event: {status or event})")


def main():
    parser = argparse.ArgumentParser(description="IndexTTS-2.5 via HF demo Space")
    parser.add_argument("text", help="text to synthesize")
    parser.add_argument("-o", "--output", default="itts_out.wav")
    parser.add_argument("--ref", default=None,
                        help="local .wav voice reference "
                             "(default: downloads an example voice)")
    parser.add_argument("--lang", default="EN",
                        choices=["EN", "ZH", "JA", "ES", "AR"])
    args = parser.parse_args()

    if args.ref:
        with open(args.ref, "rb") as fh:
            ref_bytes = fh.read()
    else:
        print("no --ref given, downloading example voice...", file=sys.stderr)
        url = ("https://huggingface.co/spaces/IndexTeam/IndexTTS-2.5-Demo/"
               "resolve/main/examples/voice_01.wav")
        ref_bytes = _post(url, b"", {}, timeout=180)

    print(f"uploading reference ({len(ref_bytes)} bytes)...", file=sys.stderr)
    server_path = upload_ref(ref_bytes)

    print("queued on ZeroGPU, waiting...", file=sys.stderr)
    audio = generate(args.text, server_path, args.lang)
    with open(args.output, "wb") as fh:
        fh.write(audio)
    print(f"wrote {len(audio)} bytes to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
