#!/usr/bin/env python3
"""fish-tts-bot: Telegram front end for tts.py.

Send the bot a message -> it replies with spoken audio (mp3).
Send a .txt / .md / plain-text document -> same, using the file's contents.

Setup:
    Put TELEGRAM_BOT_TOKEN=<token from @BotFather> in ./env next to this
    script (or ~/.config/fish-tts/.env, or export it).

Run:
    python3 bot.py
"""

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

from tts import get_api_key, synthesize

TG_API = "https://api.telegram.org"
MAX_CHUNK = 900          # characters per synthesis request
POLL_TIMEOUT = 50
ITT_SPACE = os.environ.get("INDEX_TTS_SPACE", "D300274/IndexTTS-2.5-Demo")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_REF = os.environ.get("ITTS_DEFAULT_REF") or os.path.join(
    SCRIPT_DIR, "refs", "voice_01.wav")
if not os.path.isabs(BUNDLED_REF):
    BUNDLED_REF = os.path.join(SCRIPT_DIR, BUNDLED_REF)
SERVERLESS = bool(os.environ.get("VERCEL"))


def _cap(default_value, env_name):
    try:
        return max(1, int(os.environ[env_name]))
    except (KeyError, ValueError):
        return default_value


# serverless invocations must finish inside Vercel's maxDuration, so cap work
MAX_CHUNKS = _cap(8 if SERVERLESS else 30, "FISH_MAX_CHUNKS")
ITT_MAX_CHUNKS = _cap(1 if SERVERLESS else 6, "ITTS_MAX_CHUNKS")
DATA_DIR = "/tmp/fish-tts" if SERVERLESS else SCRIPT_DIR
REF_DIR = os.path.join(DATA_DIR, "refs")   # writable custom-voice refs
ENGINES_FILE = os.path.join(DATA_DIR, "engines.json")
SPEEDS_FILE = os.path.join(DATA_DIR, "speeds.json")
PENDING_VOICE = set()    # chat_ids waiting to record a reference clip


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def log(*parts):
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def call_tg(token, method, payload=None, retries=4):
    """POST JSON to the Bot API with basic network retries."""
    url = f"{TG_API}/bot{token}/{method}"
    data = json.dumps(payload or {}).encode("utf-8")
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=POLL_TIMEOUT + 30) as resp:
                body = json.loads(resp.read())
            if not body.get("ok"):
                raise RuntimeError(f"{method}: {body}")
            return body["result"]
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            log(f"net error on {method} (attempt {attempt + 1}): {exc}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{method} failed after {retries} attempts: {last}")


def download_doc(token, file_id):
    meta = call_tg(token, "getFile", {"file_id": file_id})
    url = f"{TG_API}/file/bot{token}/{meta['file_path']}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def split_text(text):
    """Split into synthesis-friendly chunks at sentence boundaries."""
    pieces = [p for p in re.split(r"(?<=[.!?;。！？])\s+|\n{2,}", text.strip()) if p]
    chunks, cur = [], ""
    for piece in pieces:
        while len(piece) > MAX_CHUNK:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(piece[:MAX_CHUNK])
            piece = piece[MAX_CHUNK:]
        if cur and len(cur) + len(piece) + 1 > MAX_CHUNK:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip() if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def synthesize_full(text, api_key, model, voice):
    chunks = split_text(text)[:MAX_CHUNKS]
    if len(chunks) == 1:
        return synthesize(chunks[0], api_key, model, voice, "mp3", None, None), 1
    parts = []
    for i, chunk in enumerate(chunks, 1):
        log(f"synthesizing chunk {i}/{len(chunks)}")
        parts.append(synthesize(chunk, api_key, model, voice, "mp3", None, None))
    return b"".join(parts), len(chunks)


# ---------- IndexTTS-2.5 engine (private HF Space) ----------

_ITTS_CLIENT = None


def itts_client():
    global _ITTS_CLIENT
    if _ITTS_CLIENT is None:
        from gradio_client import Client
        _ITTS_CLIENT = Client(ITT_SPACE,
                              token=os.environ.get("HF_TOKEN"), verbose=False,
                              httpx_kwargs={"timeout": 300})
    return _ITTS_CLIENT


def synthesize_itts_chunk(text, ref_path, lang="EN", speed=1.0):
    from gradio_client import handle_file
    out = itts_client().predict(
        emo_control_method="Same as the voice reference",
        prompt=handle_file(ref_path),
        text=text,
        lang_choice=lang,
        emo_ref_path=None,
        emo_weight=0.65,
        vec1=0.0, vec2=0.0, vec3=0.0, vec4=0.0,
        vec5=0.0, vec6=0.0, vec7=0.0, vec8=0.0,
        emo_text="", emo_random=False,
        max_text_tokens_per_segment=120,
        duration_factor=float(speed),
        param_18=True, param_19=0.8, param_20=30, param_21=0.8,
        param_22=0.0, param_23=3, param_24=10.0, param_25=1500,
        api_name="/gen_single",
    )
    b64 = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else None
    if not b64 or not b64.startswith("data:audio/wav;base64,"):
        raise RuntimeError(f"unexpected space response: {str(out)[:200]}")
    return base64.b64decode(b64.split(",", 1)[1])


def concat_wavs(parts):
    """Concatenate same-format wav bytes into one wav."""
    out = None
    frames = []
    params = None
    for blob in parts:
        with wave.open(io.BytesIO(blob)) as w:
            if params is None:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams(params)
        for f in frames:
            w.writeframes(f)
    return buf.getvalue()


ITT_WARMUP = "Hi there. "


def _trim_wav_intro(wav_bytes, min_keep=1.0):
    """Cut the robotic warm-up intro: drop everything before the end of the
    first silence detected after `min_keep` seconds."""
    import tempfile
    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        return wav_bytes
    fin = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    fout = fin.name + ".trim.wav"
    try:
        fin.write(wav_bytes)
        fin.close()
        proc = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "info", "-i", fin.name,
             "-af", "silencedetect=noise=-35dB:d=0.22", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120)
        ends = [float(m) for m in
                re.findall(r"silence_end:\s*([0-9.]+)", proc.stderr)]
        cuts = [t for t in ends if t >= min_keep]
        if not cuts:
            return wav_bytes
        r = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{cuts[0]:.2f}",
             "-i", fin.name, "-c:a", "pcm_s16le", fout],
            capture_output=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(fout):
            return wav_bytes
        with open(fout, "rb") as fh:
            return fh.read()
    except Exception:
        log(traceback.format_exc())
        return wav_bytes
    finally:
        try:
            os.remove(fin.name)
        except OSError:
            pass
        if os.path.exists(fout):
            os.remove(fout)


def synthesize_full_itts(text, ref_path, lang="EN", speed=1.0):
    chunks = split_text(text)[:ITT_MAX_CHUNKS]
    parts = []
    for i, chunk in enumerate(chunks, 1):
        log(f"itts chunk {i}/{len(chunks)}")
        raw = synthesize_itts_chunk(ITT_WARMUP + chunk, ref_path, lang, speed)
        parts.append(_trim_wav_intro(raw))
    if len(parts) == 1:
        return parts[0], 1
    return concat_wavs(parts), len(chunks)


# ---------- engine preference (per chat) ----------

def get_engine(chat_id):
    fallback = os.environ.get("DEFAULT_ENGINE", "fish")
    try:
        with open(ENGINES_FILE) as fh:
            return json.load(fh).get(str(chat_id)) or fallback
    except Exception:
        return fallback


def set_engine(chat_id, name):
    data = {}
    try:
        with open(ENGINES_FILE) as fh:
            data = json.load(fh)
    except Exception:
        pass
    data[str(chat_id)] = name
    with open(ENGINES_FILE, "w") as fh:
        json.dump(data, fh)


def custom_ref(chat_id):
    path = os.path.join(REF_DIR, f"custom_{chat_id}.wav")
    return path if os.path.exists(path) else BUNDLED_REF


def get_speed(chat_id):
    try:
        with open(SPEEDS_FILE) as fh:
            v = float(json.load(fh)[str(chat_id)])
            if 0.5 <= v <= 2.0:
                return v
    except Exception:
        pass
    return 1.0


def set_speed(chat_id, value):
    data = {}
    try:
        with open(SPEEDS_FILE) as fh:
            data = json.load(fh)
    except Exception:
        pass
    data[str(chat_id)] = value
    with open(SPEEDS_FILE, "w") as fh:
        json.dump(data, fh)


def send_audio(token, chat_id, audio, caption, fmt="mp3"):
    mime = "audio/wav" if fmt == "wav" else "audio/mpeg"
    boundary = "fishtts" + uuid.uuid4().hex
    fields = {"chat_id": str(chat_id), "caption": caption[:1000]}
    parts = []
    for key, value in fields.items():
        if value:
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
            )
    parts.append(
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="audio"; '
        f'filename="tts.{fmt}"\r\nContent-Type: {mime}\r\n\r\n'
    )
    body = "".join(parts).encode("utf-8") + audio + f"\r\n--{boundary}--\r\n".encode()
    url = f"{TG_API}/bot{token}/sendAudio"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            if not result.get("ok"):
                raise RuntimeError(f"sendAudio: {result}")
            return
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            log(f"net error on sendAudio (attempt {attempt + 1}): {exc}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"sendAudio failed after retries: {last}")


def reply(token, chat_id, text):
    call_tg(token, "sendMessage", {
        "chat_id": chat_id,
        "text": text[:4000],
        "parse_mode": "HTML",
    })


def handle_message(token, api_key, model, voice, msg):
    chat_id = msg["chat"]["id"]
    user = msg.get("from", {}).get("first_name", "there")

    text_in = msg.get("text", "").strip()
    if text_in.startswith("/"):
        cmd = text_in.split()[0].split("@")[0]
        if cmd in ("/start", "/help"):
            reply(token, chat_id,
                  "<b>fish-tts bot</b>\n"
                  "Send me any text and I'll speak it back as audio.\n"
                  "Or send a .txt / .md file and I'll read the whole thing.\n\n"
                  "<b>Engines</b>\n"
                  "/engine - show current engine\n"
                  "/engine fish - Fish Audio s2.1 (free, fast)\n"
                  f"/engine itts - IndexTTS-2.5 voice clone ({ITT_MAX_CHUNKS * MAX_CHUNK // 3} chars max)\n"
                  "/setvoice - clone a custom voice: attach or send a "
                  "voice message right after this command (~10s of clean speech)\n"
                  "/cancel - abort pending voice setup")
        elif cmd == "/engine":
            parts = text_in.split(maxsplit=1)
            if len(parts) == 1:
                cur = get_engine(chat_id)
                reply(token, chat_id,
                      f"Current engine: <b>{cur}</b>.\n"
                      "Switch with /engine fish or /engine itts")
            elif parts[1].strip().lower() in ("fish", "itts", "indextts"):
                name = "itts" if parts[1].strip().lower().startswith("i") else "fish"
                set_engine(chat_id, name)
                ref_note = ""
                if name == "itts" and not os.path.exists(
                        os.path.join(REF_DIR, f"custom_{chat_id}.wav")):
                    ref_note = ("\nHeads-up: you're on the default sample voice. "
                                "/setvoice to clone your own.")
                reply(token, chat_id, f"Engine set to <b>{name}</b>.{ref_note}")
            else:
                reply(token, chat_id, "Unknown engine. Use /engine fish or /engine itts")
        elif cmd == "/speed":
            parts = text_in.split(maxsplit=1)
            if len(parts) == 1:
                reply(token, chat_id,
                      f"Current speed factor: <b>{get_speed(chat_id)}</b> "
                      "(1.0 = normal, higher = slower, range 0.5-2.0). "
                      "Try: /speed 1.15")
            else:
                try:
                    v = float(parts[1])
                    if not 0.5 <= v <= 2.0:
                        raise ValueError
                except ValueError:
                    reply(token, chat_id, "Give me a number between 0.5 and 2.0, e.g. /speed 1.15")
                    return
                set_speed(chat_id, v)
                reply(token, chat_id, f"Speed factor set to <b>{v}</b> (itts engine only).")
        elif cmd == "/setvoice":
            PENDING_VOICE.add(chat_id)
            reply(token, chat_id,
                  "Send me a voice message or audio clip now - about 5-30 seconds "
                  "of clean speech from the person to clone.\n/cancel to abort.")
        elif cmd == "/cancel":
            PENDING_VOICE.discard(chat_id)
            reply(token, chat_id, "Okay, cancelled.")
        else:
            reply(token, chat_id, "Unknown command. Try /help")
        return

    # custom-voice capture
    if chat_id in PENDING_VOICE and (msg.get("voice") or msg.get("audio")):
        src = msg.get("voice") or msg.get("audio")
        try:
            ffmpeg = ffmpeg_bin()
            if not ffmpeg:
                raise RuntimeError("ffmpeg is not available in this environment")
            raw_bytes = download_doc(token, src["file_id"])
            os.makedirs(REF_DIR, exist_ok=True)
            tmp_in = os.path.join(REF_DIR, f"in_{chat_id}.bin")
            with open(tmp_in, "wb") as fh:
                fh.write(raw_bytes)
            out_path = os.path.join(REF_DIR, f"custom_{chat_id}.wav")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                            "-i", tmp_in, "-ar", "22050", "-ac", "1", out_path],
                           check=True, timeout=120)
            os.remove(tmp_in)
            PENDING_VOICE.discard(chat_id)
            reply(token, chat_id,
                  "Voice saved! I'll use it for IndexTTS from now on. "
                  "Make sure /engine is set to itts.")
        except Exception as exc:
            log(traceback.format_exc())
            reply(token, chat_id, f"Couldn't process that audio: {exc}")
        return

    doc = msg.get("document")
    source_text = None
    label = "your message"
    if doc:
        name = doc.get("file_name", "")
        mime = doc.get("mime_type", "")
        ok_ext = name.lower().endswith((".txt", ".md", ".csv", ".log"))
        if not (ok_ext or mime.startswith("text/")):
            reply(token, chat_id, f"'{name}' doesn't look like a text file "
                                  "(.txt/.md/csv/log only).")
            return
        if doc.get("file_size", 0) > 10_000_000:
            reply(token, chat_id, "File too large (max ~10 MB of text).")
            return
        raw = download_doc(token, doc["file_id"])
        source_text = raw.decode("utf-8", "replace")
        label = f"'{name}'"
    else:
        source_text = msg.get("text", "")

    source_text = (source_text or "").strip()
    if not source_text:
        reply(token, chat_id, "I didn't find any text to speak. Try /help")
        return

    total_chunks = len(split_text(source_text))
    engine = get_engine(chat_id)
    if engine == "itts":
        max_chunks = ITT_MAX_CHUNKS
        limit_note = (f"That's ~{total_chunks} chunks but IndexTTS can only do "
                      f"{ITT_MAX_CHUNKS} per request - I'll read the first part only.")
    else:
        max_chunks = MAX_CHUNKS
        limit_note = (f"That's ~{total_chunks} chunks but I can only do "
                      f"{MAX_CHUNKS} per request - I'll read the first part only.")
    if total_chunks > max_chunks:
        reply(token, chat_id, limit_note)
        truncated = True
    else:
        truncated = False

    reply(token, chat_id, f"One sec {user}, synthesizing {label} ({engine})...")
    call_tg(token, "sendChatAction", {"chat_id": chat_id, "action": "record_voice"})

    try:
        if engine == "itts":
            audio, used = synthesize_full_itts(
                source_text, custom_ref(chat_id), speed=get_speed(chat_id))
            fmt = "wav"
        else:
            audio, used = synthesize_full(source_text, api_key, model, voice)
            fmt = "mp3"
    except SystemExit:
        raise
    except Exception as exc:
        log(traceback.format_exc())
        msg_str = str(exc)
        if "quota" in msg_str.lower():
            reply(token, chat_id,
                  "Daily ZeroGPU quota is used up - it resets 24h after your "
                  "first GPU use today. /engine fish meanwhile?")
        else:
            reply(token, chat_id, f"Synthesis failed: {msg_str[:300]}")
        return

    note = " (truncated)" if truncated else ""
    send_audio(token, chat_id, audio,
               f"[{engine}] spoken {label}{note} - "
               f"{used} chunk{'s' if used != 1 else ''}", fmt)
    log(f"served {len(audio)} bytes to chat {chat_id}")


_SEEN_UPDATES = {}


def seen_before(update_id):
    """Best-effort dedupe of Telegram webhook retries across warm instances."""
    now = time.time()
    if len(_SEEN_UPDATES) > 2000:
        for key in [k for k, ts in _SEEN_UPDATES.items() if now - ts > 21600]:
            del _SEEN_UPDATES[key]
    if update_id in _SEEN_UPDATES:
        return True
    _SEEN_UPDATES[update_id] = now
    return False


def process_update(update):
    from tts import load_env_file
    load_env_file(os.path.join(SCRIPT_DIR, ".env"))
    load_env_file(os.path.expanduser("~/.config/fish-tts/.env"))

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        log("error: TELEGRAM_BOT_TOKEN not set")
        return
    api_key = get_api_key(None)
    if not api_key:
        log("error: FISH_API_KEY not set")
        return
    model = os.environ.get("FISH_TTS_MODEL", "s2.1-pro-free")
    voice = os.environ.get("FISH_VOICE") or None

    msg = update.get("message")
    if not msg:
        return
    uid = update.get("update_id")
    if uid is not None and seen_before(uid):
        return
    try:
        handle_message(token, api_key, model, voice, msg)
    except SystemExit:
        raise
    except Exception as exc:
        log(traceback.format_exc())
        try:
            reply(token, msg["chat"]["id"], f"Something broke: {exc}")
        except Exception:
            pass


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        from tts import load_env_file, script_dir
        load_env_file(os.path.join(script_dir(), ".env"))
        load_env_file(os.path.expanduser("~/.config/fish-tts/.env"))
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("error: TELEGRAM_BOT_TOKEN not set. Get one from @BotFather and "
              "put it in ./env as TELEGRAM_BOT_TOKEN=<token>", file=sys.stderr)
        return 2
    api_key = get_api_key(None)
    if not api_key:
        print("error: FISH_API_KEY not set (needed by tts.py)", file=sys.stderr)
        return 2
    model = os.environ.get("FISH_TTS_MODEL", "s2.1-pro-free")
    voice = os.environ.get("FISH_VOICE") or None

    me = call_tg(token, "getMe")
    log(f"running as @{me['username']} (model={model}, voice={voice or 'default'})")

    offset = 0
    while True:
        try:
            updates = call_tg(token, "getUpdates", {
                "offset": offset,
                "timeout": POLL_TIMEOUT,
                "allowed_updates": ["message"],
            })
        except Exception as exc:
            log(f"poll error, backing off: {exc}")
            time.sleep(5)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            msg = update.get("message")
            if not msg:
                continue
            try:
                handle_message(token, api_key, model, voice, msg)
            except SystemExit:
                raise
            except Exception as exc:
                log(traceback.format_exc())
                try:
                    reply(token, msg["chat"]["id"], f"Something broke: {exc}")
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
