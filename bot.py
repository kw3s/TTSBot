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

import auk
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

from tts import create_voice_model, get_api_key, get_voice_model, synthesize

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
AUK_MAX_CHUNKS = _cap(1 if SERVERLESS else 4, "AUK_MAX_CHUNKS")
# AuK runs on a ZeroGPU Space with a per-request budget, so keep chunks short
# (~300 chars is roughly 20 s of speech) instead of the 900 used for fish.
AUK_CHUNK = _cap(300, "AUK_CHUNK")
AUK_VARIANT = os.environ.get("AUK_VARIANT", "flash")
AUK_USE_PE = os.environ.get("AUK_USE_PE", "").strip().lower() in ("1", "true", "yes")
DATA_DIR = "/tmp/fish-tts" if SERVERLESS else SCRIPT_DIR
REF_DIR = os.path.join(DATA_DIR, "refs")   # writable custom-voice refs
ENGINES_FILE = os.path.join(DATA_DIR, "engines.json")
SPEEDS_FILE = os.path.join(DATA_DIR, "speeds.json")
STATE_FILE = os.path.join(DATA_DIR, "voice_state.json")
PENDING_VOICE = set()    # chat_ids waiting to record a reference clip


def load_state():
    return read_json(STATE_FILE, {})


# Vercel kills the whole invocation at maxDuration, and the GPU Spaces can sit
# in a queue far longer than that. Run those calls on a deadline so the user
# always gets a reply instead of silence.
SYNTH_BUDGET = float(os.environ.get("SYNTH_BUDGET_SECONDS")
                     or (45 if SERVERLESS else 0))


def with_budget(fn, label):
    """Run fn(), raising TimeoutError once SYNTH_BUDGET seconds have passed."""
    if not SYNTH_BUDGET:
        return fn()
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # re-raised on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(SYNTH_BUDGET)
    if thread.is_alive():
        raise TimeoutError(
            f"{label} was still queued after {SYNTH_BUDGET:.0f}s - the shared "
            "GPU Space is busy right now. Try again in a moment, or /engine "
            "fish for instant speech.")
    if "error" in box:
        raise box["error"]
    return box["value"]


def ensure_parent(path):
    """State files live under /tmp on serverless, which starts empty."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------- durable state ----------
# Each serverless instance gets a fresh /tmp, so anything under DATA_DIR
# evaporates between cold starts: engine choice, speed, cloned-voice refs. When
# a Blob store is configured we keep the JSON in Blob and treat /tmp as a cache.
BLOB_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN")
BLOB_BASE = "https://blob.vercel-storage.com"
BLOB_PREFIX = os.environ.get("BLOB_STATE_PREFIX", "state")


def _blob_url(name):
    return f"{BLOB_BASE}/{BLOB_PREFIX}/{name}"


# Blob reads are served from the store's own domain, not the API host (the API
# host 404s on GET), so resolve it once per instance and remember it.
_BLOB_STORE_BASE = None
_BLOB_CACHE = {}          # name -> (fetched_at, data)
_BLOB_TTL = float(os.environ.get("BLOB_CACHE_TTL", "30"))


def _remember_store_base(url):
    global _BLOB_STORE_BASE
    parts = (url or "").split("/")
    if len(parts) >= 3 and parts[2].endswith("blob.vercel-storage.com"):
        _BLOB_STORE_BASE = "/".join(parts[:3])


def _blob_store_base():
    global _BLOB_STORE_BASE
    if _BLOB_STORE_BASE is not None:
        return _BLOB_STORE_BASE or None
    _BLOB_STORE_BASE = ""
    try:
        req = urllib.request.Request(
            f"{BLOB_BASE}/?limit=1",
            headers={"authorization": f"Bearer {BLOB_TOKEN}",
                     "x-api-version": "7"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            for item in (json.load(resp).get("blobs") or []):
                _remember_store_base(item.get("url"))
                if _BLOB_STORE_BASE:
                    break
    except Exception:
        pass
    return _BLOB_STORE_BASE or None


def _blob_read(name):
    """Parsed JSON from Blob, or None when missing or unreachable."""
    now = time.time()
    cached = _BLOB_CACHE.get(name)
    if cached and now - cached[0] < _BLOB_TTL:
        return cached[1]
    base = _blob_store_base()
    if not base:
        return None
    try:
        # Cache-bust: blob reads are served with s-maxage=300, so the edge can
        # hand back a copy that is minutes out of date after a write.
        req = urllib.request.Request(
            f"{base}/{BLOB_PREFIX}/{name}?ts={int(now)}",
            headers={"authorization": f"Bearer {BLOB_TOKEN}"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        _BLOB_CACHE[name] = (now, data)
        return data
    except Exception:
        return None


def _blob_write(name, data):
    """Best-effort durable write; the /tmp copy stays as a fallback."""
    try:
        req = urllib.request.Request(
            _blob_url(name), data=json.dumps(data).encode(), method="PUT",
            headers={
                "authorization": f"Bearer {BLOB_TOKEN}",
                "x-api-version": "7",
                "x-content-type": "application/json",
                "x-add-random-suffix": "0",
                "x-allow-overwrite": "1",
                "x-cache-control-max-age": "60",
            })
        with urllib.request.urlopen(req, timeout=20) as resp:
            try:
                _remember_store_base(json.load(resp).get("url"))
            except Exception:
                pass
        _BLOB_CACHE[name] = (time.time(), data)
        return True
    except Exception as exc:
        log(f"blob write failed for {name}: {exc}")
        return False


def read_json(path, default):
    """Blob is the source of truth when configured, /tmp is the cache."""
    if BLOB_TOKEN:
        data = _blob_read(os.path.basename(path))
        if isinstance(data, (dict, list)):
            try:
                ensure_parent(path)
                with open(path, "w") as fh:
                    json.dump(data, fh)
            except Exception:
                pass
            return data
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def write_json(path, data):
    try:
        ensure_parent(path)
        with open(path, "w") as fh:
            json.dump(data, fh)
    except Exception as exc:
        log(f"local state write failed for {path}: {exc}")
    if BLOB_TOKEN:
        _blob_write(os.path.basename(path), data)


def save_state(state):
    write_json(STATE_FILE, state)


def get_fish_voice(chat_id):
    return load_state().get("fish_voice", {}).get(str(chat_id))


def set_fish_voice(chat_id, model_id):
    state = load_state()
    if model_id:
        state.setdefault("fish_voice", {})[str(chat_id)] = model_id
    else:
        state.get("fish_voice", {}).pop(str(chat_id), None)
    save_state(state)


def itts_active(chat_id):
    return bool(load_state().get("itts_active", {}).get(str(chat_id)))


def set_itts_active(chat_id, active):
    state = load_state()
    state.setdefault("itts_active", {})[str(chat_id)] = bool(active)
    save_state(state)


def auk_active(chat_id):
    return bool(load_state().get("auk_active", {}).get(str(chat_id)))


def set_auk_active(chat_id, active):
    state = load_state()
    state.setdefault("auk_active", {})[str(chat_id)] = bool(active)
    save_state(state)


def last_clone(chat_id):
    return load_state().get("last", {}).get(str(chat_id), {})


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


def split_text(text, max_len=MAX_CHUNK):
    """Split into synthesis-friendly chunks at sentence boundaries."""
    pieces = [p for p in re.split(r"(?<=[.!?;。！？])\s+|\n{2,}", text.strip()) if p]
    chunks, cur = [], ""
    for piece in pieces:
        while len(piece) > max_len:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(piece[:max_len])
            piece = piece[max_len:]
        if cur and len(cur) + len(piece) + 1 > max_len:
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


# ---------- AuK engine (official Tencent-Hunyuan Space) ----------

def synthesize_auk_chunk(text, ref_path, speed=1.0):
    """One AuK request. The instruction carries the text (zero-shot template),
    and the reference clip drives the cloned voice."""
    instruction = f'Say the following with the same voice: "{text}"'
    return auk.generate(
        instruction,
        ref_path=ref_path,
        variant=AUK_VARIANT,
        use_pe=AUK_USE_PE,
        gen_seconds=auk.estimate_seconds(text, speed),
    )


def synthesize_full_auk(text, ref_path, speed=1.0):
    chunks = split_text(text, AUK_CHUNK)[:AUK_MAX_CHUNKS]
    parts = []
    for i, chunk in enumerate(chunks, 1):
        log(f"auk chunk {i}/{len(chunks)}")
        parts.append(synthesize_auk_chunk(chunk, ref_path, speed))
    if len(parts) == 1:
        return parts[0], 1
    return concat_wavs(parts), len(chunks)


# ---------- engine preference (per chat) ----------

def get_engine(chat_id):
    fallback = os.environ.get("DEFAULT_ENGINE", "fish")
    return read_json(ENGINES_FILE, {}).get(str(chat_id)) or fallback


def set_engine(chat_id, name):
    data = read_json(ENGINES_FILE, {})
    data[str(chat_id)] = name
    write_json(ENGINES_FILE, data)


def custom_ref(chat_id):
    path = os.path.join(REF_DIR, f"custom_{chat_id}.wav")
    if os.path.exists(path) and (itts_active(chat_id) or auk_active(chat_id)):
        return path
    return BUNDLED_REF


def get_speed(chat_id):
    try:
        v = float(read_json(SPEEDS_FILE, {})[str(chat_id)])
        if 0.5 <= v <= 2.0:
            return v
    except Exception:
        pass
    return 1.0


def set_speed(chat_id, value):
    data = read_json(SPEEDS_FILE, {})
    data[str(chat_id)] = value
    write_json(SPEEDS_FILE, data)


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
                  f"/engine auk - Tencent AuK voice clone ({AUK_MAX_CHUNKS * AUK_CHUNK // 3} chars max)\n\n"
                  "<b>Voice cloning</b>\n"
                  "/clone - send a voice message after this (~5-30s clean "
                  "speech); I'll clone it on BOTH engines and send samples\n"
                  "/usefish [id] - make the last clone (or a fish.audio model "
                  "id) your active Fish voice\n"
                  "/useitts - make the last clone your active IndexTTS voice\n"
                  "/useauk - make the last clone your active AuK voice\n"
                  "/useboth - activate the last clone on every engine\n"
                  "/voices - show active voices\n"
                  "/resetvoice - back to defaults (Sarah / Rick Warren)\n"
                  "/cancel - abort pending voice setup")
        elif cmd == "/engine":
            parts = text_in.split(maxsplit=1)
            if len(parts) == 1:
                cur = get_engine(chat_id)
                reply(token, chat_id,
                      f"Current engine: <b>{cur}</b>.\n"
                      "Switch with /engine fish, /engine itts or /engine auk")
            elif parts[1].strip().lower() in ("fish", "itts", "indextts", "auk"):
                want = parts[1].strip().lower()
                name = "itts" if want.startswith("i") else ("auk" if want.startswith("a") else "fish")
                set_engine(chat_id, name)
                ref_note = ""
                has_custom = os.path.exists(os.path.join(REF_DIR, f"custom_{chat_id}.wav"))
                if name == "itts" and not (has_custom and itts_active(chat_id)):
                    ref_note = ("\nHeads-up: you're on the default sample voice. "
                                "/clone to clone your own.")
                if name == "auk" and not (has_custom and auk_active(chat_id)):
                    ref_note = ("\nHeads-up: you're on the default sample voice. "
                                "/clone then /useauk to clone your own.")
                reply(token, chat_id, f"Engine set to <b>{name}</b>.{ref_note}")
            else:
                reply(token, chat_id,
                      "Unknown engine. Use /engine fish, /engine itts or /engine auk")
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
        elif cmd in ("/clone", "/setvoice"):
            PENDING_VOICE.add(chat_id)
            reply(token, chat_id,
                  "Send me a voice message or audio clip now - about 5-30 "
                  "seconds of clean speech from the person to clone.\n"
                  "I'll clone it on both engines and send you samples, then "
                  "you choose which to activate.\n/cancel to abort.")
        elif cmd == "/usefish":
            parts = text_in.split(maxsplit=1)
            if len(parts) > 1:
                model_id = parts[1].strip()
                try:
                    info = get_voice_model(api_key, model_id)
                except Exception as exc:
                    reply(token, chat_id,
                          f"Couldn't find that fish.audio model: {exc}")
                    return
                set_fish_voice(chat_id, model_id)
                reply(token, chat_id,
                      f"Fish voice set to <b>{info.get('title', model_id)}</b>.")
            else:
                clone = last_clone(chat_id)
                if not clone.get("fish_id"):
                    reply(token, chat_id,
                          "No recent clone. Send /clone first, or use "
                          "/usefish &lt;model_id&gt;.")
                    return
                set_fish_voice(chat_id, clone["fish_id"])
                reply(token, chat_id,
                      f"Fish voice set to your clone <b>{clone.get('title', clone['fish_id'])}</b>.")
        elif cmd == "/useitts":
            pending_ref = os.path.join(REF_DIR, f"pending_{chat_id}.wav")
            if not os.path.exists(pending_ref):
                reply(token, chat_id, "No recent clone. Send /clone first.")
                return
            custom_path = os.path.join(REF_DIR, f"custom_{chat_id}.wav")
            shutil.copyfile(pending_ref, custom_path)
            set_itts_active(chat_id, True)
            reply(token, chat_id,
                  "<b>IndexTTS voice activated</b> - send /engine itts to use it.")
        elif cmd == "/useauk":
            pending_ref = os.path.join(REF_DIR, f"pending_{chat_id}.wav")
            if not os.path.exists(pending_ref):
                reply(token, chat_id, "No recent clone. Send /clone first.")
                return
            shutil.copyfile(pending_ref, os.path.join(REF_DIR, f"custom_{chat_id}.wav"))
            set_auk_active(chat_id, True)
            reply(token, chat_id,
                  "<b>AuK voice activated</b> - send /engine auk to use it.")
        elif cmd == "/useboth":
            parts = text_in.split(maxsplit=1)
            clone = last_clone(chat_id)
            if clone.get("fish_id"):
                set_fish_voice(chat_id, clone["fish_id"])
            pending_ref = os.path.join(REF_DIR, f"pending_{chat_id}.wav")
            if os.path.exists(pending_ref):
                shutil.copyfile(pending_ref,
                                os.path.join(REF_DIR, f"custom_{chat_id}.wav"))
                set_itts_active(chat_id, True)
                set_auk_active(chat_id, True)
            if not clone.get("fish_id") and not os.path.exists(pending_ref):
                reply(token, chat_id, "No recent clone. Send /clone first.")
                return
            reply(token, chat_id, "<b>Clone activated on every engine.</b>")
        elif cmd == "/voices":
            fv = get_fish_voice(chat_id)
            itts_note = (f"your clone ({ITT_SPACE})" if itts_active(chat_id)
                         else f"default ({os.path.basename(BUNDLED_REF)})")
            auk_note = (f"your clone ({auk.SPACE})" if auk_active(chat_id)
                        else f"default ({os.path.basename(BUNDLED_REF)})")
            engine = get_engine(chat_id)
            reply(token, chat_id,
                  f"Engine: <b>{engine}</b>\n"
                  f"Fish voice: <b>{fv or 'Sarah (default)'}</b>\n"
                  f"IndexTTS ref: <b>{itts_note}</b>\n"
                  f"AuK ref: <b>{auk_note}</b>")
        elif cmd == "/resetvoice":
            set_fish_voice(chat_id, None)
            set_itts_active(chat_id, False)
            set_auk_active(chat_id, False)
            reply(token, chat_id,
                  "Back to defaults on every engine (Sarah / Rick Warren).")
        elif cmd == "/cancel":
            PENDING_VOICE.discard(chat_id)
            reply(token, chat_id, "Okay, cancelled.")
        else:
            reply(token, chat_id, "Unknown command. Try /help")
        return

    # custom-voice capture: clone on both engines, send samples
    if chat_id in PENDING_VOICE and (msg.get("voice") or msg.get("audio")):
        src = msg.get("voice") or msg.get("audio")
        PENDING_VOICE.discard(chat_id)
        try:
            ffmpeg = ffmpeg_bin()
            if not ffmpeg:
                raise RuntimeError("ffmpeg is not available in this environment")
            raw_bytes = download_doc(token, src["file_id"])
            os.makedirs(REF_DIR, exist_ok=True)
            tmp_in = os.path.join(REF_DIR, f"in_{chat_id}.bin")
            with open(tmp_in, "wb") as fh:
                fh.write(raw_bytes)
            pending_ref = os.path.join(REF_DIR, f"pending_{chat_id}.wav")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                            "-i", tmp_in, "-ar", "22050", "-ac", "1",
                            pending_ref],
                           check=True, timeout=120)
            os.remove(tmp_in)
        except Exception as exc:
            log(traceback.format_exc())
            reply(token, chat_id, f"Couldn't process that audio: {exc}")
            return

        started = time.time()
        sample_text = ("Hi there. This is your brand new cloned voice, "
                       "reading a short sample so you can hear how it sounds.")
        state = load_state()
        state.setdefault("last", {})[str(chat_id)] = {
            "title": f"clone-{chat_id}-{time.strftime('%Y%m%d')}",
            "ref": pending_ref,
        }

        fish_id = None
        fish_state = None
        try:
            with open(pending_ref, "rb") as fh:
                wav_bytes = fh.read()
            info = create_voice_model(
                api_key, wav_bytes,
                title=f"TTSBot-{chat_id}-{time.strftime('%Y%m%d%H%M')}")
            fish_id = info.get("_id")
            fish_state = info.get("state")
            state["last"][str(chat_id)]["fish_id"] = fish_id
            save_state(state)
            log(f"fish model created {fish_id} (state={fish_state})")
        except Exception as exc:
            log(traceback.format_exc())
            reply(token, chat_id, f"Fish cloning failed: {exc}")

        if fish_id:
            try:
                if fish_state not in (None, "trained", "created"):
                    raise RuntimeError(f"model state is '{fish_state}'")
                audio, _used = synthesize_full(sample_text, api_key, model,
                                               fish_id)
                send_audio(token, chat_id, audio,
                           "[fish-clone] sample - /usefish to activate "
                           "this voice", "mp3")
            except Exception as exc:
                log(traceback.format_exc())
                reply(token, chat_id,
                      f"Fish clone saved ({fish_id}) but the sample failed: "
                      f"{exc}. Try /usefish later.")

        elapsed = time.time() - started
        if SERVERLESS and elapsed > 40:
            reply(token, chat_id,
                  "IndexTTS sample skipped (GPU queue too slow for the 60s "
                  "serverless window). The reference is stored - /useitts to "
                  "activate it, then just send text.")
        else:
            try:
                audio, _used = synthesize_full_itts(sample_text, pending_ref)
                send_audio(token, chat_id, audio,
                           "[itts-clone] sample - /useitts to activate "
                           "this voice", "wav")
            except Exception as exc:
                log(traceback.format_exc())
                reply(token, chat_id,
                      f"IndexTTS cloning failed: {exc}. Fish side is unaffected.")

            try:
                audio, _used = synthesize_full_auk(sample_text, pending_ref)
                send_audio(token, chat_id, audio,
                           "[auk-clone] sample - /useauk to activate "
                           "this voice", "wav")
            except Exception as exc:
                log(traceback.format_exc())
                reply(token, chat_id,
                      f"AuK cloning failed: {exc}. Other engines are unaffected.")

        reply(token, chat_id,
              "<b>Done!</b> Activate with /usefish, /useitts, /useauk, "
              "/useboth, or /voices to check what's active. /resetvoice "
              "reverts every engine to its default.")
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
    elif engine == "auk":
        max_chunks = AUK_MAX_CHUNKS
        limit_note = (f"That's ~{total_chunks} chunks but AuK can only do "
                      f"{AUK_MAX_CHUNKS} per request - I'll read the first part only.")
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
            audio, used = with_budget(
                lambda: synthesize_full_itts(
                    source_text, custom_ref(chat_id),
                    speed=get_speed(chat_id)), "IndexTTS")
            fmt = "wav"
        elif engine == "auk":
            audio, used = with_budget(
                lambda: synthesize_full_auk(
                    source_text, custom_ref(chat_id),
                    speed=get_speed(chat_id)), "AuK")
            fmt = "wav"
        else:
            chat_voice = get_fish_voice(chat_id) or voice
            audio, used = synthesize_full(source_text, api_key, model,
                                          chat_voice)
            fmt = "mp3"
    except SystemExit as exc:
        # tts.py's CLI paths call sys.exit(); convert so the webhook survives
        raise RuntimeError(f"TTS engine failed (exit {exc.code})") from None
    except Exception as exc:
        log(traceback.format_exc())
        msg_str = str(exc)
        low = msg_str.lower()
        if isinstance(exc, TimeoutError):
            reply(token, chat_id, msg_str)
        elif "quota" in low or "zerogpu" in low or "runs limit" in low:
            reply(token, chat_id,
                  "AuK/IndexTTS GPU quota is used up for today (the HF Space "
                  "runs on shared ZeroGPU). It resets ~24h after your first GPU "
                  "use today. /engine fish still works, and /engine auk will "
                  "come back on its own.")
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
