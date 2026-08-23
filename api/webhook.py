"""Vercel serverless entrypoint: Telegram webhook -> bot.process_update()."""

import os

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/", defaults={"p": ""}, methods=["GET", "HEAD", "POST"])
@app.route("/<path:p>", methods=["GET", "HEAD", "POST"])
def entry(p):
    print(f"hit: method={request.method} path={request.path!r} "
          f"root={request.script_root!r}", flush=True)
    if request.method in ("GET", "HEAD"):
        return jsonify(ok=True, service="fish-tts-bot")
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    if secret and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != secret:
        return jsonify(ok=False, error="bad secret token"), 401
    update = request.get_json(force=True, silent=True)
    if not isinstance(update, dict):
        return jsonify(ok=False, error="invalid update"), 400
    from bot import process_update
    process_update(update)
    return jsonify(ok=True)
