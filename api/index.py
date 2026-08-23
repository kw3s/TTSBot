"""Health-check endpoint (GET /api)."""

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET", "HEAD"])
@app.route("/api", methods=["GET", "HEAD"])
def index():
    return jsonify(ok=True, service="fish-tts-bot")
