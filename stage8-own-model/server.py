"""
Flask server exposing the fine-tuned model over HTTP.

Endpoints:
    GET  /health       — check server is alive
    POST /generate     — { "message": "hi" } → { "reply": "Hello!" }

Run:
    python server.py
Listens on http://localhost:5000
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flask import Flask, jsonify, request
from flask_cors import CORS

from generate import load_model, generate_reply


app = Flask(__name__)
CORS(app)  # allow requests from http://localhost:5173 (Vite dev server)

print("Loading model…")
model, tokenizer = load_model()
print("Model ready. Server starting on http://localhost:5000")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        reply = generate_reply(model, tokenizer, message)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)