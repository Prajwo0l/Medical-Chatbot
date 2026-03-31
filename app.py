from flask import Flask, render_template, jsonify, request
from backend import (
    build_rag_chain,
    answer_question,
    create_thread,
    get_all_threads,
    get_messages,
    delete_thread,
    get_thread,
)
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Build the RAG chain once at startup
print("Initialising RAG pipeline...")
rag_chain, _ = build_rag_chain()
print("RAG pipeline ready.")


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("chat.html")


# ── Thread API ────────────────────────────────────────────────────────────────

@app.route("/api/threads", methods=["GET"])
def api_get_threads():
    """Return all threads ordered by most recent."""
    return jsonify(get_all_threads())


@app.route("/api/threads", methods=["POST"])
def api_create_thread():
    """Create a new thread and return it."""
    thread_id = create_thread("New Chat")
    return jsonify(get_thread(thread_id))


@app.route("/api/threads/<thread_id>", methods=["DELETE"])
def api_delete_thread(thread_id):
    """Delete a thread and all its messages."""
    delete_thread(thread_id)
    return jsonify({"ok": True})


@app.route("/api/threads/<thread_id>/messages", methods=["GET"])
def api_get_messages(thread_id):
    """Return all messages for a thread."""
    return jsonify(get_messages(thread_id))


# ── Chat API ──────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Expects JSON: { "thread_id": "...", "message": "..." }
    Returns JSON: { "answer": "...", "thread_id": "..." }
    """
    data      = request.get_json(force=True)
    thread_id = data.get("thread_id", "").strip()
    message   = data.get("message", "").strip()

    if not thread_id or not message:
        return jsonify({"error": "thread_id and message are required"}), 400

    # Ensure thread exists
    if not get_thread(thread_id):
        create_thread("New Chat")

    answer = answer_question(rag_chain, thread_id, message)
    thread = get_thread(thread_id)

    return jsonify({
        "answer":    answer,
        "thread_id": thread_id,
        "title":     thread["title"] if thread else "New Chat",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
