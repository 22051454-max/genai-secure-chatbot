import os
import secrets
from collections import defaultdict, deque

from flask import Flask, g, jsonify, request, send_from_directory

from . import auth
from .files import extract_text
from .providers import Attachment, OpenAIProvider, build_providers, call
from .ranker import ResponseRanker
from .security import (RateLimiter, check_upload, detect_prompt_injection, sanitize_text, wrap_untrusted)
from concurrent.futures import ThreadPoolExecutor

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


def create_app(test_config=None):
    app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
    app.config.update(
        JWT_SECRET=os.environ.get("JWT_SECRET") or secrets.token_hex(32),
        JWT_TTL_MIN=int(os.environ.get("JWT_TTL_MIN", "60")),
        DB_PATH=os.environ.get("DB_PATH", os.path.join(app.instance_path, "chatbot.db")),
        MAX_CONTENT_LENGTH=12 * 1024 * 1024,
        CHAT_RATE=(int(os.environ.get("CHAT_BURST", "10")), float(os.environ.get("CHAT_PER_MIN", "20")) / 60),
        AUTH_RATE=(5, 5 / 60),
    )
    if test_config:
        app.config.update(test_config)
    os.makedirs(os.path.dirname(app.config["DB_PATH"]) or ".", exist_ok=True)
    auth.init_db(app.config["DB_PATH"])

    chat_limiter = RateLimiter(*app.config["CHAT_RATE"])
    auth_limiter = RateLimiter(*app.config["AUTH_RATE"])
    ranker = ResponseRanker()
    providers = build_providers()
    histories = defaultdict(lambda: deque(maxlen=12))
    pool = ThreadPoolExecutor(max_workers=4)

    @app.teardown_appcontext
    def close_db(_):
        db = g.pop("db", None)
        if db:
            db.close()

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
            "connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return resp

    def client_ip():
        return request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()

    def limited(limiter, key):
        ok, retry = limiter.allow(key)
        if not ok:
            r = jsonify(error="Rate limit exceeded", retry_after=retry)
            r.status_code, r.headers["Retry-After"] = 429, str(int(retry) + 1)
            return r
        return None

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", providers=[p.name for p in providers], ranker_weights=ranker.weights)

    @app.post("/api/register")
    def register():
        if (r := limited(auth_limiter, "auth:" + client_ip())):
            return r
        body = request.get_json(silent=True) or {}
        uid, err = auth.register(sanitize_text(body.get("username", ""), 32), body.get("password", ""))
        if err:
            return jsonify(error=err), 400
        return jsonify(token=auth.issue_token(uid, body["username"]), username=body["username"]), 201

    @app.post("/api/login")
    def login():
        if (r := limited(auth_limiter, "auth:" + client_ip())):
            return r
        body = request.get_json(silent=True) or {}
        username = sanitize_text(body.get("username", ""), 32)
        uid = auth.authenticate(username, body.get("password", ""))
        if not uid:
            return jsonify(error="Invalid credentials"), 401
        return jsonify(token=auth.issue_token(uid, username), username=username)

    @app.post("/api/chat")
    @auth.jwt_required
    def chat():
        if (r := limited(chat_limiter, f"user:{g.user_id}")):
            auth.log_event("rate_limited", client_ip())
            return r
        message = sanitize_text(request.form.get("message") or (request.get_json(silent=True) or {}).get("message", ""))
        attachments, notices = [], []
        for f in request.files.getlist("files")[:4]:
            data = f.read()
            err = check_upload(f.filename, data)
            if err:
                auth.log_event("upload_rejected", f"{f.filename}: {err}")
                return jsonify(error=err), 400
            if (f.mimetype or "").startswith("image/"):
                attachments.append(Attachment("image", f.filename, f.mimetype, data))
            else:
                text = sanitize_text(extract_text(f.filename, data), 12000)
                verdict = detect_prompt_injection(text)
                if verdict.labels:
                    notices.append(f"{f.filename}: possible injected instructions ({', '.join(verdict.labels)}); treated as data")
                    auth.log_event("doc_injection_flag", f"{f.filename}: {verdict.labels}")
                attachments.append(Attachment("document", f.filename, f.mimetype, text=text))
        if not message and not attachments:
            return jsonify(error="Empty message"), 400

        verdict = detect_prompt_injection(message)
        if verdict.blocked:
            auth.log_event("prompt_injection_blocked", f"score={verdict.score} {verdict.labels}: {message[:200]}")
            return jsonify(blocked=True, reason="Message looks like a prompt-injection attempt",
                           score=verdict.score, labels=verdict.labels), 422

        prompt = message or "Please review the attached file(s)."
        for a in attachments:
            if a.kind == "document":
                prompt += "\n\n" + wrap_untrusted(f"FILE {a.name}", a.text)

        history = list(histories[g.user_id])
        results = list(pool.map(lambda p: call(p, history, prompt, attachments), providers))
        ranked = ranker.rank(message or prompt, results)
        best, best_score = ranked[0]
        if not best.ok:
            return jsonify(error="All providers failed", details=[{"provider": r.provider, "error": r.error} for r in results]), 502
        histories[g.user_id].extend([{"role": "user", "content": message or "(file)"},
                                     {"role": "assistant", "content": best.text}])
        return jsonify(
            reply=best.text, provider=best.provider, notices=notices, injection_score=verdict.score,
            candidates=[{"provider": r.provider, "ok": r.ok, "score": None if s == float("-inf") else round(s, 3),
                         "latency_ms": r.latency_ms, "error": r.error, "preview": r.text[:160]} for r, s in ranked],
        )

    @app.post("/api/transcribe")
    @auth.jwt_required
    def transcribe():
        if (r := limited(chat_limiter, f"user:{g.user_id}")):
            return r
        f = request.files.get("audio")
        if not f:
            return jsonify(error="No audio"), 400
        data = f.read()
        if (err := check_upload(f.filename or "audio.webm", data)):
            return jsonify(error=err), 400
        oa = OpenAIProvider()
        if not oa.available:
            return jsonify(error="Server-side transcription needs OPENAI_API_KEY. The browser's built-in speech recognition is used instead."), 501
        try:
            return jsonify(text=sanitize_text(oa.transcribe(data, f.filename or "audio.webm")))
        except Exception as exc:
            return jsonify(error=f"Transcription failed: {type(exc).__name__}"), 502

    @app.post("/api/reset")
    @auth.jwt_required
    def reset():
        histories.pop(g.user_id, None)
        return jsonify(ok=True)

    @app.get("/api/security/events")
    @auth.jwt_required
    def events():
        rows = auth.get_db().execute(
            "SELECT ts, kind, detail FROM security_events WHERE user_id=? ORDER BY id DESC LIMIT 50", (g.user_id,)).fetchall()
        return jsonify([dict(r) for r in rows])

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Request too large"), 413

    return app
