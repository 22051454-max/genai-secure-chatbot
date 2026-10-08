"""JWT authentication backed by SQLite."""
import datetime as dt
import re
import sqlite3
from functools import wraps

import jwt
from flask import current_app, g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
    return g.db


def init_db(path):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS security_events(id INTEGER PRIMARY KEY, ts TEXT, user_id INTEGER, kind TEXT, detail TEXT);
    """)
    con.commit()
    con.close()


USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def register(username, password):
    if not USERNAME_RE.match(username or ""):
        return None, "Username must be 3-32 characters: letters, digits, _ . -"
    if len(password or "") < 8:
        return None, "Password must be at least 8 characters"
    db = get_db()
    try:
        cur = db.execute("INSERT INTO users(username, pw) VALUES(?, ?)", (username, generate_password_hash(password)))
        db.commit()
    except sqlite3.IntegrityError:
        return None, "Username already taken"
    return cur.lastrowid, None


def authenticate(username, password):
    row = get_db().execute("SELECT id, pw FROM users WHERE username=?", (username,)).fetchone()
    if row and check_password_hash(row["pw"], password or ""):
        return row["id"]
    return None


def issue_token(user_id, username):
    now = dt.datetime.now(dt.timezone.utc)
    payload = {"sub": str(user_id), "name": username, "iat": now,
               "exp": now + dt.timedelta(minutes=current_app.config["JWT_TTL_MIN"])}
    return jwt.encode(payload, current_app.config["JWT_SECRET"], algorithm="HS256")


def jwt_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify(error="Missing bearer token"), 401
        try:
            claims = jwt.decode(header[7:], current_app.config["JWT_SECRET"], algorithms=["HS256"],
                                options={"require": ["exp", "sub"]})
        except jwt.ExpiredSignatureError:
            return jsonify(error="Token expired"), 401
        except jwt.InvalidTokenError:
            return jsonify(error="Invalid token"), 401
        g.user_id, g.username = int(claims["sub"]), claims.get("name")
        return view(*a, **kw)
    return wrapped


def log_event(kind, detail):
    db = get_db()
    db.execute("INSERT INTO security_events(ts, user_id, kind, detail) VALUES(?,?,?,?)",
               (dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), getattr(g, "user_id", None), kind, detail[:500]))
    db.commit()
