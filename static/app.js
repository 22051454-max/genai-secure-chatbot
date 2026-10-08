const $ = (id) => document.getElementById(id);
let token = sessionStorage.getItem("jwt"), files = [];

function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function md(s) { return esc(s).replace(/```([\s\S]*?)```/g, "<code>$1</code>").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>"); }

async function api(path, opts = {}) {
  opts.headers = Object.assign({}, opts.headers, token ? { Authorization: "Bearer " + token } : {});
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (r.status === 401 && token) { logout(); }
  return { status: r.status, data };
}

async function auth(kind) {
  const { status, data } = await api("/api/" + kind, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: $("u").value, password: $("p").value }) });
  if (status >= 400) { $("authErr").textContent = data.error || "Failed"; return; }
  token = data.token; sessionStorage.setItem("jwt", token); show();
}
function logout() { token = null; sessionStorage.removeItem("jwt"); show(); }
async function show() {
  $("auth").hidden = !!token; $("app").hidden = !token;
  if (token) { const h = await fetch("/api/health").then(r => r.json()); $("prov").textContent = "Providers: " + h.providers.join(", "); }
}

function bubble(cls, html) { const d = document.createElement("div"); d.className = "msg " + cls; d.innerHTML = html; $("chat").appendChild(d); $("chat").scrollTop = 1e9; return d; }

async function send(e) {
  e.preventDefault();
  const text = $("msg").value.trim(); if (!text && !files.length) return;
  bubble("user", esc(text) + (files.length ? `<div class="meta">${files.map(f => "📎 " + esc(f.name)).join(" ")}</div>` : ""));
  const fd = new FormData(); fd.append("message", text); files.forEach(f => fd.append("files", f));
  $("msg").value = ""; files = []; renderChips();
  const pending = bubble("bot", "<span class='muted'>Thinking across providers…</span>");
  const { status, data } = await api("/api/chat", { method: "POST", body: fd });
  if (status === 422 && data.blocked) {
    pending.className = "msg blocked";
    pending.innerHTML = `🛡️ <b>Blocked:</b> ${esc(data.reason)}<div class="meta">risk score ${data.score} · ${esc(data.labels.join(", "))}</div>`;
  } else if (status >= 400) {
    pending.className = "msg blocked"; pending.innerHTML = "⚠️ " + esc(data.error || "Error") + (data.retry_after ? ` (retry in ${data.retry_after}s)` : "");
  } else {
    const cands = data.candidates.map(c => `<li>${esc(c.provider)}: score ${c.score ?? "n/a"}, ${c.latency_ms} ms ${c.error ? "(" + esc(c.error) + ")" : ""}</li>`).join("");
    pending.innerHTML = md(data.reply) + `<div class="meta">Selected: <b>${esc(data.provider)}</b>` +
      (data.notices.length ? `<br>⚠️ ${data.notices.map(esc).join("<br>")}` : "") +
      `<details><summary>Ranker scores</summary><ul>${cands}</ul></details></div>`;
  }
  if (!$("side").hidden) loadEvents();
}

function renderChips() { $("chips").innerHTML = files.map(f => `<span class="chip">${esc(f.name)}</span>`).join(""); }

async function loadEvents() {
  const { data } = await api("/api/security/events");
  $("events").innerHTML = (data.length ? data : [{ kind: "none", detail: "No events yet", ts: "" }])
    .map(e => `<div class="ev"><b>${esc(e.kind)}</b> <span class="muted">${esc(e.ts)}</span><br>${esc(e.detail)}</div>`).join("");
}

// Voice: browser speech recognition when available, otherwise record and send to /api/transcribe (Whisper).
let rec = null, recorder = null, chunks = [];
async function toggleMic() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (rec || recorder) { rec?.stop(); recorder?.stop(); return; }
  $("mic").classList.add("rec");
  if (SR) {
    rec = new SR(); rec.lang = "en-US"; rec.interimResults = false;
    rec.onresult = (ev) => { $("msg").value += (($("msg").value ? " " : "") + ev.results[0][0].transcript); };
    rec.onend = () => { rec = null; $("mic").classList.remove("rec"); };
    rec.start();
  } else {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream); chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop()); $("mic").classList.remove("rec");
      const fd = new FormData(); fd.append("audio", new Blob(chunks, { type: "audio/webm" }), "voice.webm"); recorder = null;
      const { status, data } = await api("/api/transcribe", { method: "POST", body: fd });
      if (status === 200) $("msg").value += data.text; else alert(data.error);
    };
    recorder.start();
  }
}

$("loginBtn").onclick = () => auth("login");
$("regBtn").onclick = () => auth("register");
$("logoutBtn").onclick = logout;
$("resetBtn").onclick = async () => { await api("/api/reset", { method: "POST" }); $("chat").innerHTML = ""; };
$("eventsBtn").onclick = () => { $("side").hidden = !$("side").hidden; if (!$("side").hidden) loadEvents(); };
$("file").onchange = (e) => { files = [...files, ...e.target.files].slice(0, 4); renderChips(); e.target.value = ""; };
$("mic").onclick = toggleMic;
$("composer").onsubmit = send;
$("msg").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); } };
show();
