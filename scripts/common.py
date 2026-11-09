"""Shared helpers for the API-vs-UI study. Keys are read from files outside the vault
(SECRETS dir) so nothing secret lands in git."""
import gzip, json, os, subprocess, sys, threading, time, urllib.error, urllib.request
from datetime import datetime, timezone

# python -I ignores PYTHONIOENCODING; a redirected stdout on Windows is cp1252 and would crash on
# Latvian/Estonian/Lithuanian text. Force UTF-8 with replacement.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(STUDY, "data")
SECRETS = os.environ.get("STUDY_SECRETS", os.path.join(os.path.expanduser("~"), ".study-secrets"))
LOG = os.path.join(STUDY, "run-log.md")
GCP_PROJECT = os.environ.get("STUDY_GCP_PROJECT", "")  # your own Google Cloud project with Vertex AI enabled
GEMINI_MODEL = "gemini-3.5-flash-lite"   # matches the "Flash-Lite" the Gemini app reports via Bright Data
NUDGE = "Search the web for current information and cite your sources."  # also sent to the Gemini API, so both arms get the same prompt

_lock = threading.Lock()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg):
    line = f"- {now()} {msg}"
    print(line, flush=True)
    with _lock, open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def secret(name):
    with open(os.path.join(SECRETS, name), encoding="utf-8") as f:
        return f.read().strip()


def append_jsonl_gz(path, rows):
    with _lock, gzip.open(path, "at", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl_gz(path):
    if not os.path.exists(path):
        return []
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut by a crash mid-write
    return out


def load_prompts():
    with open(os.path.join(DATA, "prompts.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def http(method, url, body=None, headers=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), dict(e.headers or {})
    except Exception as e:  # transport failure
        return 0, repr(e), {}


class GcpToken:
    """gcloud access token, refreshed every 40 minutes (tokens live 60)."""
    def __init__(self):
        self._tok, self._at = None, 0
        self._l = threading.Lock()

    def get(self):
        with self._l:
            if not self._tok or time.time() - self._at > 2400:
                gcloud = "gcloud.cmd" if os.name == "nt" else "gcloud"
                self._tok = subprocess.run([gcloud, "auth", "print-access-token"], capture_output=True,
                                           text=True, check=True, timeout=60).stdout.strip()
                self._at = time.time()
            return self._tok


TOKEN = GcpToken()


def gemini(contents_text, *, model=GEMINI_MODEL, tools=None, json_schema=None, temperature=None, timeout=180):
    """One Vertex generateContent call. Returns (status, parsed_json_or_text)."""
    body = {"contents": [{"role": "user", "parts": [{"text": contents_text}]}]}
    if tools:
        body["tools"] = tools
    gen = {}
    if json_schema:
        gen["responseMimeType"] = "application/json"
        gen["responseSchema"] = json_schema
    if temperature is not None:
        gen["temperature"] = temperature
    if gen:
        body["generationConfig"] = gen
    url = (f"https://aiplatform.googleapis.com/v1/projects/{GCP_PROJECT}/locations/global/"
           f"publishers/google/models/{model}:generateContent")
    st, txt, _ = http("POST", url, body, {"Authorization": f"Bearer {TOKEN.get()}",
                                          "x-goog-user-project": GCP_PROJECT,
                                          "Content-Type": "application/json"}, timeout=timeout)
    try:
        return st, json.loads(txt)
    except Exception:
        return st, txt


def gemini_text(resp):
    try:
        return "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
    except Exception:
        return ""
