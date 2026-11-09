"""Website (UI) answers through Bright Data: ChatGPT, Gemini, Google AI Mode, 5 runs per prompt.

Mechanics (Bright Data Datasets API):
- async /trigger -> snapshot_id, poll /progress, download /snapshot?format=jsonl
- ChatGPT: web_search=True, require_sources=False (forces the browse toggle)
- Gemini: a short "search the web" nudge appended to the prompt (Gemini has no search switch)
- AI Mode: always searches; `hl` = prompt language
- country = market (LV/LT/EE for Baltic prompts, US for global)
- custom_output_fields drops the multi-MB answer_html blobs
Resumable: job state in data/bd_jobs.json, answers appended to data/bd_<engine>.jsonl.gz.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, NUDGE, append_jsonl_gz, http, load_prompts, log, read_jsonl_gz, secret

BASE = "https://api.brightdata.com/datasets/v3"
RUNS = 5
BATCH = 250
MAX_INFLIGHT = 24
MAX_SUCCESS = 18_900  # budget guard: 18,750 planned records at $1.50/1k = $28.13
ENGINES = {
    "chatgpt": {"dataset": "gd_m7aof0k82r803d5bjm", "url": "https://chatgpt.com/",
                "fields": ["answer_text_markdown", "answer_text", "citations", "links_attached",
                           "web_search_query", "web_search_triggered", "index", "input", "error", "error_code"]},
    "gemini": {"dataset": "gd_mbz66arm2mf9cu856y", "url": "https://gemini.google.com/",
               "fields": ["answer_text_markdown", "answer_text", "citations", "links_attached", "model",
                          "index", "input", "error", "error_code"]},
    "google": {"dataset": "gd_mcswdt6z2elth3zqr2", "url": "https://google.com/aimode",
               "fields": ["answer_text_markdown", "answer_text", "citations", "links_attached",
                          "index", "input", "error", "error_code"]},
}
JOBS = os.path.join(DATA, "bd_jobs.json")
KEY = secret("brightdata.key")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def row_for(engine, p, run):
    e = ENGINES[engine]
    r = {"url": e["url"], "index": p["n"] * 10 + run}
    if engine == "chatgpt":
        r.update(prompt=p["text"], web_search=True, require_sources=False, country=p["country"])
    elif engine == "gemini":
        r.update(prompt=f"{p['text']} {NUDGE}", country=p["country"])
    else:
        r.update(prompt=p["text"], country=p["country"], hl=p["lang"])
    return r


def usable(rec):
    return not rec.get("error") and bool((rec.get("answer_text_markdown") or rec.get("answer_text") or "").strip())


def load_state():
    if os.path.exists(JOBS):
        return json.load(open(JOBS, encoding="utf-8"))
    return {"jobs": []}


def save_state(s):
    tmp = JOBS + ".tmp"
    json.dump(s, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, JOBS)


def done_keys():
    keys, succ = set(), 0
    for eng in ENGINES:
        for r in read_jsonl_gz(os.path.join(DATA, f"bd_{eng}.jsonl.gz")):
            if r.get("usable"):
                keys.add((eng, r["n"], r["run"]))
                succ += 1
    return keys, succ


def trigger(engine, rows):
    e = ENGINES[engine]
    url = f"{BASE}/trigger?dataset_id={e['dataset']}&include_errors=true&custom_output_fields={'|'.join(e['fields'])}"
    for attempt in range(8):
        st, txt, hdr = http("POST", url, rows, H, timeout=120)
        if st == 200:
            return json.loads(txt)["snapshot_id"]
        wait = float(hdr.get("Retry-After") or hdr.get("retry-after") or 30)
        log(f"bd: trigger {engine} HTTP {st} {txt[:160]!r}, retry in {wait:.0f}s")
        if st in (400, 401, 402, 403):
            return None
        time.sleep(min(wait, 120))
    return None


def download(job, pmap):
    st, txt, _ = http("GET", f"{BASE}/snapshot/{job['sid']}?format=jsonl", headers=H, timeout=900)
    if st != 200:
        return None
    out = []
    for line in txt.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        idx = rec.get("index")
        if idx is None:
            idx = (rec.get("input") or {}).get("index")
        if idx is None:
            continue
        n, run = int(idx) // 10, int(idx) % 10
        p = pmap.get(n)
        if not p:
            continue
        out.append({"engine": job["engine"], "n": n, "run": run, "pid": p["pid"], "sid": job["sid"],
                    "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "usable": usable(rec),
                    **{k: rec.get(k) for k in ENGINES[job["engine"]]["fields"] if k not in ("input",)}})
    return out


def plan(prompts, keys, state):
    """Batches of (engine, run) rows not yet answered and not already in a live job."""
    live = set()
    for j in state["jobs"]:
        if j["status"] in ("running", "triggered"):
            live.update((j["engine"], i // 10, i % 10) for i in j["indexes"])
    todo = []
    for engine in ENGINES:
        for run in range(1, RUNS + 1):
            rows = [row_for(engine, p, run) for p in prompts
                    if (engine, p["n"], run) not in keys and (engine, p["n"], run) not in live]
            for i in range(0, len(rows), BATCH):
                todo.append((engine, run, rows[i:i + BATCH]))
    # interleave so all three engines progress together (AI Mode is the slowest)
    todo.sort(key=lambda t: (t[1], list(ENGINES).index(t[0])))
    return todo


def main(rounds=3):
    prompts = load_prompts()
    pmap = {p["n"]: p for p in prompts}
    state = load_state()
    for rnd in range(1, rounds + 1):
        keys, succ = done_keys()
        todo = plan(prompts, keys, state)
        if not todo and not any(j["status"] == "running" for j in state["jobs"]):
            break
        log(f"bd: round {rnd}: {succ} usable answers so far, {sum(len(t[2]) for t in todo)} rows to send in {len(todo)} batches")
        while todo or any(j["status"] == "running" for j in state["jobs"]):
            running = [j for j in state["jobs"] if j["status"] == "running"]
            while todo and len(running) < MAX_INFLIGHT:
                _, succ = done_keys() if len(state["jobs"]) % 10 == 0 else (None, succ)
                if succ >= MAX_SUCCESS:
                    log(f"bd: budget guard hit ({succ} usable records), not sending more")
                    todo = []
                    break
                engine, run, rows = todo.pop(0)
                sid = trigger(engine, rows)
                if sid:
                    j = {"engine": engine, "run": run, "sid": sid, "status": "running", "rows": len(rows),
                         "indexes": [r["index"] for r in rows], "t0": time.time()}
                    state["jobs"].append(j)
                    running.append(j)
                    save_state(state)
                    log(f"bd: triggered {engine} run {run} ({len(rows)} rows) -> {sid}")
            time.sleep(30)
            print(f"{time.strftime('%H:%M:%S')} tick: polling {len(running)} running jobs, {len(todo)} batches queued", flush=True)
            for j in running:
                st, txt, _ = http("GET", f"{BASE}/progress/{j['sid']}", headers=H, timeout=60)
                status = json.loads(txt).get("status") if st == 200 else None
                if status == "ready":
                    recs = download(j, pmap)
                    if recs is None:
                        continue  # retry download next tick
                    append_jsonl_gz(os.path.join(DATA, f"bd_{j['engine']}.jsonl.gz"), recs)
                    ok = sum(r["usable"] for r in recs)
                    j.update(status="done", got=len(recs), usable=ok, mins=round((time.time() - j["t0"]) / 60, 1))
                    save_state(state)
                    log(f"bd: {j['engine']} run {j['run']} {j['sid']} ready: {ok}/{j['rows']} usable in {j['mins']} min")
                elif status == "failed" or (time.time() - j["t0"] > 4 * 3600):
                    j.update(status="failed")
                    save_state(state)
                    log(f"bd: {j['engine']} run {j['run']} {j['sid']} {status or 'timed out after 4h'}")
    keys, succ = done_keys()
    log(f"bd: finished: {succ} usable website answers (planned {len(prompts) * RUNS * len(ENGINES)})")


if __name__ == "__main__":
    main()
