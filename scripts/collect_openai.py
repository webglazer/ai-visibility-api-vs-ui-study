"""ChatGPT API answers: OpenAI Responses API, web_search tool forced (tool_choice=required),
default search context. Model: gpt-5.6-luna. Planned model was `chat-latest` (the model behind
ChatGPT), but this OpenAI org's usage tier caps chat-latest at 10,000 tokens/min and one web-search
answer needs ~11,500, so every call was rejected ("Request too large"), even with low search context
and capped output. gpt-5.6-luna has a 60,000 tokens/min cap and answered with search in testing.
Prompts go in a seeded random order, one at a time, until the account balance runs out
(insufficient_quota). Usage is logged per call so cost per prompt can be computed afterwards. Resumable."""
import json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, append_jsonl_gz, http, load_prompts, log, read_jsonl_gz, secret

OUT = os.path.join(DATA, "api_chatgpt.jsonl.gz")
MODEL = "gpt-5.4-mini"  # gpt-5.6-luna hit a 50 requests/day cap after 9 answers; see run-log.md
KEY = secret("openai.key")
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def body(p):
    return {"model": MODEL, "input": p["text"],
            "tools": [{"type": "web_search"}], "tool_choice": "required",
            "include": ["web_search_call.action.sources"]}


def parse(p, d, ms):
    text, ann, searches = "", [], []
    for o in d.get("output", []):
        if o.get("type") == "web_search_call":
            a = o.get("action") or {}
            searches.append({"query": a.get("query"), "queries": a.get("queries"),
                             "sources": [s.get("url") for s in (a.get("sources") or [])]})
        elif o.get("type") == "message":
            for c in o.get("content", []):
                if c.get("type") == "output_text":
                    text += c.get("text", "")
                    ann += [{"url": a.get("url"), "title": a.get("title")} for a in c.get("annotations", [])
                            if a.get("type") == "url_citation"]
    return {"engine": "chatgpt_api", "n": p["n"], "run": 1, "pid": p["pid"], "model": d.get("model"),
            "response_id": d.get("id"), "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ms": ms, "usable": bool(text.strip()), "answer_text": text, "citations": ann,
            "web_search_calls": searches, "usage": d.get("usage")}


def main():
    prompts = load_prompts()
    random.Random(20261105).shuffle(prompts)
    done = {r["n"] for r in read_jsonl_gz(OUT) if r.get("usable") or r.get("skip")}
    todo = [p for p in prompts if p["n"] not in done]
    log(f"chatgpt_api: model {MODEL}, {len(done)} done, {len(todo)} to go (runs until balance is used up)")
    ok = 0
    for i, p in enumerate(todo, 1):
        for attempt in range(12):
            t0 = time.time()
            st, txt, hdr = http("POST", "https://api.openai.com/v1/responses", body(p), H, timeout=300)
            ms = int((time.time() - t0) * 1000)
            try:
                d = json.loads(txt)
            except Exception:
                d = {}
            if st == 200 and not d.get("error"):
                row = parse(p, d, ms)
                append_jsonl_gz(OUT, [row])
                ok += row["usable"]
                break
            err = (d.get("error") or {}) if isinstance(d, dict) else {}
            code, msg = err.get("code"), err.get("message", txt[:200])
            if "requests per day" in str(msg):
                log(f"chatgpt_api: hit the account's requests-per-day cap after {ok} answers this pass. Stopping; rerun tomorrow to continue. {str(msg)[:200]}")
                return
            if code == "insufficient_quota" or "exceeded your current quota" in str(msg):
                log(f"chatgpt_api: balance used up after {ok} answers this pass ({len(done) + ok} total). Stopping.")
                return
            if "Request too large" in str(msg):
                append_jsonl_gz(OUT, [{"engine": "chatgpt_api", "n": p["n"], "pid": p["pid"], "usable": False,
                                       "skip": True, "error": msg[:300]}])
                log(f"chatgpt_api: {p['pid']} too large for the 10k TPM limit, skipped")
                break
            m = re.search(r"try again in ([\d.]+)(ms|s)", str(msg))
            wait = (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)) + 1 if m else min(120, 10 * (attempt + 1))
            time.sleep(wait)
        else:
            log(f"chatgpt_api: gave up on {p['pid']} after 12 attempts: {str(msg)[:160]}")
        if i % 25 == 0:
            log(f"chatgpt_api: {i} processed this pass, {ok} usable")
    log(f"chatgpt_api: all prompts done, {ok} usable this pass")


if __name__ == "__main__":
    main()
