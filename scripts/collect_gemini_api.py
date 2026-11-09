"""Gemini API answers (Vertex AI, Google Search grounding), 1 run per prompt.
Same text the Gemini app gets through Bright Data (prompt + source nudge), same model family (Flash-Lite).
Resumable: skips prompts already answered in data/api_gemini.jsonl.gz."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, GEMINI_MODEL, NUDGE, append_jsonl_gz, gemini, gemini_text, load_prompts, log, read_jsonl_gz

OUT = os.path.join(DATA, "api_gemini.jsonl.gz")


def one(p):
    text = f"{p['text']} {NUDGE}"
    last = None
    for attempt in range(6):
        t0 = time.time()
        st, resp = gemini(text, tools=[{"googleSearch": {}}])
        if st == 200 and isinstance(resp, dict) and resp.get("candidates"):
            c = resp["candidates"][0]
            g = c.get("groundingMetadata", {}) or {}
            ans = gemini_text(resp)
            row = {"engine": "gemini_api", "n": p["n"], "run": 1, "pid": p["pid"], "model": resp.get("modelVersion"),
                   "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ms": int((time.time() - t0) * 1000),
                   "usable": bool(ans.strip()), "answer_text": ans, "finish_reason": c.get("finishReason"),
                   "web_search_queries": g.get("webSearchQueries") or [],
                   "sources": [{"title": (x.get("web") or {}).get("title"), "uri": (x.get("web") or {}).get("uri"),
                                "domain": (x.get("web") or {}).get("domain")} for x in g.get("groundingChunks", []) or []],
                   "supports": len(g.get("groundingSupports", []) or []),
                   "usage": resp.get("usageMetadata")}
            append_jsonl_gz(OUT, [row])
            return True
        last = (st, str(resp)[:200])
        time.sleep(min(60, 5 * 2 ** attempt))
    log(f"gemini_api: gave up on {p['pid']}: {last}")
    append_jsonl_gz(OUT, [{"engine": "gemini_api", "n": p["n"], "run": 1, "pid": p["pid"], "usable": False,
                           "error": str(last)}])
    return False


def main():
    prompts = load_prompts()
    done = {r["n"] for r in read_jsonl_gz(OUT) if r.get("usable")}
    todo = [p for p in prompts if p["n"] not in done]
    log(f"gemini_api: model {GEMINI_MODEL}, {len(done)} done, {len(todo)} to go")
    ok = 0
    with ThreadPoolExecutor(8) as ex:
        for i, res in enumerate(ex.map(one, todo), 1):
            ok += res
            if i % 100 == 0:
                log(f"gemini_api: {i}/{len(todo)} processed, {ok} usable")
    log(f"gemini_api: finished, {ok}/{len(todo)} usable this pass")


if __name__ == "__main__":
    main()
