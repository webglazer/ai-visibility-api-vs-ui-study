"""ChatGPT API answers through OpenRouter (added 2026-11-06). The direct OpenAI account hit a
50 requests/day cap per model after 21 answers, so the rest go through OpenRouter's Responses API
with the same model (openai/gpt-5.4-mini), the same OpenAI web_search tool forced, provider pinned to
OpenAI. Tested: OpenRouter returns OpenAI's own web_search_call (ws_ ids) and url_citation
annotations. One difference: OpenRouter rejects include=web_search_call.action.sources, so the full
list of pages searched is not saved; cited URLs are. Same seeded prompt order as collect_openai.py,
appends to the same file. Stops when OpenRouter reports no credit (HTTP 402). Resumable."""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from collect_openai import OUT, parse
from common import append_jsonl_gz, http, load_prompts, log, read_jsonl_gz, secret

MODEL = "openai/gpt-5.4-mini"
URL = "https://openrouter.ai/api/v1/responses"
H = {"Authorization": f"Bearer {secret('openrouter.key')}", "Content-Type": "application/json"}
THREADS = 4  # 8 threads drew many upstream 429s
stop = False
spent = []
n429 = []


def body(p):
    return {"model": MODEL, "input": p["text"], "tools": [{"type": "web_search"}],
            "tool_choice": "required", "provider": {"only": ["openai"]}}


def one(p):
    global stop
    for attempt in range(8):
        if stop:
            return 0
        t0 = time.time()
        st, txt, _ = http("POST", URL, body(p), H, timeout=300)
        ms = int((time.time() - t0) * 1000)
        try:
            d = json.loads(txt.strip())
        except Exception:
            d = {}
        if st == 200 and d.get("status") == "completed":
            row = parse(p, d, ms)
            row["via"] = "openrouter"
            append_jsonl_gz(OUT, [row])
            spent.append((d.get("usage") or {}).get("cost") or 0)
            return int(row["usable"])
        if st == 402:
            stop = True
            log(f"chatgpt_api/openrouter: out of credit (402): {txt[:160]}")
            return 0
        if st == 429:
            n429.append(1)
            if len(n429) % 25 == 1:  # log the reason, not every retry
                log(f"chatgpt_api/openrouter: 429 #{len(n429)}: {txt.strip()[:300]}")
        time.sleep(min(60, 3 * 2 ** attempt))
    log(f"chatgpt_api/openrouter: gave up on {p['pid']}: {st} {txt[:160]}")
    return 0


def main():
    prompts = load_prompts()
    random.Random(20261105).shuffle(prompts)
    done = {r["n"] for r in read_jsonl_gz(OUT) if r.get("usable")}
    todo = [p for p in prompts if p["n"] not in done]
    log(f"chatgpt_api/openrouter: {MODEL}, {len(done)} done, {len(todo)} to go, {THREADS} threads")
    ok = 0
    with ThreadPoolExecutor(THREADS) as ex:
        for i, res in enumerate(ex.map(one, todo), 1):
            ok += res
            if i % 50 == 0:
                log(f"chatgpt_api/openrouter: {i}/{len(todo)} processed, {ok} usable, ${sum(spent):.2f} spent, "
                    f"${sum(spent) / max(1, len(spent)):.4f}/prompt")
    log(f"chatgpt_api/openrouter: finished pass, {ok} usable, ${sum(spent):.2f} spent")


if __name__ == "__main__":
    main()
