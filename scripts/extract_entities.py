"""Extract the brands/companies/products named in every collected answer (all channels).
Gemini (Vertex, Google credits) with a JSON schema, temperature 0. URLs are stripped first so a
cited website only counts when the answer text itself names it. Resumable: data/entities.jsonl.gz."""
import os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, append_jsonl_gz, gemini, gemini_text, log, read_jsonl_gz
import json

OUT = os.path.join(DATA, "entities.jsonl.gz")
MODEL = "gemini-3.6-flash"  # 3.5-flash-lite ran out of quota mid-run; one extractor for every answer
PER_CALL = 10  # was 5 for the first ~15.5k answers; raised to halve calls under Vertex 429s
SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "id": {"type": "INTEGER"}, "brands": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["id", "brands"]}}
PROMPT = """For EACH numbered AI answer below, list every company, brand, product, service, app, tool, store, organisation or named business that the answer mentions, in order of first mention, each once.
Use the name as written (short form, no legal suffix needed). Do NOT include websites that only appear as a citation/source label. Do NOT include: people, countries, cities or regions on their own, generic categories ("supermarkets", "banks"), or the AI assistant itself.
Return a JSON array with one object per answer: {"id": <answer number>, "brands": [...]} (empty list if none). Judge each answer on its own.
"""


def answer_of(r):
    return (r.get("answer_text_markdown") or r.get("answer_text") or "")


def _link(m):
    text = m.group(1).strip()
    # a link whose label is a bare domain/source label ("lsm.lv", "LA.LV +1") is a citation pill: drop it
    return "" if re.fullmatch(r"[\w.\-]+\.[a-z]{2,}(\s*\+\d+)?", text, re.I) else text


def clean(t):
    t = re.sub(r"\(\s*\[[^\]]*\]\([^)]*\)\s*\)", "", t)  # "([source](url))" inline citations
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", _link, t)       # [label](url) -> label unless it is a domain
    t = re.sub(r"https?://\S+", "", t)                   # bare URLs
    t = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", t)            # [1] style refs
    return t[:8000]


def all_answers():
    rows = []
    for name in ("bd_chatgpt", "bd_gemini", "bd_google", "api_chatgpt", "api_gemini"):
        seen = set()
        for r in read_jsonl_gz(os.path.join(DATA, f"{name}.jsonl.gz")):
            if not r.get("usable"):
                continue
            key = (r["engine"], int(r["n"]), int(r.get("run") or 1))
            if key in seen:
                continue
            seen.add(key)
            rows.append((key, answer_of(r)))
    return rows


def one(group):
    body = PROMPT + "".join(f"\n=== ANSWER {i} ===\n{clean(t)[:6000]}\n" for i, (_, t) in enumerate(group))
    for attempt in range(8):
        st, resp = gemini(body, model=MODEL, json_schema=SCHEMA, temperature=0, timeout=240)
        if st == 200:
            try:
                res = {x["id"]: x.get("brands", []) for x in json.loads(gemini_text(resp))}
                if set(res) >= set(range(len(group))):
                    append_jsonl_gz(OUT, [{"engine": k[0], "n": k[1], "run": k[2], "brands": res[i]}
                                          for i, (k, _) in enumerate(group)])
                    return len(group)
            except Exception:
                pass
        time.sleep(min(60, 2 * 2 ** attempt))
    return 0


def main():
    done = {(r["engine"], r["n"], r["run"]) for r in read_jsonl_gz(OUT)}
    todo = [x for x in all_answers() if x[0] not in done]
    log(f"entities: {len(done)} done, {len(todo)} answers to extract with {MODEL}")
    groups = [todo[i:i + PER_CALL] for i in range(0, len(todo), PER_CALL)]
    ok = 0
    with ThreadPoolExecutor(8) as ex:
        for i, res in enumerate(ex.map(one, groups), 1):
            ok += res
            if i % 200 == 0:
                log(f"entities: {i * PER_CALL}/{len(todo)} sent, {ok} extracted")
    log(f"entities: finished, {ok}/{len(todo)} extracted this pass")


if __name__ == "__main__":
    main()
